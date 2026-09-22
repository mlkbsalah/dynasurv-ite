"""Treatment mix recommended by the seed ensemble, with its uncertainty.

Reads a `reports/recommendations/<cohort>/<stamp>_<kind>_M<M>/` report written
by `scripts/RecommendEnsemble.py`. Two rows of panels, one column per line:

  (a) 100 % stacked bars: the arms the clinicians gave the holdout patients next
      to the arm the ensemble leads with. Ensemble segments are coloured by the
      leader arm; the solid part is patients with a confident recommendation,
      the hatched part patients whose leader could not be separated from a
      rival (undecided).
  (b) the decision plane: one point per patient, x = pessimistic gap between the
      leader and its closest rival (months of RMST), y = share of members that
      vote for the leader. Dashed lines are the two thresholds; the shaded
      quadrant is where a recommendation is issued. Points beyond the left edge
      are pinned there as "<"; patients whose leader has no rival are pinned at
      the right edge as ">".

Run:  python data_analysis/recommendation_mix.py [report_dir] [--kind bestCALIB]
Figure -> latex/figs/recommendation_mix.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from km_line1_by_year import GRID_C, SEC
from matplotlib.patches import Patch
from paper_style import panel_label, savefig, use_paper_style
from recommendation_style import (
    ARM_COLOR,
    ARM_LABEL,
    ARM_ORDER,
    HATCH,
    HATCH_LINEWIDTH,
    OTHER_COLOR,
    OTHER_LABEL,
    SOLID,
    leader_table,
    newest_report,
)

X_MIN, X_MAX = -3.0, 6.5  # months; gaps outside are pinned at the edges
PIN_RIGHT = 6.3
Y_MIN, Y_MAX = 0.2, 1.04
LABEL_FLOOR = 6.0  # % of the bar; no label inside a thinner segment
JITTER = 0.006  # vote share is quantised to 1/M; spread the stacks a little


def load_report(report_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(report_dir / "recommendations.parquet")
    meta = json.loads((report_dir / "summary.json").read_text())
    return df, meta


def mix_shares(table: pd.DataFrame) -> pd.DataFrame:
    """Per line and bar, the share of patients in each (arm, part) segment."""
    rows = []
    for line, group in table.groupby("line"):
        n = len(group)

        def add(bar: str, arm: str, part: str, k: int) -> None:
            rows.append(
                dict(
                    line=line + 1, bar=bar, arm=arm, part=part, n=k, share=100.0 * k / n
                )
            )

        observed = group.observed_name.where(
            group.observed_name.isin(ARM_ORDER), OTHER_LABEL
        )
        for arm in ARM_ORDER + [OTHER_LABEL]:
            add("clinicians", arm, "observed", int((observed == arm).sum()))
        for arm in ARM_ORDER:
            for part in ("confident", "undecided"):
                add(
                    "ensemble",
                    arm,
                    part,
                    int(((group.leader_name == arm) & (group.decision == part)).sum()),
                )
        add(
            "ensemble",
            OTHER_LABEL,
            "no_support",
            int((group.decision == "no_support").sum()),
        )
    return pd.DataFrame(rows)


def draw_mix(ax, shares: pd.DataFrame) -> None:
    for x, bar in ((0, "clinicians"), (1, "ensemble")):
        bottom = 0.0
        for r in shares[shares.bar == bar].itertuples():
            if r.n == 0:
                continue
            color = OTHER_COLOR if r.arm == OTHER_LABEL else ARM_COLOR[r.arm]
            style = HATCH if r.part == "undecided" else SOLID
            ax.bar(x, r.share, bottom=bottom, width=0.62, facecolor=color, **style)
            if r.share >= LABEL_FLOOR:
                ax.text(
                    x,
                    bottom + r.share / 2,
                    f"{r.share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white",
                )
            bottom += r.share
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["clinicians", "ensemble"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 100)


def draw_plane(
    ax, table: pd.DataFrame, margin: float, p_min: float, rng
) -> tuple[int, int]:
    """Scatter of (gap, vote share); returns how many points were pinned left/right."""
    sup = table[table.leader_idx >= 0]
    gap = sup.gap.to_numpy(dtype=float)
    y = sup.p_best_leader.to_numpy(dtype=float) + rng.uniform(-JITTER, JITTER, len(sup))
    colors = np.array([ARM_COLOR.get(a, OTHER_COLOR) for a in sup.leader_name])

    ax.fill_between([margin, X_MAX], p_min, Y_MAX, color=GRID_C, linewidth=0, zorder=0)
    ax.axvline(margin, ls="--", color=SEC, lw=0.8)
    ax.axhline(p_min, ls="--", color=SEC, lw=0.8)

    inner = np.isfinite(gap) & (gap >= X_MIN)
    ax.scatter(
        gap[inner],
        y[inner],
        c=colors[inner],
        s=6,
        alpha=0.35,
        linewidths=0,
        rasterized=True,
    )
    left = np.isfinite(gap) & (gap < X_MIN)
    right = ~np.isfinite(gap)
    for sel, x, marker in ((left, X_MIN, "<"), (right, PIN_RIGHT, ">")):
        if sel.any():
            ax.scatter(
                np.full(sel.sum(), x),
                y[sel],
                c=colors[sel],
                marker=marker,
                s=12,
                alpha=0.8,
                linewidths=0,
            )
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([-2, 0, 2, 4, 6])
    return int(left.sum()), int(right.sum())


def make_mix_mpl(report_dir: Path):
    """Paper figure. Returns (pdf path, shares table, pinned counts per line)."""
    df, meta = load_report(report_dir)
    table = leader_table(df)
    shares = mix_shares(table)
    margin = float(meta["recommendation"]["margin_months"])
    p_min = float(meta["recommendation"]["p_best_min"])

    use_paper_style()
    plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH
    lines = sorted(table.line.unique())
    fig, axes = plt.subplots(2, len(lines), figsize=(9.0, 5.6), sharey="row")
    axes = np.asarray(axes).reshape(2, len(lines))
    rng = np.random.default_rng(0)
    pinned = {}
    for j, line in enumerate(lines):
        rows = table[table.line == line]
        draw_mix(axes[0, j], shares[shares.line == line + 1])
        axes[0, j].set_title(f"line {line + 1}  (n = {len(rows):,})")
        pinned[line + 1] = draw_plane(axes[1, j], rows, margin, p_min, rng)
        axes[1, j].set_xlabel("gap to runner-up (months)")
    axes[0, 0].set_ylabel("share of patients (%)")
    axes[1, 0].set_ylabel("leader vote share")
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")

    handles = [Patch(facecolor=ARM_COLOR[a], label=ARM_LABEL[a]) for a in ARM_ORDER]
    handles += [
        Patch(facecolor=OTHER_COLOR, label=OTHER_LABEL),
        Patch(facecolor=SEC, label="confident", **SOLID),
        Patch(facecolor=SEC, label="undecided", **HATCH),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(handles),
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = savefig(fig, "recommendation_mix")
    plt.close(fig)
    return out, shares, pinned, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report_dir", nargs="?", type=Path)
    parser.add_argument("--kind", default="bestCALIB")
    args = parser.parse_args()
    report_dir = args.report_dir or newest_report(args.kind)
    print("report:", report_dir)

    out, shares, pinned, table = make_mix_mpl(report_dir)
    piv = shares.pivot_table(
        index=["line", "bar", "arm"], columns="part", values="share", aggfunc="sum"
    )
    print(piv.round(1).to_string(na_rep="·"))
    other = table[~table.observed_name.isin(ARM_ORDER)]
    print("\n'other' observed arms per line:")
    print(other.groupby(["line", "observed_name"]).size().to_string())
    for line, (left, right) in pinned.items():
        print(
            f"line {line}: {left} patients pinned at gap < {X_MIN:g}, "
            f"{right} with no rival pinned at the right edge"
        )
    print("\nsaved", out)


if __name__ == "__main__":
    main()
