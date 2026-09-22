"""Robustness of the ensemble recommendation to the RMST horizon.

Reads a `reports/recommendations/<cohort>/<stamp>_<kind>_M<M>_horizons/` sweep
written by `scripts/HorizonRobustness.py`. Two rows of panels, one column per
line:

  (a) the ensemble's leader mix at each scored horizon, stacked to the share of
      patients; solid = confident recommendation, hatched = undecided.
  (b) agreement with the configured reference horizon: the share of patients
      whose leader is the same (solid), the mean Jaccard overlap of the
      equivalence sets (dashed), and the share of holdout records observed to
      that horizon (dotted).

The dotted vertical line is the reference horizon of that line; the grey band
starts at the line's longest observed follow-up, beyond which every RMST is
model extrapolation.

Run:  python data_analysis/recommendation_horizon.py [report_dir] [--kind bestCALIB]
Figure -> latex/figs/recommendation_horizon.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from km_line1_by_year import GRID_C, INK, MUTED, SEC
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paper_style import panel_label, savefig, use_paper_style
from recommendation_style import (
    ARM_COLOR,
    ARM_LABEL,
    ARM_ORDER,
    HATCH,
    HATCH_LINEWIDTH,
    OTHER_COLOR,
    SOLID,
    newest_report,
)

X_MIN, X_MAX = 1.5, 37.5  # months
BAR_WIDTH = 2.2  # on the default 3-month grid


def records_frame(meta: dict) -> pd.DataFrame:
    rows = []
    for key, block in meta["per_line"].items():
        line = int(key.split("_")[1])
        for r in block["records"]:
            rows.append(
                {
                    "line": line,
                    "horizon": r["horizon"],
                    "is_reference": r["is_reference"],
                    "n": r["n"],
                    "coverage": r["coverage"],
                    "confident": r["rates"]["confident"],
                    "undecided": r["rates"]["undecided"],
                    "no_support": r["rates"]["no_support"],
                    "set_size": r["set_size_mean"],
                    "leader_same": r["leader_same"],
                    "set_jaccard": r["set_jaccard"],
                    "mix_confident": r["leader_mix_confident"],
                    "mix_undecided": r["leader_mix_undecided"],
                }
            )
    return pd.DataFrame(rows).sort_values(["line", "is_reference", "horizon"])


def draw_mix(ax, recs: pd.DataFrame) -> None:
    for r in recs.itertuples():
        bottom = 0.0
        for arm in ARM_ORDER:
            for mix, style in ((r.mix_confident, SOLID), (r.mix_undecided, HATCH)):
                k = mix.get(arm, 0)
                if k == 0:
                    continue
                share = k / r.n
                ax.bar(
                    r.horizon,
                    share,
                    bottom=bottom,
                    width=BAR_WIDTH,
                    facecolor=ARM_COLOR[arm],
                    **style,
                )
                bottom += share
        if r.no_support > 0:
            ax.bar(
                r.horizon,
                r.no_support,
                bottom=bottom,
                width=BAR_WIDTH,
                facecolor=OTHER_COLOR,
                **SOLID,
            )
    ax.set_ylim(0, 1)


def draw_agreement(ax, recs: pd.DataFrame) -> None:
    ax.plot(recs.horizon, recs.leader_same, color=INK, marker="o", ms=3, lw=1.2)
    ax.plot(
        recs.horizon, recs.set_jaccard, color=SEC, ls="--", marker="s", ms=2.6, lw=1.2
    )
    ax.plot(recs.horizon, recs.coverage, color=MUTED, ls=":", lw=1.2)
    ax.set_ylim(0, 1.02)


def make_horizon_mpl(report_dir: Path):
    """Paper figure. Returns (pdf path, per-line-and-horizon table)."""
    meta = json.loads((report_dir / "horizon_summary.json").read_text())
    recs = records_frame(meta)
    swept = recs[~recs.is_reference]
    lines = sorted(recs.line.unique())

    use_paper_style()
    plt.rcParams["hatch.linewidth"] = HATCH_LINEWIDTH
    fig, axes = plt.subplots(
        2, len(lines), figsize=(9.0, 5.4), sharex=True, sharey="row"
    )
    axes = np.asarray(axes).reshape(2, len(lines))
    for j, line in enumerate(lines):
        block = meta["per_line"][f"line_{line}"]
        rows = swept[swept.line == line]
        draw_mix(axes[0, j], rows)
        draw_agreement(axes[1, j], rows)
        for ax in axes[:, j]:
            ax.axvline(block["reference_horizon"], ls=":", color=INK, lw=0.9)
            if block["max_observed_time"] < X_MAX:
                ax.axvspan(
                    block["max_observed_time"],
                    X_MAX,
                    color=GRID_C,
                    alpha=0.6,
                    linewidth=0,
                    zorder=0,
                )
            ax.set_xlim(X_MIN, X_MAX)
        axes[0, j].set_title(f"line {line}  (n = {block['n']:,})")
        axes[1, j].set_xlabel("RMST horizon (months)")
        axes[1, j].set_xticks([6, 12, 18, 24, 30, 36])
    axes[0, 0].set_ylabel("share of patients")
    axes[1, 0].set_ylabel("agreement with reference")
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")

    handles = [Patch(facecolor=ARM_COLOR[a], label=ARM_LABEL[a]) for a in ARM_ORDER]
    handles += [
        Patch(facecolor=SEC, label="confident", **SOLID),
        Patch(facecolor=SEC, label="undecided", **HATCH),
        Line2D([], [], color=INK, marker="o", ms=3, lw=1.2, label="same leader"),
        Line2D(
            [],
            [],
            color=SEC,
            ls="--",
            marker="s",
            ms=2.6,
            lw=1.2,
            label="set overlap (Jaccard)",
        ),
        Line2D([], [], color=MUTED, ls=":", lw=1.2, label="observed to horizon"),
        Patch(facecolor=GRID_C, alpha=0.6, label="beyond last observed follow-up"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        columnspacing=1.2,
        handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    out = savefig(fig, "recommendation_horizon")
    plt.close(fig)
    table = recs[
        [
            "line",
            "horizon",
            "is_reference",
            "coverage",
            "confident",
            "undecided",
            "set_size",
            "leader_same",
            "set_jaccard",
        ]
    ].reset_index(drop=True)
    return out, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("report_dir", nargs="?", type=Path)
    parser.add_argument("--kind", default="bestCALIB")
    args = parser.parse_args()
    report_dir = args.report_dir or newest_report(args.kind, horizons=True)
    print("report:", report_dir)
    out, table = make_horizon_mpl(report_dir)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
