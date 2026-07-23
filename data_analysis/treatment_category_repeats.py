"""Consecutive repetition of treatment category over the first 4 treatment lines.

For the HR+HER2- cohort, restricts to patients with at least 4 treatment lines,
takes lines 1-4, run-length-encodes each patient's ``T_treatment_category``
sequence, and summarises the consecutive-repeat runs (runs of the same category
in adjacent lines, length >= 2):

  * how many repeat blocks a patient has (0 / 1 / 2), and
  * how long those blocks are (2 / 3 / 4 lines), broken down by category.

Run:  python data_analysis/treatment_category_repeats.py
Figure -> data_analysis/plots/treatment_category_repeats.png
"""

from collections import Counter
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
N_LINES = 4

# ---- palette / chrome (dataviz reference palette, light mode) ----
SURFACE, INK, SEC, MUTED, GRID, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE = "#2a78d6"  # categorical slot 1
LEN_COLORS = {2: "#86b6ef", 3: "#2a78d6", 4: "#104281"}  # ordinal blue ramp light->dark


def load_repeats():
    """Return per-patient repeat counts and (category, length) repeat tallies."""
    df = pd.read_parquet(DATA_PATH)

    # drop patients with duplicated line numbers (data anomaly)
    bad = df.groupby("usubjid")["X_line_number"].apply(lambda s: s.duplicated().any())
    df = df[~df["usubjid"].isin(bad[bad].index)].copy()

    # keep patients reaching >= N_LINES, restrict to the first N_LINES lines
    maxline = df.groupby("usubjid")["X_line_number"].max()
    keep = maxline[maxline >= N_LINES].index
    sub = df[df["usubjid"].isin(keep) & (df["X_line_number"] <= N_LINES)]
    sub = sub.sort_values(["usubjid", "X_line_number"])

    per_patient_nrep = []
    rep_by_cat = Counter()  # (category, run_length) -> n blocks
    cat_total = Counter()  # category -> total repeat blocks
    len_pool = Counter()  # run_length -> n blocks
    for _, grp in sub.groupby("usubjid"):
        runs = [(k, sum(1 for _ in g)) for k, g in groupby(grp["T_treatment_category"])]
        reps = [(k, length) for k, length in runs if length >= 2]
        per_patient_nrep.append(len(reps))
        for k, length in reps:
            rep_by_cat[(k, length)] += 1
            cat_total[k] += 1
            len_pool[length] += 1

    return (
        sub["usubjid"].nunique(),
        Counter(per_patient_nrep),
        rep_by_cat,
        cat_total,
        len_pool,
    )


def make_figure(n_pat, nrep, rep_by_cat, cat_total, len_pool):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "text.color": INK,
            "axes.edgecolor": BASE,
            "axes.labelcolor": SEC,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.55]}
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.11, wspace=0.28)

    # --- Panel A: repeat blocks per patient ---
    xs = [0, 1, 2]
    pct = [100 * nrep.get(k, 0) / n_pat for k in xs]
    bars = axA.bar(xs, pct, width=0.62, color=BLUE, zorder=3)
    axA.set_xticks(xs)
    axA.set_xlabel("consecutive-repeat blocks per patient", fontsize=10.5)
    axA.set_ylabel("% of patients", fontsize=10.5)
    axA.set_ylim(0, max(pct) * 1.18)
    for b, p, k in zip(bars, pct, xs):
        axA.text(
            b.get_x() + b.get_width() / 2,
            p + max(pct) * 0.02,
            f"{p:.1f}%\n({nrep.get(k, 0):,})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
            linespacing=1.25,
        )
    axA.set_title(
        "How often patients repeat a category",
        fontsize=12,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    axA.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    axA.set_axisbelow(True)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # --- Panel B: repeats by category x run length ---
    cats = [c for c, _ in cat_total.most_common()][::-1]  # largest on top
    y = list(range(len(cats)))
    left = [0] * len(cats)
    for length in (2, 3, 4):
        vals = [rep_by_cat.get((c, length), 0) for c in cats]
        axB.barh(
            y,
            vals,
            left=left,
            color=LEN_COLORS[length],
            height=0.66,
            edgecolor=SURFACE,
            linewidth=1.4,
            zorder=3,
        )
        left = [base + v for base, v in zip(left, vals)]
    for yi, c in zip(y, cats):
        axB.text(
            cat_total[c] + max(cat_total.values()) * 0.01,
            yi,
            f"{cat_total[c]:,}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    axB.set_yticks(y)
    axB.set_yticklabels(cats, fontsize=9.5, color=INK)
    axB.set_xlabel("number of consecutive-repeat blocks", fontsize=10.5)
    axB.set_xlim(0, max(cat_total.values()) * 1.10)
    axB.set_title(
        "Which categories are repeated — and for how long",
        fontsize=12,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    axB.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    for s in ("top", "right", "left"):
        axB.spines[s].set_visible(False)
    axB.tick_params(axis="y", length=0)

    handles = [
        Patch(facecolor=LEN_COLORS[length], label=f"{length} lines")
        for length in (2, 3, 4)
    ]
    leg = axB.legend(
        handles=handles,
        title="repeat length",
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        title_fontsize=9.5,
        handlelength=1.1,
        borderaxespad=0.4,
    )
    leg.get_title().set_color(SEC)

    fig.suptitle(
        "Consecutive repetition of treatment category over the first 4 lines",
        x=0.055,
        y=0.955,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )
    pct_any = 100 * (1 - nrep.get(0, 0) / n_pat)
    fig.text(
        0.055,
        0.885,
        f"HR+HER2− cohort · patients with ≥ 4 treatment lines "
        f"(n = {n_pat:,}) · a repeat = same category in consecutive lines "
        f"· {pct_any:.0f}% repeat ≥ 1 category · "
        f"{sum(len_pool.values()):,} repeat blocks total",
        ha="left",
        fontsize=10,
        color=SEC,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "treatment_category_repeats.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    return out


def main():
    n_pat, nrep, rep_by_cat, cat_total, len_pool = load_repeats()
    print(f"patients with >= {N_LINES} lines: {n_pat:,}")
    print(f"repeat >= 1 category: {100 * (1 - nrep.get(0, 0) / n_pat):.1f}%")
    print("repeat blocks per patient:", dict(sorted(nrep.items())))
    print("repeat-block lengths:", dict(sorted(len_pool.items())))
    out = make_figure(n_pat, nrep, rep_by_cat, cat_total, len_pool)
    print("saved", out)


if __name__ == "__main__":
    main()
