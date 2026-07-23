"""Consecutive repetition of treatment category over the first 4 treatment lines.

For the HR+HER2- cohort, restricts to patients with at least 4 treatment lines,
takes lines 1-4, run-length-encodes each patient's ``T_treatment_category``
sequence, and summarises the consecutive-repeat runs (runs of the same category
in adjacent lines, length >= 2):

  * how many repeat blocks a patient has (0 / 1 / 2), and
  * how long those blocks are (2 / 3 / 4 lines), broken down by category.

Renders an interactive two-panel Plotly figure: hover any bar/segment for the
underlying counts and percentages.

Run:  python data_analysis/treatment_category_repeats.py
Figure -> data_analysis/plots/treatment_category_repeats.html
"""

from collections import Counter
from itertools import groupby
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Build the interactive two-panel figure and write it to an HTML file."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.34, 0.66],
        horizontal_spacing=0.13,
        subplot_titles=(
            "<b>How often patients repeat a category</b>",
            "<b>Which categories are repeated — and for how long</b>",
        ),
    )

    # --- Panel A: repeat blocks per patient ---
    xs = [0, 1, 2]
    counts = [nrep.get(k, 0) for k in xs]
    pct = [100 * c / n_pat for c in counts]
    fig.add_trace(
        go.Bar(
            x=[str(k) for k in xs],
            y=pct,
            marker_color=BLUE,
            width=0.62,
            text=[f"{p:.1f}%" for p in pct],
            textposition="outside",
            textfont=dict(color=INK, size=12),
            customdata=[[c] for c in counts],
            hovertemplate=(
                "<b>%{x} repeat block(s)</b><br>"
                "%{customdata[0]:,} patients<br>"
                "%{y:.1f}% of patients"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Panel B: repeats by category x run length (stacked horizontal) ---
    # ascending by total so the largest category ends up on top
    cats = [c for c, _ in cat_total.most_common()][::-1]
    for length in (2, 3, 4):
        vals = [rep_by_cat.get((c, length), 0) for c in cats]
        cust = [[100 * v / cat_total[c], cat_total[c]] for v, c in zip(vals, cats)]
        fig.add_trace(
            go.Bar(
                y=cats,
                x=vals,
                orientation="h",
                name=f"{length} lines",
                marker_color=LEN_COLORS[length],
                marker_line=dict(color=SURFACE, width=1.2),
                customdata=cust,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "repeat length: %{fullData.name}<br>"
                    "%{x:,} blocks (%{customdata[0]:.1f}% of this category)<br>"
                    "category total: %{customdata[1]:,} blocks"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # category-total labels at the end of each stacked bar
    xmax = max(cat_total.values())
    for c in cats:
        fig.add_annotation(
            x=cat_total[c] + xmax * 0.01,
            y=c,
            text=f"{cat_total[c]:,}",
            xref="x2",
            yref="y2",
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(color=INK, size=11),
        )

    # --- layout ---
    pct_any = 100 * (1 - nrep.get(0, 0) / n_pat)
    fig.update_layout(
        barmode="stack",
        bargap=0.35,
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Helvetica Neue, Helvetica, Arial, sans-serif",
            color=INK,
            size=12,
        ),
        title=dict(
            text=(
                "Consecutive repetition of treatment category over the first 4 lines"
                "<br><span style='font-size:13px;color:" + SEC + "'>"
                f"HR+HER2− cohort · patients with ≥ 4 treatment lines (n = {n_pat:,}) · "
                "a repeat = same category in consecutive lines · "
                f"{pct_any:.0f}% repeat ≥ 1 category · "
                f"{sum(len_pool.values()):,} repeat blocks total</span>"
            ),
            x=0.012,
            xanchor="left",
            font=dict(size=19),
        ),
        legend=dict(
            title_text="repeat length",
            title_font_color=SEC,
            orientation="v",
            x=0.995,
            xanchor="right",
            y=0.02,
            yanchor="bottom",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor=BASE,
            borderwidth=1,
        ),
        margin=dict(l=10, r=20, t=120, b=60),
        height=560,
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=BASE,
            font=dict(color=INK, size=12),
        ),
    )

    # axes
    fig.update_yaxes(
        title_text="% of patients", gridcolor=GRID, zeroline=False, row=1, col=1
    )
    fig.update_xaxes(
        title_text="consecutive-repeat blocks per patient",
        type="category",
        gridcolor=GRID,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="number of consecutive-repeat blocks",
        gridcolor=GRID,
        zeroline=False,
        range=[0, xmax * 1.12],
        row=1,
        col=2,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=cats,
        gridcolor="rgba(0,0,0,0)",
        row=1,
        col=2,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "treatment_category_repeats.html"
    fig.write_html(
        out,
        include_plotlyjs=True,  # self-contained (works offline / in sandboxes)
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
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
