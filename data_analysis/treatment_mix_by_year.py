"""Evolution of the treatment-category mix over calendar years, lines 1-4.

For each treatment line, the share of each `T_treatment_category` among the
lines that started in a given year (100% stacked bars). Supports Finding 2 of
the paper: ET+ANTI-CDK enters from 2014 and dominates line 1 by 2022, while
CT+ANTI-ANGIO all but disappears after 2016. 2023 is excluded (20 line-1
patients). OTHER and NO TREATMENT are left out, so shares are over the
remaining categories.

Run:  python data_analysis/treatment_mix_by_year.py
Figures -> data_analysis/plots/treatment_mix_by_year.html
           latex/figs/treatment_mix_by_year.png  (line 1 only)
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
FIGS_DIR = REPO_ROOT / "latex" / "figs"
YEARS = (2008, 2022)
LINES = (1, 2, 3, 4)
WINDOW_START = 2018  # first line-1 year kept in the analysis cohort

# stacking order, bottom to top: the two period-bound categories sit on the axis
CATEGORIES = [
    "ET+ANTI-CDK wo CT",
    "CT+ANTI-ANGIO",
    "ET alone",
    "ET+TT",
    "MONOCT std alone",
    "POLYCT alone",
    "CT+TT",
    "CT+ANTI-HER2",
    "CT+IT",
]
# not well-defined interventions (see the Identifiability section); dropped
# before computing shares, so each bar is 100% of the remaining categories
EXCLUDED = ("OTHER", "NO TREATMENT")
COLORS = dict(zip(CATEGORIES, px.colors.qualitative.Safe))


def load():
    df = pd.read_parquet(
        DATA_PATH,
        columns=["usubjid", "X_line_number", "T_treatment_category", "line_start_date"],
    )
    df = df.drop_duplicates(["usubjid", "X_line_number"])
    df = df[~df["T_treatment_category"].isin(EXCLUDED)]
    df["year"] = pd.to_datetime(df["line_start_date"]).dt.year
    return df[df["year"].between(*YEARS) & df["X_line_number"].isin(LINES)]


def counts_by_year(df, line):
    sub = df[df["X_line_number"] == line]
    return pd.crosstab(sub["year"], sub["T_treatment_category"]).reindex(
        columns=CATEGORIES, fill_value=0
    )


def add_bars(fig, counts, row=None, col=None, showlegend=True):
    share = counts.div(counts.sum(axis=1), axis=0) * 100
    for cat in CATEGORIES:
        fig.add_trace(
            go.Bar(
                x=counts.index,
                y=share[cat],
                name=cat,
                legendgroup=cat,
                showlegend=showlegend,
                marker_color=COLORS[cat],
                customdata=counts[cat],
                hovertemplate=(
                    f"<b>{cat}</b><br>%{{x}}: %{{y:.1f}}% (%{{customdata:,}} patients)"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )


def mark_window(fig, row=None, col=None):
    fig.add_vline(
        x=WINDOW_START - 0.5,
        line=dict(color="black", width=1.5, dash="dot"),
        row=row,
        col=col,
    )


def make_all_lines(df):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[f"Line {k}" for k in LINES],
        horizontal_spacing=0.07,
        vertical_spacing=0.12,
    )
    for i, line in enumerate(LINES):
        row, col = i // 2 + 1, i % 2 + 1
        add_bars(fig, counts_by_year(df, line), row, col, showlegend=i == 0)
        mark_window(fig, row, col)
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        title="Treatment category mix by year of line start (HR+HER2−)",
        legend_title_text="treatment category",
        height=850,
        width=1300,
    )
    fig.update_xaxes(dtick=2, title_text="year of line start")
    fig.update_yaxes(range=[0, 100], ticksuffix="%", title_text="share of lines")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "treatment_mix_by_year.html"
    fig.write_html(out, include_plotlyjs=True)
    return out


def make_line1_png(df):
    fig = go.Figure()
    add_bars(fig, counts_by_year(df, 1))
    mark_window(fig)
    fig.add_annotation(
        x=WINDOW_START - 0.5,
        y=102,
        text="cohort window",
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
    )
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        legend_title_text="treatment category",
        width=1000,
        height=520,
        margin=dict(l=60, r=20, t=40, b=50),
    )
    fig.update_xaxes(dtick=1, title_text="year of line-1 start")
    fig.update_yaxes(
        range=[0, 106],
        tickvals=list(range(0, 101, 20)),
        ticksuffix="%",
        title_text="share of first-line treatments",
    )
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "treatment_mix_by_year.png"
    fig.write_image(out, scale=2)
    return out


def main():
    df = load()
    pd.set_option("display.width", 250)
    for line in LINES:
        counts = counts_by_year(df, line)
        share = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1)
        share["n"] = counts.sum(axis=1)
        print(f"\nline {line} — row % by year of line start")
        print(share.to_string())
    print("\nsaved", make_all_lines(df))
    print("saved", make_line1_png(df))


if __name__ == "__main__":
    main()
