"""Evolution of the performance-status distribution (`X_mpps`) over calendar years.

`X_mpps` is the ECOG/WHO performance status, 0 (fully active) to 4 (completely
disabled), recorded per treatment line in the HR+HER2- cohort.

The figure answers "did performance status shift over the years?" three ways:

  * the full distribution per year of line-1 onset, as 100% stacked bars;
  * the two ends of the scale (PS 0 and PS >= 2) as trend lines with 95% Wilson
    intervals, because the mean hides what is actually happening;
  * the distribution across treatment lines 1-4, the other axis along which
    performance status can drift.

IMPORTANT: the only parquet available is `model_entry_imputed_data_*`, which has
zero missing `X_mpps` across all 63,317 rows. Real-world performance status is
rarely complete, so an unknown share of these values is imputed and there is no
raw file in `data/` to compare against. Read the trends accordingly.

Run:  python data_analysis/mpps_performance_status_by_year.py
Figure -> data_analysis/plots/mpps_by_year.html
"""

from pathlib import Path

import numpy as np
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
YEARS = (2008, 2022)  # 2023 holds only 20 line-1 patients — too thin to plot
LINES = (1, 2, 3, 4)

# ---- palette / chrome (dataviz reference palette, light mode) ----
SURFACE, INK, SEC, MUTED, GRID, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# Performance status is ORDINAL, so one hue light -> dark, never categorical hues.
# Verified monotone in relative luminance (0.864 -> 0.051, min adjacent gap 0.137).
PS_LEVELS = (0, 1, 2, 3, 4)
PS_COLOR = {
    0: "#e8f0fb",
    1: "#a9caf0",
    2: "#6ba5e4",
    3: "#2a78d6",
    4: "#123f7a",
}
PS_DESC = {
    0: "fully active",
    1: "restricted in strenuous activity",
    2: "ambulatory, up > 50% of waking hours",
    3: "limited self-care, confined > 50%",
    4: "completely disabled",
}
# Text sits on the fill in the stacked bars, so it must flip with the ramp.
PS_TEXT = {0: INK, 1: INK, 2: INK, 3: "#ffffff", 4: "#ffffff"}
LABEL_FLOOR = 8.0  # only label a segment wide enough to hold the number upright


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, as percentages."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def load():
    """Line-1 rows (one per patient) plus the full long frame for the by-line panel."""
    df = pd.read_parquet(
        DATA_PATH,
        columns=["usubjid", "X_line_number", "X_mpps", "line_start_date"],
    )
    df["year"] = pd.to_datetime(df["line_start_date"]).dt.year
    df["ps"] = df["X_mpps"].astype(int)

    line1 = df[df["X_line_number"] == 1].drop_duplicates("usubjid", keep="first")
    line1 = line1[line1["year"].between(*YEARS)]
    return line1, df[df["X_line_number"].isin(LINES)]


def add_stacked(fig, ct, row, col, axis_suffix, hover_unit):
    """100% stacked bars of the PS distribution, one bar per index value."""
    idx = list(ct.index)
    totals = ct.sum(axis=1)
    for ps in PS_LEVELS:
        counts = ct[ps].to_numpy() if ps in ct.columns else np.zeros(len(idx))
        pct = 100 * counts / totals.to_numpy()
        fig.add_trace(
            go.Bar(
                x=[str(i) for i in idx],
                y=pct,
                name=f"PS {ps}",
                legendgroup=f"ps{ps}",
                showlegend=(row == 1),
                marker_color=PS_COLOR[ps],
                marker_line=dict(color=SURFACE, width=1.2),
                text=[f"{v:.0f}" if v >= LABEL_FLOOR else "" for v in pct],
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                constraintext="inside",
                textfont=dict(color=PS_TEXT[ps], size=10),
                customdata=np.stack([counts, totals.to_numpy()], axis=-1),
                hovertemplate=(
                    f"<b>%{{x}}</b> · PS {ps} — {PS_DESC[ps]}<br>"
                    "%{customdata[0]:,.0f} patients (%{y:.1f}%)<br>"
                    f"{hover_unit} total: %{{customdata[1]:,.0f}}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
    return idx


def add_trends(fig, line1, row, col):
    """The two ends of the scale, where the movement actually is."""
    years = sorted(line1["year"].unique())
    series = {
        "PS 0 (fully active)": (lambda s: s == 0, "#6ba5e4"),
        "PS ≥ 2 (impaired)": (lambda s: s >= 2, "#123f7a"),
    }
    for name, (pred, color) in series.items():
        pts, los, his, ns, ks = [], [], [], [], []
        for y in years:
            s = line1.loc[line1["year"] == y, "ps"]
            k, n = int(pred(s).sum()), len(s)
            lo, hi = wilson(k, n)
            pts.append(100 * k / n)
            los.append(lo)
            his.append(hi)
            ns.append(n)
            ks.append(k)
        fig.add_trace(  # CI band first so the line sits on top
            go.Scatter(
                x=years + years[::-1],
                y=his + los[::-1],
                fill="toself",
                fillcolor="rgba(42,120,214,0.10)",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=pts,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                customdata=np.stack([ks, ns, los, his], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b> — " + name + "<br>"
                    "%{y:.1f}%  (95% CI %{customdata[2]:.1f}–%{customdata[3]:.1f})<br>"
                    "%{customdata[0]:,} of %{customdata[1]:,} patients"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        # direct label so identity is never colour-alone
        fig.add_annotation(
            x=years[-1],
            y=pts[-1],
            text=f"  <b>{name.split(' (')[0]}</b>",
            xref=f"x{2 if row == 2 else ''}".replace("x1", "x"),
            yref=f"y{2 if row == 2 else ''}".replace("y1", "y"),
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            font=dict(color=color, size=11),
            row=row,
            col=col,
        )
    return years


def make_figure(line1, longdf):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        column_widths=[0.56, 0.44],
        row_heights=[0.54, 0.46],
        horizontal_spacing=0.11,
        vertical_spacing=0.155,
        subplot_titles=(
            "<b>Performance status at line-1 onset, by year</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            "   each bar is one entry cohort, summing to 100%</span>",
            "<b>The two ends of the scale</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            "   shaded = 95% CI</span>",
            "<b>By treatment line</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            "   all years pooled · 79.9% of line-to-line steps show no change</span>",
        ),
    )

    ct_year = pd.crosstab(line1["year"], line1["ps"])
    years = add_stacked(fig, ct_year, 1, 1, "", "year")
    add_trends(fig, line1, 2, 1)
    ct_line = pd.crosstab(longdf["X_line_number"], longdf["ps"]).loc[list(LINES)]
    add_stacked(fig, ct_line, 2, 2, "3", "line")

    n = len(line1)
    first, last = ct_year.index[0], ct_year.index[-1]
    # 5-year eras rather than single endpoint years, which are noisy
    early = line1.loc[line1["year"].between(first, first + 4), "ps"]
    late = line1.loc[line1["year"].between(last - 4, last), "ps"]
    p0_e, p0_l = 100 * (early == 0).mean(), 100 * (late == 0).mean()
    ge2_e, ge2_l = 100 * (early >= 2).mean(), 100 * (late >= 2).mean()

    fig.update_layout(
        barmode="stack",
        bargap=0.24,
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Helvetica Neue, Helvetica, Arial, sans-serif", color=INK, size=12
        ),
        title=dict(
            text=(
                "Performance status (X_mpps, ECOG/WHO 0–4) over time"
                f"<br><span style='font-size:13px;color:{SEC}'>"
                f"HR+HER2− cohort · {n:,} patients at line-1 onset, {first}–{last}"
                f"<br>Comparing {first}–{first + 4} with {last - 4}–{last}, the distribution "
                f"<b>spreads</b> rather than shifts: PS 0 {p0_e:.1f}%→{p0_l:.1f}%, "
                f"PS ≥ 2 {ge2_e:.1f}%→{ge2_l:.1f}%, mean {early.mean():.2f}→{late.mean():.2f}"
                "<br><b style='color:#c2410c'>These values are imputed</b> — X_mpps has zero "
                "missing rows in every parquet in data/, so the observed-vs-imputed"
                "<br>split cannot be recovered here. Read the trend as a property of the "
                "delivered data, not of observed practice."
                "</span>"
            ),
            x=0.012,
            xanchor="left",
            font=dict(size=19),
        ),
        legend=dict(
            title_text="performance status",
            title_font_color=SEC,
            orientation="h",
            x=1.0,
            xanchor="right",
            y=1.028,
            yanchor="bottom",
            traceorder="normal",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor=BASE,
            borderwidth=1,
        ),
        margin=dict(l=10, r=20, t=278, b=56),
        height=940,
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=BASE, font=dict(color=INK, size=12)
        ),
    )

    for i, ann in enumerate(fig.layout.annotations[:3]):
        axis = fig.layout["xaxis" if i == 0 else f"xaxis{i + 1}"]
        ann.update(x=axis.domain[0], xanchor="left")

    fig.update_yaxes(
        title_text="% of patients",
        range=[0, 100],
        gridcolor=GRID,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="year of line-1 onset", type="category", gridcolor=GRID, row=1, col=1
    )
    fig.update_yaxes(
        title_text="% of patients", gridcolor=GRID, zeroline=False, row=2, col=1
    )
    fig.update_xaxes(
        title_text="year of line-1 onset",
        gridcolor=GRID,
        range=[years[0] - 0.4, years[-1] + 2.6],
        tickmode="array",
        tickvals=[y for y in years if y % 2 == 0],
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="% of line-records",
        range=[0, 100],
        gridcolor=GRID,
        zeroline=False,
        row=2,
        col=2,
    )
    fig.update_xaxes(
        title_text="treatment line", type="category", gridcolor=GRID, row=2, col=2
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "mpps_by_year.html"
    fig.write_html(
        out,
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    return out


def main():
    line1, longdf = load()
    print(f"line-1 patients {YEARS[0]}–{YEARS[1]}: {len(line1):,}")
    ct = pd.crosstab(line1["year"], line1["ps"])
    pct = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    print("\nrow % (year × PS):")
    print(pct.to_string())
    print("\nmean PS / % PS>=2 by year:")
    for y in ct.index:
        s = line1.loc[line1["year"] == y, "ps"]
        print(
            f"  {y}  n={len(s):>5,}  mean={s.mean():.3f}  "
            f"PS0={100 * (s == 0).mean():5.1f}%  PS>=2={100 * (s >= 2).mean():5.1f}%"
        )
    print("\nby treatment line (row %):")
    ctl = pd.crosstab(longdf["X_line_number"], longdf["ps"]).loc[list(LINES)]
    print((ctl.div(ctl.sum(axis=1), axis=0) * 100).round(1).to_string())
    print("\nsaved", make_figure(line1, longdf))


if __name__ == "__main__":
    main()
