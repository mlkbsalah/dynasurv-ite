"""Evolution of the performance-status distribution (`mpps`) over calendar years.

Performance status is ECOG/WHO 0 (fully active) to 4 (completely disabled).

Two sources, and they disagree in DIRECTION, which is the point of this figure:

  * `data_raw/metperf.parquet` — the RAW table of dated measurements
    (`usubjid`, `mpdt`, `mpps`). This is observed data. Restricted here to the
    `usubjid` set of the HR+HER2- V2 model-entry file.
  * `X_mpps` in `data/model_entry_imputed_data_HR+HER2-_..._V2.parquet` — one
    value per treatment line, with zero missing rows because it is IMPUTED.

Only 30.8% of 2008 line-1 records have a real measurement within +/-90 days,
rising to 79.9% by 2022. The imputation fills the gap toward the mode (PS 1), so
as coverage improves the imputed distribution drifts toward the observed one.
That manufactures a rising-PS-0 trend which the observed data does not show —
observed PS 0 in fact FALLS. Always read the observed panel.

Run:  python data_analysis/mpps_performance_status_by_year.py
Figure -> data_analysis/plots/mpps_by_year.html
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)
RAW_PATH = REPO_ROOT / "data_raw" / "metperf.parquet"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
YEARS = (2008, 2022)  # 2023 holds only 20 line-1 patients — too thin to plot
WINDOW_DAYS = 90  # a measurement this close to line-1 start counts as "at onset"

SURFACE, INK, SEC, MUTED, GRID, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# Performance status is ORDINAL: one hue light -> dark, never categorical hues.
# Verified monotone in relative luminance (0.864 -> 0.051, min adjacent gap 0.137).
PS_LEVELS = (0, 1, 2, 3, 4)
PS_COLOR = {0: "#e8f0fb", 1: "#a9caf0", 2: "#6ba5e4", 3: "#2a78d6", 4: "#123f7a"}
PS_DESC = {
    0: "fully active",
    1: "restricted in strenuous activity",
    2: "ambulatory, up > 50% of waking hours",
    3: "limited self-care, confined > 50%",
    4: "completely disabled",
}
PS_TEXT = {0: INK, 1: INK, 2: INK, 3: "#ffffff", 4: "#ffffff"}
LABEL_FLOOR = 8.0
LIGHT_END, DARK_END = "#6ba5e4", "#123f7a"
WARN = "#c2410c"


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, as percentages."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def load():
    """Line-1 rows joined to the nearest observed measurement, cohort-restricted."""
    mod = pd.read_parquet(
        MODEL_PATH, columns=["usubjid", "X_line_number", "X_mpps", "line_start_date"]
    )
    cohort = set(mod["usubjid"].unique())

    raw = pd.read_parquet(RAW_PATH, columns=["usubjid", "mpdt", "mpps"])
    raw = raw[raw["usubjid"].isin(cohort)]  # restrict to the HR+HER2- V2 patients

    line1 = (
        mod[mod["X_line_number"] == 1]
        .drop_duplicates("usubjid")
        .assign(line_start_date=lambda d: pd.to_datetime(d["line_start_date"]))
        .rename(columns={"X_mpps": "imputed"})
    )
    line1 = line1[line1["line_start_date"].dt.year.between(*YEARS)]

    pair = line1[["usubjid", "line_start_date", "imputed"]].merge(
        raw, on="usubjid", how="left"
    )
    pair["gap"] = (pair["mpdt"] - pair["line_start_date"]).dt.days
    near = pair[pair["gap"].abs() <= WINDOW_DAYS]
    best = near.loc[near["gap"].abs().groupby(near["usubjid"]).idxmin()]

    joined = line1.merge(best[["usubjid", "mpps"]], on="usubjid", how="left").assign(
        year=lambda d: d["line_start_date"].dt.year,
        imputed=lambda d: d["imputed"].astype(int),
    )
    return joined, raw


def add_stacked(fig, ct, row, col):
    """100% stacked bars of the observed PS distribution, one bar per year."""
    totals = ct.sum(axis=1)
    for ps in PS_LEVELS:
        counts = ct[ps].to_numpy() if ps in ct.columns else np.zeros(len(ct))
        pct = 100 * counts / totals.to_numpy()
        fig.add_trace(
            go.Bar(
                x=[str(i) for i in ct.index],
                y=pct,
                name=f"PS {ps}",
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
                    "measured at onset that year: %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )


def _share(frame, col, pred):
    """Per-year percentage plus Wilson interval for a predicate on ``col``."""
    years = sorted(frame["year"].unique())
    pts, los, his, ks, ns = [], [], [], [], []
    for y in years:
        s = frame.loc[frame["year"] == y, col].dropna().astype(int)
        k, n = int(pred(s).sum()), len(s)
        lo, hi = wilson(k, n)
        pts.append(100 * k / n if n else 0.0)
        los.append(lo)
        his.append(hi)
        ks.append(k)
        ns.append(n)
    return years, pts, los, his, ks, ns


def add_comparison(fig, joined, row, col):
    """Observed vs imputed, same two summaries — the direction disagreement."""
    obs = joined.dropna(subset=["mpps"])
    specs = [
        ("PS 0, observed", obs, "mpps", lambda s: s == 0, LIGHT_END, "solid"),
        ("PS 0, imputed", joined, "imputed", lambda s: s == 0, LIGHT_END, "dot"),
        ("PS ≥ 2, observed", obs, "mpps", lambda s: s >= 2, DARK_END, "solid"),
        ("PS ≥ 2, imputed", joined, "imputed", lambda s: s >= 2, DARK_END, "dot"),
    ]
    for i, (name, frame, col_name, pred, color, dash) in enumerate(specs):
        years, pts, los, his, ks, ns = _share(frame, col_name, pred)
        if dash == "solid":  # CI band on the observed series only
            fig.add_trace(
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
                mode="lines+markers" if dash == "solid" else "lines",
                name=name,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=5, color=color),
                customdata=np.stack([ks, ns], axis=-1),
                hovertemplate=(
                    "<b>%{x}</b> — " + name + "<br>"
                    "%{y:.1f}%  (%{customdata[0]:,} of %{customdata[1]:,})"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            x=years[-1],
            y=pts[-1],
            text=f"  {name}",
            xanchor="left",
            yanchor="middle",
            yshift=8 if i < 2 else -8,  # keep the four end-labels from colliding
            showarrow=False,
            font=dict(color=color, size=10),
            row=row,
            col=col,
        )
    return sorted(joined["year"].unique())


def add_coverage(fig, joined, row, col):
    """Why the imputed series drifts: measurement coverage improves steeply."""
    cov = joined.groupby("year").agg(
        n=("usubjid", "size"), matched=("mpps", lambda s: s.notna().sum())
    )
    pct = 100 * cov["matched"] / cov["n"]
    fig.add_trace(
        go.Bar(
            x=[str(y) for y in cov.index],
            y=pct,
            marker_color=LIGHT_END,
            width=0.62,
            text=[f"{v:.0f}" for v in pct],
            textposition="outside",
            textfont=dict(color=SEC, size=9),
            customdata=np.stack([cov["matched"], cov["n"]], axis=-1),
            hovertemplate=(
                "<b>%{x}</b><br>%{y:.1f}% of line-1 patients have a measured PS<br>"
                "%{customdata[0]:,} of %{customdata[1]:,}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def make_figure(joined):
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"colspan": 2}, None], [{}, {}]],
        column_widths=[0.56, 0.44],
        row_heights=[0.52, 0.48],
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
        subplot_titles=(
            "<b>Observed performance status at line-1 onset, by year</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            f"   measured within ±{WINDOW_DAYS} days of line-1 start · imputed values "
            "excluded</span>",
            "<b>Observed vs imputed — opposite directions</b>"
            f"<span style='font-size:12px;color:{SEC}'>   dotted = imputed</span>",
            "<b>Why: measurement coverage</b>"
            f"<span style='font-size:12px;color:{SEC}'>"
            "   % of line-1 patients with a real measurement</span>",
        ),
    )

    obs = joined.dropna(subset=["mpps"]).astype({"mpps": int})
    ct = pd.crosstab(obs["year"], obs["mpps"])
    add_stacked(fig, ct, 1, 1)
    years = add_comparison(fig, joined, 2, 1)
    add_coverage(fig, joined, 2, 2)

    first, last = years[0], years[-1]

    def era(frame, col, lo, hi):
        s = frame.loc[frame["year"].between(lo, hi), col].dropna().astype(int)
        return 100 * (s == 0).mean(), 100 * (s >= 2).mean(), s.mean(), len(s)

    o_e = era(obs, "mpps", first, first + 4)
    o_l = era(obs, "mpps", last - 4, last)
    i_e = era(joined, "imputed", first, first + 4)
    i_l = era(joined, "imputed", last - 4, last)

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
                "Performance status (ECOG/WHO 0–4) at line-1 onset, over time"
                f"<br><span style='font-size:13px;color:{SEC}'>"
                "HR+HER2− cohort · observed measurements from data_raw/metperf.parquet, "
                "restricted to the V2 file's patient ids"
                f"<br><b>Observed: performance status gets WORSE.</b> "
                f"{first}–{first + 4} → {last - 4}–{last}: PS 0 falls "
                f"{o_e[0]:.1f}%→{o_l[0]:.1f}%, PS ≥ 2 rises {o_e[1]:.1f}%→{o_l[1]:.1f}%, "
                f"mean {o_e[2]:.2f}→{o_l[2]:.2f}"
                f"<br><b style='color:{WARN}'>The imputed X_mpps reverses this</b> — it has "
                f"PS 0 <i>rising</i> {i_e[0]:.1f}%→{i_l[0]:.1f}%. Coverage climbs 31%→80% "
                "over the period, so early years are mostly"
                "<br>mode-filled at PS 1; the imputed curve drifts toward the truth as real "
                "data arrives, and that drift looks like a trend. Use the observed panel."
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
        margin=dict(l=10, r=20, t=286, b=56),
        height=960,
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=BASE, font=dict(color=INK, size=12)
        ),
    )

    for i, ann in enumerate(fig.layout.annotations[:3]):
        axis = fig.layout["xaxis" if i == 0 else f"xaxis{i + 1}"]
        ann.update(x=axis.domain[0], xanchor="left")

    fig.update_yaxes(
        title_text="% of measured patients",
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
        range=[first - 0.4, last + 4.2],
        tickmode="array",
        tickvals=[y for y in years if y % 2 == 0],
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="% with a measured PS",
        range=[0, 100],
        gridcolor=GRID,
        zeroline=False,
        row=2,
        col=2,
    )
    fig.update_xaxes(
        title_text="year of line-1 onset", type="category", gridcolor=GRID, row=2, col=2
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
    joined, raw = load()
    obs = joined.dropna(subset=["mpps"]).astype({"mpps": int})
    print(f"line-1 patients {YEARS[0]}–{YEARS[1]}: {len(joined):,}")
    print(
        f"with an observed PS within ±{WINDOW_DAYS}d: {len(obs):,} "
        f"({100 * len(obs) / len(joined):.1f}%)"
    )
    print(
        f"agreement where observed exists: "
        f"{100 * (obs['mpps'] == obs['imputed']).mean():.1f}%"
    )

    print("\nobserved PS distribution by year (row %):")
    ct = pd.crosstab(obs["year"], obs["mpps"])
    p = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
    p["n"] = ct.sum(axis=1)
    print(p.to_string())

    print("\nera comparison:")
    for nm, frame, col in (("observed", obs, "mpps"), ("imputed ", joined, "imputed")):
        for lo, hi in ((2008, 2012), (2018, 2022)):
            s = frame.loc[frame["year"].between(lo, hi), col].dropna().astype(int)
            print(
                f"  {nm} {lo}-{hi}  n={len(s):>6,}  PS0={100 * (s == 0).mean():5.1f}%  "
                f"PS>=2={100 * (s >= 2).mean():5.1f}%  mean={s.mean():.3f}"
            )

    print("\nsaved", make_figure(joined))


if __name__ == "__main__":
    main()
