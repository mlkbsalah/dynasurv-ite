"""Overall survival from line-1 onset, per treatment category, by year of onset.

For the HR+HER2- cohort, takes each patient's first treatment line and builds
Kaplan-Meier overall-survival curves measured from the line-1 start date. Within
each treatment category, one curve per calendar year of line-1 onset, so a shift
of the curves with calendar year is directly visible.

Two interactive Plotly figures are written:

  * ``km_line1_by_year.html``     - small multiples, one panel per treatment
    category, one KM curve per onset year (light = early years, dark = recent).
    Clicking a year in the legend toggles it in every panel at once.
  * ``km_line1_24mo_trend.html``  - the same information collapsed to a single
    milestone: 24-month OS by onset year, one line per treatment category,
    with 95% CIs.

Caveats the figures are annotated with:
  * follow-up is administratively censored at the database lock (2024-03), so
    recent-year cohorts have short, noisier tails; curves stop at each cohort's
    own last follow-up rather than being extrapolated flat.
  * the treatment mix itself moves over calendar time (ET+ANTI-CDK only appears
    from 2017, CT+ANTI-ANGIO collapses after 2015), so a within-category shift
    over years is partly case-mix, not only a change in efficacy.

Run:  python data_analysis/km_line1_by_year.py
Figures -> data_analysis/plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

MIN_N = 40  # smallest year-cohort we will draw a curve for
HORIZON = 60.0  # months of follow-up shown on the x-axis
GRID = np.arange(0.0, HORIZON + 0.5, 0.5)
MILESTONE = 24.0  # months, for the trend figure
N_PANELS = 6  # treatment categories shown (largest by line-1 volume)

# ---- palette / chrome (dataviz reference palette, light mode) ----
SURFACE, INK, SEC, MUTED, GRID_C, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# blue ordinal ramp, steps 250->700 (light-mode ordinal floor is step 250)
BLUE_RAMP = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
# categorical slots 1-6 (validated: adjacent CVD dE 9.1, normal-vision 19.6)
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def ramp_color(frac):
    """Sample the blue ordinal ramp at ``frac`` in [0, 1] (light -> dark)."""
    pos = frac * (len(BLUE_RAMP) - 1)
    i = int(np.clip(np.floor(pos), 0, len(BLUE_RAMP) - 2))
    w = pos - i
    a = BLUE_RAMP[i].lstrip("#")
    b = BLUE_RAMP[i + 1].lstrip("#")
    rgb = [
        round(int(a[k : k + 2], 16) * (1 - w) + int(b[k : k + 2], 16) * w)
        for k in (0, 2, 4)
    ]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def load_line1():
    """First treatment line per patient with OS measured from the line-1 start."""
    df = pd.read_parquet(DATA_PATH)
    line1 = df[df["X_line_number"] == 1].copy()
    line1 = line1.drop_duplicates("usubjid")
    line1["year"] = line1["line_start_date"].dt.year
    return line1[
        [
            "usubjid",
            "year",
            "T_treatment_category",
            "Y_onset_to_death",
            "Y_global_death_status",
        ]
    ].rename(
        columns={
            "T_treatment_category": "category",
            "Y_onset_to_death": "time",
            "Y_global_death_status": "event",
        }
    )


def km_curve(times, events):
    """Step-evaluate KM survival + 95% CI + at-risk on GRID, truncated at last follow-up."""
    kmf = KaplanMeierFitter().fit(times, events)
    timeline = kmf.survival_function_.index.values
    surv = kmf.survival_function_.iloc[:, 0].values
    ci = kmf.confidence_interval_.values

    max_fu = float(np.max(times))
    grid = GRID[GRID <= max_fu]
    if len(grid) < 2:
        return None

    idx = np.clip(
        np.searchsorted(timeline, grid, side="right") - 1, 0, len(timeline) - 1
    )
    at_risk = np.array([(np.asarray(times) >= g).sum() for g in grid])
    return {
        "t": grid,
        "s": surv[idx],
        "lo": ci[idx, 0],
        "hi": ci[idx, 1],
        "at_risk": at_risk,
        "max_fu": max_fu,
        "median": kmf.median_survival_time_,
    }


def milestone_os(times, events, t_star=MILESTONE):
    """KM survival at ``t_star`` with 95% CI, or None if follow-up is too short."""
    if float(np.max(times)) < t_star:
        return None
    kmf = KaplanMeierFitter().fit(times, events)
    timeline = kmf.survival_function_.index.values
    i = int(
        np.clip(
            np.searchsorted(timeline, t_star, side="right") - 1, 0, len(timeline) - 1
        )
    )
    ci = kmf.confidence_interval_.values
    return float(kmf.survival_function_.iloc[i, 0]), float(ci[i, 0]), float(ci[i, 1])


def cells(df):
    """(category, year) cohorts with at least MIN_N patients, largest categories first."""
    order = [c for c, _ in df["category"].value_counts().items()][:N_PANELS]
    out = {}
    for cat in order:
        sub = df[df["category"] == cat]
        years = sorted(y for y, n in sub["year"].value_counts().items() if n >= MIN_N)
        if years:
            out[cat] = {y: sub[sub["year"] == y] for y in years}
    return out


def make_km_grid(df, by_cat):
    """Small multiples: one panel per category, one KM curve per onset year."""
    cats = list(by_cat)
    n_col = 3
    n_row = int(np.ceil(len(cats) / n_col))
    all_years = sorted({y for cells_ in by_cat.values() for y in cells_})
    span = max(all_years) - min(all_years)
    color = {y: ramp_color((y - min(all_years)) / span) for y in all_years}

    titles = [
        f"<b>{cat}</b><span style='color:{SEC};font-weight:400'>"
        f"  n = {sum(len(g) for g in by_cat[cat].values()):,}</span>"
        for cat in cats
    ]
    fig = make_subplots(
        rows=n_row,
        cols=n_col,
        subplot_titles=titles,
        horizontal_spacing=0.055,
        vertical_spacing=0.13,
        shared_yaxes=False,
    )

    seen = set()
    for i, cat in enumerate(cats):
        row, col = i // n_col + 1, i % n_col + 1
        for year, grp in by_cat[cat].items():
            curve = km_curve(grp["time"].values, grp["event"].values)
            if curve is None:
                continue
            med = curve["median"]
            med_txt = "not reached" if not np.isfinite(med) else f"{med:.0f} mo"
            cust = np.column_stack(
                [
                    curve["lo"] * 100,
                    curve["hi"] * 100,
                    curve["at_risk"],
                    np.full(len(curve["t"]), len(grp)),
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=curve["t"],
                    y=curve["s"] * 100,
                    mode="lines",
                    line=dict(color=color[year], width=2, shape="hv"),
                    name=str(year),
                    legendgroup=str(year),
                    showlegend=year not in seen,
                    customdata=cust,
                    hovertemplate=(
                        f"<b>{cat} · {year}</b><br>"
                        "%{x:.0f} months from line-1 onset<br>"
                        "OS %{y:.1f}%  (95% CI %{customdata[0]:.1f}–%{customdata[1]:.1f})<br>"
                        "at risk: %{customdata[2]:,} of %{customdata[3]:,}<br>"
                        f"median OS: {med_txt}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            seen.add(year)

        fig.add_hline(
            y=50,
            line=dict(color=BASE, width=1, dash="dot"),
            row=row,
            col=col,
        )

    n_pat = sum(len(g) for cells_ in by_cat.values() for g in cells_.values())
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Helvetica Neue, Helvetica, Arial, sans-serif", color=INK, size=12
        ),
        title=dict(
            text=(
                "Overall survival from line-1 onset, by treatment category and year of onset"
                "<br><span style='font-size:13px;color:" + SEC + "'>"
                f"HR+HER2− cohort · {n_pat:,} patients in year-cohorts of ≥ {MIN_N} · "
                "light = early years, dark = recent · dotted line = median · "
                "click a year in the legend to isolate it in every panel</span>"
            ),
            x=0.008,
            xanchor="left",
            font=dict(size=19),
        ),
        legend=dict(
            title_text="year of line-1 onset",
            title_font_color=SEC,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.09,
            yanchor="top",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor=BASE,
            borderwidth=1,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=BASE, font=dict(color=INK, size=12)
        ),
        margin=dict(l=60, r=24, t=118, b=118),
        height=300 * n_row + 190,
    )
    fig.update_xaxes(range=[0, HORIZON], gridcolor=GRID_C, zeroline=False, dtick=12)
    fig.update_yaxes(range=[0, 100], gridcolor=GRID_C, zeroline=False, ticksuffix="%")
    # axis titles only on the outer edge of the grid
    for i in range(len(cats)):
        row, col = i // n_col + 1, i % n_col + 1
        if i + n_col >= len(cats):
            fig.update_xaxes(
                title_text="months since line-1 onset",
                title_font=dict(size=11, color=SEC),
                row=row,
                col=col,
            )
        if col == 1:
            fig.update_yaxes(
                title_text="overall survival",
                title_font=dict(size=11, color=SEC),
                row=row,
                col=col,
            )
    # left-align each panel title over its own panel
    for i, ann in enumerate(fig.layout.annotations[: len(cats)]):
        axis_key = "xaxis" if i == 0 else f"xaxis{i + 1}"
        ann.update(
            x=fig.layout[axis_key].domain[0],
            xanchor="left",
            font=dict(size=13, color=INK),
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "km_line1_by_year.html"
    fig.write_html(
        out,
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    return out


def make_trend(by_cat):
    """24-month OS by onset year, one line per treatment category, with 95% CIs."""
    fig = go.Figure()
    rows = []
    for i, (cat, cells_) in enumerate(by_cat.items()):
        xs, ys, lo, hi, ns = [], [], [], [], []
        for year, grp in cells_.items():
            est = milestone_os(grp["time"].values, grp["event"].values)
            if est is None:
                continue
            s, cl, ch = est
            xs.append(year)
            ys.append(s * 100)
            lo.append((s - cl) * 100)
            hi.append((ch - s) * 100)
            ns.append(len(grp))
            rows.append((cat, year, len(grp), s, cl, ch))
        if not xs:
            continue
        color = CAT_COLORS[i % len(CAT_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=cat,
                line=dict(color=color, width=2),
                marker=dict(color=color, size=8, line=dict(color=SURFACE, width=1.5)),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=hi,
                    arrayminus=lo,
                    color=color,
                    thickness=1,
                    width=0,
                ),
                customdata=np.column_stack(
                    [ns, np.array(ys) - np.array(lo), np.array(ys) + np.array(hi)]
                ),
                hovertemplate=(
                    f"<b>{cat}</b><br>"
                    "line-1 onset in %{x}<br>"
                    f"{MILESTONE:.0f}-month OS: " + "%{y:.1f}%"
                    " (95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f})<br>"
                    "cohort: %{customdata[0]:,} patients"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Helvetica Neue, Helvetica, Arial, sans-serif", color=INK, size=12
        ),
        title=dict(
            text=(
                f"{MILESTONE:.0f}-month overall survival from line-1 onset, by year of onset"
                "<br><span style='font-size:13px;color:" + SEC + "'>"
                f"HR+HER2− cohort · year-cohorts of ≥ {MIN_N} patients with ≥ {MILESTONE:.0f} "
                "months of potential follow-up · whiskers = 95% CI</span>"
            ),
            x=0.008,
            xanchor="left",
            font=dict(size=19),
        ),
        legend=dict(
            title_text="treatment category at line 1",
            title_font_color=SEC,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.16,
            yanchor="top",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor=BASE,
            borderwidth=1,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=BASE, font=dict(color=INK, size=12)
        ),
        margin=dict(l=64, r=28, t=110, b=110),
        height=560,
    )
    fig.update_xaxes(
        title_text="year of line-1 onset", gridcolor=GRID_C, zeroline=False, dtick=1
    )
    fig.update_yaxes(
        title_text=f"{MILESTONE:.0f}-month overall survival",
        ticksuffix="%",
        gridcolor=GRID_C,
        zeroline=False,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "km_line1_24mo_trend.html"
    fig.write_html(
        out,
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    return out, pd.DataFrame(
        rows, columns=["category", "year", "n", "os24", "ci_lo", "ci_hi"]
    )


def main():
    df = load_line1()
    by_cat = cells(df)
    print(f"line-1 patients: {len(df):,}  ({df['year'].min()}-{df['year'].max()})")
    for cat, cells_ in by_cat.items():
        years = list(cells_)
        print(
            f"  {cat:<20} {sum(len(g) for g in cells_.values()):>6,} patients "
            f"in {len(years)} year-cohorts ({years[0]}-{years[-1]})"
        )
    print("saved", make_km_grid(df, by_cat))
    out, tbl = make_trend(by_cat)
    print("saved", out)
    print()
    print(f"{MILESTONE:.0f}-month OS by category and year (%):")
    piv = tbl.assign(os24=(tbl.os24 * 100).round(1)).pivot(
        index="category", columns="year", values="os24"
    )
    print(piv.to_string(na_rep="·"))


if __name__ == "__main__":
    main()
