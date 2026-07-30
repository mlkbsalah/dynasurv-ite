"""Overall survival from line-1 onset, crossed by treatment category and onset year.

For the HR+HER2- cohort, takes each patient's first treatment line and builds
Kaplan-Meier overall-survival curves measured from the line-1 start date. The same
(category x year) cells are shown three ways:

  * ``km_line1_panel_per_year.html``     - one panel per calendar year of line-1
    onset, stratified inside the year by treatment category. Answers "in a given
    year, how did the treatments compare?".
  * ``km_line1_panel_per_category.html`` - one panel per treatment category, one
    curve per onset year (light = early, dark = recent). Answers "did this
    treatment's survival shift over the years?".
  * ``km_line1_24mo_trend.html``         - both collapsed to 24-month OS by year,
    one line per category, with 95% CIs.

Clicking a series in any legend isolates it across every panel at once.

All 11 treatment categories are in scope. A curve is only drawn for a (category,
year) cell with at least ``MIN_N`` patients, because a Kaplan-Meier curve on a
handful of patients is noise rather than signal; ``main`` prints the full
coverage table so the suppressed cells stay visible. Three categories
(CT+ANTI-HER2, CT+IT, CT+TT) never reach ``MIN_N`` in any single year -- they are
only estimable pooled across years, which ``km_line1_adjusted.py`` does.

Caveats the figures are annotated with:
  * follow-up is administratively censored at the database lock (2024-03), so
    recent-year cohorts have short, noisier tails; curves stop at each cohort's
    own last follow-up rather than being extrapolated flat.
  * these are crude, unadjusted curves -- within a year the treatment groups are
    not exchangeable (confounding by indication). See ``km_line1_adjusted.py``
    for IPTW/IPCW-adjusted curves.

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

MIN_N = 20  # smallest (category, year) cell we will draw a curve for
HORIZON = 60.0  # months of follow-up shown on the x-axis
GRID = np.arange(0.0, HORIZON + 0.5, 0.5)
MILESTONE = 24.0  # months, for the trend figure

# ---- palette / chrome (dataviz reference palette, light mode) ----
SURFACE, INK, SEC, MUTED, GRID_C, BASE = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
# categorical slots 1-8, validated: adjacent CVD dE 9.1, normal-vision dE 19.6
CAT_COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]
# past slot 8 the palette is out of validated hues: fall back to muted + dashed
# (secondary encoding) rather than inventing a 9th competing colour.
OVERFLOW_COLOR = MUTED
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
    """Step-evaluate KM survival + 95% CI + at-risk on GRID, cut at last follow-up."""
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


def category_order(df):
    """All treatment categories, largest line-1 volume first (a stable colour key)."""
    return list(df["category"].value_counts().index)


def cat_style(cats):
    """Colour + dash per category; past the 8 validated slots use muted + dotted."""
    style = {}
    for i, cat in enumerate(cats):
        if i < len(CAT_COLORS):
            style[cat] = (CAT_COLORS[i], "solid")
        else:
            style[cat] = (OVERFLOW_COLOR, "dot")
    return style


def coverage(df, cats):
    """(category, year) -> patient frame, keeping only cells of at least MIN_N."""
    counts = pd.crosstab(df["category"], df["year"]).reindex(cats).fillna(0).astype(int)
    kept = {}
    for cat in cats:
        sub = df[df["category"] == cat]
        for year, n in counts.loc[cat].items():
            if n >= MIN_N:
                kept[(cat, year)] = sub[sub["year"] == year]
    return counts, kept


def apply_chrome(
    fig, title, subtitle, legend_title, height, legend_y, glossary=(), gloss_shift=112
):
    """Chrome plus an optional term glossary along the bottom.

    Panel titles stay bare; anything that only describes the chart goes in
    ``glossary`` as (term, definition) pairs.
    """
    gloss_h = 19 * len(glossary)
    if glossary:
        height += gloss_h
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(
            family="Helvetica Neue, Helvetica, Arial, sans-serif", color=INK, size=12
        ),
        title=dict(
            text=f"{title}<br><span style='font-size:13px;color:{SEC}'>{subtitle}</span>",
            x=0.008,
            xanchor="left",
            font=dict(size=19),
        ),
        legend=dict(
            title_text=legend_title,
            title_font_color=SEC,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=legend_y,
            yanchor="top",
            bgcolor="rgba(252,252,251,0.85)",
            bordercolor=BASE,
            borderwidth=1,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor=BASE, font=dict(color=INK, size=12)
        ),
        height=height,
        # reserve the strip the glossary will occupy; callers that set their own
        # margins after this call must budget for it themselves
        margin=dict(b=gloss_shift + gloss_h + 16) if glossary else None,
    )
    if glossary:
        # pinned a fixed number of pixels under the plot area so it clears the
        # bottom legend, whose offset is a paper fraction and so height-dependent
        fig.add_annotation(
            text="<br>".join(f"<b>{t}</b> — {d}" for t, d in glossary),
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yshift=-gloss_shift,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font=dict(color=SEC, size=11),
        )


def _outer_axis_titles(fig, n_panels, n_col):
    """x title only on the bottom edge of the grid, y title only on the left edge."""
    for i in range(n_panels):
        row, col = i // n_col + 1, i % n_col + 1
        if i + n_col >= n_panels:
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


def _left_align_titles(fig, n_panels):
    for i, ann in enumerate(fig.layout.annotations[:n_panels]):
        axis_key = "xaxis" if i == 0 else f"xaxis{i + 1}"
        ann.update(
            x=fig.layout[axis_key].domain[0],
            xanchor="left",
            font=dict(size=13, color=INK),
        )


def _add_km(fig, curve, grp, label, hover_head, color, dash, group, show, row, col):
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
            line=dict(color=color, width=2, shape="hv", dash=dash),
            name=label,
            legendgroup=group,
            showlegend=show,
            customdata=cust,
            hovertemplate=(
                f"<b>{hover_head}</b><br>"
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


def write_html(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / name
    fig.write_html(
        out,
        include_plotlyjs=True,
        full_html=True,
        config={"displayModeBar": True, "responsive": True},
    )
    return out


def make_year_panels(cats, kept, counts):
    """One panel per onset year, stratified inside the year by treatment category."""
    style = cat_style(cats)
    years = sorted({y for _, y in kept})
    n_col = 5
    n_row = int(np.ceil(len(years) / n_col))

    titles = []
    for year in years:
        n = sum(len(g) for (_, y), g in kept.items() if y == year)
        titles.append(
            f"<b>{year}</b><span style='color:{SEC};font-weight:400'>  n = {n:,}</span>"
        )
    fig = make_subplots(
        rows=n_row,
        cols=n_col,
        subplot_titles=titles,
        horizontal_spacing=0.035,
        vertical_spacing=0.10,
    )

    seen = set()
    for i, year in enumerate(years):
        row, col = i // n_col + 1, i % n_col + 1
        for cat in cats:  # global order keeps colours and legend stable
            grp = kept.get((cat, year))
            if grp is None:
                continue
            curve = km_curve(grp["time"].values, grp["event"].values)
            if curve is None:
                continue
            color, dash = style[cat]
            _add_km(
                fig,
                curve,
                grp,
                cat,
                f"{cat} · {year}",
                color,
                dash,
                cat,
                cat not in seen,
                row,
                col,
            )
            seen.add(cat)
        fig.add_hline(
            y=50, line=dict(color=BASE, width=1, dash="dot"), row=row, col=col
        )

    n_pat = sum(len(g) for g in kept.values())
    never = [c for c in cats if not any(k[0] == c for k in kept)]
    never_txt = (
        f" · never ≥ {MIN_N} in a single year, so not drawn here: "
        + ", ".join(f"{c} ({counts.loc[c].sum():,} in total)" for c in never)
        if never
        else ""
    )
    apply_chrome(
        fig,
        "Survival by treatment, one panel per year",
        f"HR+HER2− · {n_pat:,} patients · click a treatment in the legend to follow it "
        f"across every year{never_txt}",
        "treatment category at line 1",
        260 * n_row + 210,
        -0.055 if n_row > 2 else -0.10,
        [
            (
                "the curve",
                "Kaplan-Meier overall survival — the share of that group still alive, "
                "measured from the day first-line treatment started",
            ),
            (
                "dotted line",
                "the 50% mark. Where a curve crosses it is that group's median survival — "
                "the month by which half of them had died",
            ),
            (
                "why curves stop early",
                "each one ends at its own group's last follow-up rather than being drawn "
                "flat. Recent years are short because the database closed in March 2024",
            ),
        ],
    )
    fig.update_xaxes(range=[0, HORIZON], gridcolor=GRID_C, zeroline=False, dtick=24)
    fig.update_yaxes(range=[0, 100], gridcolor=GRID_C, zeroline=False, ticksuffix="%")
    _outer_axis_titles(fig, len(years), n_col)
    _left_align_titles(fig, len(years))
    return write_html(fig, "km_line1_panel_per_year.html")


def make_category_panels(cats, kept):
    """One panel per treatment category, one curve per onset year (light -> dark)."""
    drawn = [c for c in cats if any(k[0] == c for k in kept)]
    n_col = 4
    n_row = int(np.ceil(len(drawn) / n_col))
    all_years = sorted({y for _, y in kept})
    span = max(max(all_years) - min(all_years), 1)
    color = {y: ramp_color((y - min(all_years)) / span) for y in all_years}

    titles = [
        f"<b>{cat}</b><span style='color:{SEC};font-weight:400'>"
        f"  n = {sum(len(g) for (c, _), g in kept.items() if c == cat):,}</span>"
        for cat in drawn
    ]
    fig = make_subplots(
        rows=n_row,
        cols=n_col,
        subplot_titles=titles,
        horizontal_spacing=0.045,
        vertical_spacing=0.13,
    )

    seen = set()
    for i, cat in enumerate(drawn):
        row, col = i // n_col + 1, i % n_col + 1
        for year in all_years:
            grp = kept.get((cat, year))
            if grp is None:
                continue
            curve = km_curve(grp["time"].values, grp["event"].values)
            if curve is None:
                continue
            _add_km(
                fig,
                curve,
                grp,
                str(year),
                f"{cat} · {year}",
                color[year],
                "solid",
                str(year),
                year not in seen,
                row,
                col,
            )
            seen.add(year)
        fig.add_hline(
            y=50, line=dict(color=BASE, width=1, dash="dot"), row=row, col=col
        )

    n_pat = sum(len(g) for g in kept.values())
    apply_chrome(
        fig,
        "Survival by year, one panel per treatment",
        f"HR+HER2− · {n_pat:,} patients · click a year in the legend to follow it across "
        "every treatment",
        "year of line-1 onset",
        300 * n_row + 200,
        -0.09 if n_row > 1 else -0.16,
        [
            (
                "the curve",
                "Kaplan-Meier overall survival — the share still alive, measured from the "
                "day first-line treatment started",
            ),
            (
                "colour",
                "pale = earliest years, dark = most recent. Reading pale to dark shows "
                "whether that treatment's survival changed over time",
            ),
            (
                "dotted line",
                "the 50% mark — where a curve crosses it is that year's median survival",
            ),
            (
                "why curves stop early",
                "each ends at its own year's last follow-up; recent years are short "
                "because the database closed in March 2024",
            ),
        ],
    )
    fig.update_xaxes(range=[0, HORIZON], gridcolor=GRID_C, zeroline=False, dtick=12)
    fig.update_yaxes(range=[0, 100], gridcolor=GRID_C, zeroline=False, ticksuffix="%")
    _outer_axis_titles(fig, len(drawn), n_col)
    _left_align_titles(fig, len(drawn))
    return write_html(fig, "km_line1_panel_per_category.html")


def make_trend(cats, kept):
    """24-month OS by onset year, one line per treatment category, with 95% CIs."""
    style = cat_style(cats)
    fig = go.Figure()
    rows = []
    for cat in cats:
        years = sorted(y for c, y in kept if c == cat)
        xs, ys, lo, hi, ns = [], [], [], [], []
        for year in years:
            grp = kept[(cat, year)]
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
        color, dash = style[cat]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name=cat,
                line=dict(color=color, width=2, dash=dash),
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

    apply_chrome(
        fig,
        f"Share still alive {MILESTONE:.0f} months after starting treatment",
        "HR+HER2− · by year of first-line onset",
        "treatment category at line 1",
        560,
        -0.16,
        [
            (
                "each point",
                f"the share of that year's patients still alive {MILESTONE:.0f} months "
                "after their first-line start. Higher is better",
            ),
            (
                "whiskers",
                "95% confidence interval. A year is plotted only if it had enough "
                f"patients and enough elapsed time to reach {MILESTONE:.0f} months",
            ),
            (
                "crude",
                "no adjustment — the groups differ in who was in them, so gaps between "
                "treatments are not treatment effects",
            ),
        ],
    )
    fig.update_layout(margin=dict(l=64, r=28, t=110, b=232))
    fig.update_xaxes(
        title_text="year of line-1 onset", gridcolor=GRID_C, zeroline=False, dtick=1
    )
    fig.update_yaxes(
        title_text=f"{MILESTONE:.0f}-month overall survival",
        ticksuffix="%",
        gridcolor=GRID_C,
        zeroline=False,
    )
    out = write_html(fig, "km_line1_24mo_trend.html")
    return out, pd.DataFrame(
        rows, columns=["category", "year", "n", "os24", "ci_lo", "ci_hi"]
    )


def main():
    df = load_line1()
    cats = category_order(df)
    counts, kept = coverage(df, cats)

    print(f"line-1 patients: {len(df):,}  ({df['year'].min()}-{df['year'].max()})")
    print(f"treatment categories: {len(cats)}")
    print(f"\npatients per (category, year) — cells drawn at MIN_N={MIN_N} marked *")
    marked = counts.astype(str)
    for cat in counts.index:
        for year in counts.columns:
            if (cat, year) in kept:
                marked.loc[cat, year] = marked.loc[cat, year] + "*"
    print(marked.to_string())
    never = [c for c in cats if not any(k[0] == c for k in kept)]
    if never:
        print(
            f"\nnever drawable in a single year (< {MIN_N}/year): " + ", ".join(never)
        )

    print("\nsaved", make_year_panels(cats, kept, counts))
    print("saved", make_category_panels(cats, kept))
    out, tbl = make_trend(cats, kept)
    print("saved", out)
    print(f"\n{MILESTONE:.0f}-month OS by category and year (%):")
    piv = tbl.assign(os24=(tbl.os24 * 100).round(1)).pivot(
        index="category", columns="year", values="os24"
    )
    print(piv.reindex([c for c in cats if c in piv.index]).to_string(na_rep="·"))


if __name__ == "__main__":
    main()
