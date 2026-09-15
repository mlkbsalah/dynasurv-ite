"""Overall survival from lines 2-4, by calendar era, stratified on prior treatment.

``km_line1_by_year.py`` asked whether first-line survival improved over the years
(it did not, materially). This script asks the same question at lines 2, 3 and 4 --
where the naive answer is a large apparent gain, and where that gain is mostly an
artefact of *who reaches the line*.

The problem: the patients starting a second line in 2021 have a different treatment
history from those starting one in 2010, so a raw year-on-year comparison at line 2
confounds "later lines got better" with "the mix of prior treatment changed". The fix
here is to hold prior treatment fixed. Each line-k row is labelled by the modalities
the patient had already received in lines 1..k-1:

    prior CT only / prior ET only / prior ET+CT, each optionally "+CDK4/6"

which is coarse enough to keep >= ``MIN_N`` patients per (stratum, era) cell at every
line, unlike the full ordered category sequence.

Figures:
  * ``km_later_lines_by_era.html``    - KM grid, one row per prior-exposure stratum
    (top row = unstratified, for contrast), one column per line, one curve per era
    (pale = early, dark = recent). Answers "holding prior treatment fixed, did this
    line's survival move?".
  * ``km_later_lines_trend.html``     - ``MILESTONE``-month OS. Top row is year-granular
    per stratum; bottom row puts the crude estimate beside the history-standardised one
    per era, so the size of the composition artefact is readable straight off the bars.
  * ``km_later_lines_diagnostics.html`` - why the stratification is needed and what it
    still cannot fix: the prior-exposure mix per line over time, how long reaching each
    line takes, how many patients get there, and how many start each line each year.

Standardisation: the grey bars are direct standardisation over the three prior-exposure
strata that exist across the whole period (CT only, ET only, ET+CT) with CDK4/6 exposure
**collapsed into its base stratum**, weighted to that line's pooled distribution.
Collapsing keeps every patient in scope and keeps the weights estimable in every year --
the cost is that a CDK4/6 benefit stays inside a stratum and is therefore *not* removed
by the standardisation. So the grey bars are "what the trend would look like if only the
prior ET/CT mix had been held constant", not "the trend with modern drugs removed".

What it finds: the crude gain from 2008-2011 to 2020-2024 is +4 pt at line 2, +11 pt at
line 3 and +12 pt at line 4; standardised it is +1, +5 and +6. What survives lands
entirely between 2008-2011 and 2012-2015 and is flat afterwards, including across the
CDK4/6 rollout.

Caveats the figures are annotated with:
  * conditioning on reaching line k is conditioning on having progressed and survived,
    which no covariate adjustment repairs. On a fixed 48-month landmark the recent
    cohorts are a *smaller* slice of their line-1 cohort (line 2: 71% of 2016 starts
    reach it within four years, 59% of 2019) arriving at much the same point in the
    disease course (median 11.9 -> 13.1 months), so the later-line groups are more
    selected than they used to be.
  * the selection panels must be indexed by the *line-1* year, not by the year the later
    line started. The cohort opens in 2008, so a line-4 start in 2010 cannot be more
    than two years past its line-1 start; indexed that way the median time to line 4
    appears to grow 22 -> 35 months, which is entirely a boundary artefact.
  * these are crude curves within a stratum: prior modality is held fixed, everything
    else (performance status, metastatic burden, age) is not. See
    ``km_line1_adjusted.py`` for what full IPTW/IPCW adjustment can and cannot do.

Run:  python data_analysis/km_later_lines_by_history.py
Figures -> data_analysis/plots/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from km_line1_by_year import (
    BASE,
    CAT_COLORS,
    GRID_C,
    HORIZON,
    INK,
    SEC,
    SURFACE,
    _left_align_titles,
    apply_chrome,
    km_curve,
    milestone_os,
    ramp_color,
    write_html,
)
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)

LINES = (2, 3, 4)
MIN_N = 20  # smallest cell we will draw a curve or plot a point for
MILESTONE = 12.0  # months; reachable for every line start up to 2023-03
FU_COMPLETE = 0.8  # a milestone needs this share of the cell able to reach it
LANDMARK = 48.0  # months allowed to reach a later line, in the selection panels
ERAS = ((2008, 2011), (2012, 2015), (2016, 2019), (2020, 2024))
LOCK = pd.Timestamp("2024-03-08")  # last event/censoring date in the extract

ALL = "all patients"
# prior-exposure strata, prognostically ordered; the two +CDK4/6 ones only exist
# from 2016 and are drawn dashed wherever they sit next to era-spanning series
BASE_STRATA = ("CT only", "ET only", "ET+CT")
STRATA = (
    "CT only",
    "ET only",
    "ET only +CDK4/6",
    "ET+CT",
    "ET+CT +CDK4/6",
)
ERA_CONFINED = {"ET only +CDK4/6", "ET+CT +CDK4/6"}
# legend order is otherwise trace order, which interleaves the bar series into the
# middle of the strata whenever a stratum's first year is dropped for thin follow-up
LEGEND_RANK = {name: i for i, name in enumerate((ALL,) + STRATA + ("std",), start=1)}
STRATUM_COLOR = {
    ALL: INK,
    "CT only": CAT_COLORS[1],
    "ET only": CAT_COLORS[0],
    "ET only +CDK4/6": CAT_COLORS[2],
    "ET+CT": CAT_COLORS[3],
    "ET+CT +CDK4/6": CAT_COLORS[4],
}
LINE_COLOR = {2: CAT_COLORS[0], 3: CAT_COLORS[2], 4: CAT_COLORS[7]}

ERA_NAMES = tuple(f"{a}-{b}" for a, b in ERAS)


def era_of(year):
    for (a, b), name in zip(ERAS, ERA_NAMES):
        if a <= year <= b:
            return name
    return None


def load_lines():
    """One row per (patient, treatment line), with prior-exposure history attached.

    ``Y_onset_to_death`` is already measured from *that line's* start date, so it
    carries over from the line-1 analysis unchanged; ``Y_global_death_status`` is
    death by the database lock.
    """
    df = pd.read_parquet(DATA_PATH)
    df = df.drop_duplicates(["usubjid", "X_line_number"]).sort_values(
        ["usubjid", "X_line_number"]
    )

    # modalities received strictly before the current line
    for col in ("T_endocrine", "T_chemotherapy", "T_anti_cdk"):
        df["prior_" + col] = df.groupby("usubjid")[col].transform(
            lambda s: s.fillna(0).gt(0).cumsum().shift(1)
        )
    et, ct = df["prior_T_endocrine"].gt(0), df["prior_T_chemotherapy"].gt(0)
    df["base_stratum"] = np.select(
        [et & ct, et, ct], ["ET+CT", "ET only", "CT only"], default="neither"
    )
    df["stratum"] = df["base_stratum"] + np.where(
        df["prior_T_anti_cdk"].gt(0), " +CDK4/6", ""
    )

    first_start = (
        df[df["X_line_number"] == 1].set_index("usubjid")["line_start_date"].to_dict()
    )
    df["months_since_line1"] = (
        df["line_start_date"] - df["usubjid"].map(first_start)
    ).dt.days / 30.44
    df["potential_fu"] = (LOCK - df["line_start_date"]).dt.days / 30.44
    df["year"] = df["line_start_date"].dt.year
    df["era"] = df["year"].map(era_of)

    keep = df["X_line_number"].isin((1,) + LINES) & df["base_stratum"].ne("neither")
    # line 1 has no prior exposure by construction; kept only for the attrition panel
    return df[keep | df["X_line_number"].eq(1)][
        [
            "usubjid",
            "X_line_number",
            "year",
            "era",
            "stratum",
            "base_stratum",
            "months_since_line1",
            "potential_fu",
            "Y_onset_to_death",
            "Y_global_death_status",
        ]
    ].rename(
        columns={
            "X_line_number": "line",
            "Y_onset_to_death": "time",
            "Y_global_death_status": "event",
        }
    )


def milestone(grp):
    """``MILESTONE``-month OS as (est, lo, hi) in %, or None if unestimable.

    Rejects a cell unless ``FU_COMPLETE`` of it started early enough to *reach* the
    milestone before the data lock. Kaplan-Meier will happily return an estimate
    from the handful of earliest entrants in a part-observed year, and for the most
    recent year that estimate swings by tens of points on a few dozen patients.
    """
    if len(grp) < MIN_N or grp["potential_fu"].ge(MILESTONE).mean() < FU_COMPLETE:
        return None
    est = milestone_os(grp["time"].values, grp["event"].values, MILESTONE)
    if est is None:
        return None
    return tuple(100 * v for v in est)


def standardised_trend(line_df, weights, by):
    """Direct standardisation of ``MILESTONE``-month OS over ``weights``.

    Returns {group -> (estimate, coverage)} where a group is only estimated when
    every weighted stratum clears ``MIN_N``, so the weights never have to be
    renormalised onto a partial set of strata.
    """
    out = {}
    for key, grp in line_df.groupby(by, observed=True):
        parts, wts = [], []
        for stratum, w in weights.items():
            est = milestone(grp[grp["base_stratum"] == stratum])
            if est is None:
                break
            parts.append(est[0])
            wts.append(w)
        else:
            out[key] = (float(np.average(parts, weights=wts)), len(grp))
    return out


def _axis_labels(fig, n_panels, n_col, x_title, y_title):
    """x title on the bottom edge of the grid only, y title on the left edge only."""
    for i in range(n_panels):
        row, col = i // n_col + 1, i % n_col + 1
        if i + n_col >= n_panels:
            fig.update_xaxes(
                title_text=x_title,
                title_font=dict(size=11, color=SEC),
                row=row,
                col=col,
            )
        if col == 1:
            fig.update_yaxes(
                title_text=y_title,
                title_font=dict(size=11, color=SEC),
                row=row,
                col=col,
            )


def _panel_title(head, tail):
    return f"<b>{head}</b><span style='color:{SEC};font-weight:400'>  {tail}</span>"


def make_era_panels(df):
    """KM grid: rows = prior-exposure stratum (+ unstratified), cols = line."""
    rows = (ALL,) + STRATA
    n_col = len(LINES)
    era_color = {
        name: ramp_color(i / (len(ERA_NAMES) - 1)) for i, name in enumerate(ERA_NAMES)
    }

    cells = {}
    for stratum in rows:
        for line in LINES:
            sub = df[df["line"] == line]
            if stratum != ALL:
                sub = sub[sub["stratum"] == stratum]
            cells[(stratum, line)] = sub

    titles = [
        _panel_title(f"line {line} · {stratum}", f"n = {len(cells[(stratum, line)]):,}")
        for stratum in rows
        for line in LINES
    ]
    fig = make_subplots(
        rows=len(rows),
        cols=n_col,
        subplot_titles=titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.055,
    )

    seen = set()
    for i, stratum in enumerate(rows):
        for j, line in enumerate(LINES):
            sub = cells[(stratum, line)]
            for era in ERA_NAMES:
                grp = sub[sub["era"] == era]
                if len(grp) < MIN_N:
                    continue
                curve = km_curve(grp["time"].values, grp["event"].values)
                if curve is None:
                    continue
                med = curve["median"]
                med_txt = "not reached" if not np.isfinite(med) else f"{med:.0f} mo"
                fig.add_trace(
                    go.Scatter(
                        x=curve["t"],
                        y=curve["s"] * 100,
                        mode="lines",
                        line=dict(color=era_color[era], width=2, shape="hv"),
                        name=era,
                        legendgroup=era,
                        showlegend=era not in seen,
                        customdata=np.column_stack(
                            [
                                curve["lo"] * 100,
                                curve["hi"] * 100,
                                curve["at_risk"],
                                np.full(len(curve["t"]), len(grp)),
                            ]
                        ),
                        hovertemplate=(
                            f"<b>line {line} · {stratum} · {era}</b><br>"
                            "%{x:.0f} months into the line<br>"
                            "OS %{y:.1f}%  (95% CI %{customdata[0]:.1f}–%{customdata[1]:.1f})<br>"
                            "at risk: %{customdata[2]:,} of %{customdata[3]:,}<br>"
                            f"median OS: {med_txt}"
                            "<extra></extra>"
                        ),
                    ),
                    row=i + 1,
                    col=j + 1,
                )
                seen.add(era)
            fig.add_hline(
                y=50, line=dict(color=BASE, width=1, dash="dot"), row=i + 1, col=j + 1
            )

    n_pat = sum(len(cells[(s, line)]) for s in STRATA for line in LINES)
    apply_chrome(
        fig,
        "Survival at lines 2-4, by era, holding prior treatment fixed",
        f"HR+HER2− · {n_pat:,} treatment lines · top row is unstratified; every row below "
        "fixes what the patient had already received · click an era to follow it everywhere",
        "era in which this treatment line started",
        250 * len(rows) + 220,
        -0.025,
        glossary=[
            (
                "the curve",
                "Kaplan-Meier overall survival — the share of that group still alive, "
                "measured from the day this treatment line started",
            ),
            (
                "the row label",
                "what the patient had already received in earlier lines. ET = endocrine "
                "therapy, CT = chemotherapy, CDK4/6 = a CDK4/6 inhibitor",
            ),
            (
                "top row vs the rest",
                "the top row mixes histories together, so its era gap includes the change "
                "in who reaches the line. The rows below hold that fixed",
            ),
            (
                "dotted line",
                "the 50% mark — where a curve crosses it is that group's median survival",
            ),
            (
                "empty panels",
                f"fewer than {MIN_N} patients in that cell. CDK4/6 rows are blank before "
                "2016 because the drug class was not in use",
            ),
        ],
    )
    # a hair of left pad, or the "0" x tick sits on top of the "0%" y tick
    fig.update_xaxes(range=[-1.5, HORIZON], gridcolor=GRID_C, zeroline=False, dtick=24)
    fig.update_yaxes(range=[0, 100], gridcolor=GRID_C, zeroline=False, ticksuffix="%")
    _axis_labels(
        fig,
        len(rows) * n_col,
        n_col,
        "months since this line started",
        "overall survival",
    )
    _left_align_titles(fig, len(rows) * n_col)
    return write_html(fig, "km_later_lines_by_era.html")


def make_trend(df, weights):
    """Milestone OS per line: year-granular per stratum, then crude vs standardised.

    The standardisation is drawn per era rather than per year on purpose. It needs all
    three base strata to clear ``MIN_N`` in the same cell -- at year granularity that
    fails often enough at line 4 to leave the series full of holes, and a series with
    holes invites reading a gap as a dip.
    """
    n_col = len(LINES)
    fig = make_subplots(
        rows=2,
        cols=n_col,
        subplot_titles=[
            _panel_title(f"line {line}", f"n = {len(df[df['line'] == line]):,}")
            for line in LINES
        ]
        + [_panel_title(f"line {line}", "crude vs standardised") for line in LINES],
        horizontal_spacing=0.055,
        vertical_spacing=0.19,
        row_heights=[0.62, 0.38],
    )

    seen = set()
    rows = []
    for j, line in enumerate(LINES):
        sub = df[df["line"] == line]

        for stratum in (ALL,) + STRATA:
            s = sub if stratum == ALL else sub[sub["stratum"] == stratum]
            pts = {}
            for year, grp in s.groupby("year"):
                est = milestone(grp)
                if est is not None:
                    pts[year] = (est, len(grp))
            if not pts:
                continue
            years = sorted(pts)
            est = np.array([pts[y][0] for y in years])
            ns = np.array([pts[y][1] for y in years])
            color = STRATUM_COLOR[stratum]
            wide = stratum == ALL
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=est[:, 0],
                    mode="lines+markers",
                    name=stratum,
                    legendgroup=stratum,
                    legendrank=LEGEND_RANK[stratum],
                    showlegend=stratum not in seen,
                    line=dict(
                        color=color,
                        width=3 if wide else 2,
                        dash="dot" if stratum in ERA_CONFINED else "solid",
                    ),
                    marker=dict(
                        color=color,
                        size=9 if wide else 6,
                        line=dict(color=SURFACE, width=1.5),
                    ),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=est[:, 2] - est[:, 0],
                        arrayminus=est[:, 0] - est[:, 1],
                        color=color,
                        thickness=1,
                        width=0,
                    ),
                    customdata=np.column_stack([ns, est[:, 1], est[:, 2]]),
                    hovertemplate=(
                        f"<b>line {line} · {stratum}</b><br>"
                        "line started in %{x}<br>"
                        f"{MILESTONE:.0f}-month OS: " + "%{y:.1f}%"
                        " (95% CI %{customdata[1]:.1f}–%{customdata[2]:.1f})<br>"
                        "%{customdata[0]:,} patients"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=j + 1,
            )
            seen.add(stratum)
            for y, e, n in zip(years, est, ns):
                rows.append((line, stratum, y, n, e[0]))

        # bottom row: the same question at era granularity, crude beside standardised
        std = standardised_trend(sub, weights[line], "era")
        eras = [e for e in ERA_NAMES if e in std and milestone(sub[sub["era"] == e])]
        crude = [milestone(sub[sub["era"] == e])[0] for e in eras]
        for label, vals, color, ns in (
            (ALL, crude, INK, [len(sub[sub["era"] == e]) for e in eras]),
            (
                "history-standardised",
                [std[e][0] for e in eras],
                SEC,
                [std[e][1] for e in eras],
            ),
        ):
            fig.add_trace(
                go.Bar(
                    x=eras,
                    y=vals,
                    name=label,
                    legendgroup=label if label == ALL else "std",
                    legendrank=LEGEND_RANK[ALL if label == ALL else "std"],
                    showlegend=label != ALL and "std" not in seen,
                    marker=dict(color=color, line_width=0),
                    text=[f"{v:.0f}%" for v in vals],
                    textposition="outside",
                    textfont=dict(size=10, color=SEC),
                    customdata=ns,
                    hovertemplate=(
                        f"<b>line {line} · {label}</b><br>%{{x}}<br>"
                        f"{MILESTONE:.0f}-month OS: " + "%{y:.1f}%<br>"
                        "%{customdata:,} patients<extra></extra>"
                    ),
                ),
                row=2,
                col=j + 1,
            )
            if label != ALL:
                seen.add("std")
                for e, v, n in zip(eras, vals, ns):
                    rows.append((line, "history-standardised", e, n, v))

    apply_chrome(
        fig,
        f"Share still alive {MILESTONE:.0f} months into lines 2-4",
        "HR+HER2− · the gap between the black and grey bars is how much of the apparent "
        "improvement is a change in who reaches the line rather than better survival",
        "prior treatment at the start of this line",
        980,
        -0.10,
        gloss_shift=170,  # 7 legend entries can wrap; keep the strip clear of them
        glossary=[
            (
                "top row",
                f"the share of each year's patients still alive {MILESTONE:.0f} months "
                "after starting the line, split by prior treatment. Whiskers are 95% "
                "confidence intervals",
            ),
            (
                "black",
                "all patients starting the line, histories mixed together — the naive "
                "trend, and the one that looks like a large improvement",
            ),
            (
                "grey bars",
                "the same patients re-weighted so the prior ET/CT mix is held at that "
                "line's pooled average. Shorter than black means the black rise was mix",
            ),
            (
                "dotted series",
                "prior CDK4/6 exposure — only exists from 2016, so it has no early years "
                "to be compared against",
            ),
            (
                "missing years",
                f"a year is plotted only with at least {MIN_N} patients and enough elapsed "
                f"time for most of them to reach {MILESTONE:.0f} months, which drops 2023",
            ),
            (
                "still crude",
                "prior modality is held fixed; performance status, metastatic burden and "
                "age are not",
            ),
        ],
    )
    # b must cover gloss_shift + 19 px per glossary row + 16 (apply_chrome's budget)
    fig.update_layout(
        barmode="group", bargap=0.28, margin=dict(l=64, r=28, t=112, b=310)
    )
    fig.update_xaxes(gridcolor=GRID_C, zeroline=False)
    fig.update_xaxes(dtick=2, row=1)
    fig.update_yaxes(range=[0, 100], gridcolor=GRID_C, zeroline=False, ticksuffix="%")
    for col in range(1, n_col + 1):
        fig.update_xaxes(
            title_text="year this line started",
            title_font=dict(size=11, color=SEC),
            row=1,
            col=col,
        )
        fig.update_xaxes(
            title_text="era", title_font=dict(size=11, color=SEC), row=2, col=col
        )
    for row in (1, 2):
        fig.update_yaxes(
            title_text=f"{MILESTONE:.0f}-month overall survival",
            title_font=dict(size=11, color=SEC),
            row=row,
            col=1,
        )
    _left_align_titles(fig, 2 * n_col)
    out = write_html(fig, "km_later_lines_trend.html")
    return out, pd.DataFrame(rows, columns=["line", "stratum", "period", "n", "os"])


def landmark_selection(df):
    """Per year of line-1 start: who reaches line k within ``LANDMARK`` months, and when.

    Both quantities have to be indexed by the *line-1* year, not by the year the later
    line started. Indexed the other way, both ends are wrong: the cohort opens in 2008,
    so a line-4 start in 2010 cannot be more than two years past its line-1 start, and
    the 2024 lock cuts the other end off. Fixing a ``LANDMARK`` window and keeping only
    line-1 years that have the whole window available makes the years comparable.
    """
    line1 = df[df["line"] == 1].drop_duplicates("usubjid")
    complete = line1.groupby("year")["potential_fu"].median().ge(LANDMARK)
    line1 = line1[line1["year"].isin([y for y, ok in complete.items() if ok])]

    out = {}
    for line in LINES:
        reached = df[df["line"].eq(line) & df["months_since_line1"].le(LANDMARK)]
        months = line1["usubjid"].map(
            reached.set_index("usubjid")["months_since_line1"]
        )
        tab = (
            line1.assign(months=months.values)
            .groupby("year")
            .agg(
                n=("usubjid", "size"),
                share=("months", lambda s: s.notna().mean() * 100),
                median_months=("months", "median"),
            )
        )
        out[line] = tab[tab["n"] >= MIN_N]
    return out


def make_diagnostics(df):
    """What the stratification fixes, and the selection it cannot fix."""
    titles = [_panel_title(f"line {line}", "prior-treatment mix") for line in LINES] + [
        _panel_title("A", "how long reaching the line takes"),
        _panel_title("B", "how many get there at all"),
        _panel_title("C", "how many start each line"),
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.065,
        vertical_spacing=0.17,
    )

    # row 1 - prior-exposure composition per line over time
    for j, line in enumerate(LINES):
        sub = df[df["line"] == line]
        comp = pd.crosstab(sub["year"], sub["stratum"], normalize="index") * 100
        years = [y for y in comp.index if (sub["year"] == y).sum() >= MIN_N]
        comp = comp.loc[years]
        for stratum in STRATA:
            if stratum not in comp:
                continue
            fig.add_trace(
                go.Bar(
                    x=comp.index,
                    y=comp[stratum],
                    name=stratum,
                    legendgroup=stratum,
                    showlegend=j == 0,
                    marker=dict(color=STRATUM_COLOR[stratum], line_width=0),
                    hovertemplate=(
                        f"<b>line {line} · {stratum}</b><br>"
                        "%{x}: %{y:.1f}% of that year's line starts<extra></extra>"
                    ),
                ),
                row=1,
                col=j + 1,
            )

    # row 2A/2B - how long it takes to reach a line, and how many get there
    sel = landmark_selection(df)
    for col, (field, label) in enumerate(
        ((("median_months"), "median months"), (("share"), "share")), start=1
    ):
        for line in LINES:
            tab = sel[line]
            fig.add_trace(
                go.Scatter(
                    x=tab.index,
                    y=tab[field],
                    mode="lines+markers",
                    name=f"line {line}",
                    legendgroup=f"line {line}",
                    showlegend=col == 1,
                    line=dict(color=LINE_COLOR[line], width=2),
                    marker=dict(color=LINE_COLOR[line], size=6),
                    customdata=tab["n"],
                    hovertemplate=(
                        f"<b>line {line}</b><br>line 1 started in %{{x}}<br>"
                        + (
                            "median %{y:.1f} months from the line-1 start"
                            if label == "median months"
                            else "%{y:.0f}% of that year's %{customdata:,} line-1 patients"
                        )
                        + f"<br>within {LANDMARK:.0f} months<extra></extra>"
                    ),
                ),
                row=2,
                col=col,
            )

    # row 2C - how many line starts each year contributes
    for line in LINES:
        counts = df[df["line"] == line]["year"].value_counts().sort_index()
        counts = counts[counts >= MIN_N]  # 2024 is a partial month, not a real dip
        fig.add_trace(
            go.Scatter(
                x=counts.index,
                y=counts.values,
                mode="lines+markers",
                name=f"line {line}",
                legendgroup=f"line {line}",
                showlegend=False,
                line=dict(color=LINE_COLOR[line], width=2),
                marker=dict(color=LINE_COLOR[line], size=6),
                hovertemplate=(
                    f"<b>line {line}</b><br>%{{x}}<br>"
                    "%{y:,} patients started the line<extra></extra>"
                ),
            ),
            row=2,
            col=3,
        )

    apply_chrome(
        fig,
        "Why later-line survival cannot be compared across years directly",
        "HR+HER2− · the top row is the confounder the stratification removes; the bottom "
        "row is what it cannot remove",
        "prior treatment / line",
        800,
        -0.16,
        gloss_shift=180,  # 8 legend entries wrap to two rows
        glossary=[
            (
                "top row",
                "what share of each year's line starts came from each prior-treatment "
                "history. By 2021 most line-4 starts follow CDK4/6, a group that did not "
                "exist before 2016 — so a raw year comparison compares different patients",
            ),
            (
                "A and B",
                f"both follow each year's first-line patients forward for a fixed "
                f"{LANDMARK:.0f} months, and only years with that whole window available "
                "before the 2024-03 lock are shown — otherwise the recent years would "
                "look artificially fast and artificially selective",
            ),
            (
                "A",
                "of those who get there, how many months after their first-line start. "
                "Rising means recent patients reach each line later in their disease "
                "course, so less of it is left — which works against an apparent gain",
            ),
            (
                "B",
                f"share of a year's first-line patients who reach the line within "
                f"{LANDMARK:.0f} months. Falling means the group arriving at that line is "
                "a smaller, more selected slice than it used to be",
            ),
            (
                "C",
                "how many patients started each line each year — where a year is thin, "
                "the year-by-year estimates in the other figures wobble. The fall after "
                "2019 is the data lock arriving, not treatment stopping",
            ),
        ],
    )
    fig.update_layout(barmode="stack", bargap=0.15, margin=dict(b=300))
    fig.update_xaxes(gridcolor=GRID_C, zeroline=False, dtick=4)
    fig.update_yaxes(
        range=[0, 100], ticksuffix="%", gridcolor=GRID_C, zeroline=False, row=1
    )
    for col in (1, 2, 3):
        fig.update_xaxes(
            title_text="year line 1 started" if col < 3 else "year this line started",
            title_font=dict(size=11, color=SEC),
            row=2,
            col=col,
        )
        fig.update_xaxes(
            title_text="year this line started",
            title_font=dict(size=11, color=SEC),
            row=1,
            col=col,
        )
    fig.update_yaxes(
        title_text="share of line starts",
        title_font=dict(size=11, color=SEC),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="months from the line-1 start",
        title_font=dict(size=11, color=SEC),
        rangemode="tozero",
        gridcolor=GRID_C,
        zeroline=False,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"reached within {LANDMARK:.0f} months",
        title_font=dict(size=11, color=SEC),
        range=[0, 100],
        ticksuffix="%",
        gridcolor=GRID_C,
        zeroline=False,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text="patients starting the line",
        title_font=dict(size=11, color=SEC),
        rangemode="tozero",
        gridcolor=GRID_C,
        zeroline=False,
        row=2,
        col=3,
    )
    years = df.loc[df["line"].isin(LINES), "year"]
    fig.update_xaxes(range=[years.min() - 0.8, years.max() + 0.8], row=1)
    _left_align_titles(fig, 6)
    return write_html(fig, "km_later_lines_diagnostics.html")


def report(df, weights):
    """Terminal tables: the crude-versus-standardised era contrast per line."""
    print(f"treatment lines in scope: {len(df[df['line'].isin(LINES)]):,}")
    print(f"eras: {', '.join(ERA_NAMES)}   milestone: {MILESTONE:.0f} months\n")

    for line in LINES:
        sub = df[df["line"] == line]
        print(f"=== line {line} — {MILESTONE:.0f}-month OS % (median OS mo) [n] ===")
        table = {}
        for stratum in (ALL,) + STRATA:
            s = sub if stratum == ALL else sub[sub["stratum"] == stratum]
            row = {}
            for era in ERA_NAMES:
                grp = s[s["era"] == era]
                est = milestone(grp)
                if est is None:
                    row[era] = "·"
                    continue
                med = km_curve(grp["time"].values, grp["event"].values)["median"]
                med_txt = "NR" if not np.isfinite(med) else f"{med:.0f}"
                row[era] = f"{est[0]:.0f}% ({med_txt}) [{len(grp):,}]"
            table[stratum] = row
        print(pd.DataFrame(table).T.to_string())

        std = standardised_trend(sub, weights[line], "era")
        crude = {}
        for era in ERA_NAMES:
            est = milestone(sub[sub["era"] == era])
            if est is not None:
                crude[era] = est[0]
        print(
            "  weights (prior ET/CT mix, CDK4/6 collapsed in): "
            + ", ".join(f"{k} {v:.2f}" for k, v in weights[line].items())
        )
        print(
            "  standardised: "
            + ", ".join(f"{e} {std[e][0]:.1f}%" for e in ERA_NAMES if e in std)
        )
        first, last = ERA_NAMES[0], ERA_NAMES[-1]
        if first in crude and last in crude and first in std and last in std:
            print(
                f"  {first} -> {last}:  crude {crude[last] - crude[first]:+.1f} pt"
                f"   |  history-standardised {std[last][0] - std[first][0]:+.1f} pt\n"
            )
        else:
            print()


def main():
    df = load_lines()
    weights = {
        line: (
            df[df["line"] == line]["base_stratum"]
            .value_counts(normalize=True)
            .reindex(BASE_STRATA)
            .dropna()
            .to_dict()
        )
        for line in LINES
    }

    report(df, weights)

    sel = landmark_selection(df)
    print(
        f"=== selection into later lines, within {LANDMARK:.0f} months of the line-1 "
        "start ===\n(only line-1 years with the full window before the 2024-03 lock)"
    )
    print(
        pd.concat(
            {
                f"line {line}": tab[["share", "median_months"]].round(1)
                for line, tab in sel.items()
            },
            axis=1,
        ).to_string(na_rep="·")
    )

    print("\nsaved", make_era_panels(df))
    out, tbl = make_trend(df, weights)
    print("saved", out)
    print("saved", make_diagnostics(df))

    print(f"\n{MILESTONE:.0f}-month OS by year (%), all patients, per line:")
    piv = tbl[tbl["stratum"].eq(ALL) & tbl["period"].map(lambda p: isinstance(p, int))]
    print(
        piv.pivot_table(index="line", columns="period", values="os")
        .round(1)
        .to_string(na_rep="·")
    )


if __name__ == "__main__":
    main()
