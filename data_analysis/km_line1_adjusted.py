"""Comparable overall-survival curves by line-1 treatment: IPTW + IPCW adjusted.

The crude Kaplan-Meier curves in ``km_line1_by_year.py`` are not comparable across
treatments, for two distinct reasons:

**Heterogeneity / confounding by indication.** Treatment at line 1 is chosen on the
basis of the patient's state (metastatic burden, histology, receptor status, and --
above all -- calendar era, since ET+ANTI-CDK only enters practice from 2017 and
CT+ANTI-ANGIO all but disappears after 2015). The groups are therefore not
exchangeable and a difference between crude curves mixes treatment effect with
case mix. Corrected here with **stabilised inverse-probability-of-treatment
weights** from a multinomial-logistic propensity model over all 11 categories,
which re-weights every treatment group to the covariate distribution of the whole
line-1 cohort. The estimand each adjusted curve targets is the marginal survival
that would have been observed had the entire cohort received that treatment.

**Censoring bias.** Censoring in this cohort is essentially administrative: 95% of
censored patients were last seen in late 2023 / early 2024, and their observed
follow-up is 92-94% of the maximum possible given their entry date. Follow-up
length is therefore determined by calendar year of entry -- which is exactly what
also determines treatment availability. Plain Kaplan-Meier assumes censoring is
independent of survival within a curve; here it is not, because a curve pools
entry years. Corrected with **inverse-probability-of-censoring weights** from a Cox
model for the censoring hazard conditional on the same covariates plus treatment.

Both weights are stabilised and truncated; the weighted curves are then computed
with a discrete-time weighted product-limit estimator on a 0.5-month grid, and each
curve stops once the effective sample size in its risk set falls below ``ESS_FLOOR``
rather than being drawn into a region where the weights carry it alone.

Outputs:
  * ``km_line1_adjusted.html``               - crude vs IPTW+IPCW-adjusted curves.
  * ``km_line1_adjustment_diagnostics.html`` - covariate balance before/after,
    effective sample size, and the calendar-era overlap that limits how far the
    adjustment can be believed.

**Read the diagnostics before the curves.** Weighting cannot manufacture overlap
that does not exist: no patient treated before 2017 could have received
ET+ANTI-CDK, and none after 2016 received CT+ANTI-ANGIO, so for those two
categories the adjusted curve is an extrapolation outside the region of common
support, not an estimate the data can support. Age is not present in this parquet
and so is absent from the propensity model -- a known unmeasured confounder for
the ET-versus-chemotherapy choice.

Run:  python data_analysis/km_line1_adjusted.py
Figures -> data_analysis/plots/
"""

import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from km_line1_by_year import (
    BASE,
    GRID_C,
    INK,
    MUTED,
    SEC,
    SURFACE,
    apply_chrome,
    cat_style,
    write_html,
)
from lifelines import CoxPHFitter, KaplanMeierFitter
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)

HORIZON = 60.0
GRID = np.arange(0.0, HORIZON + 0.5, 0.5)
MIN_POOLED_N = 50  # categories smaller than this get no curve at all
ESS_FLOOR = 25.0  # stop a curve when its weighted risk set falls below this
SW_TRUNC = (0.01, 0.99)  # IPTW truncation quantiles
IPCW_CAP = 10.0  # hard cap on the censoring weight
SMD_THRESHOLD = 0.1  # conventional "balanced" line

# post-baseline / degenerate columns that must not enter the models
DROP_COVARIATES = {
    "X_line_number",
    "X_onset_to_progression",  # time to the NEXT progression: an outcome
    "X_time_between_onsets",
    "X_buffer_time",
}

SEQ_BLUE = [
    [0.0, "#cde2fb"],
    [0.25, "#9ec5f4"],
    [0.5, "#5598e7"],
    [0.75, "#256abf"],
    [1.0, "#0d366b"],
]


# --------------------------------------------------------------------------- data


def load_cohort():
    """Line-1 rows with OS from line-1 onset, treatment, year, and covariates."""
    df = pd.read_parquet(DATA_PATH)
    line1 = df[df["X_line_number"] == 1].drop_duplicates("usubjid").copy()
    line1["year"] = line1["line_start_date"].dt.year
    line1 = line1.rename(
        columns={
            "T_treatment_category": "category",
            "Y_onset_to_death": "time",
            "Y_global_death_status": "event",
        }
    )
    return line1.reset_index(drop=True)


def build_covariates(raw):
    """Baseline design matrix: X_ features (minus post-baseline) + year dummies."""
    x_cols = [c for c in raw.columns if c.startswith("X_") and c not in DROP_COVARIATES]
    X = raw[x_cols].astype(float)
    X = X.loc[:, X.std(axis=0) > 0]  # constants carry no information
    X = X.loc[:, ~X.T.duplicated()]  # exact duplicates are collinear
    years = pd.get_dummies(raw["year"].astype(int), prefix="yr", drop_first=True)
    return pd.concat([X, years.astype(float)], axis=1)


# ---------------------------------------------------------------------- weighting


def fit_iptw(X, a_idx, n_cat):
    """Stabilised IPTW from a multinomial-logistic propensity model over all categories."""
    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(max_iter=5000, C=1.0, solver="lbfgs")
    model.fit(Xs, a_idx)
    ps = model.predict_proba(Xs)
    p_marg = np.bincount(a_idx, minlength=n_cat) / len(a_idx)
    p_obs = np.clip(ps[np.arange(len(a_idx)), a_idx], 1e-8, None)
    sw = p_marg[a_idx] / p_obs
    lo, hi = np.quantile(sw, SW_TRUNC)
    return np.clip(sw, lo, hi), ps, float(np.mean((sw < lo) | (sw > hi)))


def fit_ipcw(X, a_dummies, times, events):
    """Stabilised IPCW on GRID from a Cox model for the censoring hazard."""
    design = pd.concat(
        [X.reset_index(drop=True), a_dummies.reset_index(drop=True)], axis=1
    )
    fit_df = design.copy()
    fit_df["_T"] = np.asarray(times, dtype=float)
    fit_df["_C"] = (~np.asarray(events, dtype=bool)).astype(int)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(fit_df, duration_col="_T", event_col="_C")

    sc = cph.predict_survival_function(design, times=GRID).values  # (len(GRID), n)
    marg = KaplanMeierFitter().fit(times, (~np.asarray(events, dtype=bool)).astype(int))
    tl = marg.survival_function_.index.values
    sv = marg.survival_function_.iloc[:, 0].values
    idx = np.clip(np.searchsorted(tl, GRID, side="right") - 1, 0, len(tl) - 1)
    sc_marg = sv[idx][:, None]  # (len(GRID), 1)

    w = sc_marg / np.clip(sc, 1e-8, None)
    capped = float(np.mean(w > IPCW_CAP))
    return np.clip(w, 0.0, IPCW_CAP).T, capped  # -> (n, len(GRID))


# ----------------------------------------------------------------------- estimator


def weighted_km(times, events, sw, ipcw):
    """Discrete-time weighted product-limit on GRID, cut when the risk set thins out."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    t_out, s_out, ess_out, n_out = [GRID[0]], [1.0], [np.nan], [len(times)]
    surv = 1.0
    for k in range(1, len(GRID)):
        t0, t1 = GRID[k - 1], GRID[k]
        w = sw * ipcw[:, k]
        at_risk = times > t0
        if not at_risk.any():
            break
        wr = w[at_risk]
        denom = wr.sum()
        ess = denom**2 / np.sum(wr**2) if denom > 0 else 0.0
        if denom <= 0 or ess < ESS_FLOOR:
            break
        died = at_risk & (times <= t1) & events
        surv *= 1.0 - w[died].sum() / denom
        t_out.append(t1)
        s_out.append(max(surv, 0.0))
        ess_out.append(ess)
        n_out.append(int(at_risk.sum()))
    return (
        np.array(t_out),
        np.array(s_out),
        np.array(ess_out),
        np.array(n_out),
    )


def crude_km(times, events):
    """Unweighted KM on the same grid, for the side-by-side comparison."""
    ones = np.ones(len(times))
    return weighted_km(times, events, ones, np.ones((len(times), len(GRID))))


# --------------------------------------------------------------------- diagnostics


def _wmean(x, w):
    return np.sum(w[:, None] * x, axis=0) / np.sum(w)


def _wvar(x, w):
    m = _wmean(x, w)
    denom = np.sum(w) - np.sum(w**2) / np.sum(w)
    return np.sum(w[:, None] * (x - m) ** 2, axis=0) / max(denom, 1e-8)


def smd(X, mask, w):
    """|standardised mean difference| per covariate, group vs the pooled remainder."""
    xa, wa = X[mask], w[mask]
    xb, wb = X[~mask], w[~mask]
    if len(xa) < 2 or len(xb) < 2:
        return np.full(X.shape[1], np.nan)
    pooled = np.sqrt((_wvar(xa, wa) + _wvar(xb, wb)) / 2)
    diff = np.abs(_wmean(xa, wa) - _wmean(xb, wb))
    return np.where(pooled > 1e-8, diff / np.where(pooled > 1e-8, pooled, 1.0), np.nan)


# ------------------------------------------------------------------------- figures


def make_curves_figure(cats, drawn, crude, adjusted, n_by_cat, style, bal):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"<b>Crude</b><span style='color:{SEC};font-weight:400'>"
            "  as observed — not comparable</span>",
            f"<b>IPTW + IPCW adjusted</b><span style='color:{SEC};font-weight:400'>"
            "  standardised to the whole cohort</span>",
        ),
        horizontal_spacing=0.08,
    )
    for col, book in ((1, crude), (2, adjusted)):
        for cat in drawn:
            t, s, ess, n_at = book[cat]
            color, dash = style[cat]
            unbalanced = bal[cat]["after"] > SMD_THRESHOLD
            note = ""
            if col == 2 and unbalanced:
                dash = "dash"
                note = (
                    "<br><i>not comparable: max |SMD| still "
                    f"{bal[cat]['after']:.2f} on {bal[cat]['worst']}</i>"
                )
            cust = np.column_stack([n_at, ess, np.full(len(t), n_by_cat[cat])])
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=s * 100,
                    mode="lines",
                    line=dict(color=color, width=2, shape="hv", dash=dash),
                    name=cat,
                    legendgroup=cat,
                    showlegend=col == 1,
                    customdata=cust,
                    hovertemplate=(
                        f"<b>{cat}</b><br>"
                        "%{x:.0f} months from line-1 onset<br>"
                        "OS %{y:.1f}%<br>"
                        "at risk: %{customdata[0]:,} of %{customdata[2]:,}"
                        "  (effective %{customdata[1]:.0f})"
                        f"{note}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.add_hline(y=50, line=dict(color=BASE, width=1, dash="dot"), row=1, col=col)

    apply_chrome(
        fig,
        "Survival by treatment — before and after making the groups comparable",
        f"HR+HER2− · {sum(n_by_cat.values()):,} patients · dashed curves are still not "
        "comparable",
        "treatment category at line 1",
        660,
        -0.19,
        gloss_shift=150,
        glossary=[
            (
                "crude",
                "survival exactly as observed. The groups contain different kinds of "
                "patient, so a gap between two curves is not a treatment effect",
            ),
            (
                "adjusted",
                "each group re-weighted to look like the whole cohort on 80 recorded "
                "characteristics, and corrected for patients leaving follow-up early. "
                "Gaps here are closer to a like-for-like comparison",
            ),
            (
                "dashed",
                "the re-weighting did not work for this group — it is only ever given in "
                "one stretch of calendar years, so there is nobody comparable in the "
                "other years. Read these as extrapolation, not as an estimate",
            ),
            (
                "why curves stop",
                "a curve ends once too few patients remain for the weighted estimate to "
                "mean anything",
            ),
        ],
    )
    fig.update_layout(margin=dict(l=64, r=28, t=118, b=252))
    fig.update_xaxes(
        range=[0, HORIZON],
        gridcolor=GRID_C,
        zeroline=False,
        dtick=12,
        title_text="months since line-1 onset",
        title_font=dict(size=11, color=SEC),
    )
    fig.update_yaxes(
        range=[0, 100],
        gridcolor=GRID_C,
        zeroline=False,
        ticksuffix="%",
        title_text="overall survival",
        title_font=dict(size=11, color=SEC),
    )
    for ann in fig.layout.annotations[:2]:
        ann.update(font=dict(size=13, color=INK))
    return write_html(fig, "km_line1_adjusted.html")


def make_diagnostics_figure(cats, drawn, bal, ess_tbl, era, style):
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.30, 0.26, 0.44],
        horizontal_spacing=0.10,
        subplot_titles=(
            "<b>Covariate balance</b>",
            "<b>Effective sample size</b>",
            "<b>Calendar-era overlap</b>",
        ),
    )

    ys = drawn[::-1]
    for name, key, wkey, color in (
        ("before weighting", "before", "worst_before", "#eb6834"),
        ("after weighting", "after", "worst", "#2a78d6"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[bal[c][key] for c in ys],
                y=ys,
                mode="markers",
                name=name,
                marker=dict(color=color, size=10, line=dict(color=SURFACE, width=1.5)),
                customdata=[
                    [bal[c][wkey], bal[c][key.split("_")[0] + "_mean"]] for c in ys
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"max |SMD| {name}: " + "%{x:.3f}<br>"
                    "worst covariate: %{customdata[0]}<br>"
                    "mean |SMD| over all covariates: %{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    fig.add_vline(
        x=SMD_THRESHOLD,
        line=dict(color=BASE, width=1, dash="dot"),
        annotation_text=f"{SMD_THRESHOLD:g} = balanced",
        annotation_position="top",
        annotation_font=dict(size=10, color=SEC),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            y=ys,
            x=[ess_tbl[c]["n"] for c in ys],
            orientation="h",
            name="patients",
            marker_color=MUTED,
            hovertemplate="<b>%{y}</b><br>%{x:,} patients<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            y=ys,
            x=[ess_tbl[c]["ess"] for c in ys],
            orientation="h",
            name="effective (weighted)",
            marker_color="#2a78d6",
            hovertemplate=(
                "<b>%{y}</b><br>effective sample size after weighting: "
                "%{x:,.0f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Heatmap(
            z=era.values * 100,
            x=[str(y) for y in era.columns],
            y=list(era.index),
            colorscale=SEQ_BLUE,
            zmin=0,
            colorbar=dict(
                title=dict(text="% of<br>group", font=dict(size=10)), x=1.005
            ),
            hovertemplate=(
                "<b>%{y}</b><br>%{x}: %{z:.1f}% of this treatment's patients"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=3,
    )

    apply_chrome(
        fig,
        "Did the adjustment work?",
        f"one row per treatment · {sum(bal[c]['after'] <= SMD_THRESHOLD for c in drawn)}"
        f" of {len(drawn)} groups end up comparable",
        "",
        520,
        -0.22,
        gloss_shift=126,
        glossary=[
            (
                "balance (left)",
                f"how different this group still is from everyone else, across "
                f"{bal[drawn[0]]['n_cov']} recorded characteristics. Orange = before "
                f"weighting, blue = after. Below {SMD_THRESHOLD} counts as comparable, so "
                "blue dots left of the line are the groups you can trust",
            ),
            (
                "effective sample size (middle)",
                "weighting buys comparability by leaning on fewer patients. Grey is how "
                "many there are; blue is how many they are worth once weighted",
            ),
            (
                "calendar-era overlap (right)",
                "which years each treatment was actually used. A row concentrated in one "
                "block of years has no counterpart in other years — that is why "
                "weighting cannot fix it",
            ),
        ],
    )
    fig.update_layout(barmode="overlay", margin=dict(l=150, r=90, t=118, b=212))
    smd_max = max(max(bal[c]["before"], bal[c]["after"]) for c in drawn)
    fig.update_xaxes(
        title_text="max |SMD| vs the pooled remainder",
        title_font=dict(size=11, color=SEC),
        gridcolor=GRID_C,
        zeroline=False,
        range=[0, smd_max * 1.12],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="patients (grey) / effective (blue)",
        title_font=dict(size=11, color=SEC),
        gridcolor=GRID_C,
        zeroline=False,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text="year of line-1 onset",
        title_font=dict(size=11, color=SEC),
        row=1,
        col=3,
    )
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    for ann in fig.layout.annotations[:3]:
        ann.update(font=dict(size=13, color=INK))
    return write_html(fig, "km_line1_adjustment_diagnostics.html")


# ----------------------------------------------------------------------------- run


def main():
    t_start = _time.time()
    raw = load_cohort()
    cats = list(raw["category"].value_counts().index)
    style = cat_style(cats)
    a_idx = raw["category"].map({c: i for i, c in enumerate(cats)}).values
    X = build_covariates(raw)
    times = raw["time"].values
    events = raw["event"].values.astype(bool)
    print(f"cohort: {len(raw):,} line-1 patients · {len(cats)} treatment categories")
    print(
        f"covariates in the models: {X.shape[1]} ({X.shape[1] - 15} clinical + year dummies)"
    )

    sw, ps, sw_trunc_frac = fit_iptw(X, a_idx, len(cats))
    print(
        f"IPTW: stabilised weights mean {sw.mean():.3f} "
        f"[{sw.min():.3f}, {sw.max():.3f}] · {sw_trunc_frac:.1%} truncated"
    )

    a_dummies = pd.get_dummies(raw["category"], prefix="tx", drop_first=True).astype(
        float
    )
    ipcw, ipcw_capped = fit_ipcw(X, a_dummies, times, events)
    print(
        f"IPCW: {ipcw_capped:.2%} of (patient, time) weights hit the cap of {IPCW_CAP:g}"
    )

    Xv = X.values
    names = list(X.columns)
    n_by_cat, crude, adjusted, bal, ess_tbl = {}, {}, {}, {}, {}
    drawn = []
    for cat in cats:
        mask = raw["category"].values == cat
        n_by_cat[cat] = int(mask.sum())
        before = smd(Xv, mask, np.ones(len(Xv)))
        after = smd(Xv, mask, sw)
        bal[cat] = {
            "before": float(np.nanmax(before)),
            "after": float(np.nanmax(after)),
            "before_mean": float(np.nanmean(before)),
            "after_mean": float(np.nanmean(after)),
            "worst_before": names[int(np.nanargmax(before))],
            "worst": names[int(np.nanargmax(after))],
            "n_cov": int(np.sum(~np.isnan(before))),
        }
        wc = sw[mask]
        ess_tbl[cat] = {
            "n": int(mask.sum()),
            "ess": float(wc.sum() ** 2 / np.sum(wc**2)),
        }
        if mask.sum() >= MIN_POOLED_N:
            drawn.append(cat)
            crude[cat] = crude_km(times[mask], events[mask])
            adjusted[cat] = weighted_km(times[mask], events[mask], sw[mask], ipcw[mask])

    skipped = [c for c in cats if c not in drawn]
    print(
        f"\ncurves drawn for {len(drawn)} of {len(cats)} categories; "
        f"below MIN_POOLED_N={MIN_POOLED_N}: {', '.join(skipped) if skipped else 'none'}"
    )

    print("\nbalance (max |SMD| vs pooled remainder) and effective sample size")
    print(
        f"{'category':<20}{'n':>7}{'ESS':>8}{'maxSMD pre':>12}{'post':>8}"
        f"{'meanSMD post':>14}  worst covariate after weighting"
    )
    for cat in cats:
        b = bal[cat]
        flag = "  <-- NOT BALANCED" if b["after"] > SMD_THRESHOLD else ""
        print(
            f"{cat:<20}{ess_tbl[cat]['n']:>7,}{ess_tbl[cat]['ess']:>8,.0f}"
            f"{b['before']:>12.3f}{b['after']:>8.3f}{b['after_mean']:>14.3f}"
            f"  {b['worst']}{flag}"
        )

    era = (
        pd.crosstab(raw["category"], raw["year"], normalize="index")
        .reindex(drawn)
        .iloc[::-1]
    )

    print(
        "\nsaved",
        make_curves_figure(cats, drawn, crude, adjusted, n_by_cat, style, bal),
    )
    print("saved", make_diagnostics_figure(cats, drawn, bal, ess_tbl, era, style))

    print("\nsurvival at 12 / 24 / 36 months (%), crude vs adjusted")
    print(
        f"{'category':<20}{'12 crude':>10}{'12 adj':>9}{'24 crude':>10}{'24 adj':>9}{'36 crude':>10}{'36 adj':>9}"
    )
    for cat in drawn:
        row = f"{cat:<20}"
        for ms in (12.0, 24.0, 36.0):
            for book in (crude, adjusted):
                t, s, _, _ = book[cat]
                if t[-1] >= ms:
                    row += f"{100 * s[np.searchsorted(t, ms, side='right') - 1]:>{10 if book is crude else 9}.1f}"
                else:
                    row += f"{'·':>{10 if book is crude else 9}}"
        print(row)
    print(f"\nelapsed {_time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
