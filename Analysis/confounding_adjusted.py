"""Adjusted confounding analysis for the HR+/HER2- V2 cohort.

Both axes are ADJUSTED for every other covariate (unlike the marginal screen):
  treatment axis : partial pseudo-R2 of each covariate in a multinomial propensity model
  survival axis  : partial pseudo-R2 of each covariate in a Cox model
Both are drop-one likelihood-ratio / null-deviance, so they share the same unit.
A third figure gives the change-in-estimate: the % shift in one treatment log-HR
as each covariate is added -- the only quantity actually in units of bias.

Run from the Analysis/ directory:  python confounding_adjusted.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA = os.environ.get(
    "DATA_PATH",
    "../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet",
)
LINES = [int(x) for x in os.environ.get("LINES", "1,2,3,4").split(",")]
MIN_ARM = 200
CONTRAST = ("ET+ANTI-CDK wo CT", "ET alone")
OUTDIR = "figures"
PAL = {
    "Static baseline biology": "#2a78d6",
    "Time-varying disease state": "#eb6834",
    "Calendar era": "#1baf7a",
}
os.makedirs(OUTDIR, exist_ok=True)

TIME_VARYING = {
    "buffer_time",
    "new_metastatic_site_count",
    "cumulative_new_metastatic_sites",
    "progression_of_old_site_count",
    "cumulative_progression_of_old_sites",
    "local_locoregional_relapse_count",
    "brain_met_count",
    "csf_meningeal_met_count",
    "other_cns_met_count",
    "bone_met_count",
    "pulmonary_met_count",
    "metastatic_lymph_node_count",
    "pleural_met_count",
    "cutaneous_met_count",
    "liver_met_count",
    "other_met_count",
    "contralateral_BC_progression_count",
    "onset_to_progression",
    "time_between_onsets",
    "mpps",
}


def family_of(v):
    if v == "entry_year":
        return "Calendar era"
    return (
        "Time-varying disease state" if v in TIME_VARYING else "Static baseline biology"
    )


df = pd.read_parquet(DATA)
df["entry_year"] = pd.to_datetime(df["line_start_date"]).dt.year
db_lock = pd.to_datetime(
    pd.concat([df.line_start_date, df.line_end_date, df.death_date])
).max()


def informative(s):
    p = s.value_counts(normalize=True)
    return len(p) > 1 and p.iloc[1:].sum() >= 0.01


def prep_line(L):
    d = df[df.X_line_number == L].copy()
    keep = d.T_treatment_category.value_counts()
    d = d[d.T_treatment_category.isin(keep[keep >= MIN_ARM].index)].copy()
    end = pd.to_datetime(d.death_date).where(d.Y_global_death_status, db_lock)
    time = (end - pd.to_datetime(d.line_start_date)).dt.days / 30.44
    ok = time.notna() & (time > 0)
    d, time = d[ok], time[ok]
    cols = [c for c in d.columns if c.startswith("X_") and c != "X_line_number"] + [
        "entry_year"
    ]
    X = d[cols].copy()
    X.columns = [c[2:] if c.startswith("X_") else c for c in X.columns]
    X = X.loc[:, [c for c in X.columns if informative(X[c])]]
    X = X.loc[:, ~X.T.duplicated()].reset_index(drop=True)
    return (
        X,
        time.values,
        d.Y_global_death_status.values.astype(bool),
        d.T_treatment_category.values,
    )


def fit_cox(frame, penalizer):
    return CoxPHFitter(penalizer=penalizer).fit(frame, "_t", "_e")


def cox_partial_r2(X, time, event):
    base = X.copy()
    base["_t"] = time
    base["_e"] = event.astype(int)
    full, pen = None, 0.0
    for p in (0.0, 0.1):
        try:
            full = fit_cox(base, p)
            pen = p
            break
        except Exception:
            continue
    ll_full = full.log_likelihood_
    d_null = -2 * ll_full + full.log_likelihood_ratio_test().test_statistic
    return {
        v: max(
            2
            * (ll_full - fit_cox(base.drop(columns=[v]), pen).log_likelihood_)
            / d_null,
            0,
        )
        for v in X.columns
    }


def mnom_partial_r2(X, arm):
    Xs = StandardScaler().fit_transform(X.values.astype(float))
    y = pd.factorize(arm)[0]
    freq = np.bincount(y) / len(y)
    ll_null = np.sum(np.log(freq[y]))

    def ll(idx):
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000).fit(
            Xs[:, idx], y
        )
        proba = clf.predict_proba(Xs[:, idx])
        return np.sum(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1)))

    cols = np.arange(Xs.shape[1])
    ll_full = ll(cols)
    d_null = -2 * ll_null
    return {
        v: max(2 * (ll_full - ll(np.delete(cols, j))) / d_null, 0)
        for j, v in enumerate(X.columns)
    }


def score_line(L):
    X, time, event, arm = prep_line(L)
    outcome = cox_partial_r2(X, time, event)
    treat = mnom_partial_r2(X, arm)
    return pd.DataFrame(
        {
            "line": f"Line {L}",
            "covariate": list(X.columns),
            "family": [family_of(v) for v in X.columns],
            "treat": [treat[v] for v in X.columns],
            "outcome": [outcome[v] for v in X.columns],
        }
    )


def change_in_estimate(L, arms):
    X, time, event, arm = prep_line(L)
    mask = np.isin(arm, arms)
    X = X[mask].reset_index(drop=True)
    base = X.copy()
    base["treat"] = (arm[mask] == arms[0]).astype(int)
    base["_t"] = time[mask]
    base["_e"] = event[mask].astype(int)
    b0 = fit_cox(base[["treat", "_t", "_e"]], 0.0).params_["treat"]
    rows = []
    for v in X.columns:
        try:
            b = fit_cox(base[["treat", v, "_t", "_e"]], 0.0).params_["treat"]
            rows.append(
                {
                    "covariate": v,
                    "family": family_of(v),
                    "pct": 100 * (b - b0) / abs(b0),
                }
            )
        except Exception:
            pass
    b_full = fit_cox(base, 0.0).params_["treat"]
    return pd.DataFrame(rows), 100 * (b_full - b0) / abs(b0)


res = pd.concat([score_line(L) for L in LINES], ignore_index=True)
res["score"] = res.treat * res.outcome
res["label"] = ""
for _, g in res.groupby("line"):
    res.loc[g.score.nlargest(4).index, "label"] = res.loc[
        g.score.nlargest(4).index, "covariate"
    ]

quad = px.scatter(
    res,
    x="treat",
    y="outcome",
    color="family",
    size="score",
    text="label",
    facet_col="line",
    facet_col_wrap=2,
    hover_name="covariate",
    size_max=20,
    hover_data={
        "treat": ":.3f",
        "outcome": ":.3f",
        "score": ":.4f",
        "label": False,
        "family": False,
    },
    color_discrete_map=PAL,
    labels={
        "treat": "Adjusted assoc. with TREATMENT (partial pseudo-R²)",
        "outcome": "Adjusted assoc. with SURVIVAL (partial pseudo-R²)",
        "family": "Confounder family",
    },
    title="Adjusted confounding map — HR+/HER2− V2 (each axis conditions on all other covariates)",
)
quad.update_traces(textposition="top center", textfont_size=9)
quad.write_html(
    os.path.join(OUTDIR, "confounding_quadrant_adjusted.html"), include_plotlyjs=True
)

cie, full_shift = change_in_estimate(1, CONTRAST)
cie = cie.reindex(cie.pct.abs().sort_values(ascending=False).index).head(15)
bar = px.bar(
    cie,
    x="pct",
    y="covariate",
    color="family",
    orientation="h",
    color_discrete_map=PAL,
    labels={
        "pct": "% change in treatment log-HR when the covariate is added",
        "covariate": "",
        "family": "Confounder family",
    },
    title=f"Change-in-estimate — {CONTRAST[0]} vs {CONTRAST[1]} (line 1) · full adjustment shifts log-HR {full_shift:+.0f}%",
)
bar.update_yaxes(categoryorder="total ascending")
bar.write_html(
    os.path.join(OUTDIR, "change_in_estimate_line1.html"), include_plotlyjs=True
)

res.sort_values(["line", "score"], ascending=[True, False]).to_csv(
    os.path.join(OUTDIR, "confounding_scores_adjusted.csv"), index=False
)

print("done. figures + csv in", OUTDIR)
print(
    res.sort_values("score", ascending=False)[
        ["line", "covariate", "treat", "outcome", "score"]
    ]
    .head(10)
    .to_string(index=False)
)
