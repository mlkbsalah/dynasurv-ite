from pathlib import Path

import numpy as np
import pandas as pd
import torch
from icecream import ic
from joblib import Parallel, delayed
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import kaplan_meier_estimator
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw

STATIC_PATH = Path(
    "/Users/malek/TheLAB/DynaSurv/data/model_entry_imputes_data_STATIC_no_staging.parquet"
)
DYNAMIC_PATH = Path(
    "/Users/malek/TheLAB/DynaSurv/data/model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet"
)
MAX_LINE = 4

PAT_ID_COL = "usubjid"
LINE_ID_COL = "lineid"
TIME_COL = "Y_onset_to_death"
EVENT_COL = "Y_global_death_status"

# Must match DynaSurv configs/config.toml evaluation_horizon_times and brier_integration_step
EVALUATION_HORIZON_TIMES = [100.0, 75.0, 50.0, 30.0]
BRIER_INTEGRATION_STEPS = 6
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_BOOTSTRAP = 30
CI_ALPHA = 0.95

# RSF hyperparameters (Ishwaran et al. 2008 recommendations)
N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 15  # key regulariser for survival forests
MAX_FEATURES = "sqrt"
N_JOBS = -1


def make_data():
    df_dynamic = pd.read_parquet(DYNAMIC_PATH)
    df_static = pd.read_parquet(STATIC_PATH)

    df_merge = df_dynamic.merge(df_static, on="usubjid", how="inner")

    df_merge = (
        df_merge.loc[df_merge[LINE_ID_COL] <= MAX_LINE]
        .sort_values(by=[PAT_ID_COL, LINE_ID_COL])
        .reset_index(drop=True)
        .copy()
    )

    feature_prefixes = ("X_", "T_")
    cat_cols = [
        c
        for c in df_merge.columns
        if any(c.startswith(p) for p in feature_prefixes)
        and df_merge[c].dtype in ("object", "category")
    ]
    if cat_cols:
        df_merge = pd.get_dummies(df_merge, columns=cat_cols, dtype=float)

    Y_col = [TIME_COL, EVENT_COL]
    X_col = [
        col for col in df_merge.columns if col.startswith("X_") or col.startswith("T_")
    ] + [PAT_ID_COL]

    XY_list = [
        (
            df_merge.loc[df_merge[LINE_ID_COL] == 1, X_col].rename(
                columns={c: f"{c}_line1" for c in X_col if c != PAT_ID_COL}
            ),
            df_merge.loc[df_merge[LINE_ID_COL] == 1, Y_col],
        )
    ]

    for line_idx in range(1, MAX_LINE):
        Y_line = df_merge.loc[df_merge[LINE_ID_COL] == line_idx + 1, Y_col]
        X_line = df_merge.loc[df_merge[LINE_ID_COL] == line_idx + 1, X_col]
        X_merge = pd.merge(
            X_line,
            XY_list[-1][0],
            on=PAT_ID_COL,
            how="inner",
            suffixes=(f"_line{line_idx + 1}", ""),
        )
        XY_list.append((X_merge, Y_line))

    return XY_list


def _to_structured(Y: pd.DataFrame) -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for t, e in zip(Y[TIME_COL], Y[EVENT_COL])],
        dtype=[("event", bool), ("time", float)],
    )


def _eval_step_fns(step_fns, t_eval: np.ndarray, fill_before: float) -> np.ndarray:
    """Vectorised evaluation of a list of sksurv StepFunctions at times t_eval."""
    n = len(step_fns)
    m = len(t_eval)
    out = np.full((n, m), fill_before, dtype=np.float32)
    for i, fn in enumerate(step_fns):
        mask_in = t_eval >= fn.x[0]
        if mask_in.any():
            t_clipped = np.clip(t_eval[mask_in], fn.x[0], fn.x[-1])
            out[i, mask_in] = fn(t_clipped).astype(np.float32)
    return out


def compute_ece(
    pred_surv_at_t: np.ndarray,
    obs_times: np.ndarray,
    obs_events: np.ndarray,
    eval_time: float,
    n_bins: int = 10,
    bin_min_samples: int = 10,
) -> float:
    """ECE at a single evaluation time: weighted mean |KM - predicted| per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pred_surv_at_t, bins) - 1

    calib_pred, calib_obs, bin_count = [], [], []
    for i in range(n_bins):
        idx = bin_indices == i
        if idx.sum() < bin_min_samples:
            continue
        agg_pred = np.mean(pred_surv_at_t[idx])
        time_km, surv_km = kaplan_meier_estimator(
            obs_events[idx].astype(bool), obs_times[idx]
        )
        if eval_time <= time_km[0]:
            est = 1.0
        elif eval_time >= time_km[-1]:
            est = surv_km[-1]
        else:
            est = surv_km[time_km <= eval_time][-1]
        calib_pred.append(agg_pred)
        calib_obs.append(est)
        bin_count.append(idx.sum())

    if not bin_count:
        return float("nan")
    bin_count = np.array(bin_count)
    return float(
        np.sum(
            bin_count
            / bin_count.sum()
            * np.abs(np.array(calib_obs) - np.array(calib_pred))
        )
    )


def _bootstrap_single(score_fn, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_test, size=n_test)
    try:
        return score_fn(idx)
    except Exception:
        return None


def bootstrap_ci(score_fn, n_test: int, seed: int = 0):
    """Percentile bootstrap CI over test-set resamples (parallelised via joblib)."""
    raw = Parallel(n_jobs=1, backend="threading")(
        delayed(_bootstrap_single)(score_fn, n_test, seed + i)
        for i in range(N_BOOTSTRAP)
    )
    arr = np.array([r for r in raw if r is not None])
    lo_pct = (1 - CI_ALPHA) / 2 * 100
    hi_pct = (1 + CI_ALPHA) / 2 * 100
    return (
        tuple(np.nanpercentile(arr, lo_pct, axis=0).tolist()),
        tuple(np.nanpercentile(arr, hi_pct, axis=0).tolist()),
    )


def fmt(val: float, lo: float, hi: float) -> str:
    return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"


def fit_rsf(X: pd.DataFrame, Y: pd.DataFrame):
    feature_cols = [c for c in X.columns if c != PAT_ID_COL]
    rsf = RandomSurvivalForest(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        oob_score=True,  # out-of-bag C-index for free — useful for monitoring
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    rsf.fit(X[feature_cols].astype(float), _to_structured(Y))
    ic(f"RSF OOB C-index: {rsf.oob_score_:.3f}")
    return rsf, feature_cols


def evaluate_rsf(
    rsf, feature_cols, y_train_struct, X_test, y_test_struct, tmax, eval_time
):
    """Evaluate RSF with IPCW C-index, IBS, and ECE — identical protocol to Cox/DeepSurv."""
    train_events = torch.tensor(y_train_struct["event"].copy(), dtype=torch.bool)
    train_times = torch.tensor(y_train_struct["time"].copy(), dtype=torch.float32)
    test_events = torch.tensor(y_test_struct["event"].copy(), dtype=torch.bool)
    test_times = torch.tensor(y_test_struct["time"].copy(), dtype=torch.float32)

    X_test_f = X_test[feature_cols].astype(float)

    # ---- C-index (IPCW, time-dependent) ----------------------------------------
    cumhaz_fns = rsf.predict_cumulative_hazard_function(X_test_f)
    cumhaz_matrix = torch.tensor(
        _eval_step_fns(cumhaz_fns, test_times.numpy(), fill_before=0.0)
    )  # (n_test, n_test)

    ci_ipcw_weights = get_ipcw(
        event=train_events, time=train_times, new_time=test_times
    )
    ci_fun = ConcordanceIndex()
    c_index = ci_fun(
        estimate=cumhaz_matrix,
        event=test_events,
        time=test_times,
        weight=ci_ipcw_weights,
    ).item()

    # ---- IBS (IPCW) -------------------------------------------------------------
    bs_eval_times = torch.linspace(
        0, tmax, steps=BRIER_INTEGRATION_STEPS, dtype=torch.float32
    )
    surv_fns = rsf.predict_survival_function(X_test_f)
    surv_matrix = torch.tensor(
        _eval_step_fns(surv_fns, bs_eval_times.numpy(), fill_before=1.0)
    )  # (n_test, BRIER_INTEGRATION_STEPS)

    bs_ipcw_weights = get_ipcw(
        event=train_events, time=train_times, new_time=bs_eval_times
    )
    bs_fun = BrierScore()
    bs_fun(
        estimate=surv_matrix,
        event=test_events,
        time=test_times,
        new_time=bs_eval_times,
        weight_new_time=bs_ipcw_weights,
    )
    ibs = bs_fun.integral().item()

    # ---- ECE --------------------------------------------------------------------
    surv_fns_ece = rsf.predict_survival_function(X_test_f)
    pred_surv_at_t = _eval_step_fns(
        surv_fns_ece, np.array([eval_time]), fill_before=1.0
    )[:, 0]
    ece = compute_ece(
        pred_surv_at_t,
        y_test_struct["time"].copy(),
        y_test_struct["event"].copy(),
        eval_time,
    )

    return c_index, ibs, ece


if __name__ == "__main__":
    XY_list = make_data()

    table = PrettyTable()
    table.field_names = [
        "Line",
        "N train",
        "N test",
        f"C-index (IPCW) [{int(CI_ALPHA * 100)}% CI]",
        f"IBS [{int(CI_ALPHA * 100)}% CI]",
        f"ECE [{int(CI_ALPHA * 100)}% CI]",
    ]

    for line_idx, (X, Y) in enumerate(XY_list):
        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=Y[EVENT_COL].tolist(),
        )
        X_train, X_test = (
            X.iloc[train_idx].reset_index(drop=True),
            X.iloc[test_idx].reset_index(drop=True),
        )
        Y_train, Y_test = (
            Y.iloc[train_idx].reset_index(drop=True),
            Y.iloc[test_idx].reset_index(drop=True),
        )

        tmax = EVALUATION_HORIZON_TIMES[line_idx]
        y_train_struct = _to_structured(Y_train)
        y_test_struct = _to_structured(Y_test)

        ic(f"Line {line_idx + 1}: fitting RSF on {len(X_train)} patients...")
        rsf, feature_cols = fit_rsf(X_train, Y_train)

        c_index, ibs, ece = evaluate_rsf(
            rsf, feature_cols, y_train_struct, X_test, y_test_struct, tmax, tmax
        )

        def _score(
            idx,
            _rsf=rsf,
            _fc=feature_cols,
            _ytr=y_train_struct,
            _Xte=X_test,
            _yte=y_test_struct,
        ):
            return evaluate_rsf(
                _rsf,
                _fc,
                _ytr,
                _Xte.iloc[idx].reset_index(drop=True),
                _yte[idx],
                tmax,
                tmax,
            )

        ci_lo, ci_hi = bootstrap_ci(_score, len(X_test))

        table.add_row(
            [
                line_idx + 1,
                len(X_train),
                len(X_test),
                fmt(c_index, ci_lo[0], ci_hi[0]),
                fmt(ibs, ci_lo[1], ci_hi[1]),
                fmt(ece, ci_lo[2], ci_hi[2]),
            ]
        )

    print(table)
