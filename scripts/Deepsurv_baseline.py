from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from icecream import ic
from joblib import Parallel, delayed
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.nonparametric import kaplan_meier_estimator
from torchsurv.metrics.brier_score import BrierScore
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.stats.ipcw import get_ipcw
from tqdm import tqdm

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
RANDOM_STATE = 2691820962
N_BOOTSTRAP = 100
CI_ALPHA = 0.95

# DeepSurv hyperparameters
HIDDEN_DIMS = [64, 64]
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


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
    df_merge = df_dynamic.copy()
    # One-hot encode categorical columns to match ESMEOnlineDataModuleCV encoding
    feature_prefixes = ("X_", "T_treatment_")
    # feature_prefixes = ("X_")
    cat_cols = [
        c
        for c in df_merge.columns
        if any(c.startswith(p) for p in feature_prefixes)
        and df_merge[c].dtype in ("object", "category")
    ]
    if cat_cols:
        ic(f"One-hot encoding categorical columns: {cat_cols}")
        df_merge = pd.get_dummies(df_merge, columns=cat_cols, dtype=float)

    Y_col = [TIME_COL, EVENT_COL]
    X_col = [
        col
        for col in df_merge.columns
        if col.startswith("X_") or col.startswith("T_treatment_")
    ] + [PAT_ID_COL]
    # X_col = [
    #     col for col in df_merge.columns if col.startswith("X_") and col != "X_onset_to_progression"
    # ] + [PAT_ID_COL]
    ic(f"Total features (after encoding): {len(X_col)}")

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


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_single(score_fn, n_test: int, seed: int):
    """One bootstrap resample — module-level so loky can pickle it."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_test, size=n_test)
    try:
        return score_fn(idx)
    except Exception:
        return None


def bootstrap_ci(score_fn, n_test: int, seed: int = 0):
    """Percentile bootstrap CI over test-set resamples (parallelised via joblib).

    score_fn(idx) -> tuple[float, ...] accepts a 1-D integer index array drawn
    with replacement. Returns (lo, hi) each a matching tuple of CI bounds.
    """
    raw = Parallel(n_jobs=4)(
        delayed(_bootstrap_single)(score_fn, n_test, seed + i)
        for i in range(N_BOOTSTRAP)
    )
    arr = np.array([r for r in raw if r is not None])  # (n_valid, n_metrics)
    lo_pct = (1 - CI_ALPHA) / 2 * 100
    hi_pct = (1 + CI_ALPHA) / 2 * 100
    return (
        tuple(np.nanpercentile(arr, lo_pct, axis=0).tolist()),
        tuple(np.nanpercentile(arr, hi_pct, axis=0).tolist()),
    )


def fmt(val: float, lo: float, hi: float) -> str:
    return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DeepSurv(nn.Module):
    """MLP that outputs a scalar log-risk score h(x) for Cox partial likelihood."""

    def __init__(self, in_features: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,)


def cox_partial_likelihood_loss(
    log_risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    """Breslow approximation of Cox negative log partial likelihood."""
    # Sort by descending time so the risk set for each event is a suffix
    order = torch.argsort(times, descending=True)
    log_risk = log_risk[order]
    events = events[order]

    log_cumsum_exp = torch.logcumsumexp(log_risk, dim=0)
    loss = -torch.mean((log_risk - log_cumsum_exp) * events)
    return loss


# ---------------------------------------------------------------------------
# Breslow survival estimation
# ---------------------------------------------------------------------------


def breslow_baseline(log_risks: np.ndarray, times: np.ndarray, events: np.ndarray):
    """Compute the Breslow baseline cumulative hazard on training data.

    Returns (unique_event_times, H0_cumulative) as parallel arrays.
    """
    risks = np.exp(log_risks)
    event_times = np.sort(np.unique(times[events.astype(bool)]))

    H0 = np.zeros(len(event_times))
    for j, t in enumerate(event_times):
        at_risk_risk_sum = risks[times >= t].sum()
        n_deaths = events[times == t].sum()
        H0[j] = n_deaths / at_risk_risk_sum if at_risk_risk_sum > 0 else 0.0

    return event_times, np.cumsum(H0)


def predict_survival_matrix(
    log_risk_test: np.ndarray,
    event_times: np.ndarray,
    H0: np.ndarray,
    t_eval: np.ndarray,
) -> np.ndarray:
    """Survival probabilities S_i(t) = exp(-H0(t) * exp(h_i)).

    Returns (n_test, len(t_eval)).
    """
    H0_at_t = np.interp(t_eval, event_times, H0, left=0.0, right=H0[-1])
    risks = np.exp(log_risk_test)  # (n_test,)
    return np.exp(-np.outer(risks, H0_at_t)).astype(np.float32)  # (n_test, n_eval)


def predict_cumhaz_matrix(
    log_risk_test: np.ndarray,
    event_times: np.ndarray,
    H0: np.ndarray,
    t_eval: np.ndarray,
) -> np.ndarray:
    """Cumulative hazard H_i(t) = H0(t) * exp(h_i).

    Returns (n_test, len(t_eval)).
    """
    H0_at_t = np.interp(t_eval, event_times, H0, left=0.0, right=H0[-1])
    risks = np.exp(log_risk_test)  # (n_test,)
    return np.outer(risks, H0_at_t).astype(np.float32)  # (n_test, n_eval)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def fit_deepsurv(X: pd.DataFrame, Y: pd.DataFrame):
    feature_cols = [c for c in X.columns if c != PAT_ID_COL]
    X_np = X[feature_cols].astype(float).values

    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)

    times = Y[TIME_COL].values.astype(np.float32)
    events = Y[EVENT_COL].values.astype(np.float32)

    X_t = torch.tensor(X_np, dtype=torch.float32)
    t_t = torch.tensor(times, dtype=torch.float32)
    e_t = torch.tensor(events, dtype=torch.float32)

    model = DeepSurv(
        in_features=X_np.shape[1], hidden_dims=HIDDEN_DIMS, dropout=DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset = torch.utils.data.TensorDataset(X_t, t_t, e_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training DeepSurv for {EPOCHS} epochs on {len(X)} patients...")
    model.train()
    for epoch in tqdm(range(EPOCHS), desc="Epochs"):
        for X_batch, t_batch, e_batch in loader:
            optimizer.zero_grad()
            log_risk = model(X_batch)
            loss = cox_partial_likelihood_loss(log_risk, t_batch, e_batch)
            loss.backward()
            optimizer.step()

    # Compute log-risks on full training set for Breslow estimator
    model.eval()
    with torch.no_grad():
        log_risks_train = model(X_t).numpy()

    event_times, H0 = breslow_baseline(log_risks_train, times, events)

    return model, scaler, feature_cols, event_times, H0


def evaluate_deepsurv(
    model,
    scaler,
    feature_cols,
    event_times,
    H0,
    y_train_struct,
    X_test,
    y_test_struct,
    tmax,
    eval_time,
):
    """Evaluate DeepSurv with the same IPCW C-index and IBS as DynaSurv."""
    train_events = torch.tensor(y_train_struct["event"].copy(), dtype=torch.bool)
    train_times = torch.tensor(y_train_struct["time"].copy(), dtype=torch.float32)
    test_events = torch.tensor(y_test_struct["event"].copy(), dtype=torch.bool)
    test_times = torch.tensor(y_test_struct["time"].copy(), dtype=torch.float32)

    X_test_np = scaler.transform(X_test[feature_cols].astype(float).values)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        log_risks_test = model(X_test_t).numpy()

    # ---- C-index (IPCW, time-dependent) ------------------------------------
    # cumhaz_matrix[i, j] = H_i(t_j)  matching DynaSurv's eval_factual_cumhazard
    cumhaz_matrix = torch.tensor(
        predict_cumhaz_matrix(log_risks_test, event_times, H0, test_times.numpy())
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

    # ---- IBS (IPCW) --------------------------------------------------------
    bs_eval_times = torch.linspace(
        0, tmax, steps=BRIER_INTEGRATION_STEPS, dtype=torch.float32
    )
    surv_matrix = torch.tensor(
        predict_survival_matrix(log_risks_test, event_times, H0, bs_eval_times.numpy())
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

    # ---- ECE ---------------------------------------------------------------
    pred_surv_at_t = predict_survival_matrix(
        log_risks_test, event_times, H0, np.array([eval_time])
    )[:, 0]
    ece = compute_ece(
        pred_surv_at_t,
        y_test_struct["time"].copy(),
        y_test_struct["event"].copy(),
        eval_time,
    )

    return c_index, ibs, ece


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

        ic(f"Line {line_idx + 1}: fitting DeepSurv on {len(X_train)} patients...")
        model, scaler, feature_cols, event_times, H0 = fit_deepsurv(X_train, Y_train)

        # Point estimates on the full test set
        c_index, ibs, ece = evaluate_deepsurv(
            model,
            scaler,
            feature_cols,
            event_times,
            H0,
            y_train_struct,
            X_test,
            y_test_struct,
            tmax,
            tmax,
        )

        # Bootstrap CIs (resample test set only, model is fixed)
        def _score(
            idx,
            _m=model,
            _sc=scaler,
            _fc=feature_cols,
            _et=event_times,
            _h0=H0,
            _ytr=y_train_struct,
            _Xte=X_test,
            _yte=y_test_struct,
        ):
            return evaluate_deepsurv(
                _m,
                _sc,
                _fc,
                _et,
                _h0,
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
