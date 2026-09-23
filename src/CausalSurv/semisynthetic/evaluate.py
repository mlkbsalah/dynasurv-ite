"""Score predictions of every arm's survival curve against the simulated truth.

Every quantity is computed at each sample's own (last) line and that line's horizon
`tau` from `[eval] horizon_times`. Four tables, one row per (predictor, line, ...):

curve    S(t) error to tau and RMST error, split into the FACTUAL arm (the one the sample
         was assigned) and the COUNTERFACTUAL arms -- only the second is unobservable
         on real data, and the gap between the two is the confounding penalty.
effect   arm-pair contrasts of RMST: PEHE (root mean squared error of the individual
         effect) next to the bias of the average effect. They answer different
         questions: hidden confounding can affect both bias and PEHE.
policy   value of the arm a predictor would pick (true RMST), regret against the best
         arm, best-arm hit rate and abstention rate.
factual  exact expected Brier score and uncensored latent-outcome checks of the
         factual curve. These are simulation diagnostics, not real-data estimators.

`line == "all"` pools the lines. Arms outside the line-level support (too thin to be
recommendable) are left out of every table, so a metric is never charged for an arm the
model was told not to trust.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from sksurv.exceptions import NoComparablePairException
from sksurv.metrics import concordance_index_censored

from CausalSurv.semisynthetic.outcome import rmst, survival
from CausalSurv.semisynthetic.predictors import Prediction, truth_arrays

REFERENCE = "reference"


@dataclass(frozen=True)
class Truth:
    """Ground truth of the evaluated samples, on the evaluation grid."""

    frame: pd.DataFrame  # truth.parquet rows of the evaluated samples
    arms: tuple[str, ...]
    t_grid: np.ndarray
    tau: np.ndarray  # (n,) horizon of each sample's line
    survival: np.ndarray  # (n, A, T) true S_a(t)
    rmst: np.ndarray  # (n, A) true RMST to tau, closed form
    factual: np.ndarray  # (n,) index of the assigned arm
    line: np.ndarray  # (n,)

    @property
    def dt(self) -> float:
        return float(self.t_grid[1] - self.t_grid[0])

    def tau_index(self, tau: np.ndarray | float) -> np.ndarray:
        """Grid index of a horizon; horizons must sit on the grid."""
        idx = np.asarray(tau) / self.dt
        if not np.allclose(idx, np.round(idx)):
            raise ValueError(
                f"horizons {np.unique(tau)} are off the grid step {self.dt}"
            )
        return np.round(idx).astype(int)


def build_truth(
    frame: pd.DataFrame,
    arms: tuple[str, ...],
    t_grid: np.ndarray,
    horizon_times: list[float],
) -> Truth:
    line = frame["lineid"].to_numpy().astype(int)
    tau = np.asarray(horizon_times, dtype=float)[line - 1]
    if tau.max() > t_grid[-1] + 1e-9:
        raise ValueError(f"grid ends at {t_grid[-1]}, horizon {tau.max()} is beyond it")
    eta, k, lam = truth_arrays(frame, arms)
    true_rmst = np.empty((len(frame), len(arms)))
    for horizon in np.unique(tau):
        rows = tau == horizon
        true_rmst[rows] = rmst(horizon, eta[rows], k[rows], lam[rows])
    return Truth(
        frame=frame.reset_index(drop=True),
        arms=arms,
        t_grid=t_grid,
        tau=tau,
        survival=survival(t_grid, eta, k, lam),
        rmst=true_rmst,
        factual=frame["arm_idx"].to_numpy().astype(int),
        line=line,
    )


def _to_tau(y: np.ndarray, truth: Truth) -> np.ndarray:
    """Integral of y (n, A, T) over [0, tau_i] for each sample: (n, A)."""
    cumulative = cumulative_trapezoid(y, dx=truth.dt, axis=-1, initial=0.0)
    return cumulative[np.arange(len(y)), :, truth.tau_index(truth.tau)]


def _line_groups(truth: Truth):
    for line in [*np.unique(truth.line), "all"]:
        yield (
            line,
            (np.ones(len(truth.line), bool) if line == "all" else truth.line == line),
        )


def _allowed(truth: Truth, line_support: np.ndarray, pred: Prediction | None = None):
    allowed = line_support[truth.line - 1]
    if pred is not None and pred.patient_support is not None:
        allowed = allowed & pred.patient_support
    return allowed


def curve_errors(
    predictions: dict[str, Prediction], truth: Truth, line_support: np.ndarray
) -> pd.DataFrame:
    rows = []
    supported = _allowed(truth, line_support)
    is_factual = np.arange(len(truth.arms))[None, :] == truth.factual[:, None]
    for name, pred in predictions.items():
        ise = _to_tau((pred.survival - truth.survival) ** 2, truth) / truth.tau[:, None]
        rmst_err = _to_tau(pred.survival, truth) - truth.rmst
        for line, in_line in _line_groups(truth):
            for kind, pick in (
                ("factual", is_factual),
                ("counterfactual", ~is_factual),
            ):
                for a, arm in enumerate(truth.arms):
                    cell = in_line & supported[:, a] & pick[:, a]
                    if not cell.any():
                        continue
                    rows.append(
                        dict(
                            predictor=name,
                            line=line,
                            arm=arm,
                            kind=kind,
                            n=int(cell.sum()),
                            rmse_S=float(np.sqrt(ise[cell, a].mean())),
                            rmst_mae=float(np.abs(rmst_err[cell, a]).mean()),
                            rmst_bias=float(rmst_err[cell, a].mean()),
                        )
                    )
    return pd.DataFrame(rows)


def effect_errors(
    predictions: dict[str, Prediction], truth: Truth, line_support: np.ndarray
) -> pd.DataFrame:
    rows = []
    supported = _allowed(truth, line_support)
    for name, pred in predictions.items():
        rmst_hat = _to_tau(pred.survival, truth)
        for line, in_line in _line_groups(truth):
            for a, b in itertools.combinations(range(len(truth.arms)), 2):
                cell = in_line & supported[:, a] & supported[:, b]
                if not cell.any():
                    continue
                true_effect = truth.rmst[cell, a] - truth.rmst[cell, b]
                hat_effect = rmst_hat[cell, a] - rmst_hat[cell, b]
                rows.append(
                    dict(
                        predictor=name,
                        line=line,
                        arm_a=truth.arms[a],
                        arm_b=truth.arms[b],
                        n=int(cell.sum()),
                        pehe=float(np.sqrt(((hat_effect - true_effect) ** 2).mean())),
                        ate_true=float(true_effect.mean()),
                        ate_bias=float(hat_effect.mean() - true_effect.mean()),
                        sign_agree=float(
                            (np.sign(hat_effect) == np.sign(true_effect)).mean()
                        ),
                    )
                )
    return pd.DataFrame(rows)


def _choose(score: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best allowed arm per sample, and where none is allowed (an abstention)."""
    return np.where(allowed, score, -np.inf).argmax(axis=1), ~allowed.any(axis=1)


def policy_table(
    predictions: dict[str, Prediction], truth: Truth, line_support: np.ndarray
) -> pd.DataFrame:
    n = len(truth.line)
    idx = np.arange(n)
    allowed = _allowed(truth, line_support)
    best_value = np.where(allowed, truth.rmst, -np.inf).max(axis=1)
    best_arm, _ = _choose(truth.rmst, allowed)
    status_quo = truth.rmst[idx, truth.factual]

    # picks[(predictor, policy)] = (arm chosen, abstained)
    picks: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    picks[(REFERENCE, "factual")] = (truth.factual, np.zeros(n, bool))
    constant = np.empty(n, int)
    for line in np.unique(truth.line):
        in_line = truth.line == line
        # Line-level support is the same for every sample of a line.
        mean_rmst = np.where(
            allowed[in_line][0], truth.rmst[in_line].mean(axis=0), -np.inf
        )
        constant[in_line] = mean_rmst.argmax()
    picks[(REFERENCE, "best_constant")] = (constant, np.zeros(n, bool))

    for name, pred in predictions.items():
        rmst_hat = _to_tau(pred.survival, truth)
        picks[(name, "line_only")] = _choose(rmst_hat, allowed)
        if pred.patient_support is not None:
            picks[(name, "supported")] = _choose(
                rmst_hat, _allowed(truth, line_support, pred)
            )

    rows = []
    random_value = np.where(allowed, truth.rmst, 0).sum(1) / allowed.sum(1)
    for (name, policy), (arm, abstain) in picks.items():
        value = np.where(abstain, status_quo, truth.rmst[idx, arm])
        for line, in_line in _line_groups(truth):
            rows.append(
                dict(
                    predictor=name,
                    policy=policy,
                    line=line,
                    n=int(in_line.sum()),
                    value=float(value[in_line].mean()),
                    regret=float((best_value - value)[in_line].mean()),
                    hit_rate=float(((arm == best_arm) & ~abstain)[in_line].mean()),
                    abstain_rate=float(abstain[in_line].mean()),
                )
            )
    for line, in_line in _line_groups(truth):
        rows.append(
            dict(
                predictor=REFERENCE,
                policy="random",
                line=line,
                n=int(in_line.sum()),
                value=float(random_value[in_line].mean()),
                regret=float((best_value - random_value)[in_line].mean()),
                hit_rate=float((1 / allowed.sum(1))[in_line].mean()),
                abstain_rate=0.0,
            )
        )
    return pd.DataFrame(rows)


def factual_metrics(
    predictions: dict[str, Prediction], truth: Truth, train: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Score at the exact requested horizon, without censoring-distribution transfer.

    If p is the true factual survival probability and q the prediction, the
    conditional expected Brier loss is p(1-q)^2 + (1-p)q^2. We also score the
    uncensored simulated outcome and its concordance with horizon-specific risk.
    `brier` aliases `brier_expected`; `c_index` aliases `c_index_latent` for existing
    table consumers. `train` is retained for API compatibility, but not used.
    """
    rows = []
    for line in np.unique(truth.line):
        in_line = truth.line == line
        tau = float(truth.tau[in_line][0])
        grid_at = int(truth.tau_index(tau))
        idx = np.arange(in_line.sum())
        arm = truth.factual[in_line]
        p = truth.survival[in_line][idx, arm, grid_at]
        latent_time = truth.frame.loc[in_line, "latent_time"].to_numpy()
        for name, pred in predictions.items():
            s = pred.survival[in_line][idx, arm, grid_at]
            expected = float(np.mean(p * (1 - s) ** 2 + (1 - p) * s**2))
            latent = float(np.mean(((latent_time > tau).astype(float) - s) ** 2))
            try:
                ci = (
                    float(
                        concordance_index_censored(
                            np.ones(len(latent_time), dtype=bool), latent_time, 1 - s
                        )[0]
                    )
                    if len(latent_time) >= 2
                    else float("nan")
                )
            except NoComparablePairException:
                ci = float("nan")
            rows.append(
                dict(
                    predictor=name,
                    line=int(line),
                    n=int(in_line.sum()),
                    tau=tau,
                    tau_used=tau,
                    estimator="exact_expected_and_uncensored_latent",
                    c_index=ci,
                    c_index_latent=ci,
                    brier=expected,
                    brier_expected=expected,
                    brier_latent=latent,
                )
            )
    return pd.DataFrame(rows)


def evaluate(
    predictions: dict[str, Prediction],
    truth: Truth,
    line_support: np.ndarray,
    train: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "curve": curve_errors(predictions, truth, line_support),
        "effect": effect_errors(predictions, truth, line_support),
        "policy": policy_table(predictions, truth, line_support),
        "factual": factual_metrics(predictions, truth, train),
    }
