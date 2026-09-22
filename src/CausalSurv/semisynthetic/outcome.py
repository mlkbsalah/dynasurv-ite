"""Potential-survival model of the semi-synthetic DGP.

    S_a(t | z, u) = exp(-(t / lambda_line)^k_line * exp(eta_a))
    eta_a = f(z) + tau_a + heterogeneity * g_a(z) + hidden.outcome * u

so eta > 0 is a higher hazard and each arm's time is Weibull(k, lambda * exp(-eta/k)).
f and g_a are centred within each line, which makes (k, lambda) the baseline of the
average patient and lets them be fitted to the real per-line Kaplan-Meier. Survival
and RMST for every arm are closed-form, so the truth needs no simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc

from CausalSurv.evaluation.discrete_survival import kaplan_meier
from CausalSurv.semisynthetic.config import OUTCOME_DRIVERS, DGPConfig
from CausalSurv.semisynthetic.drivers import LINE

TIME_COL = "Y_onset_to_death"
EVENT_COL = "Y_global_death_status"

# The KM tail is fitted only while at least this share of the line is still at risk.
MIN_AT_RISK_SHARE = 0.05


@dataclass(frozen=True)
class WeibullFit:
    shape: np.ndarray  # k, (n_lines,)
    scale: np.ndarray  # lambda, (n_lines,)
    achieved_median: np.ndarray  # median of the fitted mixture, months
    real_median: np.ndarray  # median of the real KM, months (nan if never reached)


def _centred(values: np.ndarray, lines: np.ndarray) -> np.ndarray:
    out = values.copy()
    for line in np.unique(lines):
        rows = lines == line
        out[rows] -= out[rows].mean(axis=0)
    return out


def linear_predictor(
    drivers: pd.DataFrame, u: np.ndarray, lines: np.ndarray, cfg: DGPConfig
) -> np.ndarray:
    """eta, shape (n_rows, n_arms), arms sorted."""
    z = drivers[list(OUTCOME_DRIVERS)].to_numpy()

    def weights(coefs: dict[str, float]) -> np.ndarray:
        return np.array([coefs.get(d, 0.0) for d in OUTCOME_DRIVERS])

    f = _centred((z @ weights(cfg.outcome.prognostic))[:, None], lines)
    g = _centred(
        np.stack(
            [z @ weights(cfg.outcome.interactions.get(a, {})) for a in cfg.arms], 1
        ),
        lines,
    )
    tau = np.array([cfg.outcome.arm_effects[a] for a in cfg.arms])
    hidden = cfg.hidden_confounder.outcome * u[:, None]
    return f + tau + cfg.outcome.heterogeneity * g + hidden


def _arm_scale(eta: np.ndarray, shape: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Weibull scale of every (row, arm), given per-row k and lambda."""
    k = shape[:, None]
    return scale[:, None] * np.exp(-eta / k)


def survival(
    t: float | np.ndarray, eta: np.ndarray, shape: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """True S_a(t), shape (n_rows, n_arms) for scalar t or (n_rows, n_arms, len(t))."""
    s = _arm_scale(eta, shape, scale)
    k = shape[:, None]
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return np.exp(-((t / s) ** k))
    return np.exp(-((t[None, None, :] / s[..., None]) ** k[..., None]))


def rmst(
    tau: float, eta: np.ndarray, shape: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """True restricted mean survival time up to `tau`, shape (n_rows, n_arms).

    int_0^tau exp(-(t/s)^k) dt = s * Gamma(1 + 1/k) * P(1/k, (tau/s)^k)."""
    s = _arm_scale(eta, shape, scale)
    k = shape[:, None]
    return s * gamma_fn(1 + 1 / k) * gammainc(1 / k, (tau / s) ** k)


def sample_times(
    eta: np.ndarray, shape: np.ndarray, scale: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Latent event time under every arm, shape (n_rows, n_arms).

    One uniform per row is shared by all arms (a monotone coupling), so arms differ
    only through eta and the individual effect is not swamped by independent noise."""
    v = rng.random(len(eta))[:, None]
    return _arm_scale(eta, shape, scale) * (-np.log(v)) ** (1 / shape[:, None])


def _km_median(times: np.ndarray, surv: np.ndarray) -> float:
    below = np.flatnonzero(surv <= 0.5)
    return float(times[below[0]]) if below.size else float("nan")


def fit_weibull(
    cohort: pd.DataFrame,
    eta: np.ndarray,
    arm_idx: np.ndarray,
    cfg: DGPConfig,
) -> WeibullFit:
    """Per line, choose (k, lambda) so the population mixture of the factual arms
    matches the real Kaplan-Meier of the line over its reliably observed range."""
    lines = cohort[LINE].to_numpy().astype(int)
    factual_eta = eta[np.arange(len(eta)), arm_idx]
    n_lines = cfg.cohort.n_lines
    shape, scale = np.empty(n_lines), np.empty(n_lines)
    achieved, real = np.empty(n_lines), np.empty(n_lines)

    for line in range(1, n_lines + 1):
        rows = lines == line
        times = cohort.loc[rows, TIME_COL].to_numpy()
        uniq, km = kaplan_meier(times, cohort.loc[rows, EVENT_COL].to_numpy())
        at_risk = rows.sum() - np.searchsorted(np.sort(times), uniq, side="left")
        keep = at_risk >= MIN_AT_RISK_SHARE * rows.sum()
        grid, target = uniq[keep], km[keep]
        e = factual_eta[rows]

        def mixture(t, k, lam):
            return np.exp(-((t[None, :] / lam) ** k) * np.exp(e)[:, None]).mean(0)

        def residual(p):
            return mixture(grid, np.exp(p[0]), np.exp(p[1])) - target

        start = np.log([1.0, max(np.median(times), 1.0)])
        sol = least_squares(residual, start)
        shape[line - 1], scale[line - 1] = np.exp(sol.x)

        dense = np.linspace(0.05, 2 * uniq[-1], 4000)
        achieved[line - 1] = _km_median(dense, mixture(dense, *np.exp(sol.x)))
        real[line - 1] = _km_median(uniq, km)

    return WeibullFit(shape, scale, achieved, real)


@dataclass(frozen=True)
class Outcomes:
    eta: np.ndarray  # (n_rows, n_arms)
    times: np.ndarray  # (n_rows, n_arms) latent event time under every arm
    fit: WeibullFit
    row_shape: np.ndarray  # (n_rows,) k of each row's line
    row_scale: np.ndarray  # (n_rows,) lambda of each row's line


def simulate_outcomes(
    cohort: pd.DataFrame,
    drivers: pd.DataFrame,
    u: np.ndarray,
    arm_idx: np.ndarray,
    cfg: DGPConfig,
    rng: np.random.Generator,
) -> Outcomes:
    lines = cohort[LINE].to_numpy().astype(int)
    eta = linear_predictor(drivers, u, lines, cfg)
    fit = fit_weibull(cohort, eta, arm_idx, cfg)
    shape, scale = fit.shape[lines - 1], fit.scale[lines - 1]
    return Outcomes(eta, sample_times(eta, shape, scale, rng), fit, shape, scale)
