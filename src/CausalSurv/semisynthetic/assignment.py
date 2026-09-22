"""Treatment-assignment policy of the semi-synthetic DGP.

    pi_a(row) = softmax_a(alpha[line, a] + gamma * B[a] . z + strength * c[a] * u)

`alpha` is calibrated per line so the marginal simulated arm mix equals the real
within-arm mix at every gamma and hidden-confounder strength: a sweep then changes
*who* gets which arm, never *how many* get it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from CausalSurv.semisynthetic.config import ASSIGNMENT_DRIVERS, DGPConfig
from CausalSurv.semisynthetic.drivers import ARM_COL, LINE


@dataclass(frozen=True)
class Assignment:
    alpha: np.ndarray  # (n_lines, n_arms), rows sum to 0
    pi: np.ndarray  # (n_rows, n_arms) true propensities
    arm_idx: np.ndarray  # (n_rows,) index into `arms`
    arms: tuple[str, ...]

    @property
    def arm_name(self) -> np.ndarray:
        return np.asarray(self.arms)[self.arm_idx]


def coefficient_matrix(cfg: DGPConfig) -> np.ndarray:
    """B, shape (n_arms, n_drivers), arms sorted, drivers in ASSIGNMENT_DRIVERS order."""
    B = np.zeros((len(cfg.arms), len(ASSIGNMENT_DRIVERS)))
    for i, arm in enumerate(cfg.arms):
        for driver, value in cfg.assignment.coefficients.get(arm, {}).items():
            B[i, ASSIGNMENT_DRIVERS.index(driver)] = value
    return B


def hidden_vector(cfg: DGPConfig) -> np.ndarray:
    """c, shape (n_arms,): loading of the hidden confounder on each arm's logit."""
    loadings = cfg.hidden_confounder.assignment
    return np.array([loadings.get(arm, 0.0) for arm in cfg.arms])


def base_logits(drivers: pd.DataFrame, u: np.ndarray, cfg: DGPConfig) -> np.ndarray:
    """Everything in the logit except alpha, shape (n_rows, n_arms)."""
    z = drivers[list(ASSIGNMENT_DRIVERS)].to_numpy()
    observed = cfg.assignment.gamma * z @ coefficient_matrix(cfg).T
    hidden = cfg.hidden_confounder.strength * u[:, None] * hidden_vector(cfg)
    return observed + hidden


def target_mix(cohort: pd.DataFrame, cfg: DGPConfig) -> np.ndarray:
    """Real arm shares among the modelled arms, shape (n_lines, n_arms)."""
    real = cohort[cohort[ARM_COL].isin(cfg.arms)]
    counts = pd.crosstab(real[LINE], real[ARM_COL]).reindex(
        index=range(1, cfg.cohort.n_lines + 1), columns=list(cfg.arms), fill_value=0
    )
    if (counts.sum(axis=1) == 0).any():
        raise ValueError(f"no real rows in the modelled arms at some line:\n{counts}")
    return counts.div(counts.sum(axis=1), axis=0).to_numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def calibrate_alpha(
    logits: np.ndarray,
    lines: np.ndarray,
    target: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 500,
) -> np.ndarray:
    """Iterative proportional fitting of alpha[line] so mean(pi) over the line's rows
    equals `target[line]`. The mean runs over every row of the line, because under
    prefix expansion every row is given a simulated arm."""
    alpha = np.zeros_like(target)
    for line in range(1, target.shape[0] + 1):
        rows = logits[lines == line]
        a = np.log(target[line - 1])
        for _ in range(max_iter):
            mean_pi = _softmax(rows + a).mean(axis=0)
            if np.abs(mean_pi - target[line - 1]).max() < tol:
                break
            a = a + np.log(target[line - 1] / mean_pi)
        else:
            raise RuntimeError(
                f"alpha calibration did not converge at line {line} "
                f"(max error {np.abs(mean_pi - target[line - 1]).max():.2e})"
            )
        alpha[line - 1] = a - a.mean()
    return alpha


def sample_arms(pi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniform per row through the inverse CDF, so the draw is the same wherever
    pi is, which keeps sweeps over gamma / strength on common random numbers."""
    u = rng.random(len(pi))
    idx = (np.cumsum(pi, axis=1) < u[:, None]).sum(axis=1)
    return np.minimum(idx, pi.shape[1] - 1)


def assign(
    cohort: pd.DataFrame,
    drivers: pd.DataFrame,
    u: np.ndarray,
    cfg: DGPConfig,
    rng: np.random.Generator,
) -> Assignment:
    lines = cohort[LINE].to_numpy().astype(int)
    logits = base_logits(drivers, u, cfg)
    alpha = calibrate_alpha(logits, lines, target_mix(cohort, cfg))
    pi = _softmax(logits + alpha[lines - 1])
    return Assignment(alpha, pi, sample_arms(pi, rng), cfg.arms)
