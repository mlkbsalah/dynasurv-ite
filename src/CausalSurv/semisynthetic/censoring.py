"""Censoring of the simulated event times.

    C = min(administrative, dropout)
    administrative = data_cutoff - real line_start_date          (per row, fixed)
    dropout        ~ Exponential(rate_line)                      (independent of T)

Administrative censoring is a fact of the real cohort and is kept. Dropout is added
only where the cutoff alone leaves the simulated event rate above the real one, with
`rate_line` bisected so the rates agree. Dropout draws are independent conditional
on line and fitted generator parameters. Administrative follow-up depends on the
calendar date, which is associated with assignment and patient characteristics;
marginal independent censoring and transfer of training-cohort IPCW weights to a
later test cohort are NOT guaranteed. Use the known truth/latent outcomes for
primary simulation scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from CausalSurv.semisynthetic.config import DGPConfig
from CausalSurv.semisynthetic.drivers import LINE, START_COL
from CausalSurv.semisynthetic.outcome import EVENT_COL

DAYS_PER_MONTH = 30.44  # same conversion as the calendar driver
BISECTION_STEPS = 60


@dataclass(frozen=True)
class Censoring:
    administrative: np.ndarray  # (n_rows,) months
    dropout: np.ndarray  # (n_rows,) months, inf where the line has no dropout
    time: np.ndarray  # (n_rows,) observed time = min(T, C)
    event: np.ndarray  # (n_rows,) 1 if T <= C
    rate: np.ndarray  # (n_lines,) dropout rate per month, 0 = none
    achieved_rate: np.ndarray  # (n_lines,) simulated event rate
    target_rate: np.ndarray  # (n_lines,) real event rate
    cutoff_only_rate: np.ndarray  # (n_lines,) event rate with no dropout


def administrative_time(cohort: pd.DataFrame, cfg: DGPConfig) -> np.ndarray:
    cutoff = pd.Timestamp(cfg.cohort.data_cutoff)
    return ((cutoff - cohort[START_COL]).dt.days / DAYS_PER_MONTH).to_numpy()


def real_event_rate(cohort: pd.DataFrame, cfg: DGPConfig) -> np.ndarray:
    by_line = cohort.groupby(LINE)[EVENT_COL].mean()
    return by_line.reindex(range(1, cfg.cohort.n_lines + 1)).to_numpy()


def _find_rate(t: np.ndarray, admin: np.ndarray, expo: np.ndarray, target: float):
    """Dropout rate at which mean(T <= min(admin, expo / rate)) equals `target`.

    `expo` are fixed Exp(1) draws, so the event rate is a deterministic, decreasing
    function of the rate and plain bisection on its log applies. Returns 0 when the
    cutoff alone already gives no more than `target` events (dropout can only lower
    the event rate, so there is nothing to gain)."""
    cutoff_only = float((t <= admin).mean())
    if cutoff_only <= target:
        return 0.0, cutoff_only

    def event_rate(rate: float) -> float:
        return float((t <= np.minimum(admin, expo / rate)).mean())

    lo, hi = -12.0, 6.0  # log-rate per month; rate 6e-6 .. 400 covers every case
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        if event_rate(np.exp(mid)) > target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi))), cutoff_only


def simulate_censoring(
    cohort: pd.DataFrame,
    factual_time: np.ndarray,
    cfg: DGPConfig,
    rng: np.random.Generator,
) -> Censoring:
    """`factual_time` is the latent time under the simulated arm, one per row."""
    lines = cohort[LINE].to_numpy().astype(int)
    admin = administrative_time(cohort, cfg)
    expo = rng.exponential(size=len(cohort))
    target = real_event_rate(cohort, cfg)

    n_lines = cfg.cohort.n_lines
    rate, cutoff_only = np.zeros(n_lines), np.empty(n_lines)
    for line in range(1, n_lines + 1):
        rows = lines == line
        r, cutoff_only[line - 1] = _find_rate(
            factual_time[rows], admin[rows], expo[rows], target[line - 1]
        )
        if cfg.censoring.dropout:
            rate[line - 1] = r

    row_rate = rate[lines - 1]
    with np.errstate(divide="ignore"):
        dropout = np.where(
            row_rate > 0, expo / np.where(row_rate > 0, row_rate, 1), np.inf
        )
    censor = np.minimum(admin, dropout)
    event = (factual_time <= censor).astype(float)
    achieved = np.array([event[lines == ln].mean() for ln in range(1, n_lines + 1)])
    return Censoring(
        admin,
        dropout,
        np.minimum(factual_time, censor),
        event,
        rate,
        achieved,
        target,
        cutoff_only,
    )
