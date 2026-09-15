"""Pure discrete-time survival maths on a fixed interval grid.

Everything here is a function of tensors and the grid alone: no model, no
Lightning, no logging. The model used to own all of it, and the same
hazard -> survival transform was written out four separate times across
`validation_step`, `predict_discrete_hazard`, `predict_discrete_survival` and
`fit_hazard_calibration`. Both the evaluation path and the recommendation layer
need these, so they live in one place that neither has to reach through the
model to use.

Curve convention: a discrete curve has `n_intervals + 1` points on
`interval_bounds`, with the first point fixed at the origin -- S(0) = 1 for
survival, H(0) = 0 for cumulative hazard.
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Hazards to curves
# --------------------------------------------------------------------------- #
def hazards_to_survival(discrete_hazards: torch.Tensor) -> torch.Tensor:
    """(..., n_intervals) hazard probabilities -> (..., n_intervals + 1) S(t), S(0) = 1."""
    survival = torch.cumprod(1 - discrete_hazards, dim=-1)
    return torch.cat([torch.ones_like(survival[..., :1]), survival], dim=-1)


def pad_hazards(discrete_hazards: torch.Tensor) -> torch.Tensor:
    """(..., n_intervals) -> (..., n_intervals + 1), prepending a zero hazard at t = 0."""
    return torch.cat(
        [torch.zeros_like(discrete_hazards[..., :1]), discrete_hazards], dim=-1
    )


def hazards_to_cumhazard(discrete_hazards: torch.Tensor) -> torch.Tensor:
    """(..., n_intervals) hazard probabilities -> (..., n_intervals + 1) H(t), H(0) = 0.

    Padding before the cumulative sum rather than after is bitwise identical:
    0 + h is exactly h in IEEE arithmetic, and the partial sums accumulate in the
    same order either way.
    """
    return torch.cumsum(pad_hazards(discrete_hazards), dim=-1)


# --------------------------------------------------------------------------- #
# Reading a curve at arbitrary times
# --------------------------------------------------------------------------- #
def _value_at_times(
    discrete_curve: torch.Tensor,
    eval_time: torch.Tensor,
    interval_bounds: torch.Tensor,
    n_intervals: int | None,
) -> torch.Tensor:
    """Step-function read of a (batch, n_intervals + 1) curve at (n_eval_points,) times.

    Each time is mapped to the interval whose left edge it has passed, and the
    curve is read at that edge. Indices are clamped to `n_intervals - 1`, so the
    final grid point is never read: a time at or past the last interval's start
    returns the value at that start rather than at the end of the grid. This
    reproduces the model's historical behaviour exactly; it only affects times in
    the last interval and beyond.
    """
    if n_intervals is None:
        n_intervals = interval_bounds.numel() - 1

    device = discrete_curve.device
    bounds = interval_bounds.to(device)
    eval_time = eval_time.to(device)

    interval_idx = torch.bucketize(eval_time, bounds, right=True) - 1
    interval_idx = torch.clamp(interval_idx, min=0, max=n_intervals - 1)

    gather_idx = interval_idx.unsqueeze(0).expand(discrete_curve.shape[0], -1)
    return torch.gather(discrete_curve, dim=1, index=gather_idx)


def survival_at(
    discrete_survival: torch.Tensor,
    eval_time: torch.Tensor,
    interval_bounds: torch.Tensor,
    n_intervals: int | None = None,
) -> torch.Tensor:
    """S(t) at `eval_time` for every row: (batch, n_intervals + 1) -> (batch, n_eval_points)."""
    return _value_at_times(discrete_survival, eval_time, interval_bounds, n_intervals)


def cumhazard_at(
    discrete_cumhazards: torch.Tensor,
    eval_time: torch.Tensor,
    interval_bounds: torch.Tensor,
    n_intervals: int | None = None,
) -> torch.Tensor:
    """H(t) at `eval_time` for every row: (batch, n_intervals + 1) -> (batch, n_eval_points)."""
    return _value_at_times(discrete_cumhazards, eval_time, interval_bounds, n_intervals)


# --------------------------------------------------------------------------- #
# Restricted mean survival time
# --------------------------------------------------------------------------- #
def rmst(
    discrete_survival: torch.Tensor, tau: float, interval_bounds: torch.Tensor
) -> torch.Tensor:
    """Restricted mean survival time up to `tau`: the area under S(t) on [0, tau].

    Preferred over survival at a single distant time point for two reasons: it
    stays interpretable without proportional hazards (which the era-varying
    treatment mix makes hard to defend), and it is the natural scale on which to
    compare arms for a recommendation -- months of life gained, not a probability
    at an arbitrary landmark.

    Args:
        discrete_survival: (..., n_intervals + 1) survival on `interval_bounds`.
        tau: horizon, in the same units as `interval_bounds` (months). Clamped to
            the end of the grid.
        interval_bounds: (n_intervals + 1,) grid.

    Returns:
        Tensor of shape (...) holding RMST for each leading index.
    """
    bounds = interval_bounds.to(discrete_survival.device)
    tau_t = torch.as_tensor(
        float(tau), device=discrete_survival.device, dtype=bounds.dtype
    )
    tau_t = torch.clamp(tau_t, max=bounds[-1])

    # Trapezoid over each interval, truncated at tau. Segments beyond tau
    # contribute nothing; the segment containing tau is clipped and its survival
    # linearly interpolated at the cut point.
    left, right = bounds[:-1], bounds[1:]
    seg_lo = torch.clamp(left, max=tau_t)
    seg_hi = torch.clamp(right, max=tau_t)
    width = seg_hi - seg_lo  # (n_intervals,)

    full_width = (right - left).clamp(min=1e-12)
    frac = ((seg_hi - left) / full_width).clamp(0.0, 1.0)

    s_left = discrete_survival[..., :-1]
    s_right = discrete_survival[..., 1:]
    s_at_hi = s_left + (s_right - s_left) * frac
    return (0.5 * (s_left + s_at_hi) * width).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Kaplan-Meier
# --------------------------------------------------------------------------- #
def kaplan_meier(times: np.ndarray, events: np.ndarray):
    """Kaplan-Meier estimator as (unique observation times, survival at those times).

    Numpy rather than sksurv so it runs inside the validation loop without a
    per-epoch import and without leaving the arrays the buffers already hold.
    Ties are collapsed; censored observations leave the estimate flat.
    """
    order = np.argsort(times, kind="mergesort")
    t, e = times[order], events[order].astype(np.float64)

    uniq, inverse = np.unique(t, return_inverse=True)
    deaths = np.bincount(inverse, weights=e, minlength=uniq.size)
    observed = np.bincount(inverse, minlength=uniq.size)
    at_risk = t.size - np.concatenate([[0.0], np.cumsum(observed)[:-1]])

    return uniq, np.cumprod(1.0 - deaths / at_risk)


def km_at(uniq: np.ndarray, surv: np.ndarray, landmark: float) -> float:
    """KM survival at `landmark`, or NaN past the last observation.

    Past the last observed time the estimator is undefined, not equal to its
    final value: reporting the plateau there would silently compare a prediction
    against a number no patient contributed to.
    """
    if uniq.size == 0 or landmark > uniq[-1]:
        return float("nan")
    idx = int(np.searchsorted(uniq, landmark, side="right")) - 1
    return 1.0 if idx < 0 else float(surv[idx])
