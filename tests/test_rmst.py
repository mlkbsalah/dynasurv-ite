"""Pure-tensor tests of `rmst` and `rmst_grid`: interpolation, clamping, zero."""

import numpy as np
import torch

from CausalSurv.evaluation.discrete_survival import rmst, rmst_grid

BOUNDS = torch.linspace(0, 10, 11)
S = torch.tensor([1.0, 0.9, 0.7, 0.6, 0.6, 0.5, 0.3, 0.2, 0.2, 0.1, 0.0])

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def analytic(tau: float) -> float:
    """Area under the linearly interpolated curve on [0, tau]."""
    grid = np.append(BOUNDS.numpy()[BOUNDS.numpy() < tau], tau)
    return float(_trapezoid(np.interp(grid, BOUNDS.numpy(), S.numpy()), grid))


def test_tau_zero_is_zero():
    assert rmst(S, 0.0, BOUNDS).item() == 0.0


def test_horizon_between_grid_points_interpolates():
    assert abs(rmst(S, 2.5, BOUNDS).item() - 2.0875) < 1e-6
    assert abs(rmst(S, 2.5, BOUNDS).item() - analytic(2.5)) < 1e-6


def test_horizon_past_the_grid_is_clamped():
    at_end = rmst(S, 10.0, BOUNDS)
    assert torch.equal(rmst(S, 100.0, BOUNDS), at_end)
    assert abs(at_end.item() - analytic(10.0)) < 1e-6


def test_monotone_in_tau():
    values = torch.stack([rmst(S, float(t), BOUNDS) for t in torch.linspace(0, 12, 50)])
    assert (values[1:] >= values[:-1] - 1e-7).all()


def test_leading_dims_broadcast():
    g = torch.Generator().manual_seed(0)
    curves = torch.rand(3, 2, 11, generator=g).sort(-1, descending=True).values
    curves[..., 0] = 1.0
    out = rmst(curves, 4.5, BOUNDS)
    assert out.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert torch.equal(out[i, j], rmst(curves[i, j], 4.5, BOUNDS))


def test_rmst_grid_stacks_rmst_exactly():
    taus = [0.0, 2.5, 4.0, 10.0, 100.0]
    grid = rmst_grid(S, taus, BOUNDS)
    assert grid.shape == (len(taus),)
    for i, tau in enumerate(taus):
        assert torch.equal(grid[i], rmst(S, tau, BOUNDS))
