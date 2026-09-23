"""Horizon sweep on a fake model: one forward pass, many horizons, the same rule."""

import dataclasses

import pandas as pd
import pytest
import torch
import torch.nn as nn

from CausalSurv.config import RecommendationConfig
from CausalSurv.evaluation.discrete_survival import hazards_to_survival
from CausalSurv.recommendation import (
    EnsembleRecommender,
    TreatmentRecommender,
    to_frame,
)
from CausalSurv.recommendation.ensemble import summarize as ensemble_summarize

B, L, T, N_INT = 7, 2, 3, 12
RECOMMENDABLE = {0: [0, 2], 1: [1, 2]}
REFERENCE = [6.0, 4.0]


class FakeModel(nn.Module):
    """Returns stored survival curves; counts how often it is asked to."""

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        hazards = torch.rand(B, L, T, N_INT, generator=g) * 0.3
        self.register_buffer("survival", hazards_to_survival(hazards))
        self.register_buffer("interval_bounds", torch.linspace(0, 12, N_INT + 1))
        self.dummy = nn.Linear(1, 1)
        self.n_lines, self.n_treatments = L, T
        self.evaluation_horizon_times = list(REFERENCE)
        self.calls = 0

    def predict_discrete_survival(self, XPd, X_static, gather, factual_idx):
        self.calls += 1
        return self.survival


def inputs():
    xpd = torch.zeros(B, L, 1)
    return (
        xpd,
        (torch.zeros(B, 1), torch.zeros(B, 1)),
        torch.zeros(B, L, dtype=torch.long),
    )


class AllSupported:
    """Explicit support stub: absence of a propensity model must fail closed."""

    assignment_scope = "all_observed"

    def predict_mask(self, X, X_static, P, d, n_treatments):
        return torch.ones(X.shape[0], X.shape[1], n_treatments, dtype=torch.bool)


def recommender(seed: int) -> TreatmentRecommender:
    model = FakeModel(seed)
    model.x_input_dim, model.p_input_dim = 0, 0
    return TreatmentRecommender(model, RECOMMENDABLE, AllSupported(), list(REFERENCE))


def test_grid_entry_reproduces_arm_rmst_with_one_forward_pass():
    rec = recommender(0)
    r, m = rec.arm_rmst(*inputs())
    g, gm = rec.arm_rmst_grid(*inputs(), [[6.0, 4.0]])
    assert g.shape == (1, B, L, T)
    assert torch.equal(g[0], r) and torch.equal(gm, m)
    assert rec.model.calls == 2  # one per call, not one per horizon


def test_grid_is_monotone_in_the_horizon():
    rec = recommender(0)
    g, _ = rec.arm_rmst_grid(*inputs(), [[3.0, 3.0], [6.0, 6.0], [9.0, 9.0]])
    assert (g[1] >= g[0]).all() and (g[2] >= g[1]).all()
    assert rec.model.calls == 1


def test_short_grid_entry_raises():
    with pytest.raises(ValueError):
        recommender(0).arm_rmst_grid(*inputs(), [[6.0]])


def ensemble() -> EnsembleRecommender:
    return EnsembleRecommender(
        [recommender(1), recommender(2)], RecommendationConfig(min_members=1), REFERENCE
    )


def test_sweep_entry_equal_to_the_reference_reproduces_recommend():
    ens = ensemble()
    plain = ens.recommend(*inputs())
    summaries = ens.sweep(*inputs(), [[3.0, 3.0], list(REFERENCE), [9.0, 9.0]])
    assert len(summaries) == 3
    for s in summaries:
        s.validate()
    for field in dataclasses.fields(plain):
        a, b = getattr(plain, field.name), getattr(summaries[1], field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), field.name
        else:
            assert a == b, field.name


def test_member_rmst_grid_shapes():
    rmst, mask, member_masks = ensemble().member_rmst_grid(
        *inputs(), [[3.0, 3.0], [6.0, 4.0], [9.0, 9.0]]
    )
    assert rmst.shape == (3, 2, B, L, T)
    assert mask.shape == (B, L, T) and member_masks.shape == (2, B, L, T)
    assert torch.isfinite(rmst).all()


def summarize(rmst, mask, **config):
    return ensemble_summarize(
        torch.as_tensor(rmst, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
        RecommendationConfig(min_members=1, **config),
    )


def test_runner_up_gap_on_a_hand_built_case():
    # Leader is arm 0 (mean 10, std 0). Arm 1: paired diffs [2, 1] -> mean 1.5,
    # std 0.7071 -> lcb 0.7929. Arm 2: diffs [10, 10] -> lcb 10. Gap = 0.7929.
    rmst = torch.tensor([[[[10.0, 8.0, 0.0]]], [[[10.0, 9.0, 0.0]]]])
    s = summarize(rmst, torch.ones(1, 1, 3, dtype=torch.bool), margin_months=1.0)
    gap = s.runner_up_gap()
    assert gap.shape == (1, 1)
    assert abs(gap.item() - (1.5 - 2**0.5 / 2)) < 1e-5
    assert s.set_size.item() == 2 and s.decision.item() == 2  # undecided


def test_runner_up_gap_edge_rows():
    rmst = torch.rand(2, 3, 1, 4)
    mask = torch.ones(3, 1, 4, dtype=torch.bool)
    mask[0, 0] = torch.tensor([False, True, False, False])  # leader only
    mask[1, 0] = False  # nothing supported
    gap = summarize(rmst, mask).runner_up_gap()
    assert torch.isinf(gap[0, 0]) and gap[0, 0] > 0
    assert torch.isnan(gap[1, 0])
    assert torch.isfinite(gap[2, 0])


@pytest.mark.parametrize("leader_rule", ["lcb", "mean"])
def test_gap_and_vote_share_are_the_whole_rule(leader_rule):
    g = torch.Generator().manual_seed(3)
    rmst = torch.rand(6, 50, 4, 11, generator=g) * 20
    mask = torch.rand(50, 4, 11, generator=g) > 0.5
    mask[0, 0] = False
    cfg = dict(margin_months=1.0, p_best_min=0.6, leader_rule=leader_rule)
    s = summarize(rmst, mask, **cfg)
    gap = s.runner_up_gap()
    supported = s.mask.any(-1)
    singleton = s.set_size == 1
    assert torch.equal(singleton[supported], (gap >= 1.0)[supported])
    confident = s.decision == 1
    expected = (gap >= 1.0) & (s.p_best_leader + 1e-6 >= 0.6) & (s.mask.sum(-1) >= 2)
    assert torch.equal(confident, expected)


def test_frame_gap_matches_tensor_gap():
    """The figure scripts rebuild the gap from rows; pin that to the tensor version."""
    g = torch.Generator().manual_seed(4)
    rmst = torch.rand(5, 12, 3, 5, generator=g) * 20
    mask = torch.rand(12, 3, 5, generator=g) > 0.4
    s = summarize(rmst, mask)
    line_mask = torch.ones(12, 3, dtype=torch.long)
    df = to_frame(
        s,
        patient_id=torch.arange(12),
        line_mask=line_mask,
        observed_idx=torch.zeros(12, 3, dtype=torch.long),
        treatment_dict={i: f"arm{i}" for i in range(5)},
    )
    rivals = df[df.supported & ~df.is_leader]
    frame_gap = (
        rivals.groupby(["patient_id", "line"])
        .diff_lcb.min()
        .reindex(pd.MultiIndex.from_product([range(12), range(3)]))
    )
    tensor_gap = s.runner_up_gap()
    for (pid, line), value in frame_gap.items():
        t = tensor_gap[pid, line].item()
        if not s.mask[pid, line].any():
            assert t != t  # nan
        elif pd.isna(value):
            assert t == float("inf")
        else:
            assert abs(t - value) < 1e-6


def test_to_frame_horizon_column():
    g = torch.Generator().manual_seed(5)
    s = summarize(
        torch.rand(2, 4, 3, 4, generator=g), torch.ones(4, 3, 4, dtype=torch.bool)
    )
    common = dict(
        patient_id=torch.arange(4),
        line_mask=torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]]),
        observed_idx=torch.zeros(4, 3, dtype=torch.long),
        treatment_dict={i: f"arm{i}" for i in range(4)},
    )
    plain = to_frame(s, **common)
    assert "horizon" not in plain.columns and len(plain.columns) == 21
    with_h = to_frame(s, **common, horizon=[6.0, 4.0, 2.0])
    assert list(with_h.columns[:3]) == ["patient_id", "line", "horizon"]
    assert (with_h.horizon == with_h.line.map({0: 6.0, 1: 4.0, 2: 2.0})).all()
    assert len(with_h) == len(plain)
    with pytest.raises(ValueError):
        to_frame(s, **common, horizon=[6.0])
