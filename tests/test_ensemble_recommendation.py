"""Pure-tensor tests of the ensemble decision rule: no checkpoints, no data."""

import pytest
import torch

from CausalSurv.config import RecommendationConfig
from CausalSurv.recommendation import (
    NO_SUPPORTED_ARM,
    Decision,
    find_checkpoints,
    to_frame,
)
from CausalSurv.recommendation.ensemble import summarize as ensemble_summarize


def summarize(rmst, mask, member_masks=None, **config):
    cfg = RecommendationConfig(min_members=1, **config)
    return ensemble_summarize(
        torch.as_tensor(rmst, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
        cfg,
        member_masks,
    )


@pytest.mark.parametrize("leader_rule", ["lcb", "mean"])
def test_single_member_reproduces_masked_argmax(leader_rule):
    g = torch.Generator().manual_seed(0)
    rmst = torch.rand(1, 8, 4, 5, generator=g) * 20
    mask = torch.rand(8, 4, 5, generator=g) > 0.4
    mask[0, 0] = False  # one row with no supported arm
    s = summarize(
        rmst, mask, margin_months=0.0, p_best_min=0.0, leader_rule=leader_rule
    )
    ref = rmst[0].masked_fill(~mask, float("-inf")).argmax(-1)
    ref = ref.masked_fill(~mask.any(-1), NO_SUPPORTED_ARM)
    assert torch.equal(s.leader_idx, ref)
    assert torch.equal(
        s.recommended_idx, ref.masked_fill(mask.sum(-1) < 2, NO_SUPPORTED_ARM)
    )
    assert torch.equal(s.rmst_mean, rmst[0])
    assert not s.rmst_std.any()
    assert s.n_members == 1
    s.validate()


def test_two_members_split_p_best_and_stay_undecided():
    rmst = torch.zeros(2, 1, 1, 3)
    rmst[0, 0, 0] = torch.tensor([10.0, 9.0, 0.0])
    rmst[1, 0, 0] = torch.tensor([9.0, 10.0, 0.0])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    s = summarize(rmst, mask, leader_rule="mean", margin_months=1.0)
    assert torch.allclose(s.p_best[0, 0], torch.tensor([0.5, 0.5, 0.0]))
    assert s.leader_idx.item() == 0  # exact tie on the mean -> lowest index
    # arm 1: paired diff [+1, -1] -> mean 0, std sqrt(2) -> lcb < margin -> in set
    # arm 2: paired diff [10, 9] -> lcb ~ 8.8 -> out
    assert s.equivalence_set[0, 0].tolist() == [True, True, False]
    assert s.set_size.item() == 2
    assert s.decision.item() == Decision.UNDECIDED
    assert s.recommended_idx.item() == NO_SUPPORTED_ARM
    s.validate()


def test_row_without_support():
    rmst = torch.rand(3, 2, 1, 4)
    mask = torch.ones(2, 1, 4, dtype=torch.bool)
    mask[1] = False
    s = summarize(rmst, mask)
    assert s.leader_idx[1, 0] == NO_SUPPORTED_ARM
    assert s.decision[1, 0] == Decision.NO_SUPPORT
    assert s.set_size[1, 0] == 0
    assert not s.equivalence_set[1, 0].any()
    assert s.p_best[1, 0].sum() == 0
    assert s.recommended_idx[1, 0] == NO_SUPPORTED_ARM
    assert s.decision[0, 0] != Decision.NO_SUPPORT
    s.validate()


def test_clear_winner_is_confident():
    g = torch.Generator().manual_seed(1)
    base = torch.tensor([10.0, 15.0, 8.0])
    rmst = base + 0.1 * torch.randn(5, 1, 1, 3, generator=g)
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    s = summarize(rmst, mask, p_best_min=0.7, margin_months=1.0)
    assert s.leader_idx.item() == 1
    assert s.p_best_leader.item() == 1.0
    assert s.set_size.item() == 1
    assert s.decision.item() == Decision.CONFIDENT
    assert s.recommended_idx.item() == 1
    s.validate()


def test_pessimism_prefers_the_tight_arm():
    rmst = torch.zeros(4, 1, 1, 2)
    rmst[:, 0, 0, 0] = torch.tensor([6.0, 14.0, 6.0, 14.0])  # mean 10, std 4.6
    rmst[:, 0, 0, 1] = torch.tensor([8.9, 9.1, 8.9, 9.1])  # mean 9, std 0.1
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    assert summarize(rmst, mask, leader_rule="mean").leader_idx.item() == 0
    assert (
        summarize(rmst, mask, leader_rule="lcb", pessimism_c=1.0).leader_idx.item() == 1
    )
    assert (
        summarize(rmst, mask, leader_rule="lcb", pessimism_c=0.0).leader_idx.item() == 0
    )


def test_exact_tie_and_margin_zero():
    rmst = torch.tensor([[[[5.0, 5.0, 1.0]]], [[[5.0, 5.0, 1.0]]]])
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    # Zero margin still requires positive evidence of superiority, not a tie.
    s = summarize(rmst, mask, margin_months=0.0, leader_rule="mean")
    assert s.leader_idx.item() == 0
    assert s.equivalence_set[0, 0].tolist() == [True, True, False]
    assert s.decision.item() == Decision.UNDECIDED
    s.validate()
    # any positive margin keeps the tie undecided
    s = summarize(rmst, mask, margin_months=0.5, leader_rule="mean")
    assert s.set_size.item() == 2
    assert s.decision.item() == Decision.UNDECIDED
    s.validate()


def test_p_best_threshold_blocks_a_singleton_set():
    # arm 0 wins clearly in 2 of 3 members and loses clearly in the third.
    rmst = torch.tensor([[[[20.0, 10.0]]], [[[20.0, 10.0]]], [[[10.0, 20.0]]]])
    mask = torch.ones(1, 1, 2, dtype=torch.bool)
    common = dict(pessimism_c=0.0, margin_months=1.0, leader_rule="mean")
    s = summarize(rmst, mask, p_best_min=0.7, **common)
    assert s.set_size.item() == 1
    assert abs(s.p_best_leader.item() - 2 / 3) < 1e-6
    assert s.decision.item() == Decision.UNDECIDED
    s.validate()
    s = summarize(rmst, mask, p_best_min=0.6, **common)
    assert s.decision.item() == Decision.CONFIDENT
    s.validate()


def test_member_mask_disagreement_is_reported():
    rmst = torch.rand(2, 1, 1, 3)
    member_masks = torch.ones(2, 1, 1, 3, dtype=torch.bool)
    member_masks[1, 0, 0, 2] = False
    s = summarize(rmst, member_masks.all(0), member_masks)
    assert s.member_mask_disagreement[0, 0].tolist() == [False, False, True]
    assert not s.mask[0, 0, 2]
    s.validate()


def test_rejects_non_finite_rmst():
    rmst = torch.zeros(2, 1, 1, 2)
    rmst[0, 0, 0, 1] = float("-inf")
    with pytest.raises(ValueError, match="finite"):
        summarize(rmst, torch.ones(1, 1, 2, dtype=torch.bool))


@pytest.mark.parametrize("leader_rule", ["lcb", "mean"])
def test_invariants_hold_on_random_inputs(leader_rule):
    g = torch.Generator().manual_seed(2)
    rmst = torch.rand(6, 50, 4, 11, generator=g) * 24
    mask = torch.rand(50, 4, 11, generator=g) > 0.3
    mask[:5] = False
    s = summarize(rmst, mask, leader_rule=leader_rule)
    s.validate()
    assert set(s.decision.unique().tolist()) <= {int(d) for d in Decision}


def test_to_frame_rows_and_columns():
    g = torch.Generator().manual_seed(3)
    B, L, T = 4, 3, 5
    s = summarize(
        torch.rand(2, B, L, T, generator=g), torch.ones(B, L, T, dtype=torch.bool)
    )
    line_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]])
    df = to_frame(
        s,
        patient_id=torch.arange(100, 100 + B),
        line_mask=line_mask,
        observed_idx=torch.zeros(B, L, dtype=torch.long),
        treatment_dict={i: f"arm{i}" for i in range(T)},
    )
    assert len(df) == int(line_mask.sum()) * T
    expected = {
        "patient_id",
        "line",
        "arm_idx",
        "arm_name",
        "supported",
        "rmst_mean",
        "rmst_std",
        "p_best",
        "diff_mean",
        "diff_std",
        "diff_lcb",
        "in_equivalence_set",
        "is_leader",
        "leader_idx",
        "set_size",
        "p_best_leader",
        "recommended_idx",
        "decision",
        "observed_idx",
        "observed_name",
        "n_members",
    }
    assert expected <= set(df.columns)
    assert (df.groupby(["patient_id", "line"])["is_leader"].sum() == 1).all()
    assert df["arm_name"].iloc[0] == "arm0"
    assert set(df["decision"]) <= {
        "no_support",
        "confident",
        "undecided",
        "only_supported_option",
    }


def test_only_supported_option_is_not_comparative_confidence():
    s = summarize([[[[50.0, 1.0]]], [[[50.0, 1.0]]]], [[[True, False]]])
    assert s.leader_idx.item() == 0
    assert s.set_size.item() == 1
    assert s.decision.item() == Decision.ONLY_SUPPORTED_OPTION
    assert s.recommended_idx.item() == NO_SUPPORTED_ARM
    s.validate()


def test_find_checkpoints_one_per_kind(tmp_path):
    ckpt_dir = tmp_path / "01012026_000000_seed_7" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    for name in [
        "dynaSurvCausalOnline-epoch=04-val_loss= 1.5875.ckpt",
        "dynaSurvCausalOnline-bestCI-epoch=04-average_ci= 0.6579.ckpt",
        "dynaSurvCausalOnline-bestCALIB-epoch=19.ckpt",
        "dynaSurvCausalOnline-bestCALIB-epoch=19-recalibrated.ckpt",
        "dynaSurvCausalOnline-last-epoch=14.ckpt",
        "last.ckpt",
    ]:
        (ckpt_dir / name).touch()
    run = ckpt_dir.parent
    assert find_checkpoints([run], "val_loss")[0].name.startswith(
        "dynaSurvCausalOnline-epoch=04"
    )
    assert (
        find_checkpoints([run], "bestCALIB")[0].name
        == "dynaSurvCausalOnline-bestCALIB-epoch=19.ckpt"
    )
    assert (
        find_checkpoints([run], "final_epoch")[0].name
        == "dynaSurvCausalOnline-last-epoch=14.ckpt"
    )
    with pytest.raises(FileNotFoundError, match="bestIBS"):
        find_checkpoints([run], "bestIBS")
    with pytest.raises(ValueError):
        find_checkpoints([run], "last")
