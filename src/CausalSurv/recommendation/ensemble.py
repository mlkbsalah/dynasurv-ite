"""Ensemble treatment recommendation with uncertainty.

`TreatmentRecommender` ranks arms by one model's RMST and returns the argmax. When
two arms sit within noise of each other that argmax is a coin flip dressed as
advice. This module pools M members -- checkpoints trained on the same split from
different inits -- and turns their spread into a decision rule:

- per patient, line and arm: the mean RMST, its std over members, and `p_best`,
  the share of members in which the arm is the best supported one;
- a leader picked pessimistically (`mean - c * std`, the offline-bandit rule) or
  by the mean;
- an *equivalence set*: the supported arms not confidently worse than the leader
  by `margin_months`, judged on the paired per-member difference;
- a recommendation only when that set is a singleton and the leader wins often
  enough across members. Otherwise the decision is "undecided", which is a
  different outcome from "no supported arm".

Members are pooled on RMST, never on logits: every checkpoint carries its own
per-line hazard temperature and bias.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as TorchData

from ..config import CHECKPOINT_KINDS, RecommendationConfig
from ..model.checkpoint_compat import load_dynasurv_checkpoint
from .recommender import NO_SUPPORTED_ARM, TreatmentRecommender

# --------------------------------------------------------------------------- #
# Loading members
# --------------------------------------------------------------------------- #
# Glob per checkpoint kind, following the ModelCheckpoint callbacks in
# scripts/TrainDynasurvCausal.py. `last.ckpt` is deliberately absent: Lightning
# writes it as a copy of the val_loss winner, so the true final epoch is the
# `-last-epoch=` file. The `-recalibrated` variants written by hand are skipped.
KIND_PATTERNS: dict[str, str] = {
    "val_loss": "dynaSurvCausalOnline-epoch=*-val_loss=*.ckpt",
    "bestCI": "dynaSurvCausalOnline-bestCI-epoch=*.ckpt",
    "bestIBS": "dynaSurvCausalOnline-bestIBS-epoch=*.ckpt",
    "bestCALIB": "dynaSurvCausalOnline-bestCALIB-epoch=*.ckpt",
    "final_epoch": "dynaSurvCausalOnline-last-epoch=*.ckpt",
}
if set(KIND_PATTERNS) != set(CHECKPOINT_KINDS):
    raise RuntimeError("KIND_PATTERNS and config.CHECKPOINT_KINDS disagree")


class IncompatibleMemberError(ValueError):
    """A checkpoint cannot join the ensemble; the message says why."""


@dataclass
class Member:
    """One loaded ensemble member and where it came from."""

    recommender: TreatmentRecommender
    path: Path
    seed: int | None
    epoch: int | None

    @property
    def model(self):
        return self.recommender.model


def _matches(ckpt_dir: Path, kind: str) -> list[Path]:
    return sorted(
        p for p in ckpt_dir.glob(KIND_PATTERNS[kind]) if "recalibrated" not in p.name
    )


def find_checkpoints(run_dirs: Sequence[str | Path], kind: str) -> list[Path]:
    """One checkpoint of `kind` per run directory, in the order given.

    Raises when a run has none or several, naming the kinds it does have, so a
    run predating a callback (`bestCALIB` before 2026-09-07) fails loudly rather
    than silently shrinking the ensemble.
    """
    if kind not in KIND_PATTERNS:
        raise ValueError(f"kind must be one of {sorted(KIND_PATTERNS)}, got {kind!r}")
    found: list[Path] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            raise FileNotFoundError(f"{run_dir}: no checkpoints/ directory")
        matches = _matches(ckpt_dir, kind)
        if len(matches) != 1:
            present = sorted(k for k in KIND_PATTERNS if _matches(ckpt_dir, k))
            raise FileNotFoundError(
                f"{run_dir.name}: expected exactly one {kind!r} checkpoint, found "
                f"{len(matches)}; kinds present: {present}"
            )
        found.append(matches[0])
    return found


def _seed_from_path(path: Path) -> int | None:
    # <run>/checkpoints/<file>: the run directory is named {date}_seed_{seed}.
    match = re.search(r"_seed_(\d+)", path.parent.parent.name)
    return int(match.group(1)) if match else None


def _epoch_from_name(name: str) -> int | None:
    match = re.search(r"epoch=(\d+)", name)
    return int(match.group(1)) if match else None


def _arm_sets(per_line) -> dict[int, list[int]]:
    return {int(line): sorted(int(k) for k in arms) for line, arms in per_line.items()}


def load_member(
    path: str | Path,
    *,
    recommendable: dict[int, list[int]] | None = None,
    device: str | torch.device = "cpu",
) -> Member:
    """Load one checkpoint as a frozen, eval-mode `TreatmentRecommender`.

    `recommendable` overrides the per-line recommendable arm sets (pass the
    datamodule's, which are exact). Without it the sets come from the checkpoint,
    and for checkpoints written before they were persisted, from the arms the
    propensity model was fitted on -- the same set, for the lines it covers.
    A checkpoint without a propensity model is refused: its patient-level support
    would silently differ from every other member's.
    """
    path = Path(path)
    try:
        model = load_dynasurv_checkpoint(path, map_location=device)
    except TypeError as exc:
        raise IncompatibleMemberError(
            f"{path}: cannot be rebuilt by this code version ({exc})"
        ) from exc
    model.eval()
    model.freeze()

    propensity = getattr(model, "recommendation_propensity_model", None)
    if propensity is None:
        raise IncompatibleMemberError(
            f"{path}: checkpoint carries no propensity model, so its patient-level "
            "support (filter 4) would silently differ from the other members'"
        )
    if recommendable is None:
        recommendable = getattr(model, "recommendable_treatments_per_line", None)
    if not recommendable:
        recommendable = {
            int(line): [int(k) for k in result.arms]
            for line, result in propensity.line_results.items()
        }
        missing = sorted(set(range(model.n_lines)) - set(recommendable))
        warnings.warn(
            f"{path.name}: recommendable arm sets are not in the checkpoint; rebuilt "
            "from the arms the propensity model was fitted on"
            + (
                f". Lines {missing} have no fitted propensity model and would get "
                "NO recommendable arm -- pass `recommendable` from the datamodule"
                if missing
                else ""
            ),
            stacklevel=2,
        )
    model.recommendable_treatments_per_line = recommendable
    recommender = TreatmentRecommender.from_model(model)
    return Member(
        recommender=recommender,
        path=path,
        seed=_seed_from_path(path),
        epoch=_epoch_from_name(path.name),
    )


_STRICT_HPARAMS = (
    "x_input_dim",
    "x_static_dim",
    "p_input_dim",
    "p_static_dim",
    "n_treatments",
    "n_lines",
    "output_length",
)


def _plain(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def _unwrap(
    members: Sequence[Member | TreatmentRecommender],
) -> list[TreatmentRecommender]:
    return [m.recommender if isinstance(m, Member) else m for m in members]


def check_members(
    members: Sequence[Member | TreatmentRecommender],
    expected_recommendable: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    """Refuse members that would not be predicting the same thing.

    Strict (raises `IncompatibleMemberError`): data dimensions, survival grid,
    architecture, the recommendable arm sets, and per line the arms and threshold
    of the propensity model. With `expected_recommendable` (the datamodule's sets)
    every member's propensity arms must also match it, which is the only check
    that catches a config drift between training and now. Training
    hyperparameters (lr, weight decay, scheduler) only produce warnings: an lr
    sweep is still a set of valid random inits.
    """
    if not members:
        raise ValueError("no members")
    labels = [
        str(m.path.parent.parent.name) if isinstance(m, Member) else f"member {i}"
        for i, m in enumerate(members)
    ]
    recs = _unwrap(members)
    ref, ref_label = recs[0], labels[0]
    ref_h = dict(ref.model.hparams)
    warns: list[str] = []

    def refuse(label: str, what: str) -> None:
        raise IncompatibleMemberError(f"{label} vs {ref_label}: {what}")

    ref_rec = _arm_sets(ref.recommendable_treatments_per_line or {})
    ref_prop = ref.propensity_model
    for rec, label in zip(recs[1:], labels[1:]):
        h = dict(rec.model.hparams)
        for key in _STRICT_HPARAMS:
            if h.get(key) != ref_h.get(key):
                refuse(label, f"{key} {h.get(key)} != {ref_h.get(key)}")
        if not torch.equal(
            rec.model.interval_bounds.cpu(), ref.model.interval_bounds.cpu()
        ):
            refuse(label, "different survival grid (interval_bounds)")
        if _plain(h.get("arch")) != _plain(ref_h.get("arch")):
            refuse(
                label,
                f"architecture differs: {_plain(h.get('arch'))} != {_plain(ref_h.get('arch'))}",
            )
        if _arm_sets(rec.recommendable_treatments_per_line or {}) != ref_rec:
            refuse(label, "recommendable arm sets differ")
        prop = rec.propensity_model
        if set(prop.line_results) != set(ref_prop.line_results):
            refuse(label, "propensity model fitted on different lines")
        for line, result in prop.line_results.items():
            ref_result = ref_prop.line_results[line]
            if sorted(map(int, result.arms)) != sorted(map(int, ref_result.arms)):
                refuse(label, f"line {line}: propensity arms differ")
            if float(result.threshold) != float(ref_result.threshold):
                refuse(label, f"line {line}: propensity threshold differs")
        if _plain(h.get("training")) != _plain(ref_h.get("training")):
            warns.append(f"{label}: training hyperparameters differ from {ref_label}")
        if list(rec.horizon_times or []) != list(ref.horizon_times or []):
            warns.append(f"{label}: default horizon_times differ from {ref_label}")

    if expected_recommendable is not None:
        expected = _arm_sets(expected_recommendable)
        for rec, label in zip(recs, labels):
            for line, result in rec.propensity_model.line_results.items():
                arms = sorted(map(int, result.arms))
                if arms != expected.get(int(line)):
                    refuse(
                        label,
                        f"line {line}: propensity model was fitted on arms {arms} but "
                        f"the datamodule now yields {expected.get(int(line))}; the "
                        "cohort or filters changed since training",
                    )

    return {
        "n_members": len(recs),
        "warnings": warns,
        "dims": {k: ref_h.get(k) for k in _STRICT_HPARAMS},
        "recommendable_treatments_per_line": ref_rec,
        "horizon_times": list(ref.horizon_times or []),
    }


# --------------------------------------------------------------------------- #
# Decision rule
# --------------------------------------------------------------------------- #
class Decision(IntEnum):
    NO_SUPPORT = 0  # every arm unsupported for this patient and line
    CONFIDENT = 1  # singleton equivalence set and the leader wins often enough
    UNDECIDED = 2  # supported arms exist but the data cannot separate them


@dataclass
class RecommendationSummary:
    """Ensemble statistics and the decision, per patient and line.

    Shapes: `(batch, n_lines, n_treatments)` for the per-arm tensors,
    `(batch, n_lines)` for the per-line ones. `-1` in `leader_idx` and
    `recommended_idx` is `NO_SUPPORTED_ARM`.
    """

    rmst_mean: torch.Tensor
    rmst_std: torch.Tensor
    p_best: torch.Tensor
    diff_mean: torch.Tensor  # leader minus arm, mean over members
    diff_std: torch.Tensor
    diff_lcb: (
        torch.Tensor
    )  # diff_mean - c * diff_std: pessimistic deficit vs the leader
    mask: torch.Tensor  # support, AND over members
    equivalence_set: torch.Tensor
    member_mask_disagreement: torch.Tensor  # supported by some members, not all
    leader_idx: torch.Tensor
    recommended_idx: torch.Tensor
    set_size: torch.Tensor
    decision: torch.Tensor  # `Decision` codes
    p_best_leader: torch.Tensor
    n_members: int
    config: RecommendationConfig

    def validate(self) -> None:
        """Raise `AssertionError` listing every violated invariant."""
        has_support = self.mask.any(-1)
        safe = self.leader_idx.clamp(min=0).unsqueeze(-1)
        confident = self.decision == Decision.CONFIDENT
        eps = 1e-6
        checks = {
            "leader is -1 exactly where no arm is supported": torch.equal(
                self.leader_idx.eq(NO_SUPPORTED_ARM), ~has_support
            ),
            "leader is a supported arm": bool(
                self.mask.gather(-1, safe).squeeze(-1)[has_support].all()
            ),
            "leader belongs to its equivalence set": bool(
                self.equivalence_set.gather(-1, safe).squeeze(-1)[has_support].all()
            ),
            "equivalence set is empty exactly where no arm is supported": torch.equal(
                self.set_size.eq(0), ~has_support
            ),
            "equivalence set only holds supported arms": not bool(
                (self.equivalence_set & ~self.mask).any()
            ),
            "confident implies a singleton set": bool(
                self.set_size[confident].eq(1).all()
            ),
            "confident implies p_best(leader) >= p_best_min": bool(
                (self.p_best_leader[confident] + eps >= self.config.p_best_min).all()
            ),
            "recommended_idx is the leader where confident, -1 elsewhere": torch.equal(
                self.recommended_idx,
                torch.where(
                    confident, self.leader_idx, torch.full_like(self.leader_idx, -1)
                ),
            ),
            "decision codes match support and confidence": torch.equal(
                self.decision.eq(Decision.NO_SUPPORT), ~has_support
            )
            and bool((self.decision[confident] == Decision.CONFIDENT).all()),
            "p_best sums to 1 on supported rows and 0 elsewhere": bool(
                torch.allclose(
                    self.p_best.sum(-1), has_support.to(self.p_best.dtype), atol=1e-5
                )
            ),
            "p_best is 0 on unsupported arms": not bool(self.p_best[~self.mask].any()),
            "statistics are finite": all(
                bool(torch.isfinite(t).all())
                for t in (
                    self.rmst_mean,
                    self.rmst_std,
                    self.p_best,
                    self.diff_mean,
                    self.diff_std,
                    self.diff_lcb,
                )
            ),
            "std is non-negative": bool(
                (self.rmst_std >= 0).all() and (self.diff_std >= 0).all()
            ),
            "difference to the leader is 0 at the leader": bool(
                self.diff_mean.gather(-1, safe)
                .squeeze(-1)[has_support]
                .abs()
                .le(eps)
                .all()
            ),
        }
        if self.config.leader_rule == "mean":
            checks["no supported arm beats the leader on the mean"] = bool(
                (self.diff_mean[self.mask] >= -eps).all()
            )
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(
                "RecommendationSummary invariants violated: " + "; ".join(failed)
            )

    def runner_up_gap(self) -> torch.Tensor:
        """(batch, n_lines) pessimistic deficit of the closest rival to the leader.

        The minimum of `diff_lcb` over the supported arms other than the leader:
        `+inf` where the leader is the only supported arm (its set is trivially a
        singleton), NaN where nothing is supported. With `p_best_leader` this is
        the whole decision rule: the equivalence set is a singleton exactly when
        the gap reaches `margin_months`, and the decision is confident exactly
        when `p_best_leader` also reaches `p_best_min`.
        """
        T = self.mask.shape[-1]
        leader_one_hot = F.one_hot(self.leader_idx.clamp(min=0), T).bool()
        rivals = self.mask & ~leader_one_hot
        gap = self.diff_lcb.masked_fill(~rivals, float("inf")).amin(-1)
        return gap.masked_fill(~self.mask.any(-1), float("nan"))


def summarize(
    rmst: torch.Tensor,
    mask: torch.Tensor,
    config: RecommendationConfig,
    member_masks: torch.Tensor | None = None,
) -> RecommendationSummary:
    """Apply the decision rule to per-member RMST.

    `rmst` must be finite everywhere: unsupported arms are excluded through
    `mask`, never by filling -inf, so means and stds stay well defined. Ties
    resolve to the lowest arm index, as `TreatmentRecommender.recommend` does.
    """
    if rmst.dim() != 4:
        raise ValueError(
            f"rmst must be (M, batch, n_lines, n_treatments), got {tuple(rmst.shape)}"
        )
    M, B, L, T = rmst.shape
    if tuple(mask.shape) != (B, L, T):
        raise ValueError(f"mask must be {(B, L, T)}, got {tuple(mask.shape)}")
    if not torch.isfinite(rmst).all():
        raise ValueError(
            "member RMST must be finite; mask unsupported arms instead of filling -inf"
        )
    cfg = config
    mask = mask.bool()
    neg_inf = float("-inf")
    c, margin = cfg.pessimism_c, cfg.margin_months

    # 1. per-arm statistics over members
    rmst_mean = rmst.mean(0)
    rmst_std = rmst.std(0, correction=1) if M > 1 else torch.zeros_like(rmst_mean)

    # 3. leader: pessimistic score or plain mean, over supported arms only
    score = rmst_mean - c * rmst_std if cfg.leader_rule == "lcb" else rmst_mean
    has_support = mask.any(-1)  # (B, L)
    leader = score.masked_fill(~mask, neg_inf).argmax(-1)
    leader = leader.masked_fill(~has_support, NO_SUPPORTED_ARM)
    safe_leader = leader.clamp(min=0)

    # 1. probability of being best: share of members whose masked argmax is the arm
    member_best = rmst.masked_fill(~mask.unsqueeze(0), neg_inf).argmax(-1)  # (M, B, L)
    one_hot = F.one_hot(member_best, T).to(rmst.dtype)
    one_hot = one_hot * has_support.to(rmst.dtype)[None, :, :, None]
    p_best = one_hot.mean(0)  # (B, L, T)

    # 2. paired difference leader - arm, per member, then its pessimistic bound
    gather_idx = safe_leader.view(1, B, L, 1).expand(M, B, L, 1)
    diff = rmst.gather(-1, gather_idx) - rmst  # (M, B, L, T); exactly 0 at the leader
    diff_mean = diff.mean(0)
    diff_std = diff.std(0, correction=1) if M > 1 else torch.zeros_like(diff_mean)
    diff_lcb = diff_mean - c * diff_std

    # Arms not confidently worse than the leader by `margin`. The leader is
    # forced in so that margin == 0 does not empty its own set.
    leader_one_hot = F.one_hot(safe_leader, T).bool()
    equivalence_set = (mask & (diff_lcb < margin)) | leader_one_hot
    equivalence_set &= has_support.unsqueeze(-1)
    set_size = equivalence_set.sum(-1)

    p_best_leader = p_best.gather(-1, safe_leader.unsqueeze(-1)).squeeze(-1)
    p_best_leader = p_best_leader * has_support.to(p_best.dtype)
    # 1e-6 absorbs float rounding of k/M against the configured threshold.
    confident = has_support & (set_size == 1) & (p_best_leader + 1e-6 >= cfg.p_best_min)

    decision = torch.full((B, L), int(Decision.UNDECIDED), dtype=torch.long)
    decision[~has_support] = int(Decision.NO_SUPPORT)
    decision[confident] = int(Decision.CONFIDENT)
    recommended_idx = torch.where(
        confident, leader, torch.full_like(leader, NO_SUPPORTED_ARM)
    )
    disagreement = (
        torch.zeros_like(mask) if member_masks is None else member_masks.any(0) & ~mask
    )

    return RecommendationSummary(
        rmst_mean=rmst_mean,
        rmst_std=rmst_std,
        p_best=p_best,
        diff_mean=diff_mean,
        diff_std=diff_std,
        diff_lcb=diff_lcb,
        mask=mask,
        equivalence_set=equivalence_set,
        member_mask_disagreement=disagreement,
        leader_idx=leader,
        recommended_idx=recommended_idx,
        set_size=set_size,
        decision=decision,
        p_best_leader=p_best_leader,
        n_members=M,
        config=cfg,
    )


class EnsembleRecommender:
    """Pool several `TreatmentRecommender`s and apply the decision rule.

    Args:
        members: loaded members (`Member` or bare `TreatmentRecommender`), already
            checked with `check_members`.
        config: thresholds and the leader rule.
        horizon_times: per-line RMST horizons in months, shared by every member.
    """

    def __init__(
        self,
        members: Sequence[Member | TreatmentRecommender],
        config: RecommendationConfig,
        horizon_times: Sequence[float],
    ):
        self.members = _unwrap(members)
        if len(self.members) < config.min_members:
            raise ValueError(
                f"{len(self.members)} member(s) given, config requires at least "
                f"{config.min_members}"
            )
        self.config = config
        self.horizon_times = [float(t) for t in horizon_times]

    @torch.no_grad()
    def member_rmst(
        self, XPd, X_static, factual_idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-member RMST and support, on CPU.

        Returns:
            rmst: (M, batch, n_lines, n_treatments), finite.
            mask: (batch, n_lines, n_treatments), the AND over members.
            member_masks: (M, batch, n_lines, n_treatments).
        """
        x_static, p_static = X_static
        rmsts, masks = [], []
        for rec in self.members:
            device = next(rec.model.parameters()).device
            r, m = rec.arm_rmst(
                XPd.to(device),
                (x_static.to(device), p_static.to(device)),
                factual_idx.to(device),
                self.horizon_times,
            )
            rmsts.append(r.detach().cpu())
            masks.append(m.detach().cpu())
        rmst = torch.stack(rmsts)
        member_masks = torch.stack(masks)
        return rmst, member_masks.all(0), member_masks

    def summarize(
        self,
        rmst: torch.Tensor,
        mask: torch.Tensor,
        member_masks: torch.Tensor | None = None,
    ) -> RecommendationSummary:
        """See `summarize` at module level; this binds the ensemble's config."""
        return summarize(rmst, mask, self.config, member_masks)

    def recommend(self, XPd, X_static, factual_idx) -> RecommendationSummary:
        rmst, mask, member_masks = self.member_rmst(XPd, X_static, factual_idx)
        return self.summarize(rmst, mask, member_masks)

    @torch.no_grad()
    def member_rmst_grid(
        self, XPd, X_static, factual_idx, horizon_grid: Sequence[Sequence[float]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`member_rmst` at several per-line horizons, one forward pass per member.

        Returns:
            rmst: (H, M, batch, n_lines, n_treatments), finite.
            mask: (batch, n_lines, n_treatments), the AND over members; the same at
                every horizon, since the action set is fixed.
            member_masks: (M, batch, n_lines, n_treatments).
        """
        x_static, p_static = X_static
        rmsts, masks = [], []
        for rec in self.members:
            device = next(rec.model.parameters()).device
            r, m = rec.arm_rmst_grid(
                XPd.to(device),
                (x_static.to(device), p_static.to(device)),
                factual_idx.to(device),
                horizon_grid,
            )
            rmsts.append(r.detach().cpu())
            masks.append(m.detach().cpu())
        rmst = torch.stack(rmsts, dim=1)
        member_masks = torch.stack(masks)
        return rmst, member_masks.all(0), member_masks

    def sweep(
        self, XPd, X_static, factual_idx, horizon_grid: Sequence[Sequence[float]]
    ) -> list[RecommendationSummary]:
        """One `RecommendationSummary` per entry of `horizon_grid`.

        A scoring-only sweep: the members' curves, the support mask and the
        thresholds are shared, only the RMST integration limit changes. The entry
        equal to `self.horizon_times` reproduces `recommend`.
        """
        rmst, mask, member_masks = self.member_rmst_grid(
            XPd, X_static, factual_idx, horizon_grid
        )
        return [
            self.summarize(rmst[h], mask, member_masks) for h in range(rmst.shape[0])
        ]


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def holdout_batch(datamodule) -> tuple:
    """The whole holdout as one batch, as `test_dataloader` yields it.

    Built without a worker process: the datamodule's loader uses one, and it
    dies in headless sessions (background jobs, some notebooks) on macOS.
    """
    dataset = datamodule.test_dataset
    loader = TorchData.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=0
    )
    return next(iter(loader))


def to_frame(
    summary: RecommendationSummary,
    *,
    patient_id: torch.Tensor,
    line_mask: torch.Tensor,
    observed_idx: torch.Tensor,
    treatment_dict: dict[int, str],
    horizon: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Long table: one row per observed (patient, line) and arm.

    Args:
        patient_id: (batch,).
        line_mask: (batch, n_lines) 1 where the patient actually has that line.
        observed_idx: (batch, n_lines) the arm actually received.
        treatment_dict: arm index -> name, from the datamodule.
        horizon: the per-line RMST horizons the summary was computed at. When
            given, a `horizon` column after `line` holds each row's own; left out
            by default so the single-horizon table keeps its columns.
    """
    B, L, T = summary.mask.shape
    if horizon is not None and len(horizon) < L:
        raise ValueError(f"need one horizon per line ({L} lines), got {horizon!r}")
    rows = line_mask.bool()[:, :L].nonzero(as_tuple=False)  # (N, 2)
    b = rows[:, 0].repeat_interleave(T)
    line = rows[:, 1].repeat_interleave(T)
    arm = torch.arange(T).repeat(rows.shape[0])
    names = {int(k): str(v) for k, v in treatment_dict.items()}

    def per_arm(t: torch.Tensor):
        return t[b, line, arm].numpy()

    def per_line(t: torch.Tensor):
        return t[b, line].numpy()

    leader = per_line(summary.leader_idx)
    observed = per_line(observed_idx.long())
    frame = pd.DataFrame(
        {
            "patient_id": patient_id[b].numpy(),
            "line": line.numpy(),
            "arm_idx": arm.numpy(),
            "arm_name": [names.get(int(k), str(k)) for k in arm.tolist()],
            "supported": per_arm(summary.mask),
            "rmst_mean": per_arm(summary.rmst_mean),
            "rmst_std": per_arm(summary.rmst_std),
            "p_best": per_arm(summary.p_best),
            "diff_mean": per_arm(summary.diff_mean),
            "diff_std": per_arm(summary.diff_std),
            "diff_lcb": per_arm(summary.diff_lcb),
            "in_equivalence_set": per_arm(summary.equivalence_set),
            "is_leader": arm.numpy() == leader,
            "leader_idx": leader,
            "set_size": per_line(summary.set_size),
            "p_best_leader": per_line(summary.p_best_leader),
            "recommended_idx": per_line(summary.recommended_idx),
            "decision": [
                Decision(int(d)).name.lower() for d in per_line(summary.decision)
            ],
            "observed_idx": observed,
            "observed_name": [names.get(int(k), str(k)) for k in observed.tolist()],
            "n_members": summary.n_members,
        }
    )
    if horizon is not None:
        per_line_horizon = torch.tensor(
            [float(t) for t in horizon], dtype=torch.float64
        )
        frame.insert(2, "horizon", per_line_horizon[line].numpy())
    return frame
