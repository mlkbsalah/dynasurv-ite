"""Member assembly shared by the recommendation CLIs.

`scripts/RecommendEnsemble.py` and `scripts/HorizonRobustness.py` build the same
ensemble the same way: expand run globs, pick one checkpoint per run, build the
holdout datamodule on the checkpoints' survival grid, load the members and
cross-check them. Kept out of `recommendation/__init__.py` so the pure-tensor
tests never import the datamodule stack.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import torch

from ..config import (
    CHECKPOINT_KINDS,
    LEADER_RULES,
    ExperimentConfig,
    RecommendationConfig,
)
from ..data.datamodule_cv import ESMEOnlineDataModuleCV
from .ensemble import (
    Decision,
    IncompatibleMemberError,
    Member,
    RecommendationSummary,
    check_members,
    find_checkpoints,
    load_member,
)

# Defaults are relative to scripts/, where the CLIs run, like the training script.
CONFIG_PATH = "../configs/config.toml"
MODEL_CONFIG_PATH = "../configs/hpo_v3/best_config.json"
OUT_DIR = "../reports/recommendations"

# datamodule dimension -> checkpoint hyperparameter; mirrors the validation notebook.
DIM_KEYS = [
    ("x_input_dim", "x_input_dim"),
    ("p_input_dim", "p_input_dim"),
    ("x_static_dim", "x_static_dim"),
    ("p_static_dim", "p_static_dim"),
    ("output_dim", "output_length"),
]


class SkippedRunError(RuntimeError):
    """A run cannot join the ensemble and the caller did not ask to skip such runs."""


class EnsembleAssemblyError(RuntimeError):
    """No ensemble can be built from the given runs; the message says why."""


@dataclass
class AssembledEnsemble:
    members: list[Member]
    datamodule: ESMEOnlineDataModuleCV
    names: dict[int, str]  # arm index -> name
    recommendable: dict[int, list[int]]
    report: dict  # from `check_members` over the kept members
    n_intervals: int


def add_ensemble_args(parser: argparse.ArgumentParser) -> None:
    """The flags every ensemble CLI takes: runs, checkpoint kind, paths, thresholds."""
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directories or globs (each holds checkpoints/). Default: every "
        "*_seed_* run of the configured subtype and n_lines.",
    )
    parser.add_argument("--kind", choices=CHECKPOINT_KINDS, default=None)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument(
        "--skip-incompatible",
        action="store_true",
        help="Drop runs that lack the checkpoint kind or fail the compatibility "
        "check instead of aborting.",
    )
    parser.add_argument("--p-best-min", type=float, default=None)
    parser.add_argument("--margin-months", type=float, default=None)
    parser.add_argument("--pessimism-c", type=float, default=None)
    parser.add_argument("--leader-rule", choices=LEADER_RULES, default=None)


def recommendation_config(
    cfg: ExperimentConfig, args: argparse.Namespace
) -> RecommendationConfig:
    """The configured decision rule with the CLI overrides applied."""
    overrides = {
        "p_best_min": args.p_best_min,
        "margin_months": args.margin_months,
        "pessimism_c": args.pessimism_c,
        "leader_rule": args.leader_rule,
        "checkpoint_kind": args.kind,
    }
    return replace(
        cfg.recommendation, **{k: v for k, v in overrides.items() if v is not None}
    )


def default_run_glob(cfg: ExperimentConfig) -> str:
    return f"../models/{cfg.data.subtype}/{cfg.data.n_lines}lines/*_seed_*"


def expand_runs(patterns: Sequence[str] | None, default_glob: str) -> list[Path]:
    """Run directories from paths or globs, in order, without duplicates."""
    patterns = list(patterns) if patterns else [default_glob]
    runs: list[Path] = []
    for pattern in patterns:
        hits = [Path(h) for h in sorted(glob.glob(pattern))] or [Path(pattern)]
        runs.extend(h for h in hits if h.is_dir())
    runs = list(dict.fromkeys(runs))
    if not runs:
        raise FileNotFoundError(f"no run directories match {patterns}")
    return runs


def build_datamodule(cfg: ExperimentConfig, n_intervals: int) -> ESMEOnlineDataModuleCV:
    kwargs = cfg.datamodule_kwargs()
    # The survival grid is fixed by the checkpoints, not by the current model config.
    kwargs["n_intervals"] = n_intervals
    # split_seed only shuffles the propensity CV folds under a temporal split.
    dm = ESMEOnlineDataModuleCV(
        **kwargs, split_seed=0, final_training=True, num_workers=0
    )
    dm.prepare_data()
    dm.setup()
    return dm


def peek_n_intervals(path: str | Path) -> int:
    """The survival grid size a checkpoint was trained with, without building it."""
    return torch.load(path, map_location="cpu", weights_only=False)["hyper_parameters"][
        "output_length"
    ]


def assemble_members(
    cfg: ExperimentConfig,
    rec_cfg: RecommendationConfig,
    runs: Sequence[Path],
    *,
    skip_incompatible: bool = False,
    log: Callable[[str], None] = print,
) -> AssembledEnsemble:
    """Checkpoints of `rec_cfg.checkpoint_kind` from `runs`, loaded and cross-checked.

    A run without that checkpoint kind, or a member that fails the compatibility
    check against the first one, raises `SkippedRunError` unless
    `skip_incompatible`, in which case it is logged and dropped. Warnings from the
    compatibility report are logged too. `EnsembleAssemblyError` means nothing
    usable came out: no checkpoints, no loadable member, a datamodule whose
    dimensions no longer match the checkpoints, or fewer members than the config
    requires.
    """

    def give_up(message: str) -> None:
        if skip_incompatible:
            log(f"skip: {message}")
        else:
            raise SkippedRunError(message)

    paths: list[Path] = []
    for run in runs:
        try:
            paths.extend(find_checkpoints([run], rec_cfg.checkpoint_kind))
        except FileNotFoundError as exc:
            give_up(str(exc))
    if not paths:
        raise EnsembleAssemblyError("no checkpoints to ensemble")

    # The survival grid comes from the checkpoints; read it before the members
    # are built so the datamodule's exact arm sets can be handed to each one.
    n_intervals = peek_n_intervals(paths[0])
    dm = build_datamodule(cfg, n_intervals)
    recommendable = dm.recommendable_treatments_per_line
    names = {int(k): str(v) for k, v in dm.treatment_dict.items()}

    members: list[Member] = []
    for path in paths:
        try:
            member = load_member(path, recommendable=recommendable)
            if member.model.data_manifest != dm.data_manifest:
                raise IncompatibleMemberError(
                    f"{path}: data, preprocessing or split manifest differs from this datamodule"
                )
            members.append(member)
        except IncompatibleMemberError as exc:
            give_up(str(exc))
    if not members:
        raise EnsembleAssemblyError("no loadable members")

    dims = dm.get_data_dimensions()
    hparams = dict(members[0].model.hparams)
    for dim_key, hparam_key in DIM_KEYS:
        if dims[dim_key] != hparams[hparam_key]:
            raise EnsembleAssemblyError(
                f"{dim_key}: datamodule gives {dims[dim_key]}, checkpoint was built "
                f"for {hparams[hparam_key]}; the config drifted away from the runs"
            )

    kept = [members[0]]
    for member in members[1:]:
        try:
            check_members([members[0], member], expected_recommendable=recommendable)
            kept.append(member)
        except IncompatibleMemberError as exc:
            give_up(str(exc))
    report = check_members(kept, expected_recommendable=recommendable)
    for warning in report["warnings"]:
        log(f"warning: {warning}")
    if len(kept) < rec_cfg.min_members:
        raise EnsembleAssemblyError(
            f"{len(kept)} compatible member(s), config requires {rec_cfg.min_members}"
        )
    return AssembledEnsemble(
        members=kept,
        datamodule=dm,
        names=names,
        recommendable=recommendable,
        report=report,
        n_intervals=n_intervals,
    )


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def arm_mix(idx: torch.Tensor, names: dict[int, str]) -> dict[str, int]:
    values, counts = torch.unique(idx, return_counts=True)
    return {names.get(int(v), str(int(v))): int(c) for v, c in zip(values, counts)}


def per_line_summary(
    summary: RecommendationSummary, line_mask, observed_idx, names
) -> dict:
    """Decision counts, set sizes, agreement with the observed arm, per line."""
    out = {}
    for line in range(summary.mask.shape[1]):
        rows = line_mask[:, line].bool()
        n = int(rows.sum())
        if n == 0:
            continue
        decision = summary.decision[rows, line]
        supported = decision != Decision.NO_SUPPORT
        confident = decision == Decision.CONFIDENT
        set_size = summary.set_size[rows, line][supported].float()
        leader = summary.leader_idx[rows, line]
        recommended = summary.recommended_idx[rows, line]
        observed = observed_idx[rows, line]
        counts = {d.name.lower(): int((decision == d).sum()) for d in Decision}
        sizes, size_counts = torch.unique(set_size.long(), return_counts=True)
        out[f"line_{line + 1}"] = {
            "n": n,
            "counts": counts,
            "rates": {k: round(v / n, 4) for k, v in counts.items()},
            "set_size_among_supported": {
                "mean": float(set_size.mean()) if len(set_size) else None,
                "median": float(set_size.median()) if len(set_size) else None,
                "histogram": {int(s): int(c) for s, c in zip(sizes, size_counts)},
            },
            "agree_confident_with_observed": (
                float((recommended[confident] == observed[confident]).float().mean())
                if confident.any()
                else None
            ),
            "agree_leader_with_observed": (
                float((leader[supported] == observed[supported]).float().mean())
                if supported.any()
                else None
            ),
            "mean_p_best_leader_among_supported": (
                float(summary.p_best_leader[rows, line][supported].mean())
                if supported.any()
                else None
            ),
            "recommended_arm_mix": arm_mix(recommended[confident], names),
            "leader_arm_mix": arm_mix(leader[supported], names),
            "member_mask_disagreement_cells": int(
                summary.member_mask_disagreement[rows, line].sum()
            ),
        }
    return out


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
