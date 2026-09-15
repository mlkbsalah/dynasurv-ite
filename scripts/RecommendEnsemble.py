"""Ensemble treatment recommendation on the holdout, with uncertainty.

Pools several checkpoints (same temporal split, different inits) and applies the
decision rule in `CausalSurv.recommendation.ensemble`: pessimistic leader,
equivalence set, and a recommendation only when the members agree. Run from
`scripts/`, like the training script:

    python RecommendEnsemble.py --runs ../models/HR+HER2-/4lines/03092026_153705_seed_3177752843 \
                                        ../models/HR+HER2-/4lines/07092026_164247_seed_123099399
    python RecommendEnsemble.py --kind bestCALIB --runs "../models/HR+HER2-/4lines/1409*_seed_*"

Thresholds come from `[recommendation]` in config.toml; the flags override them.
Writes `recommendations.parquet` (one row per patient, line and arm) and
`summary.json` under --out.
"""

import argparse
import glob
import json
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import torch

from CausalSurv.config import CHECKPOINT_KINDS, LEADER_RULES, ExperimentConfig
from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
from CausalSurv.recommendation import (
    Decision,
    EnsembleRecommender,
    IncompatibleMemberError,
    check_members,
    find_checkpoints,
    holdout_batch,
    load_member,
    to_frame,
)

CONFIG_PATH = "../configs/config.toml"
MODEL_CONFIG_PATH = "../configs/best_config.json"
OUT_DIR = "../reports/recommendations"

# datamodule dimension -> checkpoint hyperparameter; mirrors the validation notebook.
DIM_KEYS = [
    ("x_input_dim", "x_input_dim"),
    ("p_input_dim", "p_input_dim"),
    ("x_static_dim", "x_static_dim"),
    ("p_static_dim", "p_static_dim"),
    ("output_dim", "output_length"),
]


def expand_runs(patterns: list[str], default_glob: str) -> list[Path]:
    patterns = patterns or [default_glob]
    runs: list[Path] = []
    for pattern in patterns:
        hits = [Path(h) for h in sorted(glob.glob(pattern))] or [Path(pattern)]
        runs.extend(h for h in hits if h.is_dir())
    runs = list(dict.fromkeys(runs))
    if not runs:
        sys.exit(f"no run directories match {patterns}")
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


def arm_mix(idx: torch.Tensor, names: dict[int, str]) -> dict[str, int]:
    values, counts = torch.unique(idx, return_counts=True)
    return {names.get(int(v), str(int(v))): int(c) for v, c in zip(values, counts)}


def per_line_summary(summary, line_mask, observed_idx, names) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
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
    args = parser.parse_args()

    cfg = ExperimentConfig.from_files(args.config, args.model_config)
    overrides = {
        "p_best_min": args.p_best_min,
        "margin_months": args.margin_months,
        "pessimism_c": args.pessimism_c,
        "leader_rule": args.leader_rule,
        "checkpoint_kind": args.kind,
    }
    rec_cfg = replace(
        cfg.recommendation, **{k: v for k, v in overrides.items() if v is not None}
    )
    print(f"decision rule: {rec_cfg}")

    default_glob = f"../models/{cfg.data.subtype}/{cfg.data.n_lines}lines/*_seed_*"
    runs = expand_runs(args.runs, default_glob)

    def give_up(message: str) -> None:
        if args.skip_incompatible:
            print(f"skip: {message}")
        else:
            sys.exit(f"abort (use --skip-incompatible to drop such runs): {message}")

    paths = []
    for run in runs:
        try:
            paths.extend(find_checkpoints([run], rec_cfg.checkpoint_kind))
        except FileNotFoundError as exc:
            give_up(str(exc))
    if not paths:
        sys.exit("no checkpoints to ensemble")

    # The survival grid comes from the checkpoints; read it before the members
    # are built so the datamodule's exact arm sets can be handed to each one.
    n_intervals = torch.load(paths[0], map_location="cpu", weights_only=False)[
        "hyper_parameters"
    ]["output_length"]
    dm = build_datamodule(cfg, n_intervals)
    recommendable = dm.recommendable_treatments_per_line
    names = {int(k): str(v) for k, v in dm.treatment_dict.items()}

    members = []
    for path in paths:
        try:
            members.append(load_member(path, recommendable=recommendable))
        except IncompatibleMemberError as exc:
            give_up(str(exc))
    if not members:
        sys.exit("no loadable members")

    dims = dm.get_data_dimensions()
    hparams = dict(members[0].model.hparams)
    for dim_key, hparam_key in DIM_KEYS:
        if dims[dim_key] != hparams[hparam_key]:
            sys.exit(
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
        print(f"warning: {warning}")
    if len(kept) < rec_cfg.min_members:
        sys.exit(
            f"{len(kept)} compatible member(s), config requires {rec_cfg.min_members}"
        )
    print(f"{len(kept)} members:")
    for member in kept:
        print(f"  seed={member.seed} epoch={member.epoch} {member.path}")

    XPd, X_static, _, treatment_indices, _, _, line_mask, patient_id = holdout_batch(dm)
    horizons = list(cfg.eval.horizon_times)
    ensemble = EnsembleRecommender(kept, rec_cfg, horizons)
    summary = ensemble.recommend(XPd, X_static, treatment_indices)
    summary.validate()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out)
        / f"{cfg.data.subtype}_{cfg.data.n_lines}lines"
        / f"{stamp}_{rec_cfg.checkpoint_kind}_M{len(kept)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = to_frame(
        summary,
        patient_id=patient_id,
        line_mask=line_mask,
        observed_idx=treatment_indices,
        treatment_dict=names,
    )
    frame.to_parquet(out_dir / "recommendations.parquet", index=False)

    lines = per_line_summary(summary, line_mask, treatment_indices, names)
    payload = {
        "timestamp": stamp,
        "git_head": git_head(),
        "config": str(Path(args.config).resolve()),
        "recommendation": asdict(rec_cfg),
        "horizon_times": horizons,
        "members": [
            {"path": str(m.path), "seed": m.seed, "epoch": m.epoch} for m in kept
        ],
        "compatibility": {
            "dims": report["dims"],
            "recommendable_treatments_per_line": {
                str(k): v
                for k, v in report["recommendable_treatments_per_line"].items()
            },
            "warnings": report["warnings"],
        },
        "holdout": {
            "n_patients": int(XPd.shape[0]),
            "observed_lines": [int(x) for x in line_mask.sum(0).tolist()],
            "rows_in_parquet": int(len(frame)),
        },
        "per_line": lines,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    print(f"\nholdout: {XPd.shape[0]} patients, {len(frame)} rows -> {out_dir}")
    print(
        f"{'line':>4} {'n':>5} {'no_support':>10} {'confident':>9} {'undecided':>9} "
        f"{'set':>5} {'agree_conf':>10} {'agree_lead':>10}"
    )
    for key, row in lines.items():
        r = row["rates"]
        conf = row["agree_confident_with_observed"]
        lead = row["agree_leader_with_observed"]
        print(
            f"{key[-1]:>4} {row['n']:>5} {r['no_support']:>10.2f} {r['confident']:>9.2f} "
            f"{r['undecided']:>9.2f} {row['set_size_among_supported']['mean'] or 0:>5.2f} "
            f"{'-' if conf is None else f'{conf:.2f}':>10} "
            f"{'-' if lead is None else f'{lead:.2f}':>10}"
        )


if __name__ == "__main__":
    main()
