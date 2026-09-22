"""Robustness of the ensemble recommendation to the RMST horizon.

Arms are ranked by RMST integrated up to a per-line horizon (`[eval]
horizon_times` in config.toml). This script re-scores the holdout at a grid of
horizons and asks whether the leader, the equivalence set and the recommended
mix move with it.

Scoring-only sweep: the action set (recommendable arms, propensity filter) and
the thresholds stay at their configured values; only the RMST integration limit
changes. The horizon also drives the arm-support filter inside the datamodule,
and the checkpoints persist those arm sets, so rebuilding the datamodule per
horizon would make the compatibility check refuse the members. Which arms are
recommendable at all is a different question and out of scope here, so the
support mask is identical at every horizon.

Run from `scripts/`:

    python HorizonRobustness.py --kind bestCALIB --runs "../models/HR+HER2-/4lines/21092026_11*_seed_*"
    python HorizonRobustness.py --kind bestCALIB --horizons 6 12 18 24 30 36 --runs ...

The configured per-line reference is always scored as its own grid entry, so the
agreement metrics compare against exactly the deployed rule. Writes
`horizon_sweep.parquet` (one row per horizon, patient, line and arm) and
`horizon_summary.json` under --out, in a `..._horizons` directory.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from CausalSurv.config import ExperimentConfig
from CausalSurv.recommendation import (
    Decision,
    EnsembleRecommender,
    holdout_batch,
    to_frame,
)
from CausalSurv.recommendation.pipeline import (
    EnsembleAssemblyError,
    SkippedRunError,
    add_ensemble_args,
    arm_mix,
    assemble_members,
    default_run_glob,
    expand_runs,
    git_head,
    per_line_summary,
    recommendation_config,
)

DEFAULT_HORIZONS = [3.0 * k for k in range(1, 13)]  # 3, 6, ..., 36 months


def sweep_records(
    summaries, grid, is_reference, ref, line_mask, observed_idx, time, event, names
) -> dict[int, list[dict]]:
    """Per line, one record per grid entry: decision rates, mixes, agreement with
    the reference entry, and how much of the holdout is observed to that horizon."""
    n_lines = line_mask.shape[1]
    per_line: dict[int, list[dict]] = {line: [] for line in range(n_lines)}
    for summary, entry, flag in zip(summaries, grid, is_reference):
        blocks = per_line_summary(summary, line_mask, observed_idx, names)
        for line in range(n_lines):
            block = blocks.get(f"line_{line + 1}")
            if block is None:
                continue
            rows = line_mask[:, line]
            h = float(entry[line])
            supported = ref.mask[rows, line].any(-1)
            leader = summary.leader_idx[rows, line]
            confident = summary.decision[rows, line] == Decision.CONFIDENT
            set_h = summary.equivalence_set[rows, line]
            set_ref = ref.equivalence_set[rows, line]
            jaccard = (set_h & set_ref).sum(-1).float() / (set_h | set_ref).sum(
                -1
            ).float().clamp(min=1)
            same = leader == ref.leader_idx[rows, line]
            # Share of records whose follow-up reaches h: an event before h or a
            # censoring after it. The datamodule's arm-support filter uses the same
            # formula; beyond a line's last observed time this is just its event rate.
            coverage = ((time[rows, line] >= h) | event[rows, line]).float().mean()
            per_line[line].append(
                {
                    "horizon": h,
                    "is_reference": bool(flag),
                    "n": block["n"],
                    "coverage": float(coverage),
                    "rates": block["rates"],
                    "set_size_mean": block["set_size_among_supported"]["mean"],
                    "leader_mix_confident": arm_mix(leader[confident], names),
                    "leader_mix_undecided": arm_mix(
                        leader[supported & ~confident], names
                    ),
                    "leader_same": (
                        float(same[supported].float().mean())
                        if supported.any()
                        else None
                    ),
                    "set_jaccard": (
                        float(jaccard[supported].mean()) if supported.any() else None
                    ),
                    "agree_leader_with_observed": block["agree_leader_with_observed"],
                    "mean_p_best_leader": block["mean_p_best_leader_among_supported"],
                }
            )
    return per_line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_ensemble_args(parser)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=float,
        default=DEFAULT_HORIZONS,
        help="RMST horizons in months, applied to every line (default 3..36 by 3).",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.from_files(args.config, args.model_config)
    rec_cfg = recommendation_config(cfg, args)
    reference = [float(t) for t in cfg.eval.horizon_times]
    horizons = sorted({float(h) for h in args.horizons})
    if not horizons or horizons[0] <= 0:
        sys.exit("--horizons must be positive months")
    print(f"decision rule: {rec_cfg}")
    print(f"reference horizons: {reference}; sweep: {horizons}")

    try:
        runs = expand_runs(args.runs, default_run_glob(cfg))
        ens = assemble_members(
            cfg, rec_cfg, runs, skip_incompatible=args.skip_incompatible
        )
    except SkippedRunError as exc:
        sys.exit(f"abort (use --skip-incompatible to drop such runs): {exc}")
    except (FileNotFoundError, EnsembleAssemblyError) as exc:
        sys.exit(str(exc))
    kept, dm, names, report = ens.members, ens.datamodule, ens.names, ens.report
    print(f"{len(kept)} members:")
    for member in kept:
        print(f"  seed={member.seed} epoch={member.epoch} {member.path}")

    grid_end = float(kept[0].model.interval_bounds[-1])
    beyond = [h for h in horizons if h > grid_end]
    if beyond:
        print(
            f"warning: horizons {beyond} exceed the survival grid ({grid_end:.1f} "
            "months); RMST is clamped to the grid end there"
        )

    XPd, X_static, _, treatment_indices, time, event, line_mask, patient_id = (
        holdout_batch(dm)
    )
    time = time.flatten(1).float()  # (batch, n_lines)
    event = event.flatten(1).bool()
    line_mask = line_mask.bool()
    n_lines = line_mask.shape[1]

    grid = [[h] * n_lines for h in horizons] + [reference]
    is_reference = [False] * len(horizons) + [True]
    ensemble = EnsembleRecommender(kept, rec_cfg, reference)
    summaries = ensemble.sweep(XPd, X_static, treatment_indices, grid)
    for summary in summaries:
        summary.validate()
    ref = summaries[-1]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out)
        / f"{cfg.data.subtype}_{cfg.data.n_lines}lines"
        / f"{stamp}_{rec_cfg.checkpoint_kind}_M{len(kept)}_horizons"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [
        to_frame(
            summary,
            patient_id=patient_id,
            line_mask=line_mask,
            observed_idx=treatment_indices,
            treatment_dict=names,
            horizon=entry,
        ).assign(is_reference=flag)
        for summary, entry, flag in zip(summaries, grid, is_reference)
    ]
    sweep = pd.concat(frames, ignore_index=True)
    sweep.to_parquet(out_dir / "horizon_sweep.parquet", index=False)

    records = sweep_records(
        summaries,
        grid,
        is_reference,
        ref,
        line_mask,
        treatment_indices,
        time,
        event,
        names,
    )
    per_line = {}
    for line, recs in records.items():
        if not recs:
            continue
        rows = line_mask[:, line]
        per_line[f"line_{line + 1}"] = {
            "n": int(rows.sum()),
            "reference_horizon": reference[line],
            "max_observed_time": float(time[rows, line].max()),
            "coverage_at_reference": next(
                r["coverage"] for r in recs if r["is_reference"]
            ),
            "records": recs,
        }
    payload = {
        "timestamp": stamp,
        "git_head": git_head(),
        "config": str(Path(args.config).resolve()),
        "recommendation": asdict(rec_cfg),
        "reference_horizon_times": reference,
        "horizons": horizons,
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
            "rows_in_parquet": int(len(sweep)),
            "rows_per_entry": int(len(frames[0])),
            "n_entries": len(grid),
        },
        "per_line": per_line,
    }
    (out_dir / "horizon_summary.json").write_text(json.dumps(payload, indent=2))

    print(
        f"\nholdout: {XPd.shape[0]} patients, {len(grid)} grid entries, "
        f"{len(sweep)} rows -> {out_dir}"
    )
    for key, block in per_line.items():
        print(
            f"\n{key.replace('_', ' ')}  n={block['n']}  reference {block['reference_horizon']:g} months  "
            f"follow-up to {block['max_observed_time']:.1f} months"
        )
        print(
            f"{'h':>6} {'coverage':>8} {'confident':>9} {'undecided':>9} {'set':>5} "
            f"{'leader_same':>11} {'jaccard':>8}"
        )
        for r in block["records"]:
            tag = "*" if r["is_reference"] else " "
            print(
                f"{r['horizon']:>5.0f}{tag} {r['coverage']:>8.2f} "
                f"{r['rates']['confident']:>9.2f} {r['rates']['undecided']:>9.2f} "
                f"{r['set_size_mean'] or 0:>5.2f} "
                f"{'-' if r['leader_same'] is None else f'{r['leader_same']:.2f}':>11} "
                f"{'-' if r['set_jaccard'] is None else f'{r['set_jaccard']:.2f}':>8}"
            )
    print("\n* = configured reference horizon")


if __name__ == "__main__":
    main()
