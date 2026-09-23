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
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from CausalSurv.config import ExperimentConfig
from CausalSurv.recommendation import EnsembleRecommender, holdout_batch, to_frame
from CausalSurv.recommendation.pipeline import (
    EnsembleAssemblyError,
    SkippedRunError,
    add_ensemble_args,
    assemble_members,
    default_run_glob,
    expand_runs,
    git_head,
    per_line_summary,
    recommendation_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_ensemble_args(parser)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_files(args.config, args.model_config)
    rec_cfg = recommendation_config(cfg, args)
    print(f"decision rule: {rec_cfg}")

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
        "calibration_status": [m.model.calibration_status for m in kept],
        "evaluation_protocol": 2,
        "dataset_sha256": dm.data_manifest["dataset_sha256"],
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
        f"{'line':>4} {'n':>5} {'no_support':>10} {'only_option':>11} {'confident':>9} {'undecided':>9} "
        f"{'set':>5} {'agree_conf':>10} {'agree_lead':>10}"
    )
    for key, row in lines.items():
        r = row["rates"]
        conf = row["agree_confident_with_observed"]
        lead = row["agree_leader_with_observed"]
        print(
            f"{key[-1]:>4} {row['n']:>5} {r['no_support']:>10.2f} {r['only_supported_option']:>11.2f} {r['confident']:>9.2f} "
            f"{r['undecided']:>9.2f} {row['set_size_among_supported']['mean'] or 0:>5.2f} "
            f"{'-' if conf is None else f'{conf:.2f}':>10} "
            f"{'-' if lead is None else f'{lead:.2f}':>10}"
        )


if __name__ == "__main__":
    main()
