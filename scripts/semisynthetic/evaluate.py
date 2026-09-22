"""Score a trained DynaSurv run against the semi-synthetic truth. Run from `scripts/`:

    python semisynthetic/evaluate.py --axis gamma --level 1.0 --rep 0 --seed 0

Reads   ../data/semisynthetic/{axis}/{level}/rep{rep}/            (truth, manifest)
        ../models/semisynthetic/{axis}/{level}/rep{rep}/seed_{seed}/checkpoints/
Writes  .../seed_{seed}/eval_{kind}/{curve,effect,policy,factual}.csv

DynaSurv is scored on the temporal holdout next to two references: the oracle (true
curves, the floor) and the naive per-arm Kaplan-Meier (no covariates, the ceiling).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from CausalSurv.config import CHECKPOINT_KINDS, ExperimentConfig
from CausalSurv.model.checkpoint_compat import load_dynasurv_checkpoint
from CausalSurv.recommendation.ensemble import find_checkpoints
from CausalSurv.semisynthetic.datamodule import SemiSyntheticDataModule
from CausalSurv.semisynthetic.evaluate import build_truth, evaluate
from CausalSurv.semisynthetic.predictors import (
    dynasurv_prediction,
    km_prediction,
    oracle_prediction,
)

CONFIG_PATH = "../configs/semisynthetic/config.toml"
MODEL_CONFIG_PATH = "../configs/best_config.json"
GRID_STEP = 0.1  # months; every horizon must be a multiple of it


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", required=True)
    parser.add_argument("--level", required=True)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=CHECKPOINT_KINDS, default="val_loss")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    args = parser.parse_args()

    cell = f"{args.axis}/{args.level}/rep{args.rep}"
    data_dir = Path(f"../data/semisynthetic/{cell}")
    run_dir = Path(f"../models/semisynthetic/{cell}/seed_{args.seed}")
    cfg = ExperimentConfig.from_files(args.config, args.model_config)
    cfg = replace(cfg, data=replace(cfg.data, data_dir=str(data_dir)))

    arms = tuple(json.loads((data_dir / "manifest.json").read_text())["arms"])
    truth = pd.read_parquet(data_dir / "truth.parquet").set_index("usubjid", drop=False)

    dm = SemiSyntheticDataModule(
        **cfg.datamodule_kwargs(),
        split_seed=args.seed,
        num_workers=0,
        final_training=True,
    )
    dm.prepare_data()
    train_set, holdout_set = dm._split_holdout()
    train_ids = dm.ESMEDataset.patient_ids[np.asarray(train_set.indices)]

    horizons = list(cfg.eval.horizon_times)
    steps = round(max(horizons) / GRID_STEP)
    t_grid = np.linspace(0.0, steps * GRID_STEP, steps + 1)

    model = load_dynasurv_checkpoint(find_checkpoints([run_dir], args.kind)[0])
    dyna, ids, line_support = dynasurv_prediction(
        model, holdout_set, dm.treatment_dict, arms, t_grid
    )

    held_out = truth.loc[ids]
    train = truth.loc[train_ids]
    scored = build_truth(held_out, arms, t_grid, horizons)
    tables = evaluate(
        {
            "dynasurv": dyna,
            "oracle": oracle_prediction(held_out, arms, t_grid),
            "naive_km": km_prediction(train, held_out, arms, t_grid),
        },
        scored,
        line_support,
        train,
    )

    out = run_dir / f"eval_{args.kind}"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(f"\n{cell} seed {args.seed} [{args.kind}]: {len(held_out)} held-out samples")
    print("line-level support:", line_support.astype(int).tolist(), "arms:", list(arms))
    policy = tables["policy"]
    print(
        "\nPOLICY (pooled)\n",
        policy[policy["line"] == "all"].round(3).to_string(index=False),
    )
    print("\nFACTUAL\n", tables["factual"].round(3).to_string(index=False))
    curve = tables["curve"]
    print(
        "\nCURVE (pooled, mean over arms)\n",
        curve[curve["line"] == "all"]
        .groupby(["predictor", "kind"])[["rmse_S", "rmst_mae", "rmst_bias"]]
        .mean()
        .round(3),
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    cli()
