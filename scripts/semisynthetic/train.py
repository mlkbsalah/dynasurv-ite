"""Train DynaSurv on one semi-synthetic replicate. Run from `scripts/`:

    python semisynthetic/train.py --axis gamma --level 1.0 --rep 0 --seed 0

Reads   ../data/semisynthetic/{axis}/{level}/rep{rep}/   (from semisynthetic/generate.py)
Writes  ../models/semisynthetic_v2/{axis}/{level}/rep{rep}/seed_{seed}/
"""

import argparse
import os
import sys
from dataclasses import replace

import lightning as L

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from TrainDynasurvCausal import MODEL_CONFIG_PATH, main  # noqa: E402

from CausalSurv.config import ExperimentConfig  # noqa: E402
from CausalSurv.semisynthetic.datamodule import SemiSyntheticDataModule  # noqa: E402

CONFIG_PATH = "../configs/semisynthetic/config.toml"


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis", required=True, help="gamma | strength | heterogeneity"
    )
    parser.add_argument("--level", required=True)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--model-config", default=MODEL_CONFIG_PATH)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--models-dir", default="../models/semisynthetic_v2")
    args = parser.parse_args()

    cell = f"{args.axis}/{args.level}/rep{args.rep}"
    cfg = ExperimentConfig.from_files(args.config, args.model_config)
    cfg = replace(cfg, data=replace(cfg.data, data_dir=f"../data/semisynthetic/{cell}"))

    L.seed_everything(args.seed, workers=True)
    main(
        cfg=cfg,
        split_seed=args.seed,
        date="",
        fast_dev_run=args.fast_dev_run,
        datamodule_cls=SemiSyntheticDataModule,
        run_dir=f"{args.models_dir}/{cell}/seed_{args.seed}",
        wandb_project="DynaSurvSemiSynthetic",
    )


if __name__ == "__main__":
    cli()
