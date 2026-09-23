import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.config import ExperimentConfig
from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
from CausalSurv.model import DynaSurvCausalOnline
from CausalSurv.semisynthetic.datamodule import SemiSyntheticDataModule

CONFIG_PATH = "../configs/config.toml"
MODEL_CONFIG_PATH = "../configs/hpo_v3/best_config.json"
DATAMODULES = {
    "esme": ESMEOnlineDataModuleCV,
    "semisynthetic": SemiSyntheticDataModule,
}


def main(
    cfg: ExperimentConfig,
    split_seed,
    date,
    fast_dev_run=False,
    datamodule_cls=ESMEOnlineDataModuleCV,
    run_dir=None,
    wandb_project=None,
    evaluate_test=False,
):
    train_config = cfg.train
    if fast_dev_run:
        train_config = replace(
            train_config, trainer=replace(train_config.trainer, max_epochs=3)
        )

    data_config = cfg.data
    data_module = datamodule_cls(
        **cfg.datamodule_kwargs(),
        split_seed=split_seed,
        num_workers=4,
        final_training=True,
    )

    data_module.prepare_data()
    # Build the recommendation action set from the training partition before
    # printing it. Trainer.fit() calls setup again, deterministically.
    data_module.setup("fit")
    data_module.describe_cohort()
    data_dims = data_module.get_data_dimensions()

    model = DynaSurvCausalOnline(
        x_input_dim=data_dims["x_input_dim"],
        x_static_dim=data_dims["x_static_dim"],
        p_input_dim=data_dims["p_input_dim"],
        p_static_dim=data_dims["p_static_dim"],
        output_length=data_dims["output_dim"],
        interval_bounds=data_dims["time_bins"],
        # Treatments are one-hot in P, so the arm count is the P feature width.
        n_treatments=data_dims["p_input_dim"],
        n_lines=data_config.n_lines,
        arch=cfg.arch,
        training=cfg.training,
        evaluation=cfg.eval,
    )

    if run_dir is None:
        run_dir = (
            f"../models/{data_config.subtype}/{data_config.n_lines}lines/"
            f"{date}_seed_{split_seed}"
        )
    ckpt_dir = f"{run_dir}/checkpoints/"
    if any(Path(ckpt_dir).glob("*.ckpt")):
        raise FileExistsError(
            "Run directory already contains checkpoints; choose a new run directory instead of mixing protocols/runs"
        )
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "data_manifest.json").write_text(
        json.dumps(data_module.data_manifest, indent=2)
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            dirpath=ckpt_dir,
            filename="dynaSurvCausalOnline-{epoch:02d}-{val_loss: .4f}",
        ),
        ModelCheckpoint(
            monitor="average_ci",
            mode="max",
            save_top_k=1,
            dirpath=ckpt_dir,
            filename="dynaSurvCausalOnline-bestCI-{epoch:02d}-{average_ci: .4f}",
        ),
        ModelCheckpoint(
            monitor="average_ibs",
            mode="min",
            save_top_k=1,
            dirpath=ckpt_dir,
            filename="dynaSurvCausalOnline-bestIBS-{epoch:02d}-{average_ibs: .4f}",
        ),
        # Marginal calibration: mean predicted S(t) against the KM estimate of the
        # same risk set, pooled over lines and landmarks. Kept separate from IBS and
        # val_loss because it peaks much later than either -- across an lr sweep the
        # gap was still falling ~10 epochs after val_loss bottomed, so neither of the
        # monitors above ever saves the best-calibrated epoch.
        ModelCheckpoint(
            monitor="val/calib_gap_abs_mean",
            mode="min",
            save_top_k=1,
            dirpath=ckpt_dir,
            # No metric in the filename: the "/" in the metric name would be read
            # as a path separator and scatter the checkpoints into subdirectories.
            filename="dynaSurvCausalOnline-bestCALIB-{epoch:02d}",
        ),
        # Always keep the final trained epoch (no metric monitored), since the
        # best-metric checkpoints above tend to land on early epochs (~15).
        ModelCheckpoint(
            monitor=None,
            save_top_k=1,
            dirpath=ckpt_dir,
            filename="dynaSurvCausalOnline-last-{epoch:02d}",
        ),
    ]

    early_stopping = train_config.early_stopping
    if early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping.monitor,
                mode=early_stopping.mode,
                patience=early_stopping.patience,
                verbose=True,
            )
        )

    logger = WandbLogger(
        project=wandb_project
        or (
            f"DynaSurvCausalOnline_{data_config.subtype}_"
            f"{data_config.n_lines}lines_new_inline_outcome"
        ),
        name=f"seed_{split_seed}_{date}",
        save_dir=run_dir,
    )

    trainer = L.Trainer(
        max_epochs=train_config.trainer.max_epochs,
        accelerator=train_config.trainer.accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=train_config.trainer.gradient_clip_val,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, datamodule=data_module)
    # Test is an explicit final evaluation action, never part of selection.
    if evaluate_test:
        from CausalSurv.recommendation.ensemble import find_checkpoints

        checkpoint = find_checkpoints([run_dir], cfg.recommendation.checkpoint_kind)[0]
        trainer.test(model, datamodule=data_module, ckpt_path=str(checkpoint))


if __name__ == "__main__":
    import os
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Enable fast development run mode",
        default=False,
    )
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--model-config", type=str, default=MODEL_CONFIG_PATH)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed for the split RNG, the propensity CV folds and torch/numpy "
            "initialisation (via seed_everything). Random when omitted."
        ),
    )

    parser.add_argument(
        "--datamodule",
        choices=sorted(DATAMODULES),
        default="esme",
        help="'semisynthetic' reads prefix-expanded data written by semisynthetic/generate.py",
    )
    parser.add_argument(
        "--data-dir", default=None, help="Override [data] data_dir of the config"
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Directory that receives {date}_seed_{seed}/ (default ../models/{subtype}/{n}lines)",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the preselected checkpoint on the reserved test set after training",
    )

    args = parser.parse_args()

    split_seed = (
        args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    )
    # Without this the run-directory seed never touched weight init or shuffling:
    # two runs with the same seed were still two independent draws.
    L.seed_everything(split_seed, workers=True)
    date = datetime.now().strftime("%d%m%Y_%H%M%S")

    cfg = ExperimentConfig.from_files(args.config, args.model_config)
    if args.data_dir is not None:
        cfg = replace(cfg, data=replace(cfg.data, data_dir=args.data_dir))
    run_dir = (
        f"{args.run_root}/{date}_seed_{split_seed}"
        if args.run_root is not None
        else None
    )

    main(
        cfg=cfg,
        split_seed=split_seed,
        date=date,
        fast_dev_run=args.fast_dev_run,
        datamodule_cls=DATAMODULES[args.datamodule],
        run_dir=run_dir,
        wandb_project=args.wandb_project,
        evaluate_test=args.evaluate_test,
    )
