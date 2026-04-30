import argparse
import os

import lightning as L
import tomllib
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.datamodule_progression import ESMEProgressionOnlineDataModuleCV
from CausalSurv.model.dynasurv_causal_online_multihead import (
    DynaSurvCausalOnlineMultiheadProgression,
)


def load_config(config_path):
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def main(
    model_config,
    train_config,
    eval_config,
    data_config,
    split_seed,
    date,
    fast_dev_run=False,
):
    if fast_dev_run:
        train_config["trainer"]["max_epochs"] = 3

    data_module = ESMEProgressionOnlineDataModuleCV(
        data_dir=data_config["data_dir"],
        subtype=data_config["subtype"],
        n_lines=data_config["n_lines"],
        n_intervals=model_config["n_intervals"],
        batch_size=data_config["batch_size"],
        split_seed=split_seed,
        num_workers=4,
        final_training=True,
        bound_split="quantile",
    )

    data_module.prepare_data()
    data_module.setup()
    data_dims = data_module.get_data_dimensions()
    print(data_dims["progression_time_bins"].dim())

    model = DynaSurvCausalOnlineMultiheadProgression(
        x_input_dim=data_dims["x_input_dim"],
        x_static_dim=data_dims["x_static_dim"],
        p_input_dim=data_dims["p_input_dim"],
        p_static_dim=data_dims["p_static_dim"],
        n_treatments=data_dims["p_input_dim"],
        n_lines=data_config["n_lines"],
        output_length=data_dims["output_dim"],
        interval_bounds=data_dims["death_time_bins"],
        progression_bounds=data_dims["progression_time_bins"],
        lambda_prop_loss=model_config["lambda_prop_loss"],
        lambda_progression_loss=model_config.get("lambda_progression_loss", 1.0),
        lambda_ipm_mmd=model_config["lambda_ipm_mmd"],
        lambda_ipm_emd2=model_config["lambda_ipm_emd2"],
        evaluation_horizon_times=eval_config["horizon_times"],
        brier_integration_step=eval_config["integration_step"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode=train_config["early_stopping"]["mode"],
            patience=train_config["early_stopping"]["patience"],
            verbose=True,
        ),
    ]

    logger = WandbLogger(
        project=f"DynaSurvCausalMultiheadProgression_{data_config['subtype']}_{data_config['n_lines']}lines",
        name=f"seed_{split_seed}_{date}",
        save_dir=f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/{date}_seed_{split_seed}",
    )

    trainer = L.Trainer(
        max_epochs=train_config["trainer"]["max_epochs"],
        accelerator="cpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    args = parser.parse_args()

    split_seed = int.from_bytes(os.urandom(4), "big")
    date = datetime.now().strftime("%d%m%Y_%H%M%S")

    config = load_config("../configs/config.toml")
    model_config = load_config("../configs/config_default_train_.toml")

    main(
        model_config=model_config,
        train_config=config["train"],
        eval_config=config["eval"],
        data_config=config["data"],
        split_seed=split_seed,
        date=date,
        fast_dev_run=args.fast_dev_run,
    )
