import argparse
import json
import os

import lightning as L
import tomllib
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.CV_online_data_utils import ESMEOnlineDataModuleCV
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline


def load_config(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def load_model_config(model_config_dir):
    with open(os.path.join(model_config_dir, "best_config.json"), "r") as f:
        best_model_config = json.load(f)
    return best_model_config["config"]


def main(
    model_config, train_config, eval_config, data_config, split_seed, fast_dev_run=False
):
    if fast_dev_run:
        train_config["trainer"]["max_epochs"] = 3

    data_module = ESMEOnlineDataModuleCV(
        data_dir=data_config["data_dir"],
        subtype=data_config["subtype"],
        n_lines=data_config["n_lines"],
        horizon=data_config["horizon"],
        n_intervals=model_config["n_intervals"],
        batch_size=data_config["batch_size"],
        split_seed=split_seed,
        num_workers=4,
        final_training=True,
    )

    data_dims = data_module.get_data_dimensions()

    model = DynaSurvCausalOnline(
        x_input_dim=data_dims["x_input_dim"],
        x_static_dim=data_dims["x_static_dim"],
        p_input_dim=data_dims["p_input_dim"],
        p_static_dim=data_dims["p_static_dim"],
        output_length=data_dims["output_dim"],
        interval_bounds=data_dims["time_bins"],
        n_treatments=data_dims["p_input_dim"],
        lstm_hidden_length=model_config["lstm_hidden_length"],
        x_embed_dim=model_config["x_embed_dim"],
        p_embed_dim=model_config["p_embed_dim"],
        init_h_hidden=model_config["init_h_hidden"],
        init_p_hidden=model_config["init_p_hidden"],
        mlpx_hidden_units=model_config["mlpx_hidden_units"],
        mlpp_hidden_units=model_config["mlpp_hidden_units"],
        mlpsa_hidden_units=model_config["mlpsa_hidden_units"],
        mlpprop_hidden_units=model_config["mlpprop_hidden_units"],
        init_h_dropout=model_config["init_h_dropout"],
        init_p_dropout=model_config["init_p_dropout"],
        mlpx_dropout=model_config["mlpx_dropout"],
        mlpp_dropout=model_config["mlpp_dropout"],
        mlpsa_dropout=model_config["mlpsa_dropout"],
        mlpprop_dropout=model_config["mlpprop_dropout"],
        lambda_prop_loss=model_config["lambda_prop_loss"],
        lr=model_config["lr"],
        weight_decay=model_config["weight_decay"],
        lr_scheduler_stepsize=model_config["lr_scheduler_stepsize"],
        lr_scheduler_gamma=model_config["lr_scheduler_gamma"],
        attention=model_config["attention"],
        evaluation_horizon_times=eval_config["horizon_times"],
        brier_integration_step=eval_config["integration_step"],
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=train_config["early_stopping"]["monitor"],
            mode=train_config["early_stopping"]["mode"],
            patience=train_config["early_stopping"]["patience"],
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/seed_{split_seed}/checkpoints/",
            filename="dynaSurvCausalOnline-{epoch:02d}-{val_loss: .4f}",
        ),
        ModelCheckpoint(
            monitor="average_ci",
            mode="max",
            save_top_k=1,
            dirpath=f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/seed_{split_seed}/checkpoints/",
            filename="dynaSurvCausalOnline-bestCI-{epoch:02d}-{average_ci: .4f}",
        ),
        ModelCheckpoint(
            monitor="average_ibs",
            mode="min",
            save_top_k=1,
            dirpath=f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/seed_{split_seed}/checkpoints/",
            filename="dynaSurvCausalOnline-bestIBS-{epoch:02d}-{average_ibs: .4f}",
        ),
    ]

    logger = WandbLogger(
        project=f"DynaSurvCausalOnline_{data_config['subtype']}_{data_config['n_lines']}lines_final_model",
        name=f"seed_{split_seed}",
        save_dir="../wandb_logs/",
    )

    trainer = L.Trainer(
        max_epochs=train_config["trainer"]["max_epochs"],
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=5,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_seed", type=int, required=True)
    parser.add_argument(
        "--fast_dev_run", action="store_true", help="Enable fast development run mode"
    )
    args = parser.parse_args()
    split_seed = args.split_seed
    fast_dev_run = args.fast_dev_run

    config = load_config("../configs/config.toml")
    data_config = config["data"]
    train_config = config["train"]
    eval_config = config["eval"]

    model_config_dir = f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/seed_{split_seed}"
    best_model_config = load_model_config(model_config_dir)

    main(
        model_config=best_model_config,
        train_config=train_config,
        eval_config=eval_config,
        data_config=data_config,
        split_seed=split_seed,
        fast_dev_run=fast_dev_run,
    )
