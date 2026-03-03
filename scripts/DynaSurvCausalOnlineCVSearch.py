import argparse
import json
import os
import random
from datetime import datetime

import lightning as L
import numpy as np
import tomllib
import wandb
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
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
    model_config,
    data_config,
    train_config,
    eval_config,
    split_seed,
    date,
    trial_id,
    n_folds,
    project_name,
    fast_dev_run=False,
):
    if fast_dev_run:
        n_folds = 3
        train_config["trainer"]["max_epochs"] = 3

    loss_folds = []
    average_ci_folds = []
    average_ibs_folds = []

    for k in range(n_folds):
        print(f"[Trial {trial_id}] Starting fold {k + 1}/{n_folds}")

        DataModuleCV = ESMEOnlineDataModuleCV(
            data_dir=data_config["data_dir"],
            subtype=data_config["subtype"],
            n_lines=data_config["n_lines"],
            n_intervals=model_config["n_intervals"],
            batch_size=model_config["train_batch_size"],
            fold_idx=k,
            num_folds=n_folds,
            split_seed=split_seed,
            num_workers=4,
        )

        DataModuleCV.prepare_data()
        data_dims = DataModuleCV.get_data_dimensions()

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
            EarlyStopping(
                monitor=train_config["early_stopping"]["monitor"],
                mode=train_config["early_stopping"]["mode"],
                patience=train_config["early_stopping"]["patience"],
                verbose=True,
            ),
        ]

        if not project_name:
            logger = False

        else:
            logger = WandbLogger(
                project=project_name,
                name=f"trial_{trial_id}_fold_{k}",
                group=f"trial_{trial_id}_CV",
                reinit=True,
                save_dir=f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/seed_{split_seed}_{date}",
            )
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))  # type: ignore

        trainer = L.Trainer(
            max_epochs=train_config["trainer"]["max_epochs"],
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=callbacks,  # type: ignore
            enable_checkpointing=False,
            enable_progress_bar=True,
            check_val_every_n_epoch=1,
        )

        trainer.fit(model, datamodule=DataModuleCV)
        val_res = trainer.validate(
            model, datamodule=DataModuleCV
        )[
            0
        ]  # We only care about the first dataloader (the other one is just for early stopping)
        loss_folds.append(val_res["val_loss/dataloader_idx_0"])
        average_ci_folds.append(val_res["average_ci/dataloader_idx_0"])
        average_ibs_folds.append(val_res["average_ibs/dataloader_idx_0"])

        wandb.finish()

    return {
        "mean_loss": float(np.mean(loss_folds)),
        "std_loss": float(np.std(loss_folds)),
        "mean_ci": float(np.mean(average_ci_folds)),
        "std_ci": float(np.std(average_ci_folds)),
        "mean_ibs": float(np.mean(average_ibs_folds)),
        "std_ibs": float(np.std(average_ibs_folds)),
    }


def sample_config():
    fc_search = [[8, 16, 32, 64, 128]]
    return {
        "n_intervals": random.choice([10, 30, 50, 80, 100]),
        "train_batch_size": random.choice([64, 128, 256, 512]),
        "lstm_hidden_length": random.choice([8, 16, 32, 64]),
        "x_embed_dim": random.choice([16, 32, 64, 128]),  # size around 80
        "p_embed_dim": random.choice([2, 8, 16, 32]),  # size around 11
        "mlpx_hidden_units": random.choice(fc_search),
        "mlpp_hidden_units": random.choice(fc_search),
        "init_h_hidden": random.choice(fc_search),
        "init_p_hidden": random.choice(fc_search),
        "mlpsa_hidden_units": random.choice(
            [
                [16, 16],
                [32, 32],
                [64, 64],
            ]
        ),
        "mlpprop_hidden_units": random.choice(fc_search),
        "attention": True,
        "lambda_prop_loss": np.random.uniform(0.0, 1.0),
        "lr": 10 ** np.random.uniform(-5, -2),
        "weight_decay": 10 ** np.random.uniform(-6, -2),
        "lr_scheduler_stepsize": random.choice([10, 30, 50]),
        "lr_scheduler_gamma": random.choice([0.1, 0.3, 0.6]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_id", type=int, required=False, default=0)
    parser.add_argument("--split_seed", type=int, required=True)
    parser.add_argument(
        "--date", type=str, default=datetime.now().strftime("%d%m%Y_%H%M%S")
    )
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--project_name", type=str, default=None)
    args = parser.parse_args()

    if args.fast_dev_run:
        print("Running in fast dev run mode: only 2 folds and 3 epochs per fold")
    else:
        print(
            f"Running trial {args.trial_id} with split seed {args.split_seed} and {args.n_folds} folds"
        )

    model_config = sample_config()
    config = load_config("../configs/config.toml")
    train_config = config["train"]
    eval_config = config["eval"]
    data_config = config["data"]

    results = main(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        eval_config=eval_config,
        split_seed=args.split_seed,
        date=args.date,
        trial_id=args.trial_id,
        n_folds=args.n_folds,
        project_name=args.project_name,
        fast_dev_run=args.fast_dev_run,
    )

    out_dir = f"../models/{data_config['subtype']}/{data_config['n_lines']}lines/seed_{args.split_seed}_{args.date}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/trial_{args.trial_id}.json", "w") as f:
        json.dump(
            {
                "trial_id": args.trial_id,
                "split_seed": args.split_seed,
                "config": model_config,
                **results,
            },
            f,
            indent=2,
        )
