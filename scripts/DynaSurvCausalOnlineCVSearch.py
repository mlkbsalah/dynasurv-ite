import argparse
import json
import os
import tomllib
import random

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.CV_online_data_utils import ESMEOnlineDataModuleCV
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline

def load_config(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


# ==============================================================================
# MAIN CV FUNCTION
# ==============================================================================
def main(model_config, train_config, split_seed, trial_id, n_folds, project_name):

    loss_folds = []
    average_ci_folds = []
    average_ibs_folds = []

    for k in range(n_folds):
        print(f"[Trial {trial_id}] Starting fold {k+1}/{n_folds}")

        DataModuleCV = ESMEOnlineDataModuleCV(
            data_dir="/workdir/bensalama/DynaSurv/data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
            # data_dir="/Users/malek/TheLAB/DynaSurv/data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
            n_lines=4,
            n_intervals=model_config["n_intervals"],
            batch_size=model_config["train_batch_size"],
            fold_idx=k,
            num_folds=n_folds,
            split_seed=split_seed,
            num_workers=4,
        )

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
        )

        callbacks = [
            EarlyStopping(monitor=train_config['early_stopping']['monitor'], 
                          mode=train_config['early_stopping']['mode'], 
                          patience=train_config['early_stopping']['patience'], 
                          verbose=True),
        ]
        
        if not project_name:
            logger = False

        else:
            logger = WandbLogger(
				project=project_name,
           		name=f"trial_{trial_id}_fold_{k}",
            	group=f"trial_{trial_id}_CV",
            	reinit=True,
            	save_dir="../training_logs",
				)	
            callbacks.append(LearningRateMonitor(logging_interval="epoch")) #type: ignore

        trainer = L.Trainer(
            max_epochs=train_config["trainer"]["max_epochs"],
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=callbacks, #type: ignore
            enable_checkpointing=False,
            enable_progress_bar=False,
            check_val_every_n_epoch=2,
        )

        trainer.fit(model, datamodule=DataModuleCV)
        val_res = trainer.validate(model, datamodule=DataModuleCV)[0]

        loss_folds.append(val_res["val_loss"])
        average_ci_folds.append(val_res["average_ci"])
        average_ibs_folds.append(val_res["average_ibs"])
	
        wandb.finish()

    return {
        "mean_loss": float(np.mean(loss_folds)),
	"std_loss":float(np.std(loss_folds)),
        "mean_ci": float(np.mean(average_ci_folds)),
        "std_ci": float(np.std(average_ci_folds)),
        "mean_ibs": float(np.mean(average_ibs_folds)),
        "std_ibs": float(np.std(average_ibs_folds)),
    }


# ==============================================================================
# RANDOM CONFIG SAMPLER
# ==============================================================================
def sample_config():
    return {
        "n_intervals": random.choice([5, 10, 15, 20]),
        "train_batch_size": random.choice([64, 128, 256, 512]),
        "lstm_hidden_length": random.choice([16, 32, 64]),
        "x_embed_dim": random.choice([16, 32, 64]),
        "p_embed_dim": random.choice([8, 16, 32]),
        "mlpx_hidden_units": random.choice([[64, 32], [128, 64], [32, 32]]),
        "mlpp_hidden_units": random.choice([[16], [32], [64]]),
        "mlpsa_hidden_units": random.choice([[32, 16], [64, 32], [128, 64]]),
        "init_h_hidden": random.choice([[32], [64], [128], [64, 64]]),
        "init_p_hidden": random.choice([[16], [32], [64], [32, 32]]),
        "mlpprop_hidden_units": random.choice([[16, 8], [32, 16], [64, 32]]),
        "attention": True,
        "mlpx_dropout": np.random.uniform(0.0, 0.5),
        "mlpp_dropout": np.random.uniform(0.0, 0.5),
        "mlpsa_dropout": np.random.uniform(0.0, 0.5),
        "init_h_dropout": np.random.uniform(0.0, 0.5),
        "init_p_dropout": np.random.uniform(0.0, 0.5),
        "mlpprop_dropout": np.random.uniform(0.0, 0.5),
        "lambda_prop_loss": np.random.uniform(0.0, 1.0),
        "lr": 10 ** np.random.uniform(-5, -2),
        "weight_decay": 10 ** np.random.uniform(-6, -2),
        "lr_scheduler_stepsize": random.choice([50, 100, 200]),
        "lr_scheduler_gamma": random.choice([0.1, 0.3, 0.6, 0.9]),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_id", type=int, required=False)
    parser.add_argument("--split_seed", type=int, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    
    parser.add_argument("--project_name", type=str, default=None)

    args = parser.parse_args()

    print(f"Running trial {args.trial_id} with split seed {args.split_seed} and {args.n_folds} folds")

    model_config = sample_config()
    train_config = load_config("../configs/train_config.toml")

    results = main(model_config, train_config, args.split_seed, args.trial_id, args.n_folds, args.project_name)

    out_dir = f"../models/HR+HER2-/4lines/seed_{args.split_seed}"
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
