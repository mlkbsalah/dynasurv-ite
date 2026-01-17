import argparse
import json
import os
import random

import lightning as L
import numpy as np
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.CV_online_data_utils import ESMEOnlineDataModuleCV
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline


# ==============================================================================
# MAIN CV FUNCTION
# ==============================================================================
def main(config, split_seed, trial_id, n_folds, project_name):

    loss_folds = []
    average_ci_folds = []
    average_ibs_folds = []

    for k in range(n_folds):
        print(f"[Trial {trial_id}] Starting fold {k+1}/{n_folds}")

        DataModuleCV = ESMEOnlineDataModuleCV(
            data_dir="/workdir/bensalama/DynaSurv/data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
            # data_dir="/Users/malek/TheLAB/DynaSurv/data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
            n_lines=4,
            n_intervals=config["n_intervals"],
            batch_size=config["train_batch_size"],
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
            lstm_hidden_length=config["lstm_hidden_length"],
            x_embed_dim=config["x_embed_dim"],
            p_embed_dim=config["p_embed_dim"],
            init_h_hidden=config["init_h_hidden"],
            init_p_hidden=config["init_p_hidden"],
            mlpx_hidden_units=config["mlpx_hidden_units"],
            mlpp_hidden_units=config["mlpp_hidden_units"],
            mlpsa_hidden_units=config["mlpsa_hidden_units"],
            mlpprop_hidden_units=config["mlpprop_hidden_units"],
            init_h_dropout=config["init_h_dropout"],
            init_p_dropout=config["init_p_dropout"],
            mlpx_dropout=config["mlpx_dropout"],
            mlpp_dropout=config["mlpp_dropout"],
            mlpsa_dropout=config["mlpsa_dropout"],
            mlpprop_dropout=config["mlpprop_dropout"],
            lambda_prop_loss=config["lambda_prop_loss"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            lr_scheduler_stepsize=config["lr_scheduler_stepsize"],
            lr_scheduler_gamma=config["lr_scheduler_gamma"],
            attention=config["attention"],
        )

        callbacks = [
            EarlyStopping(monitor="average_ci", mode="max", patience=30, verbose=True),
            # LearningRateMonitor(logging_interval="step"),
            # ModelCheckpoint(
            #     monitor="average_ci",
            #     mode="max",
            #     save_top_k=1,
            #     dirpath=f"../models/HR+HER2-/4lines/CV_sweep/seed_{split_seed}/trial_{trial_id}",
            #     filename=f"fold{k+1}" + "_{epoch:02d}_{average_ci:.4f}",
            # ),
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
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        trainer = L.Trainer(
            max_epochs=config["max_epochs"],
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

        loss_folds.append(val_res["val/loss"])
        average_ci_folds.append(val_res["average_ci"])
        average_ibs_folds.append(val_res["average_ibs"])
	
        wandb.finish()

    return {
        "mean_loss": float(np.mean(loss_folds)),
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
        "max_epochs": 300,
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

    config = sample_config()
    results = main(config, args.split_seed, args.trial_id, args.n_folds, args.project_name)

    out_dir = f"../models/HR+HER2-/4lines/seed_{args.split_seed}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/trial_{args.trial_id}.json", "w") as f:
        json.dump(
            {
                "trial_id": args.trial_id,
                "split_seed": args.split_seed,
                "config": config,
                **results,
            },
            f,
            indent=2,
        )
