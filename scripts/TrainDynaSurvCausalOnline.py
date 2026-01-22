import argparse
import json
import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.CV_online_data_utils import ESMEOnlineDataModuleCV
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline



def main(config, split_seed):
    data_module = ESMEOnlineDataModuleCV(
        data_dir="../data",
        subtype="HR+HER2-",
        n_lines=4,
        n_intervals=config['n_intervals'],
        batch_size=config['train_batch_size'],
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
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            monitor="average_ci",
            mode="max",
            save_top_k=1,
            dirpath=f"../models/HR+HER2-/4lines/seed_{split_seed}/checkpoints/",
            filename="dynaSurvCausalOnline-{epoch:02d}-{average_ci:.4f}",
        ),
    ]
    
    logger = WandbLogger(
        project="DynaSurvCausalOnline_HR+HER2-_4lines_final_model",
        name=f"seed_{split_seed}",
        save_dir="../wandb_logs/",
    )

    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_progress_bar=True,
        check_val_every_n_epoch=2,
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_seed", type=int, required=True)

    args = parser.parse_args()
    split_seed = args.split_seed

    config_dir = f"../models/HR+HER2-/4lines/seed_{split_seed}"
    with open(os.path.join(config_dir, "best_config.json"), "r") as f:
        best_config = json.load(f)
    best_config = best_config["config"]

    main(best_config, split_seed)
