import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

from CausalSurv.data.data_utils import ESMEDataModule
from CausalSurv.model.DynaSurvOnline import DynaSurv
from CausalSurv.metrics.metrics_callback import MetricsCallback

import wandb
from argparse import ArgumentParser


def main(args=None):
    L.seed_everything(1999)
    
    # data_module = ESMEDataModule(
    #     data_dir="../data/model_entry_imputed_data_HER2+_stable_types_categorized.parquet",
    #     config="../configs/data.toml",
    # )

    data_module = ESMEDataModule(
    data_dir="../data/synthetic/multi_line_survival_data.parquet",
    config="../configs/data.toml",
    )

    input_dims = data_module.get_input_dimensions()
    time_bins = data_module.time_bins
    model = DynaSurv(
        x_input_dim=input_dims["x_input_dim"],
        p_input_dim=input_dims["p_input_dim"],
        output_sa_length=input_dims["output_sa_length"],
        config="../configs/dynasurv.toml",
        time_bins=time_bins
    )
    
    logger = WandbLogger(
        project="Synthetic_data",
        tags=["DynaSurv", "ESME", "HR+HER2-"],
        save_dir="../training_logs",
        settings=wandb.Settings(
            _disable_stats=True, # type: ignore
            _disable_meta=True, # type: ignore
            )
        )
    
    # callbacks = [MetricsCallback(time_bins=data_module.time_bins),
    #             #  EarlyStopping(monitor="val/loss", patience=20, mode="min"),
    #             #  ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="best-checkpoint")
    #              LearningRateMonitor(logging_interval="step")]
    
    trainer = L.Trainer(
        max_epochs=500,
        logger=logger,
        callbacks=[MetricsCallback(time_bins=data_module.time_bins)],
        check_val_every_n_epoch=20,
        log_every_n_steps=5,
        accelerator="mps",
        enable_progress_bar=True,
    )
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")
    
    logger.experiment.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/model_entry_imputed_data_HER2+_stable_types_categorized.parquet")
    parser.add_argument("--config_dir", type=str, default="../configs/")

    parser.add_argument("--wandb_project", type=str, default="DynaSurv_EMSE_HR+HER2-")

    args = parser.parse_args()

    main()