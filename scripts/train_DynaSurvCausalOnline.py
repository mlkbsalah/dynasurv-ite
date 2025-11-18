import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from CausalSurv.data.online_data_utils import ESMEOnlineDataModule
from CausalSurv.model.DynaSurvCausalOnline import CausalDynaSurv
from CausalSurv.metrics.metrics_callback import MetricsCallback
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

import wandb


def main():
    L.seed_everything(1999)

    data_module = ESMEOnlineDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        config="../configs/data.toml",
    )

    input_dims = data_module.get_input_dimensions()

    model = CausalDynaSurv(
        x_input_dim=input_dims["x_input_dim"],
        p_input_dim=input_dims["p_input_dim"],
        output_sa_length=input_dims["output_sa_length"],
        n_treatments=input_dims["p_input_dim"],
        n_lines=data_module.n_lines,
        config="../configs/dynasurv.toml",
    )

    logger = WandbLogger(
        project="CausalDynaSurv_EMSE_HR+HER2-Online",
        tags=["CausalDynaSurv", "ESME", "HR+HER2-", "Online"],
        save_dir="../training_logs",
        settings=wandb.Settings(
            _disable_stats=True, # type: ignore
            _disable_meta=True, # type: ignore
            )
    )

    callbacks = [MetricsCallback(time_bins=data_module.time_bins),
                 EarlyStopping(monitor="val/loss", patience=20, mode="min"),
                 ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="best-checkpoint")
                 ]


    trainer = L.Trainer(
        max_epochs=500,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=5,
        log_every_n_steps=5,
        accelerator="auto",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
