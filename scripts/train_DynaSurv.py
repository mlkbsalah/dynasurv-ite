import lightning as L
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.data_utils import ESMEDataModule
from CausalSurv.model.DynaSurv import DynaSurv
from CausalSurv.metrics.metrics_callback import MetricsCallback

import wandb

def main():
    data_module = ESMEDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        config="../configs/data.toml",
    )

    input_dims = data_module.get_input_dimensions()
    print(f"Input dimensions: {input_dims}")
    model = DynaSurv(
        x_input_dim=input_dims["x_input_dim"],
        p_input_dim=input_dims["p_input_dim"],
        output_sa_length=input_dims["output_sa_length"],
        config="../configs/dynasurv.toml")
    
    logger = WandbLogger(
        project="Synthetic_data",
        tags=["DynaSurv", "ESME", "HR+HER2-"],
        save_dir="../training_logs",
        settings=wandb.Settings(
            _disable_stats=True, # type: ignore
            _disable_meta=True, # type: ignore
            )
        )
    
    trainer = L.Trainer(
        fast_dev_run=False,
        max_epochs=1000,
        callbacks=[MetricsCallback(time_bins=data_module.time_bins)],
        check_val_every_n_epoch=10,
        log_every_n_steps=5,
        accelerator="mps",
        enable_progress_bar=True,
        logger=logger
    )
    
    trainer.fit(model, datamodule=data_module)
    # trainer.test(model, datamodule=data_module, ckpt_path="best")
    
    # logger.experiment.finish()


if __name__ == "__main__":
    main()