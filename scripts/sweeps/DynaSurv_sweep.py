import lightning as L
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.data_utils import ESMEDataModule
from CausalSurv.model.DynaSurv import DynaSurv
from CausalSurv.metrics.metrics_callback import MetricsCallback

import wandb

def main():
    data_module = ESMEDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        n_lines = 3,
        n_intervals=10,
        train_batch_size=256,
        val_split=0.2,
        test_split=0.1,
    )

    input_dims = data_module.get_input_dimensions()
    print(f"Input dimensions: {input_dims}")
    model = DynaSurv(
            x_input_dim=input_dims['x_input_dim'],
            p_input_dim=input_dims['p_input_dim'],
            output_length=data_module.n_intervals,
            interval_bounds=input_dims['time_bins'],
            lstm_hidden_length = 64,
            x_embed_dim = 32,
            p_embed_dim = 16,
            mlpx_hidden_units = [64, 32],
            mlpp_hidden_units = [32, 16],
            mlpsa_hidden_units = [32, 16],
            mlpx_dropout = 0.2,
            mlpp_dropout = 0.2,
            mlpsa_dropout = 0.2,
            lr = 1e-3,
            weight_decay = 1e-4,
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
    
    trainer = L.Trainer(
        fast_dev_run=False,
        max_epochs=1000,
        check_val_every_n_epoch=10,
        log_every_n_steps=5,
        accelerator="cpu",
        enable_progress_bar=True,
        logger=logger
    )
    
    trainer.fit(model, datamodule=data_module)
    wandb.finish()


if __name__ == "__main__":
    main()