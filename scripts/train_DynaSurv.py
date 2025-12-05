import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, LearningRateFinder
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.data_utils import ESMEDataModule
from CausalSurv.model.DynaSurv import DynaSurv

import wandb

def main(config):
    data_module = ESMEDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        n_lines = 2,
        n_intervals=config.get('n_intervals'),
        train_batch_size=config.get('train_batch_size'),
        val_split=0.2,
        test_split=0.1,
    )

    input_dims = data_module.get_input_dimensions()
    print(f"Input dimensions: {input_dims}")
    model = DynaSurv(
            x_input_dim=input_dims['x_input_dim'],
            p_input_dim=input_dims['p_input_dim'],
            output_length=input_dims['output_dim'],
            interval_bounds=input_dims['time_bins'],
            lstm_hidden_length = config.get('lstm_hidden_length'),
            x_embed_dim = config.get('x_embed_dim'),
            p_embed_dim = config.get('p_embed_dim'),
            mlpx_hidden_units = config.get('mlpx_hidden_units'),
            mlpp_hidden_units = config.get('mlpp_hidden_units'),
            mlpsa_hidden_units = config.get('mlpsa_hidden_units'),
            mlpx_dropout = config.get('mlpx_dropout'),
            mlpp_dropout = config.get('mlpp_dropout'),
            mlpsa_dropout = config.get('mlpsa_dropout'),
            lr = config.get('lr'),
            weight_decay = config.get('weight_decay'),
    )
    

    logger = WandbLogger(
        project="SMOKE_TEST_DynaSurv_ESME_2lines",
        tags=["DynaSurv", "ESME", "HR+HER2-"],
        save_dir="../training_logs",
        settings=wandb.Settings(
            _disable_stats=True, # type: ignore
            _disable_meta=True, # type: ignore
            )
        )
    
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        # LearningRateFinder(),
    ]
    
    
    trainer = L.Trainer(
        fast_dev_run=False,
        max_epochs=500,
        check_val_every_n_epoch=2,
        log_every_n_steps=5,
        accelerator="cpu",
        enable_progress_bar=True,
        logger=logger, 
        callbacks=callbacks,
    )
    
    trainer.fit(model, datamodule=data_module)
    wandb.finish()


if __name__ == "__main__":
    config = {'lr': 0.008052881880009034,
              'lstm_hidden_length': 16,
              'max_epochs': 1000,
              'mlpp_dropout': 0.41651302880731145,
              'mlpp_hidden_units': [32],
              'mlpsa_dropout': 0.28089900571923276,
              'mlpsa_hidden_units': [32, 16],
              'mlpx_dropout': 0.4511565951423924,
              'mlpx_hidden_units': [32, 32],
              'n_intervals': 5,
              'output_length': 5,
              'p_embed_dim': 8,
              'p_input_dim': 11,
              'train_batch_size': 128,
              'weight_decay': 0.0012964898763545834,
              'x_embed_dim': 16,
              'x_input_dim': 68,
              }
    n_bootstraps = 50
    for i in range(n_bootstraps):
        print(f"Bootstrap iteration {i+1}/{n_bootstraps}")
        main(config)