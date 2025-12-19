import lightning as L
from pytorch_lightning.profilers import AdvancedProfiler
from lightning.pytorch.callbacks import LearningRateMonitor, LearningRateFinder
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.online_data_utils import ESMEOnlineDataModule
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline

import wandb

def main():
    wandb.init()
    config = wandb.config
    data_module = ESMEOnlineDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        n_lines = 4,
        n_intervals=config.get('n_intervals'),
        train_batch_size=config.get('train_batch_size'),
        val_split=0.2,
        test_split=0.1,
    )

    data_dims = data_module.get_data_dimensions()
    model = DynaSurvCausalOnline(
            x_input_dim=data_dims['x_input_dim'],
            x_static_dim=data_dims['x_static_dim'],
            p_input_dim=data_dims['p_input_dim'],
            p_static_dim=data_dims['p_static_dim'],
            output_length=data_dims['output_dim'],
            interval_bounds=data_dims['time_bins'],
            n_treatments=data_dims['p_input_dim'],
            lstm_hidden_length = config.lstm_hidden_length,
            x_embed_dim = config.x_embed_dim,
            p_embed_dim = config.p_embed_dim,
            init_h_hidden= config.init_h_hidden,
            init_p_hidden= config.init_p_hidden,
            mlpx_hidden_units = config.mlpx_hidden_units,
            mlpp_hidden_units = config.mlpp_hidden_units,
            mlpsa_hidden_units = config.mlpsa_hidden_units,
            mlpprop_hidden_units = config.mlpprop_hidden_units,
            init_h_dropout= config.init_h_dropout,
            init_p_dropout = config.init_p_dropout,
            mlpx_dropout = config.mlpx_dropout,
            mlpp_dropout = config.mlpp_dropout,
            mlpsa_dropout = config.mlpsa_dropout,
            mlpprop_dropout= config.mlpprop_dropout,
            lambda_prop_loss = config.lambda_prop_loss,
            lr = config.lr,
            weight_decay = config.weight_decay,
            lr_scheduler_stepsize = config.lr_scheduler_stepsize,
            lr_scheduler_gamma = config.lr_scheduler_gamma,
    )
    
    logger = WandbLogger(
        project="SMOKE_TEST_4lines_adversarial",
        tags=["DynaSurv", "ESME", "HR+HER2-", "Online"],
        save_dir="../training_logs",
        settings=wandb.Settings(
            _disable_stats=True, # type: ignore
            _disable_meta=True, # type: ignore
            )
        )
    
    callbacks = [
                 LearningRateMonitor(logging_interval="step"),
                #  LearningRateFinder(),
    ]
    
    trainer = L.Trainer(
        fast_dev_run=False,
        max_epochs=10,
        check_val_every_n_epoch=5,
        log_every_n_steps=5,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True,
        logger=logger, #type: ignore
        callbacks=callbacks, #type: ignore
    )
    
    trainer.fit(model, datamodule=data_module)
    # wandb.finish()


if __name__ == "__main__":
    sweep_config = {
        "method": "random",
        "metric": {"name": "val/loss","goal": "minimize"},
        "parameters": {
            # Data parameters
            "n_intervals": {"values": [5, 10, 15, 20]},
            "train_batch_size": {"values": [64, 128, 256, 512]},

            # Model architecture
            "lstm_hidden_length": {"values": [16, 32, 64]},
            "x_embed_dim": {"values": [16, 32, 64]},
            "p_embed_dim": {"values": [8, 16, 32]},
            "mlpx_hidden_units": {"values": [[64, 32],[128, 64],[32, 32]]},
            "mlpp_hidden_units": {"values": [[16],[32],[64]]},
            "mlpsa_hidden_units": {"values": [[32, 16],[64, 32],[128, 64]]},
            "init_h_hidden": {"values": [[32],[64],[128]]},
            "init_p_hidden": {"values": [[16],[32],[64]]},
            "mlpprop_hidden_units": {"values": [[16, 8],[32, 16],[64, 32]]},

            # Dropouts
            "mlpx_dropout": {"min": 0.0,"max": 0.5},
            "mlpp_dropout": {"min": 0.0,"max": 0.5},
            "mlpsa_dropout": {"min": 0.0,"max": 0.5},
            "init_h_dropout": {"min": 0.0,"max": 0.5},
            "init_p_dropout": {"min": 0.0,"max": 0.5},
            "mlpprop_dropout": {"min": 0.0,"max": 0.5},

            # Optimization
            "lambda_prop_loss": {"min": 0.0, "max": 1.0},
            "max_epochs": {"value": 500},
            "lr": {"min": 1e-5,"max": 1e-2,},
            "weight_decay": {"min": 1e-6,"max": 1e-2,},
            "lr_scheduler_stepsize": {"values": [50, 100, 200]},
            "lr_scheduler_gamma": {"values": [0.1, 0.3, 0.6, 0.9]},
            }
        }

    sweep_id = wandb.sweep(sweep_config, project="sweep_DynaSurv_ESME_4lines")
    wandb.agent(sweep_id, function=main, count=3)