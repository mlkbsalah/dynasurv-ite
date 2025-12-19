import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, LearningRateFinder
from lightning.pytorch.loggers import WandbLogger

from CausalSurv.data.online_data_utils import ESMEOnlineDataModule
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline

import wandb

def main(config):
    data_module = ESMEOnlineDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        n_lines = 4,
        n_intervals=config.get('n_intervals'),
        train_batch_size=config.get('train_batch_size'),
        val_split=0.2,
        test_split=0.1,
    )

    data_dims = data_module.get_data_dimensions()
    model = DynaSurvCausalOnline(x_input_dim=data_dims['x_input_dim'],
                           x_static_dim=data_dims['x_static_dim'],
                           p_input_dim=data_dims['p_input_dim'],
                           p_static_dim=data_dims['p_static_dim'],
                           output_length=data_dims['output_dim'],
                           interval_bounds=data_dims['time_bins'],
                           n_treatments=data_dims['p_input_dim'],
                           lstm_hidden_length = config.get('lstm_hidden_length'),
                           x_embed_dim = config.get('x_embed_dim'),
                           p_embed_dim = config.get('p_embed_dim'),
                           init_h_hidden= config.get('init_h_hidden'),
                           init_p_hidden= config.get('init_p_hidden'),
                           mlpx_hidden_units = config.get('mlpx_hidden_units'),
                           mlpp_hidden_units = config.get('mlpp_hidden_units'),
                           mlpsa_hidden_units = config.get('mlpsa_hidden_units'),
                           mlpprop_hidden_units = config.get('mlpprop_hidden_units'),
                           init_h_dropout = config.get('init_h_dropout'),
                           init_p_dropout = config.get('init_p_dropout'),
                           mlpprop_dropout= config.get('mlpprop_dropout'),
                           mlpx_dropout = config.get('mlpx_dropout'),
                           mlpp_dropout = config.get('mlpp_dropout'),
                           mlpsa_dropout = config.get('mlpsa_dropout'),
                           lambda_prop_loss = config.get('lambda_prop_loss'),
                           lr = config.get('lr'),
                           weight_decay = config.get('weight_decay'),
                           lr_scheduler_stepsize = config.get('lr_scheduler_stepsize'),
                           lr_scheduler_gamma = config.get('lr_scheduler_gamma'),
                           )
    
    logger = WandbLogger(
        project="4lines_sweped",
        tags=["DynaSurv", "ESME", "HR+HER2-", "Online", "4lines"],
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
        check_val_every_n_epoch=2,
        log_every_n_steps=5,
        accelerator="mps",
        enable_progress_bar=True,
        logger=logger, #type: ignore
        callbacks=callbacks, #type: ignore
        # profiler = "pytorch",
    )
    
    trainer.fit(model, datamodule=data_module)
    # wandb.finish()


if __name__ == "__main__":
    config = {
    "init_h_dropout": 0.14104338869137617,
    "init_h_hidden": [32],
    "init_p_dropout": 0.3064003851259076,
    "init_p_hidden": [32],
    "lambda_prop_loss": 0.33,
    "lr": 0.008375248313716659,
    "lr_scheduler_gamma": 0.9,
    "lr_scheduler_stepsize": 200,
    "lstm_hidden_length": 64,
    "max_epochs": 500,
    "mlpp_dropout": 0.08590933259285016,
    "mlpp_hidden_units": [32],
    "mlpsa_dropout": 0.2499182741949038,
    "mlpsa_hidden_units": [128, 64],
    "mlpx_dropout": 0.2203446155455437,
    "mlpx_hidden_units": [32, 32],
    "mlpprop_hidden_units": [32, 16],
    "mlpprop_dropout": 0.15,
    "n_intervals": 5,
    "n_treatments": 11,
    "p_embed_dim": 32,
    "p_static_dim": 17,
    "train_batch_size": 128,
    "weight_decay": 0.008887843976818027,
    "x_embed_dim": 64,
}
    
    n_bootstraps = 1
    for i in range(n_bootstraps):
        print(f"Bootstrap iteration {i+1}/{n_bootstraps}")
        main(config)