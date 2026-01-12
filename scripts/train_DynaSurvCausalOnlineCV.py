import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import numpy as np

from CausalSurv.data.CV_online_data_utils import ESMEOnlineDataModuleCV
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline

import wandb

from datetime import datetime

import random

NUMBER_OF_FOLDS = 3
SPLIT_SEED = np.random.randint(0, 10000)

def main(config):
    
    ## ==============================================================================
    #                             CROSS VALIDATION SETUP
    ## ==============================================================================
    loss_folds = []
    average_ci_folds = []
    average_ibs_folds = []
    for k in range(NUMBER_OF_FOLDS):
        print(f"Starting fold {k+1} out of {NUMBER_OF_FOLDS}...")
        DataModuleCV = ESMEOnlineDataModuleCV(
            data_dir = "/workdir/bensalama/DynaSurv/data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
            n_lines = 4,
            n_intervals=config['n_intervals'],
            batch_size=config['train_batch_size'],
            fold_idx = k,
            num_folds = NUMBER_OF_FOLDS,
            split_seed = SPLIT_SEED,
            num_workers=4,
        )
        data_dims = DataModuleCV.get_data_dimensions()

        model = DynaSurvCausalOnline(
                x_input_dim=data_dims['x_input_dim'],
                x_static_dim=data_dims['x_static_dim'],
                p_input_dim=data_dims['p_input_dim'],
                p_static_dim=data_dims['p_static_dim'],
                output_length=data_dims['output_dim'],
                interval_bounds=data_dims['time_bins'],
                n_treatments=data_dims['p_input_dim'],
                lstm_hidden_length = config['lstm_hidden_length'],
                x_embed_dim = config['x_embed_dim'],
                p_embed_dim = config['p_embed_dim'],
                init_h_hidden= config['init_h_hidden'],
                init_p_hidden= config['init_p_hidden'],
                mlpx_hidden_units = config['mlpx_hidden_units'],
                mlpp_hidden_units = config['mlpp_hidden_units'],
                mlpsa_hidden_units = config['mlpsa_hidden_units'],
                mlpprop_hidden_units = config['mlpprop_hidden_units'],
                init_h_dropout= config['init_h_dropout'],
                init_p_dropout = config['init_p_dropout'],
                mlpx_dropout = config['mlpx_dropout'],
                mlpp_dropout = config['mlpp_dropout'],
                mlpsa_dropout = config['mlpsa_dropout'],
                mlpprop_dropout= config['mlpprop_dropout'],
                lambda_prop_loss = config['lambda_prop_loss'],
                lr = config['lr'],
                weight_decay = config['weight_decay'],
                lr_scheduler_stepsize = config['lr_scheduler_stepsize'],
                lr_scheduler_gamma = config['lr_scheduler_gamma'],
                attention = config['attention'],
        )

        callbacks = [
                    LearningRateMonitor(logging_interval="step"),
                    EarlyStopping(monitor="average_ci", mode="max", patience=20, verbose=True),  
                    ModelCheckpoint(monitor="average_ci", 
                                    mode="max", 
                                    dirpath="../models/HR+HER2-/4lines/CV_sweep/seed_" + str(SPLIT_SEED),
                                    filename=f"version_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fold{str(k+1)}_"+"{epoch:02d}_{average_ci:.4f}",
                                    save_top_k=1,
                    ),
        ]
        
        logger = WandbLogger(
            project="test_sweep_CV",
            tags=["DynaSurv", "ESME", "HR+HER2-", "Online"],
            save_dir="../training_logs",
            settings=wandb.Settings(
                _disable_stats=True, # type: ignore
                _disable_meta=True, # type: ignore
                )
            )
    
        trainer = L.Trainer(
            max_epochs=config['max_epochs'],
            check_val_every_n_epoch=2,
            accelerator="mps",
            devices=1,
            enable_progress_bar=True,
            logger=logger, 
            callbacks=callbacks, 
        )
        
        trainer.fit(model, datamodule=DataModuleCV)
        
        
        #  Evaluate on the validation set
        validation_results = trainer.validate(model, datamodule=DataModuleCV)[0]
        loss_folds.append(validation_results['val/loss'])
        average_ci_folds.append(validation_results['average_ci'])
        average_ibs_folds.append(validation_results['average_ibs'])

    mean_loss = np.mean(loss_folds)
    std_loss = np.std(loss_folds)
    mean_ci = np.mean(average_ci_folds)
    std_ci = np.std(average_ci_folds)
    mean_ibs = np.mean(average_ibs_folds)
    std_ibs = np.std(average_ibs_folds)


        
    print("Cross-validation results:")
    print(f"Average Loss: {mean_loss} ± {std_loss}")
    print(f"Average C-index: {mean_ci} ± {std_ci}")
    print(f"Average IBS: {mean_ibs} ± {std_ibs}")

    return mean_ci, mean_ibs, mean_loss


def sample_config():
    return {
        # Data
        "n_intervals": random.choice([5, 10, 15, 20]),
        "train_batch_size": random.choice([64, 128, 256, 512]),

        # Architecture
        "lstm_hidden_length": random.choice([16, 32, 64]),
        "x_embed_dim": random.choice([16, 32, 64]),
        "p_embed_dim": random.choice([8, 16, 32]),
        "mlpx_hidden_units": random.choice([[64, 32], [128, 64], [32, 32]]),
        "mlpp_hidden_units": random.choice([[16], [32], [64]]),
        "mlpsa_hidden_units": random.choice([[32, 16], [64, 32], [128, 64], [64, 64], [32, 32]]),
        "init_h_hidden": random.choice([[32], [64], [128], [32, 32], [64, 64]]),
        "init_p_hidden": random.choice([[16], [32], [64], [16, 16], [32, 32]]),
        "mlpprop_hidden_units": random.choice([[16, 8], [32, 16], [64, 32]]),
        "attention": True,

        # Dropout
        "mlpx_dropout": np.random.uniform(0.0, 0.5),
        "mlpp_dropout": np.random.uniform(0.0, 0.5),
        "mlpsa_dropout": np.random.uniform(0.0, 0.5),
        "init_h_dropout": np.random.uniform(0.0, 0.5),
        "init_p_dropout": np.random.uniform(0.0, 0.5),
        "mlpprop_dropout": np.random.uniform(0.0, 0.5),

        # Optim
        "lambda_prop_loss": np.random.uniform(0.0, 1.0),
        "max_epochs": 5,
        "lr": 10 ** np.random.uniform(-5, -2),
        "weight_decay": 10 ** np.random.uniform(-6, -2),
        "lr_scheduler_stepsize": random.choice([50, 100, 200]),
        "lr_scheduler_gamma": random.choice([0.1, 0.3, 0.6, 0.9]),
    }


if __name__ == "__main__":
    import json
    N_TRIALS = 10

    best_ci = -np.inf
    best_config = None
    results = []

    for trial in range(N_TRIALS):
        print(f"\n=== Random search trial {trial+1}/{N_TRIALS} ===")
        config = sample_config()

        mean_ci, mean_ibs, mean_loss = main(config)

        results.append({
            "config": config,
            "mean_ci": mean_ci,
            "mean_ibs": mean_ibs,
            "mean_loss": mean_loss,
        })

        if mean_ci > best_ci:
            best_ci = mean_ci
            best_config = config

    print("\n====================")
    print("Best configuration:")
    print(best_config)
    print(f"Best CV mean C-index: {best_ci:.4f}")

    with open("../models/HR+HER2-/4lines/CV_sweep/seed_" + str(SPLIT_SEED) + "/best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    