import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from CausalSurv.data.online_data_utils import ESMEDataModule
from CausalSurv.model.DynaSurvCausalOnline import CausalDynaSurv
from CausalSurv.data.config_loader import load_config  # you already use this elsewhere
from CausalSurv.metrics.metrics_callback import MetricsCallback



def main():
    L.seed_everything(1999)

    data_module = ESMEDataModule(
        data_dir="../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized.parquet",
        config="../configs/data.toml",
    )

    input_dims = data_module.get_input_dimensions()
    x_input_dim = input_dims["x_input_dim"]
    p_input_dim = input_dims["p_input_dim"]
    output_sa_length = input_dims["output_sa_length"]
    n_treatments = input_dims["p_input_dim"]  
    n_lines = data_module.n_lines
    model_config = load_config("../configs/dynasurv.toml")
    print(f"Number of treatments: {n_treatments}")
    print(f"Number of lines: {n_lines}")

    model = CausalDynaSurv(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
        output_sa_length=output_sa_length,
        n_treatments=n_treatments,
        n_lines=n_lines,
        config=model_config,
    )

    logger = TensorBoardLogger(save_dir="logs", name="causal_esme_online")
    print(f"Logging to: {logger.log_dir}")

    trainer = L.Trainer(
        max_epochs=300,
        logger=logger,
        log_every_n_steps=1000,
	strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=-1,
        check_val_every_n_epoch=10000,
        # callbacks=[MetricsCallback(time_bins=data_module.time_bins), EarlyStopping(monitor="validation/loss", patience=20, mode="min")],
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
