import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from CausalSurv.data.data_utils import ESMEDataModule
from CausalSurv.model.DynaSurv import DynaSurv
from CausalSurv.data.config_loader import load_config  # you already use this elsewhere
from CausalSurv.metrics.metrics_callback import MetricsCallback

import matplotlib.pyplot as plt




def main():
    L.seed_everything(1999)

    data_module = ESMEDataModule(
        data_dir="../data/model_entry_imputed_data_HER2+_stable_types_categorized.parquet",
        config="../configs/data.toml",
    )


    
    input_dims = data_module.get_input_dimensions()
    x_input_dim = input_dims["x_input_dim"]
    p_input_dim = input_dims["p_input_dim"]
    output_sa_length = input_dims["output_sa_length"]


    model_config = load_config("../configs/dynasurv.toml")
    model = DynaSurv(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
        output_sa_length=output_sa_length,
        config=model_config,
    )

    logger = TensorBoardLogger(save_dir="logs", name="esme")
    print(f"Logging to: {logger.log_dir}")

    trainer = L.Trainer(

        max_epochs=500,
        logger=logger,
        log_every_n_steps=5,
        accelerator="auto",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=10,
        callbacks=[EarlyStopping(monitor="val/loss", patience=10, mode="min"), MetricsCallback(time_bins=data_module.time_bins)],
    )

    trainer.fit(model, datamodule=data_module)

    # predict survival for test set
    data_module.setup(stage="test")
    test_loader = data_module.test_dataloader()
    model.eval()
    
    batch = next(iter(test_loader))
    XPd, sa_true, time, event = batch
    with torch.no_grad():
        batch_prediction = model.predict_survival(XPd)
    patient_prediction = batch_prediction[0]

    # Plotting
    plt.figure(figsize=(10, 6))
    for time in range(patient_prediction.shape[0]):
        plt.plot(patient_prediction[time, :].detach().numpy(), label=f"Time {time + 1}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()