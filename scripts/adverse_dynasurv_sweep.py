import os

import lightning as L
import tomllib
from CausalSurv.model.DynaSurvCausalOnline import DynaSurvCausalOnline
from lightning.pytorch.callbacks import EarlyStopping

from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV


def load_config(config_path):
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config


def main(
    model_config,
    train_config,
    eval_config,
    data_config,
    split_seed,
    lambda_ipm: list,
):
    auc_dict = {}

    data_module = ESMEOnlineDataModuleCV(
        data_dir=data_config["data_dir"],
        subtype=data_config["subtype"],
        n_lines=data_config["n_lines"],
        n_intervals=model_config["n_intervals"],
        batch_size=data_config["batch_size"],
        split_seed=split_seed,
        num_workers=4,
        final_training=True,
    )

    data_module.prepare_data()
    data_module.setup()
    data_dims = data_module.get_data_dimensions()

    XPd, X_static, interval_idx, treatment_idx, time, event, mask, patient_id = next(
        iter(data_module.test_dataloader())
    )

    for alpha in lambda_ipm:
        model = DynaSurvCausalOnline(
            x_input_dim=data_dims["x_input_dim"],
            x_static_dim=data_dims["x_static_dim"],
            p_input_dim=data_dims["p_input_dim"],
            p_static_dim=data_dims["p_static_dim"],
            output_length=data_dims["output_dim"],
            interval_bounds=data_dims["time_bins"],
            n_treatments=data_dims["p_input_dim"],
            lstm_hidden_length=model_config["lstm_hidden_length"],
            x_embed_dim=model_config["x_embed_dim"],
            p_embed_dim=model_config["p_embed_dim"],
            init_h_hidden=model_config["init_h_hidden"],
            init_p_hidden=model_config["init_p_hidden"],
            mlpx_hidden_units=model_config["mlpx_hidden_units"],
            mlpp_hidden_units=model_config["mlpp_hidden_units"],
            mlpsa_hidden_units=model_config["mlpsa_hidden_units"],
            mlpprop_hidden_units=model_config["mlpprop_hidden_units"],
            lambda_ipm_emd2=0,
            lambda_prop_loss=alpha,
            lambda_ipm_mmd=0,
            lr=model_config["lr"],
            weight_decay=model_config["weight_decay"],
            lr_scheduler_stepsize=model_config["lr_scheduler_stepsize"],
            lr_scheduler_gamma=model_config["lr_scheduler_gamma"],
            attention=model_config["attention"],
            evaluation_horizon_times=eval_config["horizon_times"],
            brier_integration_step=eval_config["integration_step"],
        )

        trainer = L.Trainer(
            max_epochs=30,
            accelerator="gpu",
            devices=1,
            callbacks=EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=20,
            ),
            enable_checkpointing=True,
            enable_progress_bar=True,
            check_val_every_n_epoch=5,
        )
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

        aucs = model.compute_treatment_prediction_auc(
            XPd, X_static, treatment_idx, mask
        )
        auc_dict[alpha] = aucs

    return auc_dict


if __name__ == "__main__":
    import os

    import pandas as pd

    split_seed = int.from_bytes(os.urandom(4), "big")

    config = load_config("../configs/config.toml")
    data_config = config["data"]
    train_config = config["train"]
    eval_config = config["eval"]

    model_config_dir = "/Users/malek/TheLAB/DynaSurv/configs/config_default_train_.toml"
    model_config = load_config("../configs/config_default_train_.toml")

    auc_dict = main(
        model_config=model_config,
        train_config=train_config,
        eval_config=eval_config,
        data_config=data_config,
        split_seed=split_seed,
        lambda_ipm=[0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    )

    df = pd.DataFrame(auc_dict)
    df.to_csv("auc_results_adverse.csv")
