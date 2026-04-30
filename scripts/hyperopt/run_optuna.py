import argparse
import os
from pathlib import Path

import lightning as L
import optuna
import tomllib
import torch
from lightning.pytorch.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
from CausalSurv.model import DynaSurvCausalOnline

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.toml"

HPO_MAX_EPOCHS = 100
HPO_PATIENCE = 15
HPO_SPLIT_SEED = 42
HPO_N_INTERVALS = 100


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _suggest_mlp(trial: optuna.Trial, name: str, max_depth: int = 3) -> list[int]:
    depth = trial.suggest_int(f"{name}_depth", 1, max_depth)
    width = trial.suggest_categorical(f"{name}_width", [32, 64, 128, 256])
    return [width] * depth


def objective(
    trial: optuna.Trial,
    data_config: dict,
    eval_config: dict,
    data_dims: dict,
) -> float:
    # ---- architecture -------------------------------------------------------
    lstm_hidden = trial.suggest_categorical("lstm_hidden_length", [64, 128, 256])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 4)
    x_embed_dim = trial.suggest_categorical("x_embed_dim", [32, 64, 128])
    p_embed_dim = trial.suggest_categorical("p_embed_dim", [8, 16, 32])
    mlpx_hidden = _suggest_mlp(trial, "mlpx")
    mlpsa_hidden = _suggest_mlp(trial, "mlpsa")
    mlpprop_hidden = _suggest_mlp(trial, "mlpprop", max_depth=2)
    init_h_hidden = _suggest_mlp(trial, "init_h", max_depth=2)
    init_p_hidden = _suggest_mlp(trial, "init_p", max_depth=2)

    # ---- regularisation -----------------------------------------------------
    mlpx_dropout = trial.suggest_float("mlpx_dropout", 0.0, 0.4)
    mlpsa_dropout = trial.suggest_float("mlpsa_dropout", 0.0, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    # ---- optimiser ----------------------------------------------------------
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    lr_scheduler_gamma = trial.suggest_float("lr_scheduler_gamma", 0.1, 0.7)
    lr_scheduler_stepsize = trial.suggest_int("lr_scheduler_stepsize", 10, 50)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # ---- data ---------------------------------------------------------------
    data_module = ESMEOnlineDataModuleCV(
        data_dir="../../data/",
        subtype=data_config["subtype"],
        n_lines=data_config["n_lines"],
        n_intervals=HPO_N_INTERVALS,
        batch_size=batch_size,
        split_seed=HPO_SPLIT_SEED,
        final_training=True,
        num_workers=2,
    )
    data_module.prepare_data()

    # ---- model --------------------------------------------------------------
    model = DynaSurvCausalOnline(
        x_input_dim=data_dims["x_input_dim"],
        x_static_dim=data_dims["x_static_dim"],
        p_input_dim=data_dims["p_input_dim"],
        p_static_dim=data_dims["p_static_dim"],
        output_length=data_dims["output_dim"],
        interval_bounds=data_dims["time_bins"],
        n_treatments=data_dims["p_input_dim"],
        lstm_hidden_length=lstm_hidden,
        lstm_num_layers=lstm_num_layers,
        x_embed_dim=x_embed_dim,
        p_embed_dim=p_embed_dim,
        init_h_hidden=init_h_hidden,
        init_p_hidden=init_p_hidden,
        mlpx_hidden_units=mlpx_hidden,
        mlpp_hidden_units=[p_embed_dim],
        mlpsa_hidden_units=mlpsa_hidden,
        mlpprop_hidden_units=mlpprop_hidden,
        mlpx_dropout=mlpx_dropout,
        mlpp_dropout=0.0,
        mlpsa_dropout=mlpsa_dropout,
        mlpprop_dropout=0.0,
        init_h_dropout=0.0,
        init_p_dropout=0.0,
        lr=lr,
        weight_decay=weight_decay,
        lr_scheduler_stepsize=lr_scheduler_stepsize,
        lr_scheduler_gamma=lr_scheduler_gamma,
        lambda_prop_loss=0.0,
        lambda_ipm_mmd=0.0,
        lambda_ipm_emd2=0.0,
        evaluation_horizon_times=eval_config["horizon_times"],
        brier_integration_step=6,
    )

    # ---- trainer ------------------------------------------------------------
    callbacks: list = [
        EarlyStopping(monitor="val_loss", mode="min", patience=HPO_PATIENCE),
        PyTorchLightningPruningCallback(trial, monitor="average_ci"),
    ]

    trainer = L.Trainer(
        max_epochs=HPO_MAX_EPOCHS,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    try:
        trainer.fit(model, datamodule=data_module)
    except optuna.exceptions.TrialPruned:
        raise

    metric = trainer.callback_metrics.get("average_ci")
    if metric is None:
        raise optuna.exceptions.TrialPruned()
    return float(metric)


def _write_best_config(study: optuna.Study, out_path: Path) -> None:
    p = study.best_params

    def _mlp_units(name: str) -> list[int]:
        return [p[f"{name}_width"]] * p[f"{name}_depth"]

    lines = [
        f"lstm_hidden_length = {p['lstm_hidden_length']}",
        f"lstm_num_layers = {p['lstm_num_layers']}",
        f"n_intervals = {HPO_N_INTERVALS}",
        f"x_embed_dim = {p['x_embed_dim']}",
        f"p_embed_dim = {p['p_embed_dim']}",
        f"init_h_hidden = {_mlp_units('init_h')}",
        f"init_p_hidden = {_mlp_units('init_p')}",
        f"mlpx_hidden_units = {_mlp_units('mlpx')}",
        f"mlpp_hidden_units = [{p['p_embed_dim']}]",
        f"mlpsa_hidden_units = {_mlp_units('mlpsa')}",
        f"mlpprop_hidden_units = {_mlp_units('mlpprop')}",
        f"lr = {p['lr']:.2e}",
        f"lr_scheduler_stepsize = {p['lr_scheduler_stepsize']}",
        f"lr_scheduler_gamma = {p['lr_scheduler_gamma']:.4f}",
        f"weight_decay = {p['weight_decay']:.2e}",
        "attention = true",
        "init_h_dropout = 0",
        "init_p_dropout = 0",
        f"mlpx_dropout = {p['mlpx_dropout']:.4f}",
        "mlpp_dropout = 0",
        f"mlpsa_dropout = {p['mlpsa_dropout']:.4f}",
        "mlpprop_dropout = 0",
        "lambda_prop_loss = 0",
        "lambda_ipm_mmd = 0",
        "lambda_ipm_emd2 = 0",
        "evaluation_horizon_times = [100, 75, 50, 30]",
        "brier_integration_step = 6",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Best config written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="dynasurv_hpo")
    parser.add_argument(
        "--storage",
        type=str,
        default=str(Path(__file__).resolve().parent / "optuna_study.db"),
    )
    args = parser.parse_args()

    config = load_toml(CONFIG_PATH)
    data_config = config["data"]
    eval_config = config["eval"]

    # Pre-load data dims once — shared across all trials
    probe_dm = ESMEOnlineDataModuleCV(
        data_dir="../../data/",
        subtype=data_config["subtype"],
        n_lines=data_config["n_lines"],
        n_intervals=HPO_N_INTERVALS,
        batch_size=128,
        split_seed=HPO_SPLIT_SEED,
        num_workers=0,
        final_training=True,
    )
    probe_dm.prepare_data()
    data_dims = probe_dm.get_data_dimensions()

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        if PyTorchLightningPruningCallback is not None
        else optuna.pruners.NopPruner()
    )

    storage = f"sqlite:///{args.storage}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
        load_if_exists=True,
    )

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(
        lambda trial: objective(trial, data_config, eval_config, data_dims),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    print(f"\nBest trial:  {study.best_trial.number}")
    print(f"Best C-index: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = Path(__file__).resolve().parent / "best_config.toml"
    _write_best_config(study, out_path)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled"
    main()
