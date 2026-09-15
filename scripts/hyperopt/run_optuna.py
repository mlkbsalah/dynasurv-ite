import argparse
import os
from pathlib import Path

import lightning as L
import optuna
import tomllib
import torch
from lightning.pytorch.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

from CausalSurv.config import (
    ArchConfig,
    DataConfig,
    EvalConfig,
    ModelConfigFile,
    TrainingConfig,
)
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


def _identifiability_kwargs(data_config: DataConfig) -> dict:
    """Cohort-construction controls that MUST match the real training pipeline
    (TrainDynasurvCausal.py). If they diverge, HPO optimises a different cohort:
    a different set of treatment arms and feature dims (so the tuned widths do
    not even fit the production model), and a *random* holdout instead of the
    temporal one — which defeats the whole point of the temporal split, since
    the objective would no longer measure generalisation across policy eras.

    `excluded_x_columns` was missing here until 2026-09, so every trial in
    `optuna_study.db` was scored on a cohort that still contained
    X_onset_to_progression — a post-treatment mediator with corr 1.000 to the
    current line's duration. Those C-indices are leak-inflated and the tuned
    widths were fitted to a feature space that no longer exists; re-run the
    study before treating any value in best_config.json as tuned.

    The propensity-overlap controls are deliberately absent: they gate which
    arms are *recommendable*, which does not enter `average_ci`, and fitting the
    per-line propensity models would cost every trial for nothing.
    """
    return {
        "cohort_start_year": data_config.cohort_start_year,
        "temporal_split_year": data_config.temporal_split_year,
        "add_calendar_feature": data_config.add_calendar_feature,
        "excluded_treatment_arms": _as_list(data_config.excluded_treatment_arms),
        "excluded_x_columns": _as_list(data_config.excluded_x_columns),
        "min_samples_per_treatment": data_config.min_samples_per_treatment,
        "min_events_per_treatment": data_config.min_events_per_treatment,
        "min_followup_samples_per_treatment": (
            data_config.min_followup_samples_per_treatment
        ),
    }


def _as_list(value):
    return None if value is None else list(value)


def _suggest_mlp(trial: optuna.Trial, name: str, max_depth: int = 3) -> list[int]:
    depth = trial.suggest_int(f"{name}_depth", 1, max_depth)
    width = trial.suggest_categorical(f"{name}_width", [32, 64, 128, 256])
    return [width] * depth


def _suggest_configs(trial: optuna.Trial) -> tuple[ArchConfig, TrainingConfig, int]:
    """Sample one point of the search space as the model's own config objects.

    `init_h_*` and `init_p_*` used to be searched here. They were four of the
    21 dimensions and reached nothing -- the projections they sized are
    commented out in the model -- so they inflated the TPE space by ~20% for no
    signal and are gone.
    """
    arch = ArchConfig(
        lstm_hidden_length=trial.suggest_categorical(
            "lstm_hidden_length", [64, 128, 256]
        ),
        lstm_num_layers=trial.suggest_int("lstm_num_layers", 1, 4),
        x_embed_dim=trial.suggest_categorical("x_embed_dim", [32, 64, 128]),
        p_embed_dim=trial.suggest_categorical("p_embed_dim", [8, 16, 32]),
        mlpx_hidden_units=tuple(_suggest_mlp(trial, "mlpx")),
        mlpsa_hidden_units=tuple(_suggest_mlp(trial, "mlpsa")),
        mlpprop_hidden_units=tuple(_suggest_mlp(trial, "mlpprop", max_depth=2)),
        mlpx_dropout=trial.suggest_float("mlpx_dropout", 0.0, 0.4),
        mlpsa_dropout=trial.suggest_float("mlpsa_dropout", 0.0, 0.4),
        mlpprop_dropout=0.0,
        attention=True,
    )
    training = TrainingConfig(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        lr_scheduler_stepsize=trial.suggest_int("lr_scheduler_stepsize", 10, 50),
        lr_scheduler_gamma=trial.suggest_float("lr_scheduler_gamma", 0.1, 0.7),
        lambda_prop_loss=0.0,
        lambda_ipm_mmd=0.0,
        lambda_ipm_emd2=0.0,
    )
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    return arch, training, batch_size


def objective(
    trial: optuna.Trial,
    data_config: DataConfig,
    eval_config: EvalConfig,
    data_dims: dict,
    data_dir: str = "../../data/",
    gradient_clip_val: float = 0.0,
    max_epochs: int = HPO_MAX_EPOCHS,
    patience: int = HPO_PATIENCE,
) -> float:
    """One trial. Objective = ``average_ci``, which under ``final_training=True``
    with a ``temporal_split_year`` is the C-index on the *temporal holdout*
    (latest-entering patients) — the generalisation measure we care about."""
    arch, training, batch_size = _suggest_configs(trial)

    # ---- data ---------------------------------------------------------------
    data_module = ESMEOnlineDataModuleCV(
        data_dir=data_dir,
        subtype=data_config.subtype,
        n_lines=data_config.n_lines,
        n_intervals=HPO_N_INTERVALS,
        batch_size=batch_size,
        split_seed=HPO_SPLIT_SEED,
        final_training=True,
        num_workers=2,
        **_identifiability_kwargs(data_config),
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
        n_lines=data_config.n_lines,
        arch=arch,
        training=training,
        # HPO uses a coarser Brier grid than production: the objective is
        # average_ci, and the integral only has to be cheap and consistent
        # across trials.
        evaluation=EvalConfig(
            horizon_times=eval_config.horizon_times,
            integration_step=6,
            calibration_times=eval_config.calibration_times,
        ),
    )

    # ---- trainer ------------------------------------------------------------
    callbacks: list = [
        EarlyStopping(monitor="val_loss", mode="min", patience=patience),
        PyTorchLightningPruningCallback(trial, monitor="average_ci"),
    ]

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        gradient_clip_val=gradient_clip_val,
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
    """Serialise the winning trial through the same dataclasses the model reads.

    Previously this emitted hand-formatted TOML lines, so the file and the
    constructor could drift without anything noticing -- which is how six
    dropout values ended up advertised in the config and never passed. Going
    through `ModelConfigFile` means a schema change breaks here loudly.

    `batch_size` is now persisted. It is searched but used to be dropped on the
    way out, so a study that won at 64 silently handed production the 128 that
    happened to sit in config.toml.
    """
    p = study.best_params

    def _mlp_units(name: str) -> tuple[int, ...]:
        return tuple([p[f"{name}_width"]] * p[f"{name}_depth"])

    config = ModelConfigFile(
        n_intervals=HPO_N_INTERVALS,
        batch_size=p["batch_size"],
        arch=ArchConfig(
            lstm_hidden_length=p["lstm_hidden_length"],
            lstm_num_layers=p["lstm_num_layers"],
            x_embed_dim=p["x_embed_dim"],
            p_embed_dim=p["p_embed_dim"],
            mlpx_hidden_units=_mlp_units("mlpx"),
            mlpsa_hidden_units=_mlp_units("mlpsa"),
            mlpprop_hidden_units=_mlp_units("mlpprop"),
            mlpx_dropout=round(p["mlpx_dropout"], 4),
            mlpsa_dropout=round(p["mlpsa_dropout"], 4),
            mlpprop_dropout=0.0,
            attention=True,
        ),
        training=TrainingConfig(
            lr=p["lr"],
            weight_decay=p["weight_decay"],
            lr_scheduler_stepsize=p["lr_scheduler_stepsize"],
            lr_scheduler_gamma=round(p["lr_scheduler_gamma"], 4),
            lambda_prop_loss=0.0,
            lambda_ipm_mmd=0.0,
            lambda_ipm_emd2=0.0,
        ),
    )
    config.write_json(out_path)
    print(f"Best config written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--study-name", type=str, default="dynasurv_hpo")
    parser.add_argument("--data-dir", type=str, default="../../data/")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=HPO_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=HPO_PATIENCE)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "best_config.json"),
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=str(Path(__file__).resolve().parent / "optuna_study.db"),
    )
    args = parser.parse_args()

    config = load_toml(Path(args.config))
    data_config = DataConfig.from_dict(config["data"])
    eval_config = EvalConfig.from_dict(config["eval"])

    # Pre-load data dims once — shared across all trials
    probe_dm = ESMEOnlineDataModuleCV(
        data_dir=args.data_dir,
        subtype=data_config.subtype,
        n_lines=data_config.n_lines,
        n_intervals=HPO_N_INTERVALS,
        batch_size=128,
        split_seed=HPO_SPLIT_SEED,
        num_workers=0,
        final_training=True,
        **_identifiability_kwargs(data_config),
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
        lambda trial: objective(
            trial,
            data_config,
            eval_config,
            data_dims,
            data_dir=args.data_dir,
            gradient_clip_val=args.gradient_clip_val,
            max_epochs=args.max_epochs,
            patience=args.patience,
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        gc_after_trial=True,
    )

    print(f"\nBest trial:  {study.best_trial.number}")
    print(f"Best C-index: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    out_path = Path(args.out)
    _write_best_config(study, out_path)


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "disabled"
    main()
