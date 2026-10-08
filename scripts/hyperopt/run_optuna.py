"""Minimize validation survival loss with single-device Optuna workers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
from pathlib import Path

import lightning as L
import optuna
import tomllib
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.plugins.environments import LightningEnvironment
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

from CausalSurv.config import (
    ArchConfig,
    DataConfig,
    EvalConfig,
    ModelConfigFile,
    TrainingConfig,
)
from CausalSurv.data.datamodule_cv import ESMEOnlineDataModuleCV
from CausalSurv.model import DynaSurvCausalOnline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/config.toml"
PROTOCOL_VERSION = 4
PROTOCOL_NAMESPACE = "hpo_v4_val_loss"
HPO_N_TRIALS = 100
HPO_METRIC = "val_loss"
HPO_MAX_EPOCHS = 100
HPO_PATIENCE = 10
HPO_SPLIT_SEED = 42
HPO_N_INTERVALS = 100
METRICS = {
    HPO_METRIC: "min",
    "average_ci": "max",
    "average_ibs": "min",
    "val/calib_gap_abs_mean": "min",
}


def load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _identifiability_kwargs(data_config: DataConfig) -> dict:
    """Use precisely the production cohort, exclusions and validation split."""
    keys = (
        "cohort_start_year",
        "temporal_split_year",
        "validation_size",
        "validation_seed",
        "add_calendar_feature",
        "excluded_treatment_arms",
        "excluded_x_columns",
        "min_samples_per_treatment",
        "min_events_per_treatment",
        "min_followup_samples_per_treatment",
    )
    return {key: getattr(data_config, key) for key in keys}


class HPODataModule(ESMEOnlineDataModuleCV):
    """Tensorize once per process; HPO does not issue recommendations.

    Train-only support masks are still computed, but fitting a calibrated
    treatment-assignment classifier per trial is unnecessary for factual HPO.
    Refit the selected configuration with TrainDynasurvCausal.py for deployment.
    """

    def prepare_data(self):
        if self.ESMEDataset is None:
            super().prepare_data()

    def _set_training_support(self, dataset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        self.valid_treatments_per_line = self._compute_valid_treatments_per_line(
            self.ESMEDataset.treatment_indices[indices],
            self.ESMEDataset.mask[indices],
            self.min_samples_per_treatment,
        )
        self.recommendable_treatments_per_line = {}
        self.arm_support_summary = {}
        self.propensity_overlap_model = None


def _suggest_mlp(trial, name: str) -> tuple[int, ...]:
    depth = trial.suggest_int(f"{name}_depth", 1, 3)
    width = trial.suggest_categorical(f"{name}_width", [32, 64, 128, 256])
    return (width,) * depth


def suggest_config(
    trial, mlp_normalization="layer"
) -> tuple[ArchConfig, TrainingConfig, int]:
    arch = ArchConfig(
        lstm_hidden_length=trial.suggest_categorical(
            "lstm_hidden_length", [64, 128, 256]
        ),
        lstm_num_layers=trial.suggest_int("lstm_num_layers", 1, 4),
        x_embed_dim=trial.suggest_categorical("x_embed_dim", [32, 64, 128]),
        p_embed_dim=trial.suggest_categorical("p_embed_dim", [8, 16, 32]),
        mlpx_hidden_units=_suggest_mlp(trial, "mlpx"),
        mlpsa_hidden_units=_suggest_mlp(trial, "mlpsa"),
        mlpprop_hidden_units=(64,),
        mlpx_dropout=trial.suggest_float("mlpx_dropout", 0.0, 0.4),
        mlpsa_dropout=trial.suggest_float("mlpsa_dropout", 0.0, 0.4),
        mlpprop_dropout=0.0,
        attention=True,
        mlp_normalization=mlp_normalization,
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
    return arch, training, trial.suggest_categorical("batch_size", [64, 128, 256])


class NumericalTrialError(RuntimeError):
    """A non-finite objective invalidates this trial, not the whole study."""


class BestValidationMetric(Callback):
    """Read pooled metrics AFTER the module's validation-epoch hook."""

    def __init__(self, trial, metric: str):
        self.trial, self.metric, self.mode = trial, metric, METRICS[metric]
        self.best_value = None
        self.best_epoch = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.metric)
        if value is None:
            raise RuntimeError(
                f"Validation did not emit required metric {self.metric!r}"
            )
        value = float(value)
        if not math.isfinite(value):
            self.trial.set_user_attr("failure_reason", f"Non-finite {self.metric}")
            raise NumericalTrialError(f"Non-finite validation {self.metric}: {value}")
        epoch = int(trainer.current_epoch)
        improved = self.best_value is None or (
            value > self.best_value if self.mode == "max" else value < self.best_value
        )
        if improved:
            self.best_value, self.best_epoch = value, epoch
            self.trial.set_user_attr("best_epoch", epoch)
            self.trial.set_user_attr("best_validation_value", value)
            diagnostics = {
                key: float(val)
                for key, val in trainer.callback_metrics.items()
                if key in METRICS and math.isfinite(float(val))
            }
            self.trial.set_user_attr("metrics_at_best_epoch", diagnostics)
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch}: {self.metric}={value}")


def objective(
    trial,
    data_module,
    eval_config,
    *,
    max_epochs=HPO_MAX_EPOCHS,
    patience=HPO_PATIENCE,
    gradient_clip_val=0.0,
    seed=HPO_SPLIT_SEED,
    accelerator="cpu",
    precision="32-true",
    metric=HPO_METRIC,
    progress=False,
    mlp_normalization="layer",
) -> float:
    # Fixed training seed across trials; separate processes isolate their RNGs.
    L.seed_everything(seed, workers=True)
    arch, training, batch_size = suggest_config(trial, mlp_normalization)
    config = ModelConfigFile(data_module.n_intervals, batch_size, arch, training)
    trial.set_user_attr("model_config", config.to_dict())
    trial.set_user_attr("evaluation_protocol", PROTOCOL_VERSION)
    trial.set_user_attr("training_seed", seed)
    data_module.batch_size = batch_size
    data_module.prepare_data()
    dims = data_module.get_data_dimensions()
    model = DynaSurvCausalOnline(
        x_input_dim=dims["x_input_dim"],
        x_static_dim=dims["x_static_dim"],
        p_input_dim=dims["p_input_dim"],
        p_static_dim=dims["p_static_dim"],
        output_length=dims["output_dim"],
        interval_bounds=dims["time_bins"],
        n_treatments=dims["p_input_dim"],
        n_lines=data_module.n_lines,
        arch=arch,
        training=training,
        evaluation=eval_config,
        mlp_normalization=arch.mlp_normalization,
    )
    tracker = BestValidationMetric(trial, metric)
    callbacks = [
        tracker,
        EarlyStopping(
            monitor=HPO_METRIC,
            mode="min",
            min_delta=0.0,
            patience=patience,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = None
    try:
        environment = LightningEnvironment()
        environment.set_global_rank(0)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            # Slurm tasks are Optuna workers, NOT distributed-training ranks.
            plugins=[environment],
            precision=precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=progress,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            gradient_clip_val=gradient_clip_val,
            # Prefer deterministic kernels; older CUDA/PyTorch combinations
            # lack them for some survival operations (e.g. cumsum). Warn there.
            callbacks=callbacks,
            deterministic="warn",
            benchmark=False,
        )
        trainer.fit(model, datamodule=data_module)
        if tracker.best_value is None:
            raise RuntimeError("Training finished without a validation objective")
        return tracker.best_value
    except torch.cuda.OutOfMemoryError:
        trial.set_user_attr("failure_reason", "CUDA out of memory")
        raise
    finally:
        # The datamodule is reused and Lightning attaches its Trainer to it.
        # Break that reference so the previous GPU model is actually released.
        data_module.trainer = None
        model.trainer = None
        if trainer is not None:
            del trainer
        del model
        gc.collect()
        if accelerator == "gpu":
            torch.cuda.empty_cache()


def make_storage(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = JournalFileOpenLock(str(path), grace_period=None)
    return JournalStorage(JournalFileBackend(str(path), lock_obj=lock))


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def _write_best_config(study, out_path: Path, mlp_normalization: str):
    if study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        raise RuntimeError(
            "Stop all workers and resolve interrupted RUNNING trials before exporting"
        )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(trial.value)
    ]
    if not completed:
        raise RuntimeError(
            "No finite completed trials; no winning configuration was written"
        )
    if study.direction.name != "MINIMIZE":
        raise ValueError("Expected a study minimizing validation survival loss")
    details = study.user_attrs.get("protocol_details", {})
    settings = details.get("settings", {})
    if settings.get("metric") != HPO_METRIC:
        raise ValueError("Study metric does not match the validation-loss protocol")
    if settings.get("mlp_normalization") != mlp_normalization:
        raise ValueError("Study normalization does not match the requested mode")
    best = study.best_trial
    if best.user_attrs.get("evaluation_protocol") != PROTOCOL_VERSION:
        raise ValueError("Winning trial has incompatible objective provenance")
    # Exact executed config: no parameter reconstruction or floating-point rounding.
    config = ModelConfigFile.from_dict(best.user_attrs["model_config"])
    if config.arch.mlp_normalization != mlp_normalization:
        raise ValueError("Winning trial normalization does not match the study")
    atomic_json(out_path, config.to_dict())
    atomic_json(
        out_path.with_suffix(".provenance.json"),
        {
            "study_name": study.study_name,
            "trial_number": best.number,
            "value": best.value,
            "direction": study.direction.name,
            "trial_attributes": best.user_attrs,
            "study_attributes": study.user_attrs,
        },
    )
    print(
        f"Exported trial {best.number}, validation objective {best.value:.6f}: {out_path}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--mlp-normalization", choices=("layer", "batch"), default="layer"
    )
    args = parser.parse_args(argv)

    cluster = "SLURM_JOB_ID" in os.environ
    worker_id = int(os.environ.get("SLURM_PROCID", "0"))
    workers = int(os.environ.get("SLURM_NTASKS", "1")) if cluster else 1
    run_name = "cluster" if cluster else "local"
    namespace = (
        PROTOCOL_NAMESPACE
        if args.mlp_normalization == "layer"
        else f"{PROTOCOL_NAMESPACE}_batchnorm"
    )
    study_name = f"dynasurv_{namespace}_{run_name}"
    run_dir = ROOT / "studies" / namespace / run_name
    out_path = ROOT / (
        "configs/hpo_v3/best_config.json"
        if args.mlp_normalization == "layer"
        else "configs/hpo_v4_batchnorm/best_config.json"
    )
    storage = make_storage(run_dir / "study.journal")
    if args.export_only:
        study = optuna.load_study(study_name=study_name, storage=storage)
        _write_best_config(study, out_path, args.mlp_normalization)
        return

    accelerator = (
        "gpu"
        if cluster or torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    precision = "bf16-mixed" if accelerator == "gpu" else "32-true"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_float32_matmul_precision("highest")
    config = load_toml(CONFIG_PATH)
    data_config = DataConfig.from_dict(config["data"])
    evaluation = EvalConfig.from_dict(config["eval"])
    dm = HPODataModule(
        data_dir=str((CONFIG_PATH.parent / data_config.data_dir).resolve()),
        subtype=data_config.subtype,
        n_lines=data_config.n_lines,
        n_intervals=HPO_N_INTERVALS,
        batch_size=128,
        split_seed=HPO_SPLIT_SEED,
        final_training=True,
        num_workers=0,
        **_identifiability_kwargs(data_config),
    )
    dm.prepare_data()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=HPO_SPLIT_SEED + worker_id, constant_liar=True
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    if worker_id == 0 and "protocol_details" not in study.user_attrs:
        study.set_user_attr(
            "protocol_details",
            {
                "data_manifest": dm.data_manifest,
                "evaluation": evaluation.to_dict(),
                "settings": {
                    "metric": HPO_METRIC,
                    "objective": "minimum validation survival loss over completed epochs",
                    "max_epochs": HPO_MAX_EPOCHS,
                    "patience": HPO_PATIENCE,
                    "stopping_metric": HPO_METRIC,
                    "seed": HPO_SPLIT_SEED,
                    "accelerator": accelerator,
                    "precision": precision,
                    "mlp_normalization": args.mlp_normalization,
                },
            },
        )
    budget = HPO_N_TRIALS // workers + int(worker_id < HPO_N_TRIALS % workers)
    print(f"Worker {worker_id}/{workers}: {budget} trials, {accelerator}/{precision}")
    study.optimize(
        lambda trial: objective(
            trial,
            dm,
            evaluation,
            accelerator=accelerator,
            precision=precision,
            metric=HPO_METRIC,
            mlp_normalization=args.mlp_normalization,
        ),
        n_trials=budget,
        n_jobs=1,
        gc_after_trial=True,
        catch=(torch.cuda.OutOfMemoryError, NumericalTrialError),
    )
    if not cluster:
        _write_best_config(study, out_path, args.mlp_normalization)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")
    main()
