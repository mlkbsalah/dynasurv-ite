"""Protocol-v3 HPO: independent one-GPU processes sharing an Optuna study.

The temporal test set is never scored. Selection, early stopping, checkpointing
and pruning use the same development-validation metric. Run --init-only before
parallel workers and --export-only after they exit (see slurm/RunHPO.sh).
Multi-node workers use PostgreSQL/MySQL or an explicitly enabled NFS journal.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import lightning as L
import optuna
import tomllib
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.plugins.environments import LightningEnvironment
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from sqlalchemy.engine import make_url

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
PROTOCOL_VERSION = 3
HPO_MAX_EPOCHS = 100
HPO_PATIENCE = 15
HPO_SPLIT_SEED = 42
HPO_N_INTERVALS = 100
METRICS = {
    "average_ci": "max",
    "average_ibs": "min",
    "val/calib_gap_abs_mean": "min",
    "val_loss": "min",
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
    Saved trial checkpoints deliberately have NO recommendation eligibility.
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


def _suggest_configs(trial) -> tuple[ArchConfig, TrainingConfig, int]:
    arch = ArchConfig(
        lstm_hidden_length=trial.suggest_categorical(
            "lstm_hidden_length", [64, 128, 256]
        ),
        lstm_num_layers=trial.suggest_int("lstm_num_layers", 1, 4),
        x_embed_dim=trial.suggest_categorical("x_embed_dim", [32, 64, 128]),
        p_embed_dim=trial.suggest_categorical("p_embed_dim", [8, 16, 32]),
        mlpx_hidden_units=_suggest_mlp(trial, "mlpx"),
        mlpsa_hidden_units=_suggest_mlp(trial, "mlpsa"),
        # This loss has zero weight: do not tune the inactive propensity head.
        mlpprop_hidden_units=(64,),
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


def resolve_accelerator(requested: str, precision: str) -> tuple[str, str]:
    if requested == "auto":
        requested = (
            "gpu"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "gpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; check allocation/container --nv"
        )
    if precision == "auto":
        precision = "bf16-mixed" if requested == "gpu" else "32-true"
    if (
        requested == "gpu"
        and precision == "bf16-mixed"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("GPU does not support BF16; pass --precision 32-true")
    if requested == "mps" and precision != "32-true":
        raise ValueError("Use 32-true precision on MPS")
    return requested, precision


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
    metric="average_ci",
    artifact_dir: Path | None = None,
    progress=False,
) -> float:
    # Fixed training seed across trials; separate processes isolate their RNGs.
    L.seed_everything(seed, workers=True)
    arch, training, batch_size = _suggest_configs(trial)
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
    )
    tracker = BestValidationMetric(trial, metric)
    callbacks = [
        tracker,
        EarlyStopping(
            monitor=metric,
            mode=METRICS[metric],
            min_delta=0.0,
            patience=patience,
            check_on_train_epoch_end=False,
        ),
    ]
    checkpoint = None
    if artifact_dir is not None:
        trial_dir = artifact_dir / f"trial_{trial.number:06d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        config.write_json(trial_dir / "model_config.json")
        checkpoint = ModelCheckpoint(
            dirpath=trial_dir / "checkpoints",
            filename="best",
            monitor=metric,
            mode=METRICS[metric],
            save_top_k=1,
            save_on_train_epoch_end=False,
        )
        callbacks.append(checkpoint)
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
            enable_checkpointing=checkpoint is not None,
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
        if checkpoint is not None:
            trial.set_user_attr(
                "best_checkpoint",
                str(
                    Path(checkpoint.best_model_path).relative_to(artifact_dir.resolve())
                ),
            )
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


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        [Path(__file__).resolve(), *ROOT.glob("src/CausalSurv/**/*.py")]
    ):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_study_protocol(
    study, manifest, evaluation, settings=None, *, initialize=True
):
    """Reject legacy OR incompatible objectives, including initialized empty studies."""
    if manifest.get("protocol_version") != 2:
        raise ValueError("HPO requires a corrected protocol-v2 data manifest")
    payload = {
        "data_manifest": manifest,
        "evaluation": evaluation.to_dict(),
        "settings": settings or {},
        "code_sha256": code_fingerprint(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    protocol = {
        "version": PROTOCOL_VERSION,
        "fingerprint": fingerprint,
        "selection_split": "validation",
    }
    previous = study.user_attrs.get("data_protocol")
    if (previous is not None and previous != protocol) or (
        previous is None and study.trials
    ):
        raise ValueError(
            "Existing study has a legacy or different data/evaluation/training protocol. "
            "Use a new --study-name and artifact directory."
        )
    if previous is None:
        if not initialize:
            raise ValueError(
                "Initialize the shared study with --init-only before starting workers"
            )
        study.set_user_attr("data_protocol", protocol)
        study.set_user_attr("protocol_details", payload)


def verify_nfs_directory(path: Path):
    """Check the actual container mount, not just a user-provided filesystem label."""
    try:
        result = subprocess.run(
            ["findmnt", "--json", "--target", str(path), "--output", "FSTYPE,OPTIONS"],
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
        )
        mounts = json.loads(result.stdout)["filesystems"]
        if len(mounts) != 1:
            raise ValueError("Expected exactly one mount")
        mount = mounts[0]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        raise RuntimeError(
            "Cannot verify the journal's NFS mount. Install findmnt (util-linux) "
            "in the image and bind the shared directory into the container."
        ) from None
    options = set((mount.get("options") or "").split(","))
    versions = [
        option.removeprefix("vers=") for option in options if option.startswith("vers=")
    ]
    nfs3_or_later = mount.get("fstype") == "nfs4" or (
        mount.get("fstype") == "nfs"
        and any(version.split(".")[0] in ("3", "4") for version in versions)
    )
    if not nfs3_or_later or "rw" not in options or "ro" in options:
        raise ValueError(
            "--shared-journal requires a read/write NFSv3+ mount; "
            "do not use node-local scratch, SQLite, or an unverified filesystem"
        )


def require_unlocked_journal(path: Path):
    if os.path.lexists(str(path) + ".lock"):
        raise RuntimeError(
            "Journal lock exists. Stop all old workers, back up and inspect the journal, "
            "then remove only a confirmed stale .journal.lock before recovery. "
            "Locks are never forcibly stolen."
        )


def make_journal_backend(path: Path, *, shared: bool = False):
    # NFSv3+ provides O_EXCL. Never steal a lock merely because NFS is slow:
    # Optuna's default 30-second forced release could let two writers overlap.
    lock = JournalFileOpenLock(str(path), grace_period=None) if shared else None
    return JournalFileBackend(str(path), lock_obj=lock)


def make_storage(
    path: Path | None = None,
    *,
    url_env: str | None = None,
    nodes: int = 1,
    initialize: bool = True,
    heartbeat_interval: int = 60,
    grace_period: int = 300,
    shared_journal: bool = False,
):
    """Create a checked journal backend or an optional shared database backend.

    NFS journals require explicit opt-in and never forcibly expire locks. They
    have no heartbeat recovery. For RDB, credentials are read only from the
    environment; only the coordinator creates tables. Heartbeats mark abandoned
    RDB trials failed during optimize(), without silently retrying them.
    """
    if nodes < 1:
        raise ValueError("nodes must be positive")
    if url_env is not None and (path is not None or shared_journal):
        raise ValueError("Choose a journal --storage OR --storage-url-env, not both")
    if url_env is not None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", url_env):
            raise ValueError(
                "--storage-url-env must name an environment variable, not contain a URL"
            )
        value = os.environ.get(url_env)
        if not value:
            raise ValueError(
                f"Set {url_env} to a PostgreSQL/MySQL URL reachable from every node"
            )
        try:
            url = make_url(value)
        except Exception:
            raise ValueError(
                "Invalid storage URL; expected PostgreSQL/MySQL connection settings"
            ) from None
        aliases = {"postgresql": "postgresql+psycopg", "mysql": "mysql+pymysql"}
        url = url.set(drivername=aliases.get(url.drivername, url.drivername))
        if url.drivername not in ("postgresql+psycopg", "mysql+pymysql"):
            raise ValueError(
                "Shared storage requires postgresql+psycopg or mysql+pymysql; SQLite is not supported"
            )
        if not url.host or not url.database:
            raise ValueError(
                "Shared database URL must include a host and an existing database"
            )
        if nodes > 1 and url.host.lower() in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                "Multi-node workers need a shared database host, not localhost"
            )
        if heartbeat_interval < 1 or grace_period <= heartbeat_interval:
            raise ValueError(
                "Heartbeat must be positive; grace period must exceed heartbeat interval"
            )
        try:
            return RDBStorage(
                url.render_as_string(hide_password=False),
                engine_kwargs={
                    "pool_pre_ping": True,
                    "pool_recycle": 300,
                    "pool_size": 2,
                    "max_overflow": 2,
                    "connect_args": {"connect_timeout": 10},
                },
                heartbeat_interval=heartbeat_interval,
                grace_period=grace_period,
                skip_table_creation=not initialize,
            )
        except ImportError:
            raise RuntimeError(
                "Missing database driver: install requirements.txt in the HPO environment/image"
            ) from None
        except Exception:
            # Driver/SQLAlchemy errors may contain credentials or full URLs.
            raise RuntimeError(
                "Optuna database connection/schema check failed. Check driver, host, credentials, "
                "TLS settings and the --init-only step; connection details are not logged."
            ) from None
    if nodes > 1 and not shared_journal:
        raise ValueError(
            "Multi-node HPO requires --storage-url-env OR --storage with --shared-journal on NFS"
        )
    if shared_journal and path is None:
        raise ValueError("--shared-journal requires an explicit --storage path on NFS")
    path = (path or ROOT / "studies/hpo_v3/study.journal").expanduser().resolve()
    if path.suffix != ".journal":
        raise ValueError("Use a .journal storage file, not a legacy SQLite database")
    path.parent.mkdir(parents=True, exist_ok=True)
    if shared_journal:
        verify_nfs_directory(path.parent)
        if initialize:
            require_unlocked_journal(path)
        elif not path.is_file():
            raise FileNotFoundError(
                "Shared journal is missing; run --init-only before workers"
            )
    return JournalStorage(make_journal_backend(path, shared=shared_journal))


def check_shared_launch(token_file: Path | None, token: str | None):
    """The coordinator's per-job marker must be visible identically on both nodes."""
    if token_file is None and token is None:
        return
    if token_file is None or token is None:
        raise ValueError("Supply both --launch-token-file and --launch-token")
    if not token_file.is_file() or token_file.read_text().strip() != token:
        raise RuntimeError(
            "Shared project/output directory is not visible or belongs to another launch"
        )


def worker_trials(total: int, workers: int, worker_id: int) -> int:
    if total < 0 or workers < 1 or not 0 <= worker_id < workers:
        raise ValueError("Invalid trial budget or worker index")
    return total // workers + int(worker_id < total % workers)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def _write_best_config(study, out_path: Path):
    if study.user_attrs.get("data_protocol", {}).get("version") != PROTOCOL_VERSION:
        raise ValueError("Cannot export a legacy study")
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
    best = study.best_trial
    if best.user_attrs.get("evaluation_protocol") != PROTOCOL_VERSION:
        raise ValueError("Winning trial has no v3 provenance")
    # Exact executed config: no parameter reconstruction or floating-point rounding.
    config = ModelConfigFile.from_dict(best.user_attrs["model_config"])
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
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="ADDITIONAL attempts across ALL workers",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Must be 1; launch separate worker processes",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--nodes", type=int, default=int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    )
    parser.add_argument(
        "--worker-id", type=int, default=int(os.environ.get("SLURM_PROCID", "0"))
    )
    parser.add_argument("--study-name", default="dynasurv_hpo_v3")
    storage_args = parser.add_mutually_exclusive_group()
    storage_args.add_argument(
        "--storage",
        type=Path,
        help=".journal path (default: studies/hpo_v3/study.journal)",
    )
    storage_args.add_argument(
        "--storage-url-env",
        help="Environment variable containing a PostgreSQL/MySQL URL",
    )
    parser.add_argument(
        "--shared-journal",
        action="store_true",
        help="Allow a shared NFSv3+ journal, using an exclusive-create lock with no forced expiry",
    )
    parser.add_argument("--heartbeat-interval", type=int, default=60)
    parser.add_argument("--heartbeat-grace-period", type=int, default=300)
    parser.add_argument("--launch-token-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--launch-token", help=argparse.SUPPRESS)
    parser.add_argument(
        "--artifact-dir", type=Path, default=ROOT / "studies/hpo_v3/trials"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "configs/hpo_v3/best_config.json"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--data-dir", type=Path, help="Override data_dir; config-relative by default"
    )
    parser.add_argument("--max-epochs", type=int, default=HPO_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=HPO_PATIENCE)
    parser.add_argument("--n-intervals", type=int, default=HPO_N_INTERVALS)
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=HPO_SPLIT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--accelerator", choices=("auto", "gpu", "cpu", "mps"), default="auto"
    )
    parser.add_argument(
        "--precision", choices=("auto", "bf16-mixed", "32-true"), default="auto"
    )
    parser.add_argument("--metric", choices=tuple(METRICS), default="average_ci")
    parser.add_argument("--progress", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--init-only", action="store_true")
    modes.add_argument("--export-only", action="store_true")
    modes.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate initialized study/data/device; do not create trials",
    )
    modes.add_argument(
        "--fail-stale-running",
        action="store_true",
        help="Recovery ONLY after old workers stop; mark RUNNING trials FAIL and exit",
    )
    args = parser.parse_args(argv)
    if args.n_jobs != 1:
        parser.error(
            "Threaded GPU trials are unsafe; use --n-jobs 1 and independent workers"
        )
    if args.nodes < int(os.environ.get("SLURM_JOB_NUM_NODES", "1")):
        parser.error(
            "--nodes cannot understate the Slurm allocation; use RDB or an explicit shared NFS journal"
        )
    budget = worker_trials(args.n_trials, args.workers, args.worker_id)
    if (
        min(args.max_epochs, args.patience, args.n_intervals, args.threads) < 1
        or args.num_workers < 0
    ):
        parser.error(
            "Epochs/patience/intervals/threads must be positive; num-workers nonnegative"
        )
    if args.gradient_clip_val < 0:
        parser.error("gradient-clip-val must be nonnegative")
    check_shared_launch(args.launch_token_file, args.launch_token)
    if (
        args.shared_journal
        and args.storage is not None
        and (args.export_only or args.fail_stale_running)
    ):
        require_unlocked_journal(args.storage.expanduser().resolve())
    storage = make_storage(
        args.storage,
        url_env=args.storage_url_env,
        nodes=args.nodes,
        initialize=args.init_only
        or (
            args.workers == 1
            and not (args.export_only or args.fail_stale_running or args.preflight_only)
        ),
        heartbeat_interval=args.heartbeat_interval,
        grace_period=args.heartbeat_grace_period,
        shared_journal=args.shared_journal,
    )
    if args.export_only or args.fail_stale_running:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        if args.export_only:
            _write_best_config(study, args.out)
        else:
            for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
                print(f"Marked interrupted trial {trial.number} FAIL")
        return
    accelerator, precision = resolve_accelerator(args.accelerator, args.precision)
    if (
        accelerator == "gpu"
        and args.workers > 1
        and not args.init_only
        and torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            "Each worker must see exactly ONE GPU; use srun --gpus-per-task=1 --gpu-bind=single:1"
        )
    torch.set_num_threads(args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_float32_matmul_precision("highest")
    config = load_toml(args.config)
    data_config, evaluation = (
        DataConfig.from_dict(config["data"]),
        EvalConfig.from_dict(config["eval"]),
    )
    data_dir = args.data_dir or args.config.resolve().parent / data_config.data_dir
    dm = HPODataModule(
        data_dir=str(data_dir.resolve()),
        subtype=data_config.subtype,
        n_lines=data_config.n_lines,
        n_intervals=args.n_intervals,
        batch_size=128,
        split_seed=HPO_SPLIT_SEED,
        final_training=True,
        num_workers=args.num_workers,
        **_identifiability_kwargs(data_config),
    )
    dm.prepare_data()
    settings = {
        key: getattr(args, key)
        for key in (
            "metric",
            "max_epochs",
            "patience",
            "n_intervals",
            "gradient_clip_val",
            "seed",
        )
    }
    settings.update(
        accelerator=accelerator,
        precision=precision,
        torch=torch.__version__,
        lightning=L.__version__,
        optuna=optuna.__version__,
        python=platform.python_version(),
        threads=args.threads,
        search_space="predictive-v3",
        sampler="TPE-constant-liar",
        pruner="Median-startup5-warmup10",
        selection="best-validation-epoch",
        deterministic="warn",
        matmul_precision="highest",
        storage_backend="rdb"
        if args.storage_url_env
        else ("nfs-journal-openlock-no-expiry" if args.shared_journal else "journal"),
        nodes=args.nodes,
        workers=args.workers,
    )
    study_kwargs = dict(
        study_name=args.study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(
            seed=args.seed + args.worker_id, constant_liar=True
        ),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    may_initialize = args.init_only or (args.workers == 1 and not args.preflight_only)
    if may_initialize:
        study = optuna.create_study(
            **study_kwargs,
            direction="maximize" if METRICS[args.metric] == "max" else "minimize",
            load_if_exists=True,
        )
    else:
        # Parallel workers and preflight checks must not create missing studies.
        study = optuna.load_study(**study_kwargs)
    expected_direction = "MAXIMIZE" if METRICS[args.metric] == "max" else "MINIMIZE"
    if study.direction.name != expected_direction:
        raise ValueError("Study direction differs from the requested metric")
    require_study_protocol(
        study, dm.data_manifest, evaluation, settings, initialize=may_initialize
    )
    if study.get_trials(states=(optuna.trial.TrialState.RUNNING,)) and (
        args.init_only or args.preflight_only or args.workers == 1
    ):
        raise RuntimeError(
            "RUNNING trials exist. Stop old workers, then use --fail-stale-running if interrupted"
        )
    if args.init_only:
        print(f"Initialized/verified {args.study_name}; no trials run")
        return
    if args.preflight_only:
        print(
            f"Preflight passed for worker {args.worker_id} on {platform.node()}: {accelerator}/{precision}"
        )
        return
    print(
        f"Worker {args.worker_id}/{args.workers}: {budget} attempts, {accelerator}/{precision}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    study.optimize(
        lambda trial: objective(
            trial,
            dm,
            evaluation,
            max_epochs=args.max_epochs,
            patience=args.patience,
            gradient_clip_val=args.gradient_clip_val,
            seed=args.seed,
            accelerator=accelerator,
            precision=precision,
            metric=args.metric,
            artifact_dir=args.artifact_dir,
            progress=args.progress,
        ),
        n_trials=budget,
        n_jobs=1,
        gc_after_trial=True,
        catch=(torch.cuda.OutOfMemoryError, NumericalTrialError),
    )
    if args.workers == 1:
        _write_best_config(study, args.out)


if __name__ == "__main__":
    os.environ.setdefault("WANDB_MODE", "disabled")
    main()
