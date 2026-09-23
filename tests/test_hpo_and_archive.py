import importlib.util
import json
import math
import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch
from test_p1_regressions import frames

from CausalSurv.config import ArchConfig, EvalConfig, ModelConfigFile, TrainingConfig
from CausalSurv.metrics.survival_loss import NLLogisticHazard

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hpo = load_script("hpo_test", "scripts/hyperopt/run_optuna.py")
storage_probe = load_script(
    "storage_probe_test", "scripts/hyperopt/check_shared_storage.py"
)
archiver = load_script(
    "archive_test", "scripts/maintenance/archive_legacy_artifacts.py"
)


def model_config():
    return ModelConfigFile(
        6,
        8,
        ArchConfig(8, 1, 4, 4, (8,), (8,), (8,), 0.123456789, 0.234567891, 0.0, False),
        TrainingConfig(0.000123456789, 0.001, 10, 0.345678912),
    )


@pytest.mark.parametrize(
    "metric,values,best",
    [("average_ci", [0.6, 0.8, 0.7], 0.8), ("average_ibs", [0.3, 0.1, 0.2], 0.1)],
)
def test_objective_tracker_keeps_best_epoch_not_last(metric, values, best):
    study = optuna.create_study(
        direction="maximize" if hpo.METRICS[metric] == "max" else "minimize"
    )
    trial = study.ask()
    tracker = hpo.BestValidationMetric(trial, metric)
    trainer = SimpleNamespace(
        sanity_checking=True, current_epoch=0, callback_metrics={}
    )
    tracker.on_validation_end(trainer, None)
    assert tracker.best_value is None
    trainer.sanity_checking = False
    for epoch, value in enumerate(values):
        trainer.current_epoch, trainer.callback_metrics = epoch, {metric: value}
        tracker.on_validation_end(trainer, None)
    assert tracker.best_epoch == 1
    assert tracker.best_value == best
    assert trial.user_attrs["metrics_at_best_epoch"][metric] == best
    trainer.callback_metrics = {metric: float("nan")}
    with pytest.raises(hpo.NumericalTrialError):
        tracker.on_validation_end(trainer, None)
    trainer.callback_metrics = {}
    with pytest.raises(RuntimeError, match="did not emit"):
        tracker.on_validation_end(trainer, None)


def test_protocol_rejects_changed_training_settings_even_without_trials():
    study = optuna.create_study()
    manifest, evaluation = {"protocol_version": 2}, EvalConfig((3.0, 3.0))
    with pytest.raises(ValueError, match="Initialize"):
        hpo.require_study_protocol(study, manifest, evaluation, initialize=False)
    hpo.require_study_protocol(study, manifest, evaluation, {"precision": "bf16-mixed"})
    with pytest.raises(ValueError, match="different"):
        hpo.require_study_protocol(
            study, manifest, evaluation, {"precision": "32-true"}
        )


def test_winner_export_exact_roundtrip_and_no_winner_guard(tmp_path):
    study = optuna.create_study(direction="maximize")
    hpo.require_study_protocol(study, {"protocol_version": 2}, EvalConfig((3.0, 3.0)))
    target = tmp_path / "nested/best_config.json"
    with pytest.raises(RuntimeError, match="No finite completed"):
        hpo._write_best_config(study, target)
    assert not target.exists()
    config = model_config()
    study.add_trial(
        optuna.trial.create_trial(
            value=0.8,
            user_attrs={
                "model_config": config.to_dict(),
                "evaluation_protocol": 3,
                "best_epoch": 2,
            },
        )
    )
    hpo._write_best_config(study, target)
    assert ModelConfigFile.from_json(target) == config
    assert (
        json.loads(target.with_suffix(".provenance.json").read_text())["trial_number"]
        == 0
    )


def test_budget_is_total_and_no_inactive_dimensions():
    assert [hpo.worker_trials(101, 4, i) for i in range(4)] == [26, 25, 25, 25]
    assert [hpo.worker_trials(2, 4, i) for i in range(4)] == [1, 1, 0, 0]
    with pytest.raises(ValueError):
        hpo.worker_trials(10, 4, 4)
    trial = optuna.create_study().ask()
    arch, training, batch = hpo._suggest_configs(trial)
    assert not any(key.startswith(("mlpprop", "init_")) for key in trial.params)
    assert training.lambda_prop_loss == 0 and arch.mlpprop_hidden_units == (64,)


def test_cuda_precedes_mps_and_cpu_and_threaded_trials_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert hpo.resolve_accelerator("auto", "auto") == ("gpu", "bf16-mixed")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        hpo.resolve_accelerator("gpu", "auto")
    with pytest.raises(SystemExit):
        hpo.main(["--n-jobs", "4"])


@pytest.mark.parametrize("metric", list(hpo.METRICS))
def test_cli_initializes_each_metric_with_optuna_direction(
    tmp_path, monkeypatch, metric
):
    monkeypatch.setattr(
        hpo,
        "HPODataModule",
        lambda **kwargs: SimpleNamespace(
            prepare_data=lambda: None, data_manifest={"protocol_version": 2}
        ),
    )
    path = tmp_path / "study.journal"
    hpo.main(
        [
            "--init-only",
            "--accelerator",
            "cpu",
            "--precision",
            "32-true",
            "--metric",
            metric,
            "--storage",
            str(path),
            "--threads",
            "1",
        ]
    )
    study = optuna.load_study(
        study_name="dynasurv_hpo_v3", storage=hpo.make_storage(path)
    )
    assert study.direction.name == (
        "MAXIMIZE" if hpo.METRICS[metric] == "max" else "MINIMIZE"
    )


def test_bfloat16_loss_is_float32_finite_and_differentiable():
    logits = torch.randn(16, 6, dtype=torch.bfloat16, requires_grad=True)
    loss = NLLogisticHazard()(logits, torch.arange(16) % 6, torch.arange(16) % 2)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "accelerator,precision",
    [
        ("cpu", "32-true"),
        ("cpu", "bf16-mixed"),
        pytest.param(
            "gpu",
            "bf16-mixed",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="No CUDA device in local test environment",
            ),
        ),
    ],
)
def test_actual_fit_caches_data_saves_winner_and_is_not_slurm_ddp(
    tmp_path, monkeypatch, accelerator, precision
):
    torch.set_num_threads(1)
    dynamic, static = frames()
    loads = []
    dm = hpo.HPODataModule(
        data_dir="unused",
        subtype="HR+HER2-",
        n_lines=2,
        n_intervals=6,
        batch_size=8,
        split_seed=42,
        final_training=True,
        temporal_split_year=2021,
        num_workers=0,
        min_samples_per_treatment=1,
    )

    def load_data():
        loads.append(1)
        return dynamic.copy(), static.copy()

    dm._load_data = load_data
    dm.prepare_data()
    config = model_config()
    monkeypatch.setattr(
        hpo,
        "_suggest_configs",
        lambda trial: (config.arch, config.training, config.batch_size),
    )
    # A Slurm task with global rank 2 must still train and save as a single GPU/process.
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "2")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    monkeypatch.setenv("SLURM_PROCID", "2")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_NODEID", "1")
    monkeypatch.setenv("SLURM_JOB_NAME", "DynaSurvHPO4")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    study = optuna.create_study(direction="maximize")
    value = hpo.objective(
        study.ask(),
        dm,
        EvalConfig((3.0, 3.0), integration_step=3),
        max_epochs=2,
        patience=2,
        accelerator=accelerator,
        precision=precision,
        artifact_dir=tmp_path,
    )
    assert math.isfinite(value)
    assert len(loads) == 1
    assert dm.propensity_overlap_model is None
    assert dm.recommendable_treatments_per_line == {}
    checkpoint = torch.load(
        tmp_path / "trial_000000/checkpoints/best.ckpt",
        weights_only=False,
        map_location="cpu",
    )
    assert checkpoint["data_manifest"] == dm.data_manifest
    assert checkpoint["epoch"] == study.trials[0].user_attrs["best_epoch"]
    assert set(dm.train_dataset.indices).isdisjoint(dm.val_dataset.indices)
    assert set(dm.val_dataset.indices).isdisjoint(dm.test_dataset.indices)
    assert dm.trainer is None


def test_four_processes_share_journal_without_lost_trials(tmp_path):
    path = tmp_path / "shared.journal"
    optuna.create_study(study_name="parallel", storage=hpo.make_storage(path))
    code = """
import sys, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
study = optuna.load_study(study_name='parallel', storage=JournalStorage(JournalFileBackend(sys.argv[1])))
study.optimize(lambda trial: float(trial.number), n_trials=3)
"""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(4)
    ]
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=60)
        assert worker.returncode == 0, stderr.decode()
    study = optuna.load_study(study_name="parallel", storage=hpo.make_storage(path))
    assert len(study.trials) == 12
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert len({t.number for t in study.trials}) == 12
    with pytest.raises(ValueError, match="journal"):
        hpo.make_storage(tmp_path / "legacy.db")


def test_archive_moves_only_legacy_runs_preserves_links_and_checksums(tmp_path):
    old = tmp_path / "models/subtype/lines/old/checkpoints/old.ckpt"
    modern = tmp_path / "models/subtype/lines/new/checkpoints/new.ckpt"
    for path in (old, modern):
        path.parent.mkdir(parents=True)
    torch.save({"state_dict": {"weight": torch.ones(2)}}, old)
    torch.save(
        {
            "state_dict": {"init_h.weight": torch.ones(2)},
            "data_manifest": {"protocol_version": 2},
        },
        modern,
    )
    log = old.parent.parent / "log.txt"
    log.write_text("old log\n")
    (old.parent.parent / "latest.log").symlink_to("log.txt")
    destination = tmp_path / "archives/old"
    summary = archiver.archive(tmp_path, destination)
    assert summary["checkpoints"] == 1 and old.exists() and not destination.exists()
    expected_hash = archiver.sha256(old)
    archiver.archive(tmp_path, destination, apply=True)
    archived = destination / "artifacts" / old.relative_to(tmp_path)
    assert archived.exists() and not old.exists() and modern.exists()
    assert archiver.sha256(archived) == expected_hash
    assert (archived.parent.parent / "latest.log").is_symlink()
    assert json.loads((destination / "manifest.json").read_text())["verified"]
    with pytest.raises(FileExistsError):
        archiver.archive(tmp_path, destination, apply=True)
    with pytest.raises(ValueError, match="directly inside"):
        archiver.archive(tmp_path, tmp_path / "wrong")


def test_archive_append_and_database_protocol_protection(tmp_path):
    db = tmp_path / "scripts/hyperopt/optuna_study.db"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE studies (study_id INTEGER)")
        connection.execute("CREATE TABLE trials (trial_id INTEGER)")
        connection.execute("CREATE TABLE study_user_attributes (key TEXT)")
        connection.execute("INSERT INTO study_user_attributes VALUES ('data_protocol')")
    assert not archiver.legacy_study(db)
    with sqlite3.connect(db) as connection:
        connection.execute("DELETE FROM study_user_attributes")
    assert archiver.legacy_study(db)
    wal = Path(str(db) + "-wal")
    wal.touch()
    with pytest.raises(RuntimeError, match="may be open"):
        archiver.legacy_study(db)
    wal.unlink()
    destination = tmp_path / "archives/old"
    archiver.archive(tmp_path, destination, apply=True)
    model = tmp_path / "models/old/checkpoints/old.ckpt"
    model.parent.mkdir(parents=True)
    torch.save({"state_dict": {}}, model)
    archiver.archive(tmp_path, destination, apply=True, append=True)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["summary"]["checkpoints"] == 1
    assert manifest["summary"]["study_databases"] == 1
    assert manifest["verified"] and len(manifest["files"]) == 2


@pytest.mark.parametrize("backend", ["nfs-journal", "rdb"])
def test_slurm_launcher_dry_run_uses_four_typed_gpus(backend):
    result = subprocess.run(
        ["bash", str(ROOT / "slurm/RunHPO.sh")],
        env=dict(
            os.environ,
            DRY_RUN="1",
            N_TRIALS="101",
            HPO_STORAGE_BACKEND=backend,
            OPTUNA_STORAGE_URL="postgresql+psycopg://user:SENTINEL_SECRET@db/optuna",
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--gpus-per-task=h100:1" in result.stdout
    assert "--workers 4" in result.stdout and "--n-trials 101" in result.stdout
    assert "--init-only" in result.stdout and "--export-only" in result.stdout
    assert "--nodes=2" in result.stdout and "--ntasks-per-node=2" in result.stdout
    assert "--preflight-only" in result.stdout
    if backend == "rdb":
        assert "--storage-url-env OPTUNA_STORAGE_URL" in result.stdout
        assert "check_shared_storage.py" not in result.stdout
    else:
        assert "--shared-journal" in result.stdout
        assert "--storage-url-env" not in result.stdout
        assert "check_shared_storage.py" in result.stdout
        assert "--time=00:05:00" in result.stdout
    assert "/hpo_data:ro" in result.stdout and "--data-dir /hpo_data" in result.stdout
    assert "SENTINEL_SECRET" not in result.stdout + result.stderr
    script = (ROOT / "slurm/RunHPO.sh").read_text()
    assert "#SBATCH --nodes=2" in script and "#SBATCH --gres=gpu:h100:2" in script
    assert "/hpo_local/" not in script


@pytest.mark.parametrize(
    "driver,expected",
    [("postgresql", "postgresql+psycopg"), ("mysql", "mysql+pymysql")],
)
def test_rdb_storage_uses_shared_backend_without_logging_secrets(
    monkeypatch, capsys, driver, expected
):
    captured = {}

    def create_storage(url, **kwargs):
        captured.update(url=url, **kwargs)
        return captured

    monkeypatch.setattr(hpo, "RDBStorage", create_storage)
    monkeypatch.setenv(
        "TEST_OPTUNA_URL", f"{driver}://user:SENTINEL_SECRET@shared-db/studies"
    )
    storage = hpo.make_storage(url_env="TEST_OPTUNA_URL", nodes=2, initialize=False)
    assert storage["url"].startswith(expected + "://")
    assert storage["skip_table_creation"] is True
    assert storage["heartbeat_interval"] == 60 and storage["grace_period"] == 300
    assert storage["engine_kwargs"]["pool_pre_ping"] is True
    assert "SENTINEL_SECRET" not in str(capsys.readouterr())
    hpo.make_storage(url_env="TEST_OPTUNA_URL", nodes=2, initialize=True)
    assert captured["skip_table_creation"] is False


@pytest.mark.parametrize(
    "url",
    [
        "sqlite:///legacy.db",
        "not-a-url-SENTINEL_SECRET",
        "postgresql://user:SENTINEL_SECRET@localhost/db",
        "mysql://user:SENTINEL_SECRET@127.0.0.1/db",
        "postgresql://user:SENTINEL_SECRET@db",
    ],
)
def test_multinode_refuses_unsafe_urls_without_echoing_them(monkeypatch, url):
    monkeypatch.setenv("TEST_OPTUNA_URL", url)
    with pytest.raises(ValueError) as error:
        hpo.make_storage(url_env="TEST_OPTUNA_URL", nodes=2)
    assert "SENTINEL_SECRET" not in str(error.value)


def test_multinode_refuses_implicit_journal_and_missing_url_before_training(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="Multi-node"):
        hpo.make_storage(tmp_path / "study.journal", nodes=2)
    assert not (tmp_path / "study.journal").exists()
    monkeypatch.delenv("TEST_OPTUNA_URL", raising=False)
    with pytest.raises(ValueError, match="Set TEST_OPTUNA_URL"):
        hpo.make_storage(url_env="TEST_OPTUNA_URL", nodes=2)
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    with pytest.raises(SystemExit):
        hpo.main(["--nodes", "1", "--init-only"])
    with pytest.raises(ValueError, match="Multi-node"):
        hpo.main(["--init-only", "--storage", str(tmp_path / "study.journal")])


@pytest.mark.parametrize(
    "fstype,options,allowed",
    [
        ("nfs4", "rw,vers=4.1,nosuid,nodev", True),
        ("nfs", "rw,vers=3", True),
        ("nfs", "rw,vers=4.2", True),
        ("nfs4", "ro,vers=4.1", False),
        ("nfs", "rw,vers=2", False),
        ("nfs", "rw", False),
        ("ext4", "rw", False),
        ("overlay", "rw", False),
    ],
)
def test_shared_journal_requires_verified_writable_nfs(
    monkeypatch, tmp_path, fstype, options, allowed
):
    def findmnt(command, **kwargs):
        assert command[:3] == ["findmnt", "--json", "--target"]
        assert kwargs["timeout"] == 15 and kwargs["check"]
        return SimpleNamespace(
            stdout=json.dumps({"filesystems": [{"fstype": fstype, "options": options}]})
        )

    monkeypatch.setattr(hpo.subprocess, "run", findmnt)
    if allowed:
        hpo.verify_nfs_directory(tmp_path)
    else:
        with pytest.raises(ValueError, match="read/write NFSv3"):
            hpo.verify_nfs_directory(tmp_path)


def test_nfs_journal_lock_never_expires_and_startup_refuses_stale_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hpo, "verify_nfs_directory", lambda path: None)
    path = tmp_path / "study.journal"
    backend = hpo.make_journal_backend(path, shared=True)
    assert isinstance(backend._lock, hpo.JournalFileOpenLock)
    assert backend._lock.grace_period is None
    study = optuna.create_study(
        storage=hpo.make_storage(path, nodes=2, shared_journal=True)
    )
    assert study.trials == []
    lock = Path(str(path) + ".lock")
    lock.touch()
    with pytest.raises(RuntimeError, match="never forcibly stolen"):
        hpo.make_storage(path, nodes=2, shared_journal=True)
    assert lock.exists()
    with pytest.raises(ValueError, match="explicit"):
        hpo.make_storage(nodes=2, shared_journal=True)
    with pytest.raises(ValueError, match="not both"):
        hpo.make_storage(url_env="TEST_OPTUNA_URL", shared_journal=True)
    with pytest.raises(FileNotFoundError, match="init-only"):
        hpo.make_storage(
            tmp_path / "missing.journal", nodes=2, shared_journal=True, initialize=False
        )


def test_missing_findmnt_fails_closed(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("findmnt")

    monkeypatch.setattr(hpo.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="util-linux"):
        hpo.verify_nfs_directory(tmp_path)


def test_nfs_probe_four_processes_use_exclusive_create_lock(tmp_path):
    """Real concurrency on a local filesystem; NOT evidence of real NFS behavior."""
    path = tmp_path / "probe.journal"
    storage = optuna.storages.JournalStorage(
        hpo.make_journal_backend(path, shared=True)
    )
    storage_probe.initialize(storage, "test-token")
    code = """
import importlib.util, optuna, sys
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
spec = importlib.util.spec_from_file_location('probe', sys.argv[1])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
storage = optuna.storages.JournalStorage(JournalFileBackend(
    sys.argv[2], lock_obj=JournalFileOpenLock(sys.argv[2], grace_period=None)))
probe.run_worker(storage, 'test-token', int(sys.argv[3]), 4, timeout=20)
"""
    workers = []
    try:
        workers = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(ROOT / "scripts/hyperopt/check_shared_storage.py"),
                    str(path),
                    str(rank),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for rank in range(4)
        ]
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=40)
            assert worker.returncode == 0, stderr.decode()
        storage_probe.verify(storage, "test-token", workers=4, nodes=1)
        with pytest.raises(RuntimeError, match="number of nodes"):
            storage_probe.verify(storage, "test-token", workers=4, nodes=2)
        with pytest.raises(RuntimeError, match="another launch"):
            storage_probe.verify(storage, "wrong-token", workers=4, nodes=1)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)


def test_probe_refuses_missing_workers_and_incomplete_trials():
    storage = optuna.storages.InMemoryStorage()
    storage_probe.initialize(storage, "test")
    with pytest.raises(RuntimeError, match="waiting for all workers"):
        storage_probe.run_worker(storage, "test", 0, 4, timeout=0)
    with pytest.raises(RuntimeError, match="lost or duplicated"):
        storage_probe.verify(storage, "test", 4, 2)


def test_database_connection_failure_does_not_expose_url(monkeypatch):
    secret = "postgresql+psycopg://user:SENTINEL_SECRET@db/studies"
    monkeypatch.setenv("TEST_OPTUNA_URL", secret)

    def fail(*args, **kwargs):
        raise RuntimeError(f"Failed connection to {secret}")

    monkeypatch.setattr(hpo, "RDBStorage", fail)
    with pytest.raises(RuntimeError, match="connection/schema check failed") as error:
        hpo.make_storage(url_env="TEST_OPTUNA_URL", nodes=2)
    assert "SENTINEL_SECRET" not in str(error.value)
    assert error.value.__suppress_context__


def test_shared_run_marker_must_match_current_launch(tmp_path):
    marker = tmp_path / "job_id"
    with pytest.raises(RuntimeError, match="not visible"):
        hpo.check_shared_launch(marker, "1234")
    marker.write_text("1234\n")
    hpo.check_shared_launch(marker, "1234")
    with pytest.raises(RuntimeError, match="another launch"):
        hpo.check_shared_launch(marker, "5678")
    with pytest.raises(ValueError, match="both"):
        hpo.check_shared_launch(marker, None)


def test_preflight_checks_protocol_without_creating_a_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hpo,
        "HPODataModule",
        lambda **kwargs: SimpleNamespace(
            prepare_data=lambda: None, data_manifest={"protocol_version": 2}
        ),
    )
    path = tmp_path / "preflight.journal"
    common = [
        "--accelerator",
        "cpu",
        "--precision",
        "32-true",
        "--threads",
        "1",
        "--storage",
        str(path),
        "--workers",
        "4",
        "--worker-id",
        "0",
    ]
    hpo.main(common + ["--init-only"])
    hpo.main(common + ["--preflight-only"])
    study = optuna.load_study(
        study_name="dynasurv_hpo_v3", storage=hpo.make_storage(path)
    )
    assert study.trials == []
    with pytest.raises(KeyError):
        hpo.main(common + ["--preflight-only", "--study-name", "missing"])
    assert (
        len(optuna.study.get_all_study_summaries(storage=hpo.make_storage(path))) == 1
    )


@pytest.mark.parametrize(
    "backend,failure",
    [
        ("rdb", ""),
        ("rdb", "preflight"),
        ("rdb", "worker"),
        ("nfs-journal", ""),
        ("nfs-journal", "preflight"),
        ("nfs-journal", "worker"),
        ("nfs-journal", "probe_init"),
        ("nfs-journal", "probe_worker"),
        ("nfs-journal", "probe_verify"),
    ],
)
def test_two_node_launcher_flow_and_failure_cleanup(tmp_path, backend, failure):
    """Exercise real shell orchestration with mock Slurm/Apptainer commands."""
    project = tmp_path / "project"
    (project / "src/CausalSurv").mkdir(parents=True)
    data_dir = tmp_path / "separate shared data"
    data_dir.mkdir()
    (project / "dynasurv.sif").touch()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    apptainer = binaries / "apptainer"
    apptainer.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, pathlib, sys
args = sys.argv[1:]
probe = any(arg.endswith('/check_shared_storage.py') for arg in args)
phase = ('probe_' + args[args.index('--mode') + 1] if probe else
         'init' if '--init-only' in args else 'preflight' if '--preflight-only' in args
         else 'export' if '--export-only' in args else 'worker')
rank = int(os.environ.get('SLURM_PROCID', '0'))
if os.environ['HPO_STORAGE_BACKEND'] == 'rdb':
    assert os.environ['APPTAINERENV_OPTUNA_STORAGE_URL'] == os.environ['OPTUNA_STORAGE_URL']
else:
    assert '--storage-url-env' not in args
    assert probe or '--shared-journal' in args
assert os.environ['DATA_DIR'] + ':/hpo_data:ro' in args
marker = args[args.index('--launch-token-file') + 1].replace('/workspace/', os.environ['PROJECT_DIR'] + '/', 1)
assert pathlib.Path(marker).read_text().strip() == os.environ['SLURM_JOB_ID']
with open(os.environ['MOCK_LOG'], 'a') as stream:
    stream.write(json.dumps({'phase': phase, 'rank': rank}) + '\\n')
if phase == os.environ['MOCK_FAIL'] and (rank == 2 or phase in ('probe_init', 'probe_verify')):
    sys.exit(1)
"""
    )
    srun = binaries / "srun"
    srun.write_text(
        f"#!{sys.executable}\n"
        + """
import os, subprocess, sys
args = sys.argv[1:]
assert '--nodes=2' in args and '--ntasks-per-node=2' in args and '--ntasks=4' in args
command = args[args.index('apptainer'):]
for rank in range(4):
    env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_LOCALID=str(rank % 2), SLURM_NODEID=str(rank // 2))
    result = subprocess.run(command, env=env)
    if result.returncode:
        sys.exit(result.returncode)
"""
    )
    apptainer.chmod(0o700)
    srun.chmod(0o700)
    log = tmp_path / "calls.jsonl"
    env = dict(
        os.environ,
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        PROJECT_DIR=str(project),
        DATA_DIR=str(data_dir),
        DRY_RUN="0",
        N_TRIALS="4",
        HPO_STORAGE_BACKEND=backend,
        SLURM_JOB_ID="1234",
        SLURM_NTASKS="4",
        SLURM_JOB_NUM_NODES="2",
        SLURM_CPUS_PER_TASK="4",
        STUDY_TAG="hpo_v3",
        RECOVER_STALE="0",
        OPTUNA_STORAGE_URL="postgresql+psycopg://user:SENTINEL_SECRET@db/studies",
        MOCK_LOG=str(log),
        MOCK_FAIL=failure,
    )
    result = subprocess.run(
        ["bash", str(ROOT / "slurm/RunHPO.sh")],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    phases = [item["phase"] for item in calls]
    assert phases[0] == ("probe_init" if backend == "nfs-journal" else "init")
    assert not (project / "studies/hpo_v3/.launcher.lock").exists()
    assert "SENTINEL_SECRET" not in result.stdout + result.stderr
    if failure:
        assert result.returncode != 0 and "export" not in phases
        if failure == "preflight":
            assert "worker" not in phases
        if failure.startswith("probe_"):
            assert "init" not in phases and "worker" not in phases
    else:
        assert result.returncode == 0, result.stderr
        probe_phases = (
            ["probe_init"] + ["probe_worker"] * 4 + ["probe_verify"]
            if backend == "nfs-journal"
            else []
        )
        assert phases == probe_phases + ["init"] + ["preflight"] * 4 + [
            "worker"
        ] * 4 + ["export"]
        assert [item["rank"] for item in calls if item["phase"] == "worker"] == [
            0,
            1,
            2,
            3,
        ]


@pytest.mark.skipif(
    not os.environ.get("OPTUNA_TEST_STORAGE_URL"),
    reason="No dedicated PostgreSQL/MySQL test database configured",
)
def test_four_processes_share_real_rdb():
    """Opt-in integration: creates/deletes only a unique temporary test study."""
    storage = hpo.make_storage(url_env="OPTUNA_TEST_STORAGE_URL")
    name = "test_dynasurv_" + uuid.uuid4().hex
    optuna.create_study(study_name=name, storage=storage)
    code = """
import importlib.util, optuna, sys
spec = importlib.util.spec_from_file_location('hpo', sys.argv[1])
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)
storage = hpo.make_storage(url_env='OPTUNA_TEST_STORAGE_URL', initialize=False)
study = optuna.load_study(study_name=sys.argv[2], storage=storage)
study.optimize(lambda trial: float(trial.number), n_trials=3)
"""
    workers = []
    try:
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        workers = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(ROOT / "scripts/hyperopt/run_optuna.py"),
                    name,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(4)
        ]
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=60)
            assert worker.returncode == 0, stderr.decode()
        study = optuna.load_study(study_name=name, storage=storage)
        assert len(study.trials) == 12
        assert all(
            trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
        )
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
        optuna.delete_study(study_name=name, storage=storage)
