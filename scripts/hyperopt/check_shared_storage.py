"""Small, isolated cross-node NFS journal check; never reads patient data.

The launcher initializes this study once, runs one probe per GPU task, then
verifies every record before touching the real HPO study. Probe files are kept
under studies/<tag>/storage_checks/<job_id>/ for diagnosis, not winner selection.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from collections import Counter
from pathlib import Path

import optuna

STUDY_NAME = "shared_storage_probe"
TRIALS_PER_WORKER = 8


def initialize(storage, token: str):
    study = optuna.create_study(storage=storage, study_name=STUDY_NAME)
    study.set_user_attr("launch_token", token)


def load_probe(storage, token: str):
    study = optuna.load_study(storage=storage, study_name=STUDY_NAME)
    if study.user_attrs.get("launch_token") != token:
        raise RuntimeError("Storage probe belongs to another launch")
    return study


def run_worker(storage, token: str, worker_id: int, workers: int, timeout: float = 90):
    if not 0 <= worker_id < workers:
        raise ValueError("Invalid probe worker ID")
    study = load_probe(storage, token)
    hostname = platform.node()
    study.set_user_attr(f"worker_{worker_id}", hostname)
    # Ensure contention across all tasks, not four unrelated sequential checks.
    deadline = time.monotonic() + timeout
    while not all(f"worker_{rank}" in study.user_attrs for rank in range(workers)):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Storage probe timed out waiting for all workers to see the same journal"
            )
        time.sleep(0.1)

    def objective(trial):
        trial.set_user_attr("worker_id", worker_id)
        trial.set_user_attr("hostname", hostname)
        # Multiple writes exercise the same locking used by production metrics.
        for step in range(3):
            trial.report(float(trial.number), step)
            time.sleep(0.01)
        return float(trial.number)

    study.optimize(objective, n_trials=TRIALS_PER_WORKER)


def verify(storage, token: str, workers: int, nodes: int):
    study = load_probe(storage, token)
    trials = study.get_trials()
    expected = workers * TRIALS_PER_WORKER
    if len(trials) != expected or {trial.number for trial in trials} != set(
        range(expected)
    ):
        raise RuntimeError("Storage probe lost or duplicated trial records")
    counts = Counter()
    hosts = set()
    for trial in trials:
        if (
            trial.state != optuna.trial.TrialState.COMPLETE
            or trial.value != float(trial.number)
            or trial.intermediate_values
            != {step: float(trial.number) for step in range(3)}
        ):
            raise RuntimeError(
                "Storage probe has incomplete or inconsistent trial records"
            )
        rank, hostname = (
            trial.user_attrs.get("worker_id"),
            trial.user_attrs.get("hostname"),
        )
        if not hostname or study.user_attrs.get(f"worker_{rank}") != hostname:
            raise RuntimeError("Storage probe worker identity differs across processes")
        counts[rank] += 1
        hosts.add(hostname)
    if (
        counts != Counter({rank: TRIALS_PER_WORKER for rank in range(workers)})
        or len(hosts) != nodes
    ):
        raise RuntimeError(
            "Storage probe did not run the expected workers on the expected number of nodes"
        )
    print(
        f"Shared NFS journal check passed: {expected} trials, {workers} workers, {nodes} nodes"
    )


def main(argv=None):
    # Keep library functions lightweight for tests. CLI shares the production
    # mount validation and lock factory, rather than reimplementing them.
    from run_optuna import check_shared_launch, make_storage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("init", "worker", "verify"), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--launch-token-file", type=Path, required=True)
    parser.add_argument("--launch-token", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument(
        "--worker-id", type=int, default=int(os.environ.get("SLURM_PROCID", "0"))
    )
    args = parser.parse_args(argv)
    check_shared_launch(args.launch_token_file, args.launch_token)
    if args.mode == "init":
        args.directory.mkdir(parents=True, exist_ok=False)
    storage = make_storage(
        args.directory / "probe.journal",
        nodes=args.nodes,
        shared_journal=True,
        initialize=args.mode == "init",
    )
    if args.mode == "init":
        initialize(storage, args.launch_token)
    elif args.mode == "worker":
        run_worker(storage, args.launch_token, args.worker_id, args.workers)
    else:
        verify(storage, args.launch_token, args.workers, args.nodes)


if __name__ == "__main__":
    main()
