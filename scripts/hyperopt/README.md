# Corrected HPO: two nodes with two H100s each

Run from the repository root. `run_optuna.py` uses HPO protocol **v3**, built on
the corrected **v2 data protocol**. Archived SQLite studies cannot be resumed.

## Cluster launch

The launcher requests **two Slurm nodes with two H100s each**, partition `ai`,
Apptainer, and a CUDA-capable `dynasurv.sif` at the project root. Each node gets
two workers, eight CPU cores (four per worker) and 64 GB host RAM: four workers,
16 cores and 128 GB RAM in total, for at most 24 hours. Override the partition
with `sbatch --partition=...` if necessary; verify the site's `h100` GRES name.

Both nodes must see the **same shared project, data, output directories and
container image**. The launcher fixes `PROJECT_DIR` to
`/home/m-ben-salah/repos/dynasurv-ite`, with data at `PROJECT_DIR/data` and the
image at `PROJECT_DIR/dynasurv.sif`. The supplied screenshot shows
`/home/m-ben-salah` mounted `nfs4,rw` with `vers=4.1`. A read/write mount alone
does not prove directory write permission or cross-node visibility; the cluster
probe checks these before training. The default backend is an **NFS journal**.

The launcher verifies the real journal directory is on read/write NFSv3+ inside
each container. It uses Optuna's `JournalFileOpenLock` with forced lock expiry
disabled, so a slow NFS server cannot cause an active lock to be stolen. This
follows the project's pinned [Optuna 4.7 NFS journal recipe](https://optuna.readthedocs.io/en/v4.7.0/tutorial/20_recipes/011_journal_storage.html)
and [exclusive-create lock API](https://optuna.readthedocs.io/en/v4.7.0/reference/generated/optuna.storages.journal.JournalFileOpenLock.html).
Do not put a SQLite study on NFS. PostgreSQL/MySQL remain optional (below).

Submit from the **host/login shell**, not from an interactive `Apptainer>` shell.
The examples below use the fixed project path:

```bash
cd /home/m-ben-salah/repos/dynasurv-ite

# Inspect without allocating resources or creating output files.
DRY_RUN=1 bash slurm/RunHPO.sh

# Isolated cluster smoke test; its winner is not a production winner.
STUDY_TAG=hpo_v3_smoke \
  N_TRIALS=4 MAX_EPOCHS=2 PATIENCE=1 sbatch slurm/RunHPO.sh

# Production: 100 additional attempts TOTAL, 25 per GPU.
sbatch slurm/RunHPO.sh

# Optional: separate study minimizing validation IBS instead of maximizing CI.
STUDY_TAG=hpo_v3_ibs \
  METRIC=average_ibs N_TRIALS=100 sbatch slurm/RunHPO.sh
```

The launcher uses these fixed host paths regardless of the submission directory
or inherited `PROJECT_DIR`, `DATA_DIR`, or `SIF` environment variables. The
project is mounted at `/workspace`; its `data/` directory is mounted separately
at `/hpo_data` read-only. Outputs remain under the project.

The image needs the pinned Optuna 4.7, other training dependencies, CUDA PyTorch
and `findmnt` (from `util-linux`, now explicit in the Dockerfile). NFS mode does
not require Psycopg/PyMySQL. Existing compatible images can use the bind-mounted
source directly; rebuild/update only if dependencies are missing. `MAX_EPOCHS`,
`PATIENCE`, `THREADS` and `PRECISION` are configurable environment variables.

First, an isolated storage probe writes **32 tiny dummy trials across four
workers on two distinct hostnames** and verifies all trial IDs, writer IDs,
intermediate values and completed records. Workers synchronize before writing;
the probe step has a five-minute time limit. These are not model trials and do
not count toward `N_TRIALS`. Failed probes stop the job before touching the real
study. Probe files are retained separately for diagnosis.

Then one initializer creates/verifies the HPO study. A separate four-task
preflight checks the storage, data/protocol, GPU visibility and a shared per-job
filesystem marker on both nodes before any training starts.
Four independent one-GPU workers then optimize; one exporter follows successful
completion of all workers. Global Slurm task IDs are 0–3; local IDs are 0–1 on
each node. Slurm binds each task to one typed H100;
Lightning uses a single-device environment, not DDP. GPU default is BF16 mixed
precision with FP32 survival loss/metrics. CUDA/BF16 availability and one visible
GPU per worker are checked before fitting; the cluster job cannot fall back to CPU.

Study state is written directly to the shared NFS journal; it is never copied
to a node-local journal for training or merged from independent worker studies.
Trial checkpoints go directly to separate persistent directories. GPU allocation
follows [Slurm's typed GRES and binding rules](https://slurm.schedmd.com/gres.html).
The small probe is a configuration/concurrency smoke test, not a guarantee
against later NFS failures, server outages, corruption or quota exhaustion.

## Outputs and recovery

- `studies/hpo_v3/study.journal`: persistent study `dynasurv_hpo_v3` (default NFS mode).
- `studies/hpo_v3/storage_checks/<job_id>/probe.journal`: isolated probe, never a model winner.
- `studies/hpo_v3/trials/trial_000000/model_config.json`: exact executed settings.
- `studies/hpo_v3/trials/trial_000000/checkpoints/best.ckpt`: best monitored epoch.
- `configs/hpo_v3/best_config.json`: exact winning model/training/batch configuration.
- `configs/hpo_v3/best_config.provenance.json`: winning epoch, metrics, seed,
  data/evaluation/training protocol, software versions and source fingerprint.

Another `STUDY_TAG` replaces `hpo_v3` in both output roots and the default Optuna
study name. An explicit `STUDY_NAME` overrides the latter; always use a unique
study/output pair and do not share a study across independent project copies.
Training, recommendation and semi-synthetic defaults now use the new default
winner. It is intentionally absent until a study completes; use `--model-config` for a different
study's export. Do not substitute a smoke-test or archived winner.

Resubmission with identical code/data/settings adds the requested attempts; it
does not reset the study or use a lifetime trial target. Pruned/failed trials
count toward the budget. Changed objective, precision, training budget, data,
source, worker/node topology or recorded runtime requires a new study/output directory. Worker sampler
seeds differ; the model training seed is fixed across trials.
Do not edit the bind-mounted source/configuration or replace data while a job runs.

A shared study-directory lock prevents overlapping launcher jobs using that
directory. It does not protect against unrelated/manual clients. Following
interruption, first confirm all old workers have stopped. Normal cleanup releases
the lock. If an abrupt kill left `.launcher.lock`, inspect its `job_id` marker
and confirm that job and all its workers have stopped before removing just the
marker and now-empty lock directory. A separately retained `study.journal.lock`
also blocks initialization/export/recovery: **never remove it while any writer
is alive**. After all workers stop, back up and inspect the journal for interrupted
writes, then remove only a confirmed stale lock. The launcher never clears that
lock or repairs/truncates a damaged journal automatically. Then explicitly recover
interrupted trials:

```bash
RECOVER_STALE=1 sbatch slurm/RunHPO.sh
```

This marks **all** existing `RUNNING` trials failed, without resuming their
optimizer; never use it while any old worker is alive. Without opt-in, remaining
running trials block initialization/preflight/export. **Journal mode has no
heartbeat-based trial recovery.** A crashed writer can leave a lock that stalls
other workers until the job is stopped and explicit recovery is performed.

Study state does not depend on copying a journal back from a compute node.
An abrupt failure can still interrupt a journal write/checkpoint. Keep backups
when no writers are active; the launcher does not implement journal repair or
crash-proof backup. Failed/pruned trials are not automatically retried. Any
failed worker prevents winner export. A requeued job with the same job ID is
not automatically resumed: existing lock/probe artifacts require inspection;
prefer a fresh submission after recovery rather than deleting diagnostic files.

## Optional PostgreSQL/MySQL backend

Use `HPO_STORAGE_BACKEND=rdb` if an approved database is available or NFS does
not meet the checks. Supply an existing database reachable from both nodes via
`OPTUNA_STORAGE_URL`. Supported forms are `postgresql+psycopg://USER:PASSWORD@HOST/DB`
and `mysql+pymysql://USER:PASSWORD@HOST/DB`. Bare `postgresql://` and `mysql://`
select the same drivers. Encode special characters in credentials and configure
TLS according to site policy. The launcher does not provision a database.

On a trusted Bash login shell, avoid putting secrets in shell history:

```bash
read -r -s -p "Optuna database URL: " OPTUNA_STORAGE_URL
printf '\n'
export OPTUNA_STORAGE_URL
HPO_STORAGE_BACKEND=rdb sbatch slurm/RunHPO.sh
```

The URL is forwarded through the environment, not printed command arguments;
do not paste it into chat, commit it, enable shell tracing or dump the job's
environment. This mode needs Psycopg/PyMySQL from `requirements.txt` in the image.
Only the initializer requests table creation; workers load the existing schema.
RDB mode skips the NFS probe and stores study state in the database, not a journal.
Switching backends requires a separate study/output pair, not in-place migration.

RDB heartbeats default to 60 seconds with a 300-second grace period (override
`HEARTBEAT_INTERVAL` / `HEARTBEAT_GRACE_PERIOD`). Expired trials are detected when
another optimization trial starts, not by a standalone reaper. Startup/export
still refuse remaining `RUNNING` trials. See the [Optuna RDB heartbeat API](https://optuna.readthedocs.io/en/v4.7.0/reference/generated/optuna.storages.RDBStorage.html).

Both journal and database protocol records include patient split IDs. Restrict
their access like the project data; `umask 077` does not replace checking existing
permissions or site ACLs. Database availability/backups are the administrator's
responsibility.

## Statistical interpretation

Only development validation selects trials; HPO never scores the temporal test
set. Stopping, pruning, checkpointing and objective use one `--metric`:
`average_ci` (maximize, default), `average_ibs`, `val_loss`, or
`val/calib_gap_abs_mean` (minimize). The objective is the best finite epoch, not
the last epoch. Diagnostics use the configured production evaluation grid.

C-index does not guarantee calibration or treatment-effect accuracy. Choose the
objective in advance; inspect IBS, calibration and per-line support for RMST use.
Balancing weights remain zero: this is predictive-model tuning, not proof of
causal identification. The inactive propensity head is fixed, not searched.

The winning validation score is selection-optimistic. Refit a shortlist across
training seeds with a prespecified validation rule, align refit stopping/checkpoint
selection with the intended endpoint, then evaluate the reserved test once.
Production currently stops on calibration while HPO defaults to CI: make this
choice explicit. HPO checkpoints deliberately lack recommendation propensity
models/eligibility; use regular training for recommendation-ready checkpoints.

Parallel TPE arrival order and GPU operations prevent a bitwise-reproducibility
guarantee. Deterministic kernels are preferred with warning-only fallback for
unsupported operations/builds; see [PyTorch's determinism documentation](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

## Local checks

```bash
PYTHONPATH=src python scripts/hyperopt/run_optuna.py --help
PYTHONPATH=src python -m pytest -q tests/test_hpo_and_archive.py
```

For a real four-process database integration check, provide
`OPTUNA_TEST_STORAGE_URL` through the environment and run
`pytest -q tests/test_hpo_and_archive.py -k real_rdb`. Use a dedicated test
database: the test creates and deletes only its uniquely named temporary study,
but initializes Optuna tables if absent. Without this variable, the database
integration test is explicitly skipped. Launcher orchestration is also tested
using local Slurm/Apptainer mocks; those tests are not a real cluster run.

Single-host CLI work may still use `--nodes 1 --storage /path/to/local.journal`
on a local filesystem. Multi-node CLI work requires `--storage-url-env` or an
explicit `--storage /shared/path/study.journal --shared-journal`. Use separate
studies/directories for CPU, MPS and GPU because accelerator and precision are
protocol fields. Local tests exercise the NFS lock implementation with four real
processes on a local filesystem, not across actual NFS clients. H100 execution,
cross-node NFS operation and a live shared database were not tested locally;
run the isolated two-node cluster smoke test before production. No cluster job
is submitted by the local checks.
