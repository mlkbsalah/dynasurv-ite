# Legacy cleanup and H100 HPO review — 23 September 2026

## Result

Historical artifacts are quarantined, active defaults no longer select legacy
winning configs, and `scripts/hyperopt/run_optuna.py` has been corrected and
adapted to four independent H100 workers. Following the topology clarification,
the launcher now targets **two nodes with two H100s each** and defaults to a
shared **NFS journal**. The supplied screenshot identifies an NFS 4.1 read/write
mount at `/home/m-ben-salah`; the exact project/data directories remain configurable.
PostgreSQL/MySQL is optional, not mandatory. This supersedes both the initial
single-node launcher and the subsequently over-restrictive database-only setup.
No cluster job was submitted from this session.

## Archive

Location: `archives/legacy_pre_v2_2026-09-23/artifacts/`.
Inventory/checksums: `archives/legacy_pre_v2_2026-09-23/manifest.json`.

- 573 checkpoints from 96 runs: 383 real-data and 190 semi-synthetic checkpoints.
- Eight additional historical log-only/aborted runs.
- Two SQLite databases containing three historical Optuna studies, the old
  synthetic HPO log and both legacy winning configuration files.
- 2,288 files/symlinks; 2,477,195,427 bytes (about 2.31 GiB).
- Post-move SHA-256/size verification passed. Symlinks were preserved without
  following them, and original relative paths are retained for recovery.

The new v2 run `models/HR+HER2-/4lines/23092026_150314_seed_1837416121` appeared
during cleanup and was explicitly protected, including its six checkpoints and
data manifest. The final dry-run found no remaining eligible legacy artifacts
in the main workspace.

This is a recoverable **directory archive**, not deletion/compression; it cleans
active paths without claiming to free disk space. The maintenance utility is
dry-run by default. Other worktrees and remote storage were not altered; the
historical alternate worktree under `.claude/worktrees/` remains untouched.
Do not blindly `make sync` old remote runs back into the cleaned directories.

## Findings and fixes

| Finding | Consequence | Correction |
| --- | --- | --- |
| CUDA was never selected | H100 allocations could train on CPU | CUDA-first selection; cluster explicitly requires CUDA/BF16 and one visible GPU per worker |
| Final-epoch score and mismatched stopping metric | Trial rankings did not represent the selected best model | One metric for stopping, pruning, checkpointing and best-epoch objective |
| Threaded trials shared one device/process | GPU memory and RNG interference | Four independent processes; `n_jobs=1`; total budget partitioned exactly |
| SQLite/path assumptions | Fragile parallel study workflow | Explicit NFSv3+ journal with exclusive-create locking, cross-node storage probe, serial initialization/export and four-worker preflight; PostgreSQL/MySQL remains optional |
| Incomplete experiment identity | Incompatible objectives/settings could mix | v3 fingerprint covers data, evaluation, training, precision/runtime and source |
| Inactive propensity-head dimensions | Wasted search dimensions | Fix that head; search active predictive dimensions |
| Rounded/reconstructed winning settings | Export could differ from the executed trial | Exact typed config export with provenance |
| Repeated preparation and propensity fitting | Unnecessary per-trial CPU work | Cache data per worker; skip recommendation-only classifier fitting |
| Reduced-precision survival calculations | Dtype/numerical risks under BF16 | FP32 likelihood and survival metric buffers |
| Missing invalid-objective/resource handling | Bad trials could obscure selection | Reject non-finite values; record OOM failures; release per-trial resources |

The corrected cohort, train-only preprocessing, development-validation selection
and reserved temporal test remain intact. Diagnostics now use the production
evaluation grid rather than a hidden six-point Brier approximation. Unexpected
programming/data errors still fail visibly instead of being swallowed as pruning.

## Four-H100 execution

`slurm/RunHPO.sh` requests two nodes, two typed H100 GPUs and 64 GB RAM per node,
with four CPU cores per worker (16 cores/128 GB RAM total), for up to 24 hours.
Each of the four workers fits one trial at a time;
Lightning is isolated from Slurm's distributed ranks. BF16 is the default, with
FP32 survival likelihood/metrics. Model seeds are fixed across trials; TPE seeds
differ by worker. `N_TRIALS=100` means **100 additional attempts total**, not 400.

In default NFS mode, an isolated 32-trial storage probe runs first. All four
workers synchronize, write with the production lock implementation, and a
coordinator verifies records from two distinct hostnames before the real HPO
study is touched. These dummy trials do not count toward the model trial budget.
The journal directory must be a read/write NFSv3+ mount inside every container;
node-local filesystems and unverified multi-node journals are refused.

Serial initialization then precedes a four-task preflight: both nodes must access
the same study, compatible data/code, a valid single visible GPU and the shared
job marker. Global worker IDs are 0–3; each node has local IDs 0–1. One exporter
follows successful completion of every worker; worker failure prevents export.
Trial configs/checkpoints persist independently on the shared project filesystem.

The journal uses `JournalFileOpenLock(grace_period=None)`: there is no forced
30-second expiry that could steal a slow writer's lock. A stale journal lock
blocks startup/export/recovery; it is never removed automatically. Journal mode
has no heartbeat recovery. After a crash, stop all writers, back up/inspect the
journal, and resolve only confirmed stale locks before explicit trial recovery.
An interrupted journal may require repair; the launcher does not attempt it.

In optional `HPO_STORAGE_BACKEND=rdb` mode, database credentials are read from
`OPTUNA_STORAGE_URL` and forwarded through
the environment, not command arguments. Connection-construction errors are
sanitized. Only the initializer requests table creation; workers load the
existing study. Heartbeats run every 60 seconds with a 300-second grace period;
abandoned trials can be marked failed before the next optimization trial starts.
Startup/export still refuse remaining `RUNNING` trials; explicit recovery is
permitted only after all old workers stop. A shared study-directory lock guards
against a second launcher using that directory, not unrelated/manual clients.

Optuna's version-specific documentation explicitly supports shared NFS journals:
see the [Optuna 4.7 recipe](https://optuna.readthedocs.io/en/v4.7.0/tutorial/20_recipes/011_journal_storage.html)
and [NFSv3+ exclusive-create lock API](https://optuna.readthedocs.io/en/v4.7.0/reference/generated/optuna.storages.journal.JournalFileOpenLock.html).
GPU requests/binding follow [Slurm's typed GRES rules](https://slurm.schedmd.com/gres.html).

`PROJECT_DIR`, `DATA_DIR` and `SIF` accept separate absolute shared paths. Data
is bind-mounted read-only at `/hpo_data`; all study/checkpoint outputs stay under
the project. `findmnt`/`util-linux` is explicitly included in the Dockerfile.

## Verification

- Full local suite after the NFS update: **110 passed, 2 skipped**. Skipped
  integration cases require CUDA and an explicitly supplied test database,
  neither available locally. Earlier single-node/database-only revisions had
  77/91 passing tests respectively.
- Tests cover best versus final epoch, objective direction, exact config export,
  invalid metrics, protocol incompatibility, worker budgets, GPU selection,
  archive integrity/protection, and rejection of threaded GPU trials.
- CPU FP32/BF16 Lightning fits, with simulated nonzero Slurm rank, verify
  single-process training, best-checkpoint persistence and cached data.
- Four OS processes completed 12 shared-journal trials with unique IDs and no
  lost trial records; this verifies the retained single-host path, not the RDB.
- The new storage probe completed 32 trials across four real local processes
  using the same no-expiry exclusive-create locks as NFS mode. It detects missing
  workers and wrong host counts. This validates concurrency on a local filesystem,
  **not** cross-node NFS behavior; the actual cluster probe is still required.
- New mount checks cover NFSv3/v4 acceptance, read-only/old-NFS/local-filesystem
  rejection, missing `findmnt`, explicit opt-in and missing/stale journal files.
  Mock launcher tests cover both backends, external data paths, probe failures,
  training failures and successful single-export orchestration.
- New tests cover RDB configuration/schema roles, unsafe multi-node storage
  rejection, credential-redacted connection errors, shared filesystem markers
  and preflight without trial creation. Actual shell orchestration with mock
  Slurm/Apptainer verifies two-node/global-rank commands, initialization,
  preflight, optimization, export, failure propagation and lock cleanup.
- A four-process real PostgreSQL/MySQL integration test is available behind
  `OPTUNA_TEST_STORAGE_URL`; it was skipped, not represented as a successful
  live database run. The CPU Lightning smoke tests simulate a worker on node 2.
- A full-cohort, one-epoch BF16 CPU CLI trial completed and exported to temporary
  smoke-test storage. It is not a tuned result or active production winner.
- An end-to-end four-worker CLI smoke test also passed on the real cohort:
  four two-epoch CPU/BF16 trials, one shared journal, minimizing validation IBS,
  four verified best-checkpoint paths and one exported winner. The initializer,
  worker startup and exporter were exercised as separate processes. All four
  metric options are covered by CLI initialization regression tests.
- Slurm syntax/dry-run and `git diff --check` passed. Archive verification passed;
  the modern run remains in the active model directory.

## Remaining caveats

Default validation C-index measures discrimination, not calibration or treatment
effect accuracy. IBS/calibration are available as explicit alternative objectives;
choose before starting a study. Balancing weights remain zero in this predictive
search. These changes do not establish causal identification or clinical validity.

One fixed validation split/training seed does not quantify seed sensitivity.
Refit a shortlist across seeds using a prespecified validation rule, then score
the temporal test once. Align the refit stopping/checkpoint rule with the intended
endpoint: production currently stops on calibration, while HPO defaults to CI.
HPO checkpoints deliberately lack recommendation propensity models/eligibility;
use the regular training workflow for recommendation-ready models.

Actual H100 performance, two-node NFS/scheduler behavior and live RDB operation have
not been tested here. The launcher needs the site's partition/GRES spelling,
shared writable NFS project directory and data/image visible on both nodes.
The screenshot proves the filesystem type of one mount, not cross-node coherence
or the account's directory permissions. Those are exercised by the cluster probe.
Journal/database protocol records contain patient split IDs; their protections
must match those of the project data. No external database was contacted or
provisioned during this work.

NFS mode needs no Psycopg/PyMySQL driver; these dependencies are only for the
optional database backend. Ensure the image has the pinned Optuna and `findmnt`
before the isolated two-node smoke test. Study state no longer depends on copying
a journal from a compute node, but interrupted writes/checkpoints and filesystem
outages remain possible. Keep offline backups and monitor NFS latency/quotas;
the small probe is not a durability guarantee. Asynchronous TPE and GPU arithmetic
are not guaranteed bitwise reproducible.

Launch/smoke-test instructions, outputs and recovery are in
`scripts/hyperopt/README.md`. The new default winner is
`configs/hpo_v3/best_config.json`, generated only after a new study completes.
