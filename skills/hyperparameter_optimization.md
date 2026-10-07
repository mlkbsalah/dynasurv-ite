# Hyperparameter optimization

Last checked: **7 October 2026**, including uncommitted changes to
[run_optuna.py](../scripts/hyperopt/run_optuna.py) and
[RunHPO.sh](../slurm/RunHPO.sh). See the [run guide](../scripts/hyperopt/README.md)
for commands. HPO protocol **v5** uses data protocol **v2** and the active
model's LayerNorm MLPs. The v5 study is separate from historical BatchNorm v4
trials because their objective values are not directly comparable.

## Objective and selection

The default trial score is the maximum, over completed validation epochs, of
`val/hpo_score = average_ci - val/calib_gap_abs_mean`.
`HPOModel` emits it after the base model pools epoch metrics;
`BestValidationMetric` reads it in `on_validation_end` and records the best
epoch, score and available diagnostics in trial attributes.

Pruning receives each epoch's score through a median pruner (5 startup trials,
10 warm-up steps). Early stopping separately minimizes the calibration gap
with patience 10. Maximum training length is 100 epochs. Stopping and ranking
therefore do **not** use the same metric. No trial checkpoints are saved.

The training seed is fixed at 42 across trials. TPE samplers use `42 + worker_id`
with `constant_liar=True`; each process runs `n_jobs=1` on one device. Numerical
objective failures and CUDA OOM mark failed trials; unexpected errors propagate.

## Search space

| Parameter | Current values/range |
|---|---|
| LSTM width / depth | 64, 128, 256 / 1–4 |
| Clinical / treatment embedding widths | 32, 64, 128 / 8, 16, 32 |
| Clinical and survival MLPs | Independently 1–3 layers, repeated width 32, 64, 128 or 256 |
| Clinical and survival dropout | 0–0.4 |
| Learning rate | 1e-5–1e-3, log scale |
| Weight decay | 1e-5–1e-2, log scale |
| Scheduler step / gamma | 10–50 / 0.1–0.7 |
| Batch size | 64, 128, 256 |

Attention is enabled; the propensity head is fixed at `(64,)` with zero dropout.
All `lambda_prop_loss`, `lambda_ipm_mmd` and `lambda_ipm_emd2` weights are zero.
`HPO_N_INTERVALS=100` is fixed in source, not sampled or exposed through a flag.
Other settings take the typed config defaults.

## Data and runtime

The runner loads `configs/config.toml` directly and resolves `data_dir` relative
to that config's directory. It uses the production cohort, exclusions and fixed
development-validation split, with `final_training=True`. `HPODataModule`
caches tensorization per process and computes training arm-count masks. It
skips the recommendation-only assignment classifier and does not produce a
deployable recommendation checkpoint; the exported winner must be refitted
through the main training CLI.

`SLURM_JOB_ID` selects cluster mode. Worker identity/count come from
`SLURM_PROCID` and `SLURM_NTASKS`. Cluster mode requests GPU/BF16; local mode
selects CUDA, then MPS, then CPU, using full precision on non-CUDA devices.
The Lightning environment is explicitly single-device, independent of Slurm
task rank. The 100 additional trial attempts per invocation are divided across
workers (25 each with four workers).

## Storage and export

| Artifact | Current location/name |
|---|---|
| Local journal / study | `studies/hpo_v5/local/study.journal` / `dynasurv_hpo_v5_local` |
| Cluster journal / study | `studies/hpo_v5/cluster/study.journal` / `dynasurv_hpo_v5_cluster` |
| Winner, both modes | `configs/hpo_v3/best_config.json` |
| Export provenance | `configs/hpo_v3/best_config.provenance.json` |

Journal storage uses `JournalFileOpenLock` with no automatic stale-lock grace
period. Local optimization exports on completion. The Slurm launcher exports
once after all workers succeed. `--export-only` refuses any RUNNING trials,
requires a finite completed result and checks the winning trial's protocol
attribute. It atomically writes the exact executed typed configuration plus
trial/study attributes. This exports settings, not model weights.

## Current limitations

- `--export-only` is the only custom CLI option. Old v3 storage/backend,
  preflight, trial-budget, metric, interval and recovery options are removed.
- `load_if_exists=True` resumes the fixed study name. `protocol_details` is
  written only when absent; current data, source and settings are not compared
  against it on resume. The version check at export does not detect within-v5
  configuration drift.
- The previous cross-node storage probe, launcher ownership lock, heartbeat and
  stale-trial recovery commands are absent. Cluster filesystem/GPU behavior has
  not been exercised in this review.
- Local and cluster modes have separate journals but share the winner path.
  A later successful export replaces that file. Trial budget applies per launch,
  not as a lifetime cap on the resumed study.
- Despite the v5 study/objective, the output directory retains `hpo_v3` to match
  downstream defaults. The executable parser does not accept `--n-intervals`.

No active local journal or exported winner was present at this review. Source
inspection and CLI/config checks do not establish completed tuning or model quality.
