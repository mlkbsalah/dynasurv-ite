# Survival model and data pipeline

Last checked: **7 October 2026**. Scope: the active
`DynaSurvCausalOnline` workflow and data protocol **v2**.

## Entry points and configuration

- [Training CLI](../scripts/TrainDynasurvCausal.py)
- [Checkpoint validation CLI](../scripts/ValidateCheckpoint.py)
- [Model](../src/CausalSurv/model/dynasurv_causal_online.py) and
  [history encoder](../src/CausalSurv/model/embedding_C_LSTM_ITE.py)
- [Datamodule](../src/CausalSurv/data/datamodule_cv.py) and
  [dataset](../src/CausalSurv/data/dataset.py)
- [Typed configuration](../src/CausalSurv/config.py)

Run the training CLI from `scripts/`. `ExperimentConfig.from_files` combines
`configs/config.toml` with `configs/hpo_v3/best_config.json`; unknown keys are
rejected. The JSON contains `n_intervals`, `batch_size`, `arch` and `training`.
The TOML supplies cohort, trainer, evaluation and recommendation settings.
Both paths can be overridden with `--config` and `--model-config`.

The default JSON is absent locally at this review. A successful HPO export or
an explicit compatible configuration is required before training. Current HPO
fixes `n_intervals=100` in source; it does not expose `--n-intervals`.

## Data and split contract

The active loader reads the subtype V2 dynamic parquet and
`model_entry_imputes_data_STATIC_no_staging.parquet` directly from `data_dir`;
it does not read separate `data/train/` and `data/test/` directories.

Current real-data settings are HR+HER2-, four lines, first-line entry in 2018+,
a calendar feature, and exclusion of `X_onset_to_progression` from input
features. Dynamic and static records are inner-joined; exact duplicate rows are
removed, while conflicting duplicate patient-line keys raise an error.

With `final_training=True`, as used by the main CLI:

1. Entry years before 2021 form the development pool; entry in 2021+ is test.
2. A fixed 20% of development patients is reserved for validation using
   `validation_seed=0`. Early stopping and checkpoint selection use validation.
3. Continuous-feature detection/scaling and the outcome-derived time-grid maximum
   use training rows only. Support counts and the separate assignment model are
   fitted from the training partition during `setup()`.
4. `val_dataloader()` exposes validation; `test_dataloader()` exposes test.
   Test evaluation runs only when the CLI receives `--evaluate-test`.

The alternate CV path folds only the development pool and reserves a separate
early-stop subset. Treatment-category vocabulary and input column discovery
still come from the merged cohort; they are recorded in the manifest but are
not learned exclusively from the training partition.

`data_manifest` records protocol version, merged-data hash, feature order,
treatment mapping, scaler, interval bounds and split patient IDs. The run
writes `data_manifest.json` and saves the manifest in each checkpoint. Do not
copy patient-level manifest contents into documentation or commits.

## Model behavior

Each sample supplies dynamic clinical features, treatment codes, elapsed time,
static patient features and static pretreatment history. Learned `init_h`,
`init_c` and `init_p` projections use the static inputs by default. The old
description that static inputs are silently dropped applies to legacy weights,
not newly constructed models.

At line k, the encoder uses observed clinical history and prior treatment
embeddings. The current factual treatment is embedded for the following line;
it does not enter the current decision representation. A shared output MLP
produces treatment-specific discrete-hazard coordinates with shape
`(batch, observed_lines, treatments, intervals)`. Per-line temperature and bias
parameters are learned jointly with the survival loss.

New active-model runs default to LayerNorm in the encoder and both prediction-head
MLPs. This normalizes each sample across hidden features, so training behavior
does not depend on the other patients or padded rows in a batch. Set
`arch.mlp_normalization` to `batch` in the model config to use BatchNorm in
training and HPO refits; older configs without this field default to `layer`.
The shared MLP helper retains BatchNorm as its default for older model variants. Historical
BatchNorm checkpoints must be loaded through `load_dynasurv_checkpoint`; their
saved normalization is reconstructed as BatchNorm, not converted to LayerNorm.
Changing normalization changes the fitted model and requires new training and
validation before interpreting performance.

`forward_factual` gathers the observed treatment's head. All-arm predictions use
`predict_discrete_survival`, which still needs factual indices to carry observed
treatment history forward. There is no `forward_counterfactual` method or
whole-sequence treatment simulator.

The loss is masked discrete logistic-hazard negative log likelihood. Optional
gradient-reversal propensity and MMD/EMD terms remain implemented; current HPO
sets all three weights to zero. The adversarial propensity head is distinct from
the training-only assignment model used by recommendations.

## Evaluation and checkpoints

Current horizons are `[24, 18, 12, 12]` months. `integration_step=100` specifies
the number of Brier-grid points, not a month-sized integration step.
Calibration landmarks are `[6, 12, 24, 36]` months; unsupported late landmarks
are skipped. The model's IPCW Brier calculation supplies both subject-time and
evaluation-time weights from the training censoring reference.

The main CLI allows up to 100 epochs and stops on
`val/calib_gap_abs_mean` with patience 10 in the current TOML. It saves
`val_loss`, `bestCI`, `bestIBS`, `bestCALIB` and `final_epoch` checkpoint kinds.
The configured rule for optional final testing and ensemble selection is
`val_loss`; it is separate from the stopping metric.

Run `python3 scripts/ValidateCheckpoint.py /path/to/checkpoint.ckpt` to rebuild
the checkpoint's cohort and produce development-validation metrics, calibration
and Brier tables, treatment counts, and diagnostic figures. The CLI checks the
checkpoint and run manifests before evaluation and writes `validation_*` files
to the run directory, directly above `checkpoints/`. It never scores the reserved
temporal test partition. Its recommendation mix is descriptive; it does not
estimate a causal policy effect. The older `notebooks/model_validation.ipynb`
hardcodes a legacy path and uses `test_dataloader()`, so use the CLI for routine
validation of newly trained checkpoints.

Runs default to `models/{subtype}/{n_lines}lines/{date}_seed_{seed}/` with W&B
logs. Existing checkpoint directories are rejected to prevent accidental mixing.
`--fast_dev_run` currently sets **three full epochs**; it does not pass
Lightning's `fast_dev_run` flag or limit batches. It still needs data/configuration
and uses the usual run directory/logger.

Use [load_dynasurv_checkpoint](../src/CausalSurv/model/checkpoint_compat.py) for
standalone loading. It translates legacy flat hyperparameters and preserves
legacy zero-state behavior when static projection weights are absent. Loading
old weights does not convert them to corrected experiments. Checkpoints also
persist support masks, the assignment estimator and calibration status.

[HazardCalibrator](../src/CausalSurv/evaluation/hazard_calibration.py) is a
separate post-hoc utility. Default training records `joint_nll_only`; a post-hoc
fit records `posthoc_reference_bin_fit`. The CLI does not automatically perform
post-hoc calibration. Shared survival/RMST calculations live in
[discrete_survival.py](../src/CausalSurv/evaluation/discrete_survival.py).

## Boundaries and current evidence

The local saved manifest contains 3,371 training, 843 validation and 1,977 test
patients, with 100 intervals. This review checked metadata and syntax/config
loading; it did not rerun training or certify the six saved checkpoints' metrics.

Corrected split code cannot undo historical inspection of the test cohort.
Upstream imputation, feature timing, real-data censoring assumptions and
counterfactual accuracy remain outside this source review. Using static
features does not establish that every static value was measured before the
decision.

The MH, multihead and residual model files define independent Lightning modules;
base-model fixes do not automatically propagate. Legacy training/baseline
workflows require their own review before results are compared. See the
[recommendation](recommendation.md), [semisynthetic](semisynthetic_validation.md)
and [HPO](hyperparameter_optimization.md) component documents for their contracts.
