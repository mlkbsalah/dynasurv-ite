# DynaSurv

Python research framework for survival prediction at successive treatment lines,
causal treatment comparisons, supported treatment recommendations and
semisynthetic validation. The import package is `CausalSurv`.

## Current project state

Reviewed **7 October 2026**, including local changes on top of `e9dcb87`.
The active survival pipeline uses data protocol v2: static features, separate
development validation and temporal test, training-only scaling/time grid, and
checkpoint manifests. Recommendations enforce support and ensemble compatibility.
Semisynthetic evaluation uses grouped splits and known-outcome scoring.

The current local HPO runner uses protocol v5 and LayerNorm MLPs, and optimizes validation concordance
minus calibration gap. It still exports to `configs/hpo_v3/best_config.json`.
That file is absent locally, as is `studies/`. One model run with a protocol-v2
manifest is present; no corrected semisynthetic evaluation metadata was found.
Historical reports do not establish performance of the corrected pipeline.

Start with the [component index](skills/README.md):

- [Survival model and data](skills/survival_model.md)
- [Recommendation](skills/recommendation.md) and [eligibility filters](recommendation_filters.md)
- [Semisynthetic validation](skills/semisynthetic_validation.md)
- [Hyperparameter optimization](skills/hyperparameter_optimization.md) and [run guide](scripts/hyperopt/README.md)
- [Dated project check](reports/project_status_2026-10-07.md)

## Running the active workflow

Use the existing `dynasurv_env` mamba environment. Training requires the data
parquets and an exported winning configuration, or an explicit compatible
`--model-config` path. From the repository root:

```bash
mamba activate dynasurv_env
cd scripts
python3 TrainDynasurvCausal.py --seed 0
```

`--fast_dev_run` currently caps training at three full epochs; it is an execution
check, not a meaningful experiment. Test evaluation is opt-in with
`--evaluate-test`, using the checkpoint rule in the configuration.
HPO commands and prerequisites are in its run guide; running HPO starts real
optimization, not a smoke check.

`pre-commit run --all-files` runs the pinned Ruff hooks and can modify files.
`test/` is deprecated. `make build-docker` builds the Linux AMD64 container and
exports `dynasurv.tar` for HPC. No remote job was inspected or submitted during
this documentation review.

## Structure

| Directory | Purpose |
|---|---|
| `src/CausalSurv/` | Data, models, losses, evaluation, recommendations and simulation |
| `scripts/`, `slurm/` | Local workflows and cluster launchers |
| `configs/` | Typed experiment and DGP settings; generated model settings |
| `skills/` | Maintained project component documentation |
| `reports/` | Experiment reports, tables and plots |
| `models/` | Model checkpoints and run metadata |
| `notebooks/`, `data_analysis/`, `latex/` | Analysis and publication material |
| `archives/` | Quarantined historical artifacts; see its inventory documentation |
