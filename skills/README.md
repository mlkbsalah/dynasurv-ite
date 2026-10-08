# Project component documentation

Last checked: **8 October 2026**. These files describe the current
implementation and distinguish local artifacts from completed validation.

| Component | Current implementation | Maintained documentation |
|---|---|---|
| Survival model and data | Configurable LayerNorm or BatchNorm MLPs and static inputs; data protocol v2 separates development validation from temporal test; train-only scaling and survival grid | [Survival model](survival_model.md) |
| Recommendation | Training support, all-observed-class propensities, compatible ensembles, four decision statuses | [Recommendation](recommendation.md), [eligibility filters](../recommendation_filters.md) |
| Semisynthetic validation | Four-arm prefix benchmark; grouped splits; evaluation protocol v2; fixed checkpoint rule | [Semisynthetic validation](semisynthetic_validation.md) |
| Hyperparameter optimization | HPO v4 validation-loss objective with separate LayerNorm and BatchNorm studies; four independent cluster workers | [HPO component](hyperparameter_optimization.md), [run guide](../scripts/hyperopt/README.md) |

## Local execution status

The local checkout has the exported v5 winner and a trained model run with six
checkpoints. That run's checkpoint kinds were compared on development validation
only, with the existing `val_loss` rule selecting epoch 4. No v4 validation-loss
study has been run yet, and there is no corrected `eval_v2_*` metadata. These
observations establish local artifact availability, not scientific validity.

Default training, recommendation and semisynthetic training/evaluation all read
that winning JSON. HPO can produce it without an existing winner. HPO **v4**
still exports into **`configs/hpo_v3/`**; the directory name does not identify
the objective version.

The [dated project check](../reports/project_status_2026-10-07.md) records
verification, outstanding implementation gaps and historical evidence limits.
The report is tracked explicitly; other new files under `reports/` remain
ignored by default. These component documents are the durable source of current
behavior.

## Maintenance

When a component changes, update its document and any affected command examples.
Distinguish code inspection, executed checks and saved experiment results.
Record changed split rules, objectives, support rules, defaults and artifact paths.
Preserve historical reports with their original provenance rather than replacing
old numerical results with unexecuted claims.
