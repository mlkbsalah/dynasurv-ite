# Project component documentation

Last checked: **7 October 2026**, against Git HEAD `e9dcb87` plus the existing
working-tree changes. These files describe the implementation, including local
HPO edits that have not been committed.

| Component | Current implementation | Maintained documentation |
|---|---|---|
| Survival model and data | LayerNorm MLPs and static inputs enabled; data protocol v2 separates development validation from temporal test; train-only scaling and survival grid | [Survival model](survival_model.md) |
| Recommendation | Training support, all-observed-class propensities, compatible ensembles, four decision statuses | [Recommendation](recommendation.md), [eligibility filters](../recommendation_filters.md) |
| Semisynthetic validation | Four-arm prefix benchmark; grouped splits; evaluation protocol v2; fixed checkpoint rule | [Semisynthetic validation](semisynthetic_validation.md) |
| Hyperparameter optimization | HPO protocol v5 with LayerNorm; CI minus calibration-gap objective; four independent cluster workers | [HPO component](hyperparameter_optimization.md), [run guide](../scripts/hyperopt/README.md) |

## Local execution status

The local checkout has one model run with a protocol-v2 manifest but no
checkpoint files. It has no `studies/` directory, no default
`configs/hpo_v3/best_config.json`, and no `eval_v2_*` metadata under `models/`
or `reports/`. Those observations establish local artifact availability, not
cluster job status or scientific validity.

Default training, recommendation and semisynthetic training/evaluation all read
that winning JSON. HPO can produce it without an existing winner. HPO **v5**
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
