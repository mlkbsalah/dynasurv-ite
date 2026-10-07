# Semisynthetic validation

Last checked: **7 October 2026**. This describes the current four-arm benchmark
and evaluation protocol **v2**, not the earlier three-arm experiments.

## Implementation map

| Stage | Source |
|---|---|
| DGP settings and role validation | [dgp.toml](../configs/semisynthetic/dgp.toml), [config.py](../src/CausalSurv/semisynthetic/config.py) |
| Real cohort and driver construction | [drivers.py](../src/CausalSurv/semisynthetic/drivers.py) |
| Assignment, outcomes and censoring | [assignment.py](../src/CausalSurv/semisynthetic/assignment.py), [outcome.py](../src/CausalSurv/semisynthetic/outcome.py), [censoring.py](../src/CausalSurv/semisynthetic/censoring.py) |
| Prefix expansion and generation | [expand.py](../src/CausalSurv/semisynthetic/expand.py), [generate.py](../src/CausalSurv/semisynthetic/generate.py) |
| Grouped model data | [datamodule.py](../src/CausalSurv/semisynthetic/datamodule.py) |
| Prediction adapters and scoring | [predictors.py](../src/CausalSurv/semisynthetic/predictors.py), [evaluate.py](../src/CausalSurv/semisynthetic/evaluate.py) |
| Executable workflow | [generate](../scripts/semisynthetic/generate.py), [train](../scripts/semisynthetic/train.py), [evaluate](../scripts/semisynthetic/evaluate.py), [aggregate](../scripts/semisynthetic/aggregate.py) |

## What is simulated

The generator keeps real clinical covariates and prior observed treatments from
the HR+HER2- cohort, first-line entry in 2018+, up to four lines. It simulates
the current treatment and outcome for ET alone, ET+ANTI-CDK wo CT, MONOCT std
alone and POLYCT alone. Arm arrays use sorted names.

Assignment follows a softmax model with calibrated line/arm intercepts.
Potential survival follows a Weibull model with prognostic terms, arm effects
and arm/covariate interactions. Administrative censoring uses the configured
2024-02-21 cutoff, with optional exponential dropout. DGP driver scaling and
calibration use the source cohort; these generator settings are distinct from
the model's training-only preprocessing.

The three sweep controls are `gamma` (observed assignment coupling), `strength`
(hidden-confounder assignment coupling) and `heterogeneity` (outcome interactions).
The hidden variable is stored in truth artifacts and excluded from model input.
Its outcome coefficient stays fixed during a strength sweep. Typed DGP loading
rejects unknown keys, unknown arms and driver-role violations; calendar is an
assignment-only driver in this DGP.

Each patient with L lines becomes L prefix samples. Sample j contains real
history through j-1 and a simulated treatment/outcome at j. Only the endpoint
contributes to loss and support counts; earlier rows have placeholder outcomes.
All prefixes of an original patient stay in the same split. This evaluates
current decisions conditional on real histories, not simulated long-term
trajectories under repeated recommendations.

Age and menopause are still copied into dynamic columns by the generator.
The nearby source comment explaining this with ignored static model inputs is
historical: the active survival model now uses static projections.

## Training and evaluation contract

[config.toml](../configs/semisynthetic/config.toml) uses the same strict experiment
schema as real-data training. It reserves 2021+ entrants for test and a fixed
20% of earlier original patients for validation (`validation_seed=0`). Scaling
and the model's outcome-derived grid are fitted on training data; the grid
uses endpoint outcomes. The training loader shuffles with a seeded generator.

Current settings are 150 maximum epochs, `val_loss` stopping with patience 20,
horizons `[24, 18, 12, 12]`, and support thresholds 30 observations, 10 events
through the horizon and 30 known outcomes through the horizon. Non-DGP arms
remain as history but are excluded from recommendations.

Training and evaluation default to `configs/hpo_v3/best_config.json`, currently
absent locally. Both accept `--model-config`; the same compatible configuration
must be supplied to both. Current HPO uses the real-data config, so its export
is not evidence of benchmark-specific retuning.

Evaluation takes an explicit checkpoint kind and `--split validation|test`
(default `test`). It requires a matching data manifest and static-enabled model.
Choose the checkpoint rule using development data before test evaluation.
The aggregator takes a fixed rule and does not optimize it against test truth.

The wired predictors are DynaSurv, the true-curve oracle, and training-only
per-line/per-arm Kaplan–Meier (`naive_km`). Output tables contain:

- Curve error and RMST error/bias against known truth.
- Pairwise effect error, including PEHE, with support restrictions.
- Policy value, regret, hit rate and abstention for line-only and patient-supported
  decisions, alongside factual, random and oracle best-constant references.
- Exact expected factual Brier loss and uncensored latent-outcome Brier/concordance
  at the requested horizon. These replace the old transferred marginal IPCW scores.

`best_constant` uses evaluated-cohort truth and is an oracle reference, not a
policy fitted on training data. Supported-policy abstentions retain the factual
arm's true value in policy scoring. These policy scores evaluate single-model
RMST choices; they do not validate the ensemble's vote/margin confidence rule.

## Commands and artifacts

From `scripts/`, once a compatible winning configuration is available:

```bash
python3 semisynthetic/generate.py --out ../data/semisynthetic/gamma/1.0/rep0 --replicate 0 --gamma 1.0
python3 semisynthetic/train.py --axis gamma --level 1.0 --rep 0 --seed 0
python3 semisynthetic/evaluate.py --axis gamma --level 1.0 --rep 0 --seed 0 --kind val_loss --split validation
# Run only after the rule has been fixed using development data:
python3 semisynthetic/evaluate.py --axis gamma --level 1.0 --rep 0 --seed 0 --kind val_loss --split test
python3 semisynthetic/aggregate.py --kind val_loss --split test
```

Generated dynamic/static parquet, `truth.parquet` and `manifest.json` live under
`data/semisynthetic/{axis}/{level}/rep{r}/`. Checkpoints live under
`models/semisynthetic_v2/{axis}/{level}/rep{r}/seed_{seed}/`.
The evaluator currently also writes `eval_v2_{split}_{kind}/` CSVs and metadata
beside those checkpoints; this existing path differs from the repository rule
that experiment results belong in `reports/`. Aggregated tables default to
`reports/semisynthetic_sweep_v2/`. No output paths were changed in this review.

[slurm/sweep.sh](../slurm/sweep.sh) defines 10 cells × 3 replicates, compares
checkpoint kinds on validation, and uses one `TEST_KIND` (default `val_loss`)
on test. Its current wall time is four hours, unlike the repository's 24-hour
default guidance. It accepts `MODEL_CONFIG`, `N_REPS`, `VALIDATION_KINDS`,
`TEST_KIND` and `DRY_RUN` as documented in that script.

## Evidence status and limits

No corrected `eval_v2_*` metadata was found locally under `models/` or `reports/`
on this review. That does not establish what has run remotely. The
[historical benchmark report](../reports/semisynthetic_benchmark.md) and legacy
comparison tables remain exploratory; historical checkpoint selection reused
the evaluated holdout. `--legacy-exploratory` is an explicit aggregation mode,
not conversion to protocol v2.

Known simulated truth permits scoring within this DGP; it does not establish
counterfactual accuracy on real outcomes. Source-cohort calibration, chosen
functional forms, support restrictions and the limited wired comparator set
constrain what these experiments can establish. The current documentation
update reports implementation and artifact availability, not new performance.
