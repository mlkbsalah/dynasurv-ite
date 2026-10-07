# Recommendation eligibility filters

Last checked: **7 October 2026**. These are the current support rules in
[datamodule_cv.py](src/CausalSurv/data/datamodule_cv.py),
[propensity_overlap.py](src/CausalSurv/evaluation/propensity_overlap.py) and
[recommender.py](src/CausalSurv/recommendation/recommender.py).
The [recommendation component document](skills/recommendation.md) covers
ensemble decisions and compatibility requirements.

## Current configuration

| Filter | Scope | Current real-data setting |
|---|---|---|
| Training observations | Arm × line | At least 200 |
| Explicit exclusions | Arm, all lines | `NO TREATMENT`, `OTHER`, `ET+TT` |
| Events through RMST horizon | Arm × line | At least 30 |
| Outcomes known through horizon | Arm × line | At least 100 |
| Assignment probability | Arm × line × patient | At least 0.01 |

The horizons are `[24, 18, 12, 12]` months. Counts, horizon summaries and the
assignment estimator are built from the training partition in
`_set_training_support`. Explicit exclusions come from configuration; they are
not learned from validation or test. Eligibility is a statistical support rule,
not a check of individual contraindications or clinical suitability.

## Global support

`_compute_valid_treatments_per_line` counts masked training observations for
all treatment categories. An arm must meet `min_samples_per_treatment` at that
line. These valid-arm sets also enter propensity-adversary and MMD/EMD losses;
they do not alone define recommendation eligibility.

`_compute_recommendable_treatments_per_line` removes explicitly excluded arms
and applies `_compute_arm_support_summary`:

- `events_to_horizon`: observed deaths at or before the line's horizon.
- `known_to_horizon`: records followed at least through the horizon, or observed
  deaths (a death before the horizon also resolves survival status thereafter).

The current configuration requires both minimum counts above. Excluded-arm
records remain available for factual training and observed history. Historical
regimen-count arguments for the exclusions are in `improvements.md`; those
counts were not recomputed for this review.

In the normal pipeline every line/arm receives a support-summary entry. The
lower-level helper skips horizon checks if an entry is missing (`stats is
None`), so calling it without summaries is not equivalent to the fully checked
training pipeline.

## Patient-specific assignment support

`PropensityOverlapModel` fits a multinomial logistic model per line over
**all observed assignment classes**, including arms excluded from recommendation.
A one-class fit uses `DummyClassifier`. Probabilities are not renormalized over
the eligible set; otherwise low total probability of receiving any eligible arm
would be concealed.

Inputs include dynamic covariates through the current line, static patient
covariates and prior-line treatments/elapsed durations. Current treatment is
excluded. The static pretreatment-history tensor is not included by this
assignment feature builder, even though the survival model uses it.

Factual training propensities are estimated out of fold for ESS and
low-propensity diagnostics. The requested five folds are reduced for rare
classes; when stratification cannot be used, the code falls back to `KFold`.
The final estimator fits all training observations at the line. These
probability estimates and diagnostics are not proof of calibrated overlap on a
new population.

At inference, `predict_mask` permits only fitted classes reaching the probability
threshold. Missing line estimators stay false. A missing entire assignment model
also yields an all-false patient mask; legacy models without
`assignment_scope="all_observed"` raise an error. There is no bypass for lines
with one globally eligible arm.

## Combining support and decisions

`TreatmentRecommender.action_mask()` builds the global mask from persisted
recommendable sets and raises if that metadata is absent.
`patient_support_mask()` intersects it with the assignment-probability mask.
`recommend()` ranks finite RMST predictions after masking unsupported arms to
negative infinity, returning `-1` if no arm is supported.

The ensemble intersects masks across members before scoring. Its statuses are
`no_support`, `only_supported_option`, `undecided` and `confident`. Exactly one
eligible arm cannot produce a confident comparison, even with unanimous votes.
The single-model method does not implement the ensemble's vote/margin rule.

Checkpoints persist eligibility, arm summaries, assignment models and data
manifests. Ensemble loading rejects stale or missing provenance. The horizon
sweep reuses the original support mask; it measures sensitivity of RMST ranking,
not whether longer horizons have adequate additional follow-up.

## Balancing is a separate mechanism

`TrainingConfig.min_ipm_group_size` defaults to 16 observations per treatment
group within a batch. It controls which pairs contribute to MMD/EMD, not which
arms can be recommended. Current HPO fixes all balancing/adversarial weights
to zero; their runtime values for any other experiment come from its model JSON
or checkpoint, not merely the real-data TOML.
