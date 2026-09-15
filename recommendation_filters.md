# Recommendation eligibility filters

The model never recommends every observed treatment arm at every line. Four
independent filters narrow the action space before `TreatmentRecommender.recommend()`
ranks arms by RMST. This doc traces where each one is computed, where it is
applied, and what it is protecting against.

Code: `src/CausalSurv/model/dynasurv_causal_online.py`,
`src/CausalSurv/data/datamodule_cv.py`,
`src/CausalSurv/evaluation/propensity_overlap.py`,
`src/CausalSurv/recommendation/recommender.py`.
Config: `configs/config.toml` (`[data]` section).

## Summary table

| # | Filter | Level | Computed by | Threshold (current config) |
|---|--------|-------|-------------|------------------------------|
| 1 | Arm support (min count) | arm × line, global | `_compute_valid_treatments_per_line` | `min_samples_per_treatment = 200` |
| 2 | Well-definedness (excluded arms) | arm, global | `_compute_recommendable_treatments_per_line` | `excluded_treatment_arms = ["NO TREATMENT", "OTHER", "ET+TT"]` |
| 3 | Outcome/horizon support | arm × line, global | `_compute_recommendable_treatments_per_line` / `_compute_arm_support_summary` | `min_events_per_treatment = 30`, `min_followup_samples_per_treatment = 100` |
| 4 | Propensity / positivity | arm × line **× patient** | `PropensityOverlapModel` | `propensity_min_probability = 0.01` |

All four are fit **only on the training partition** (`_set_training_support`,
called from `setup()`), so validation/test outcomes can never change which
arms are eligible.

---

## 1. Arm support filter (raw count)

`ESMEOnlineDataModuleCV._compute_valid_treatments_per_line` (`datamodule_cv.py:389`)

For each line, an arm index `k` is kept only if it was observed at least
`min_samples_per_treatment` (200) times in the training partition:

```python
valid = [k for k in range(n_treatments) if (t_line == k).sum().item() >= min_samples]
```

This produces `valid_treatments_per_line`, the dataset's global notion of
"this arm has enough data at this line to say anything at all." It is the
**first and broadest** cut — everything downstream (filters 2-4) only ever
removes arms from this set, never adds one back.

`valid_treatments_per_line` has a second life inside the model that has
nothing to do with recommendation: `DynaSurvCausalOnline._valid_arm_mask`
(`dynasurv_causal_online.py:629`) uses the exact same set to decide which
observations contribute to the propensity adversary loss and the MMD/EMD
balance losses (`_compute_propensity_loss`, `_compute_ipm_mmd/emd2`). Arms
with only a handful of patients are excluded there too, so the encoder isn't
adversarially pressured to "balance" what is really just noise.

## 2. Well-definedness filter (structural exclusion)

`DEFAULT_EXCLUDED_ARMS` (`datamodule_cv.py:40`), applied in
`_compute_recommendable_treatments_per_line` (`datamodule_cv.py:410`).

Three arms are dropped from the *recommendable* set regardless of how much
data they have, because they are not a single well-defined intervention:

- **`NO TREATMENT`** — 207/0/0/1 across lines 1-4: a line-1 coding artefact,
  not a therapy. Structurally non-positive at every later line.
- **`OTHER`** — 736 distinct drug-flag combinations, modal share only 0.16.
  "Set arm = OTHER" doesn't correspond to one action (SUTVA violation).
- **`ET+TT`** — 344 distinct combinations, same problem.

Their records are **not removed from the dataset** — a patient who received
`OTHER` at line 2 still contributes that line to their history and to the
shared encoder. Only the eligibility to be *recommended* is withdrawn. See
`improvements.md` sections 3, 6, 7 for the supporting counts.

## 3. Outcome/horizon support filter

`_compute_arm_support_summary` (`datamodule_cv.py:471`) computes, per
line/arm and relative to that line's RMST horizon `tau`
(`evaluation_horizon_times`, currently `[24, 18, 12, 12]` months):

- `events_to_horizon` — deaths observed before `tau` (a death before `tau`
  fixes survival at 0 for the rest of `[0, tau]`, so it's informative).
- `known_to_horizon` — records followed at least to `tau`, *plus* the events
  above (both have a fully known survival status on `[0, tau]`; a patient
  censored before `tau` does not).

`_compute_recommendable_treatments_per_line` then requires, on top of filter
1's raw count:

```python
enough_events    = events_to_horizon  >= min_events_per_treatment      # 30
enough_followup  = known_to_horizon   >= min_followup_samples_per_treatment  # 100
```

This exists because filter 1 alone can pass an arm that has 200+ patients
but almost no one has been followed long enough to say anything about
survival at that line's horizon — raw count is not the same as identifiable
outcome information.

Filters 1-3 together produce `recommendable_treatments_per_line`, exposed on
the model as `self.recommendable_treatments_per_line`
(`_setup_valid_treatments`, `dynasurv_causal_online.py:1135`) and turned into
a `(n_lines, n_treatments)` boolean tensor by
`TreatmentRecommender.action_mask()` (`recommendation/recommender.py:71`).

## 4. Propensity / positivity filter (patient-specific)

`src/CausalSurv/evaluation/propensity_overlap.py` — `PropensityOverlapModel`,
fit in `_set_training_support` (`datamodule_cv.py:519`) and consumed by
`TreatmentRecommender.patient_support_mask()` (`recommendation/recommender.py:86`).

Filters 1-3 are **global**: an arm is either recommendable at a line or it
isn't, for every patient. This filter is the only one that is **patient
level** — it asks "does *this patient's* pre-treatment history give this arm
any real probability of having been chosen?"

- One multinomial propensity model is fit **per line**, restricted to the
  arms that already passed filters 1-3 (`eligible_arms_per_line`).
- Features are strictly pre-treatment: dynamic covariates up to and
  including the current line, static covariates, and the *prior* lines'
  treatment history/buffer time — never the current line's treatment
  (`features_from_tensors`, deliberately excludes `P[:, line]`).
- Factual propensities are estimated **out-of-fold** (`StratifiedKFold`,
  `propensity_cv_folds = 5`) so the diagnostic isn't inflated by the model
  scoring its own training rows.
- At inference, `predict_mask` marks an arm unsupported for a patient when
  its predicted probability falls below `propensity_min_probability` (0.01):

  ```python
  supported = probabilities >= fitted.threshold
  ```

- Diagnostics are logged even when nothing is masked: `effective_sample_size`
  (per arm, via propensity-based reweighting) and `low_propensity_rate` (share
  of factual assignments below the floor) are printed by
  `describe_cohort()` and meant to be read before raising the threshold —
  the code comment on `propensity_min_probability` says as much: *"Set to
  zero to audit propensities without masking recommendations; tune from the
  resulting ESS/low-propensity reports."*

This is the actual **positivity** filter in the causal-inference sense:
filters 1-3 ask "is there enough data on this arm at all," filter 4 asks "is
there enough data on this arm *for patients who look like this one*."

---

## How the filters combine at inference

`TreatmentRecommender.recommend()` (`recommendation/recommender.py:164`),
built from the model's checkpointed support state via
`TreatmentRecommender.from_model()`:

1. `patient_support_mask()` starts from the global `action_mask()`
   (filters 1-3) broadcast to the batch, then AND's in the per-patient
   `PropensityOverlapModel.predict_mask()` (filter 4) — only for lines that
   actually have a fitted propensity model, so lines without one keep only
   the global mask.
2. RMST is computed for every arm, then arms failing the combined mask are
   set to `-inf` before the argmax — a masked arm can never win the
   comparison and surface as advice, rather than being merely flagged.
3. If every arm is masked for a patient/line, `best_idx` is set to
   `NO_SUPPORTED_ARM` (`-1`) instead of defaulting to arm 0 — "no
   recommendation" is a distinct output from "recommend arm 0."

This logic used to live directly on `DynaSurvCausalOnline` as
`recommendable_mask`/`recommend_treatment`; it moved out to
`recommendation/recommender.py` since ranking arms and deciding when to
abstain is a policy built on top of the model's predictions, not something
the network itself computes. What stays on the model is the checkpointed
support state (`recommendation_propensity_model`,
`recommendable_treatments_per_line`, `valid_treatments_per_line`) that
`from_model()` reads.

## Related but *not* a recommendation filter

`min_ipm_group_size` (`src/CausalSurv/config.py`, `TrainingConfig`, default 16) looks
similar but serves a different purpose: it's the minimum per-batch group
size for a treatment pair to contribute to the MMD/EMD **balance loss**
during training. It affects what the causal regularizer sees, not which arms
can be recommended. All `lambda_ipm_*` / `lambda_prop_loss` weights are
currently 0 in every checked-in config, so these losses are inert regardless.
