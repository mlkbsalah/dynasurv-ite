# data_analysis

Standalone exploratory-analysis scripts for the ESME cohorts. Each script reads a
parquet from the (git-ignored) `data/` directory at the repo root and writes its
figure(s) into `plots/`.

## Convention

- One script per analysis, runnable directly from the repo root:
  ```bash
  python data_analysis/<script>.py
  ```
- Figures are written to `data_analysis/plots/` (git-ignored; regenerate by
  re-running the script). Interactive figures are self-contained HTML — open them
  in any browser and hover for the underlying counts.
- Scripts resolve the repo root relative to their own location, so they work whether
  run from the repo root or elsewhere.

## Scripts

| Script | What it produces |
|--------|------------------|
| `treatment_category_repeats.py` | Two interactive Plotly figures on repetition of `T_treatment_category` over the first 4 treatment lines (HR+HER2−). **Both kinds:** patients by repetition kind, repetitions per category split into **consecutive** (same category in the next line) and **non-consecutive** (the category returns after a switch), and all 15 sequence patterns four lines can take. **Consecutive only:** blocks per patient, category by run length, and where in the sequence each block sits. → `plots/treatment_category_repeats.html`, `plots/treatment_category_repeats_consecutive.html` |
| `km_line1_by_year.py` | Crude Kaplan-Meier overall survival from line-1 onset, crossed by treatment category and calendar year of onset, shown three ways: one panel per year stratified by treatment; one panel per treatment stratified by year (light = early → dark = recent); and both collapsed to 24-month OS by year. Clicking a series in any legend isolates it across every panel. → `plots/km_line1_panel_per_year.html`, `plots/km_line1_panel_per_category.html`, `plots/km_line1_24mo_trend.html` |
| `mpps_performance_status_by_year.py` | Evolution of the performance-status distribution (`X_mpps`, ECOG/WHO 0–4) over calendar time: 100% stacked composition per year of line-1 onset, the two ends of the scale (PS 0 and PS ≥ 2) as trend lines with 95% Wilson intervals, and the distribution across treatment lines 1–4. **`X_mpps` is imputed** — see the caveat below. → `plots/mpps_by_year.html` |
| `km_line1_adjusted.py` | Pooled OS by treatment category, crude vs **IPTW + IPCW adjusted**, so the curves are comparable across treatments — plus a diagnostics figure (covariate balance before/after, effective sample size, calendar-era overlap) that shows where the adjustment succeeds and where it cannot. → `plots/km_line1_adjusted.html`, `plots/km_line1_adjustment_diagnostics.html` |

## Reading the repetition figure

- A **repetition** is any line whose category the patient has already had. It is
  *consecutive* when the previous line was the same category (a continuation) and
  *non-consecutive* when the patient switched away and came back (a rechallenge).
  Counting only consecutive repeats understates reuse considerably: 58% of patients
  repeat a category in adjacent lines, but 75% repeat one somewhere in their first
  four lines.
- Panels A and C count **patients** (each patient falls in exactly one of the four
  kinds, and in exactly one of the 15 sequence patterns); panel B counts
  **repetition events**, so a patient can contribute more than one.
- The 15 patterns in panel C are the complete set — four lines admit exactly 15
  distinct repetition shapes — so the panel is exhaustive, not a top-N.

The consecutive-only figure counts **blocks** rather than repetition events: one
block is one uninterrupted run of the same category, so a 3-line block is 2
repetitions. That is why 4,551 blocks correspond to 5,357 consecutive
repetitions. Its third panel is subject to an **edge effect** and should not be
read as purely clinical: a block occupying lines 2–3 needs the category to differ
on *both* sides, while one at lines 1–2 or 3–4 has only one interior boundary to
satisfy, so the middle position is structurally rarer before any treatment
behaviour is considered.

## Reading the performance-status figure

Performance status is ECOG/WHO 0 (fully active) to 4 (completely disabled). **Two
sources exist and they disagree in direction — use the observed one.**

- **Observed:** `data_raw/metperf.parquet` (git-ignored), a table of *dated*
  measurements — `usubjid`, `mpdt`, `mpps`. The script restricts it to the `usubjid`
  set of the HR+HER2− V2 model-entry file and takes the measurement nearest each
  patient's line-1 start, within ±90 days.
- **Imputed:** `X_mpps` in the model-entry parquet, one value per treatment line with
  zero missing rows.
- **The observed trend is a modest, monotone deterioration.** 2008–2012 → 2018–2022:
  PS 0 **falls** 42.0% → 37.1%, PS ≥ 2 rises 18.9% → 22.9%, mean 0.84 → 0.95.
- **The imputed column reverses the PS 0 trend**, showing it *rising* 21.9% → 28.5%.
  This is an artefact of coverage, not a finding: only 31% of 2008 line-1 patients have
  a real measurement within ±90 days, rising to 80% by 2022. The imputation fills the
  gap toward the mode (PS 1 is 67% of the imputed early era vs 41% observed), so as real
  data arrives the imputed distribution drifts toward the truth — and that drift looks
  like a trend.
- Where an observed value exists the imputation is decent — **93.5% exact agreement** —
  but it errs toward the mode: 20.0% of observed PS 4 and 12.6% of observed PS 3 are
  imputed as PS 1. So it compresses the impaired tail specifically.
- 2023 is excluded (only 20 line-1 patients).

## Reading the line-1 survival figures

- Time is months from the line-1 start date, the event is `Y_global_death_status`,
  and follow-up is administratively censored at the database lock (2024-03).
- Each curve stops at its own cohort's last follow-up rather than being drawn flat
  to the horizon, so recent-year cohorts are short by construction.
- All 11 treatment categories are in scope. In the per-year figures a curve is drawn
  only for a (treatment × year) cell of at least `MIN_N` = 20 patients; CT+ANTI-HER2,
  CT+IT and CT+TT never reach that in any single year and appear only in the pooled
  adjusted figure. The scripts print the full coverage table.
- **Case mix moves with calendar time.** ET+ANTI-CDK only enters practice from 2017
  and CT+ANTI-ANGIO all but disappears after 2015, so a within-category shift across
  years reflects both changing practice and a changing patient mix — the crude curves
  are not treatment-effect estimates.

## What the adjustment does, and what it cannot do

`km_line1_adjusted.py` corrects two separate problems:

- **Heterogeneity (confounding by indication)** — stabilised inverse-probability-of-
  treatment weights from a multinomial-logistic propensity model over all 11
  categories and 80 covariates (clinical baseline features + calendar-year dummies),
  re-weighting each treatment group to the covariate distribution of the whole
  line-1 cohort.
- **Censoring bias** — censoring here is essentially administrative (observed
  follow-up is 92–94% of the maximum possible given entry date), so follow-up length
  is set by entry year, which is also what determines treatment availability. Plain
  KM's independent-censoring assumption fails within a curve that pools entry years.
  Corrected with inverse-probability-of-censoring weights from a Cox model for the
  censoring hazard given the same covariates plus treatment.

Both weights are stabilised and truncated, and each curve stops once the effective
sample size of its risk set drops below `ESS_FLOOR`.

**Check the diagnostics before trusting a curve.** Weighting cannot manufacture
overlap that does not exist. Balance is achieved for the categories present across
the whole period (max |SMD| ≤ 0.07 for ET alone, POLYCT, MONOCT, OTHER) but *fails*
for the era-confined ones — the residual imbalance for ET+ANTI-CDK (0.42) and
CT+ANTI-ANGIO (0.38) sits on **year dummies**, a structural positivity violation: no
patient treated before 2017 could have received ET+ANTI-CDK. Those curves are drawn
dashed and should be read as extrapolation, not as an estimate. Age is not present
in this parquet and is therefore absent from the propensity model — a known
unmeasured confounder for the endocrine-versus-chemotherapy choice.
