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
| `treatment_category_repeats.py` | Interactive Plotly figure: repetition of `T_treatment_category` over the first 4 treatment lines (HR+HER2−), counting both **consecutive** repeats (same category in the next line) and **non-consecutive** ones (the category returns after a switch) — patients by repetition kind, repetitions per category split by kind, and all 15 sequence patterns four lines can take. Hover any bar/segment for counts, run lengths and gap lengths. → `plots/treatment_category_repeats.html` |
| `km_line1_by_year.py` | Crude Kaplan-Meier overall survival from line-1 onset, crossed by treatment category and calendar year of onset, shown three ways: one panel per year stratified by treatment; one panel per treatment stratified by year (light = early → dark = recent); and both collapsed to 24-month OS by year. Clicking a series in any legend isolates it across every panel. → `plots/km_line1_panel_per_year.html`, `plots/km_line1_panel_per_category.html`, `plots/km_line1_24mo_trend.html` |
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
