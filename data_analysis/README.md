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
| `treatment_category_repeats.py` | Interactive Plotly figure: consecutive repetition of `T_treatment_category` over the first 4 treatment lines (HR+HER2−) — repeat blocks per patient, and repeats per category broken down by run length. Hover any bar/segment for counts and percentages. → `plots/treatment_category_repeats.html` |
| `km_line1_by_year.py` | Kaplan-Meier overall survival measured from line-1 onset, split by treatment category and by calendar year of onset — does a category's survival trend shift over the years? Small multiples (one panel per category, one curve per year, light = early → dark = recent; click a year in the legend to isolate it in every panel) plus a milestone view of 24-month OS by year. → `plots/km_line1_by_year.html`, `plots/km_line1_24mo_trend.html` |

## Reading the line-1 survival figures

- Time is months from the line-1 start date, the event is `Y_global_death_status`,
  and follow-up is administratively censored at the database lock (2024-03).
- Each curve stops at its own cohort's last follow-up rather than being drawn flat
  to the horizon, so recent-year cohorts are short by construction.
- Only year-cohorts with at least 40 patients are drawn (`MIN_N`).
- **Case mix moves with calendar time.** ET+ANTI-CDK only enters practice from 2017
  and CT+ANTI-ANGIO all but disappears after 2015, so a within-category shift across
  years reflects both changing practice and a changing patient mix — it is not a
  treatment-effect estimate.
