# data_analysis

Standalone exploratory-analysis scripts for the ESME cohorts. Each script reads a
parquet from the (git-ignored) `data/` directory at the repo root and writes its
figure(s) into `plots/`.

## Convention

- One script per analysis, runnable directly from the repo root:
  ```bash
  python data_analysis/<script>.py
  ```
- Figures are written to `data_analysis/plots/` (git-ignored `*.png`; regenerate by
  re-running the script).
- Scripts resolve the repo root relative to their own location, so they work whether
  run from the repo root or elsewhere.

## Scripts

| Script | What it produces |
|--------|------------------|
| `treatment_category_repeats.py` | Consecutive repetition of `T_treatment_category` over the first 4 treatment lines (HR+HER2−): repeat blocks per patient, and repeats per category broken down by run length. → `plots/treatment_category_repeats.png` |
