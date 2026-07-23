# Analysis — confounding diagnostics

`confounding_adjusted.py` measures **which patient variables confound the treatment
decision** in the HR+/HER2− V2 cohort, separately for each treatment line.

## The idea

A variable confounds a treatment → survival comparison only if it affects **both**:

1. which treatment the patient receives, and
2. how long the patient survives.

So for every covariate we compute two numbers — one per arrow — and multiply them
into a single confounding score. A variable that scores high on both is what biases
a naive treatment-effect estimate.

## How the two numbers are computed

Both are **adjusted**: each covariate is measured *after accounting for all the
others*, so correlated variables (e.g. two metastasis counts) don't get double credit.

- **Association with treatment** — fit a multinomial model that predicts the treatment
  arm from every covariate. Drop one covariate, refit, and see how much worse the fit
  gets (a likelihood-ratio). That loss, as a fraction of the model's total, is the
  covariate's adjusted contribution — a *partial pseudo-R²*.
- **Association with survival** — the same drop-one procedure on a Cox model (survival
  timed from the start of each line).

## The bias check

The scores rank variables but aren't in units of bias. So for one concrete comparison
(**ET+ANTI-CDK vs ET alone**) we also compute the **change-in-estimate**: fit the
treatment effect (a Cox hazard ratio), then add each covariate one at a time and record
how far the effect moves. A large shift = a strong confounder for that comparison.

## Run

```bash
cd Analysis
python confounding_adjusted.py          # all 4 lines (~2 min)
LINES=1 python confounding_adjusted.py  # one line, faster
```

Needs `pandas numpy plotly lifelines scikit-learn`. Reads the V2 parquet from `../data/`
(override with `DATA_PATH`). Writes to `figures/` (git-ignored):

- `confounding_quadrant_adjusted.html` — each covariate placed by treatment-association
  (x) vs survival-association (y); top-right = strongest confounders. One panel per line.
- `change_in_estimate_line1.html` — how much each covariate shifts the example effect.
- `confounding_scores_adjusted.csv` — the numbers behind the quadrant.
