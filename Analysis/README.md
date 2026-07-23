# Analysis

Confounding-mechanism diagnostics for the HR+/HER2− V2 cohort.

## `confounding_adjusted.py`

Ranks each covariate by how strongly it confounds the treatment decision, with
**both associations adjusted for every other covariate** (a partial pseudo-R²
from a drop-one likelihood-ratio), plus a change-in-estimate in true bias units.

| Quantity | Model | Meaning |
|---|---|---|
| treatment axis | multinomial propensity (`sklearn`) | partial pseudo-R² of the covariate for the treatment arm |
| survival axis | Cox (`lifelines`) | partial pseudo-R² of the covariate for overall survival from that line's start |
| change-in-estimate | Cox, one arm contrast | % shift in the treatment log-HR when the covariate is added |

A variable confounds only when **both** axes are high; the change-in-estimate is
the only quantity in units of bias on the estimand.

### Run

```bash
cd Analysis
python confounding_adjusted.py          # all 4 lines
LINES=1 python confounding_adjusted.py  # one line, faster
```

Reads `../data/model_entry_imputed_data_HR+HER2-_stable_types_categorized_V2.parquet`
(override with `DATA_PATH`). Writes to `figures/` (git-ignored):

- `confounding_quadrant_adjusted.html` — interactive quadrant, faceted by line
- `change_in_estimate_line1.html` — interactive bias bar chart
- `confounding_scores_adjusted.csv` — the ranked scores

Requires `pandas numpy plotly lifelines scikit-learn`.
