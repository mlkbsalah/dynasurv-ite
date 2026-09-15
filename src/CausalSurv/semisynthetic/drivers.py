"""Real-cohort loading and the driver matrix z_j for every (patient, line).

The cohort is built exactly as `ESMEOnlineDataModuleCV._load_data` builds it
(patient-level first-line-year filter, inner join with the static file, lines
<= n_lines), so the simulated rows line up one-to-one with what the model trains on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from CausalSurv.semisynthetic.config import ALL_DRIVERS, CohortConfig

PAT_ID = "usubjid"
LINE = "lineid"
ARM_COL = "T_treatment_category"
START_COL = "line_start_date"

# Static features never reach the model (`_init_lstm_states` returns zeros), so any
# static driver must be copied into a dynamic X_ column or it becomes an unplanned
# hidden confounder.
STATIC_COPIES = {
    "X_slct_age_static": "X_age_at_selection",
    "X_menopause_static": "X_menopause",
}

# Drivers that are z-scored; the rest are 0/1 flags kept on their natural scale.
CONTINUOUS = ("mpps", "log_prev_line", "cum_new_sites", "age", "calendar")


def load_cohort(cfg: CohortConfig) -> pd.DataFrame:
    """Dynamic rows of the modelled cohort with the static copies added."""
    data_dir = Path(cfg.data_dir)
    dynamic = pd.read_parquet(
        data_dir
        / f"model_entry_imputed_data_{cfg.subtype}_stable_types_categorized_V2.parquet"
    )
    static = pd.read_parquet(
        data_dir / "model_entry_imputes_data_STATIC_no_staging.parquet"
    )

    entry_year = dynamic.groupby(PAT_ID)[START_COL].min().dt.year
    keep = entry_year[entry_year >= cfg.cohort_start_year].index
    dynamic = dynamic[dynamic[PAT_ID].isin(keep) & (dynamic[LINE] <= cfg.n_lines)]

    static = static[[PAT_ID, *STATIC_COPIES]].rename(columns=STATIC_COPIES)
    cohort = dynamic.merge(static, on=PAT_ID, how="inner")
    # V2 carries one exact duplicate (patient, line) row; left in, it would become its
    # own prefix sample and hand the second copy a fake "previous line".
    cohort = cohort.drop_duplicates([PAT_ID, LINE], keep="first")
    return cohort.sort_values([PAT_ID, LINE], kind="stable").reset_index(drop=True)


def previous_arm_class(cohort: pd.DataFrame) -> pd.DataFrame:
    """One-hot class (CDK / ET / CT) of the REAL treatment at the previous line.

    All zeros at line 1 and after OTHER / NO TREATMENT. Relies on `cohort` being
    sorted by patient then line, as `load_cohort` returns it.
    """
    prev = cohort.groupby(PAT_ID)[ARM_COL].shift(1).fillna("")
    cdk = prev.str.contains("CDK")
    return pd.DataFrame(
        {
            "prev_cdk": cdk,
            "prev_et": prev.str.startswith("ET") & ~cdk,
            "prev_ct": prev.str.contains("CT") & ~prev.str.startswith("ET"),
        },
        index=cohort.index,
    ).astype(float)


def build_drivers(
    cohort: pd.DataFrame, cohort_start_year: int
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Driver matrix aligned with `cohort` rows, plus (mean, std) of scaled drivers."""
    visceral = (cohort["X_pulmonary_met_count"] + cohort["X_pleural_met_count"]) > 0
    liver = cohort["X_liver_met_count"] > 0
    brain = cohort["X_brain_met_count"] > 0
    origin = pd.Timestamp(year=cohort_start_year, month=1, day=1)

    raw = pd.DataFrame(
        {
            "liver": liver,
            "visceral": visceral,
            "bone_only": (cohort["X_bone_met_count"] > 0) & ~liver & ~visceral & ~brain,
            "mpps": cohort["X_mpps"],
            "log_prev_line": np.log1p(cohort["X_time_between_onsets"]),
            "cum_new_sites": cohort["X_cumulative_new_metastatic_sites"],
            "age": cohort["X_age_at_selection"],
            "menopause": cohort["X_menopause"],
            # Same formula as ESMEOnlineDataModuleCV._add_calendar_feature.
            "calendar": (cohort[START_COL] - origin).dt.days / 30.44,
            "brain": brain,
            "lobular": cohort["X_hist_invasive_lobular"],
            "brca": (cohort["X_BRCA1"] + cohort["X_BRCA2"]) > 0,
            "old_site_progression": cohort["X_progression_of_old_site_count"] > 0,
        },
        index=cohort.index,
    ).astype(float)
    raw = raw.join(previous_arm_class(cohort))

    scaling = {}
    for name in CONTINUOUS:
        mean, std = float(raw[name].mean()), float(raw[name].std())
        raw[name] = (raw[name] - mean) / std
        scaling[name] = (mean, std)

    return raw[list(ALL_DRIVERS)], scaling


def sample_hidden_confounder(
    drivers: pd.DataFrame, liver_correlation: float, rng: np.random.Generator
) -> np.ndarray:
    """Standard-normal latent u with corr(u, liver) ~= `liver_correlation`."""
    liver = drivers["liver"].to_numpy()
    liver_std = (liver - liver.mean()) / liver.std()
    noise = rng.standard_normal(len(drivers))
    return liver_correlation * liver_std + np.sqrt(1 - liver_correlation**2) * noise
