"""Prefix expansion of the cohort into the V2 schema the datamodule reads.

A patient with L lines becomes L samples. Sample j holds the patient's real rows for
lines 1..j-1 (real covariates, real past treatments) plus row j, whose treatment is the
simulated arm and whose outcome is the simulated one. The model steps through lines
unidirectionally and never sees the current line's arm before line j, so a sample can be
scored on its last line alone and `forward()` and the losses stay untouched. The
datamodule masks every row except the one with `lineid == prefix_line`.

Sample ids are `orig_usubjid * 10 + j`; all samples of a patient share `orig_usubjid`
(the unit for any train/validation grouping) and the patient's entry year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from CausalSurv.semisynthetic.drivers import ARM_COL, LINE, PAT_ID

ORIG_ID = "orig_usubjid"
PREFIX_LINE = "prefix_line"
TIME_COL = "Y_onset_to_death"
EVENT_COL = "Y_global_death_status"

# Real current-line information that no longer describes the simulated line: drug
# flags and the real outcome columns. The model reads only T_treatment_category and
# the two targets, so anything else would just be a real value sitting next to a
# simulated one.
KEEP = {ARM_COL, TIME_COL, EVENT_COL}
OUTCOME_DERIVED = ("line_end_date", "death_date")


def _drop_stale_columns(frame: pd.DataFrame) -> pd.DataFrame:
    stale = [
        c
        for c in frame.columns
        if (c.startswith(("T_", "Y_")) and c not in KEEP) or c in OUTCOME_DERIVED
    ]
    return frame.drop(columns=stale)


def expand_prefixes(
    cohort: pd.DataFrame,
    arm_name: np.ndarray,
    observed_time: np.ndarray,
    event: np.ndarray,
) -> pd.DataFrame:
    """Expanded dynamic frame. `cohort` must be sorted by patient then line with lines
    numbered 1..L, and the three arrays are aligned with its rows (one per sample)."""
    pos = cohort.groupby(PAT_ID).cumcount().to_numpy()
    lines = cohort[LINE].to_numpy().astype(int)
    if not (lines == pos + 1).all():
        raise ValueError("cohort lines must be contiguous 1..L within each patient")

    # Row i (line j of its patient) contributes rows i-j+1 .. i to its own sample.
    lengths = lines
    starts = np.arange(len(cohort)) - (lines - 1)
    within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    source = np.repeat(starts, lengths) + within

    frame = _drop_stale_columns(cohort).iloc[source].reset_index(drop=True)
    orig = np.repeat(cohort[PAT_ID].to_numpy(), lengths)
    prefix = np.repeat(lines, lengths)
    frame[ORIG_ID] = orig
    frame[PREFIX_LINE] = prefix
    frame[PAT_ID] = orig * 10 + prefix

    last = frame[LINE].to_numpy().astype(int) == prefix
    frame[TIME_COL] = 0.0
    frame[EVENT_COL] = 0.0
    frame.loc[last, ARM_COL] = arm_name
    frame.loc[last, TIME_COL] = observed_time
    frame.loc[last, EVENT_COL] = event
    return frame


def expand_static(static: pd.DataFrame, expanded: pd.DataFrame) -> pd.DataFrame:
    """One static row per sample: the patient's row re-keyed to the sample id."""
    keys = expanded[[PAT_ID, ORIG_ID]].drop_duplicates()
    out = keys.merge(
        static.rename(columns={PAT_ID: ORIG_ID}), on=ORIG_ID, how="left", validate="m:1"
    )
    return out.drop(columns=ORIG_ID).reset_index(drop=True)
