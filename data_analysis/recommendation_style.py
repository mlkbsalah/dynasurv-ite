"""Shared vocabulary of the recommendation figures: arm order, hues, hatching.

`recommendation_mix.py` and `recommendation_horizon.py` read the reports that
`scripts/RecommendEnsemble.py` and `scripts/HorizonRobustness.py` write under
`reports/recommendations/`. Both need the same arm -> colour binding so a
treatment keeps its hue across figures, and across the KM figures, which already
bind ET / ET+CDK4/6 / CT to the same palette slots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from km_line1_by_year import CAT_COLORS, MUTED

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports" / "recommendations"
DEFAULT_COHORT = "HR+HER2-_4lines"

# The arms the action set can hold; everything else the clinicians gave is "other".
ARM_ORDER = ["ET alone", "ET+ANTI-CDK wo CT", "MONOCT std alone", "POLYCT alone"]
ARM_LABEL = {
    "ET alone": "ET",
    "ET+ANTI-CDK wo CT": "ET + CDK4/6",
    "MONOCT std alone": "mono-CT",
    "POLYCT alone": "poly-CT",
}
# Same slots as STRATUM_COLOR in km_later_lines_by_history: ET blue, CT orange,
# ET + CDK4/6 green. Poly-CT takes the purple slot, well apart from orange.
ARM_COLOR = {
    "ET alone": CAT_COLORS[0],
    "ET+ANTI-CDK wo CT": CAT_COLORS[2],
    "MONOCT std alone": CAT_COLORS[1],
    "POLYCT alone": CAT_COLORS[6],
}
OTHER_LABEL = "other"
OTHER_COLOR = MUTED

# Undecided = same hue, white hatching. Hatch colour follows edgecolor in
# matplotlib; its width is the rcParam below, set after use_paper_style().
HATCH = dict(hatch="///", edgecolor="white", linewidth=0)
HATCH_LINEWIDTH = 0.6
SOLID = dict(edgecolor="white", linewidth=0.4)


def newest_report(
    kind: str = "bestCALIB", *, horizons: bool = False, cohort: str = DEFAULT_COHORT
) -> Path:
    """Newest `<stamp>_<kind>_M<M>` (or `..._horizons`) report of the cohort.

    Filtered by checkpoint kind: the newest directory overall may be a different
    kind, and a horizon sweep must not be mistaken for a plain report.
    """
    hits = sorted(
        p
        for p in (REPORTS_DIR / cohort).glob(f"*_{kind}_M*")
        if p.is_dir() and p.name.endswith("_horizons") == horizons
    )
    if not hits:
        raise FileNotFoundError(
            f"no {'horizon sweep' if horizons else 'report'} of kind {kind!r} under "
            f"{REPORTS_DIR / cohort}; run scripts/"
            f"{'HorizonRobustness' if horizons else 'RecommendEnsemble'}.py first"
        )
    return hits[-1]


def leader_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (patient, line) from the long per-arm table.

    Adds `leader_name` and `gap`, the pessimistic deficit of the closest rival
    (the smallest `diff_lcb` over supported non-leader arms): +inf when the leader
    is the only supported arm, NaN when nothing is supported. This is the same
    quantity as `RecommendationSummary.runner_up_gap`.
    """
    keys = ["patient_id", "line"]
    per_line = [
        "leader_idx",
        "decision",
        "p_best_leader",
        "set_size",
        "recommended_idx",
        "observed_idx",
        "observed_name",
    ]
    table = df.groupby(keys, sort=True)[per_line].first()
    names = dict(zip(df.arm_idx, df.arm_name))
    table["leader_name"] = table.leader_idx.map(names)
    rivals = df[df.supported & ~df.is_leader]
    gap = rivals.groupby(keys).diff_lcb.min().reindex(table.index)
    supported = table.leader_idx >= 0
    table["gap"] = np.where(supported, gap.fillna(np.inf), np.nan)
    return table.reset_index()
