"""Generate one semi-synthetic replicate on disk.

    out_dir/
      model_entry_imputed_data_{subtype}_stable_types_categorized_V2.parquet   expanded
      model_entry_imputes_data_STATIC_no_staging.parquet                       re-keyed
      truth.parquet    one row per sample: propensities, eta, latent times, censoring
      manifest.json    every parameter and every calibrated / achieved quantity

The two model files carry the real V2 names, so `out_dir` can be handed to the
datamodule as `data_dir`. The hidden confounder u appears only in `truth.parquet`.

Each stage draws from its own child of one SeedSequence keyed by (seed, replicate), so a
sweep over gamma / strength / heterogeneity reuses the same u, the same arm uniforms,
the same latent uniforms and the same dropout draws: levels differ only by the knob.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from CausalSurv.semisynthetic.assignment import Assignment, assign
from CausalSurv.semisynthetic.censoring import Censoring, simulate_censoring
from CausalSurv.semisynthetic.config import ASSIGNMENT_DRIVERS, DGPConfig
from CausalSurv.semisynthetic.drivers import (
    LINE,
    PAT_ID,
    build_drivers,
    load_cohort,
    sample_hidden_confounder,
)
from CausalSurv.semisynthetic.expand import expand_prefixes, expand_static
from CausalSurv.semisynthetic.outcome import Outcomes, simulate_outcomes

STATIC_FILE = "model_entry_imputes_data_STATIC_no_staging.parquet"


def dynamic_file(subtype: str) -> str:
    return f"model_entry_imputed_data_{subtype}_stable_types_categorized_V2.parquet"


def with_knobs(
    cfg: DGPConfig,
    gamma: float | None = None,
    strength: float | None = None,
    heterogeneity: float | None = None,
) -> DGPConfig:
    """The three sweep axes, applied to a loaded config."""
    if gamma is not None:
        cfg = dataclasses.replace(
            cfg, assignment=dataclasses.replace(cfg.assignment, gamma=gamma)
        )
    if strength is not None:
        cfg = dataclasses.replace(
            cfg,
            hidden_confounder=dataclasses.replace(
                cfg.hidden_confounder, strength=strength
            ),
        )
    if heterogeneity is not None:
        cfg = dataclasses.replace(
            cfg, outcome=dataclasses.replace(cfg.outcome, heterogeneity=heterogeneity)
        )
    return cfg


def _truth_frame(
    cohort: pd.DataFrame,
    drivers: pd.DataFrame,
    u: np.ndarray,
    assignment: Assignment,
    outcomes: Outcomes,
    censoring: Censoring,
    cfg: DGPConfig,
) -> pd.DataFrame:
    n = len(cohort)
    truth = {
        PAT_ID: cohort[PAT_ID].to_numpy() * 10 + cohort[LINE].to_numpy().astype(int),
        "orig_usubjid": cohort[PAT_ID].to_numpy(),
        "lineid": cohort[LINE].to_numpy(),
        "arm": assignment.arm_name,
        "arm_idx": assignment.arm_idx,
        "hidden_u": u,
        "weibull_k": outcomes.row_shape,
        "weibull_lambda": outcomes.row_scale,
        "latent_time": outcomes.times[np.arange(n), assignment.arm_idx],
        "admin_censor": censoring.administrative,
        "dropout_censor": censoring.dropout,
        "obs_time": censoring.time,
        "event": censoring.event,
    }
    for a, arm in enumerate(cfg.arms):
        truth[f"pi__{arm}"] = assignment.pi[:, a]
        truth[f"eta__{arm}"] = outcomes.eta[:, a]
        truth[f"latent_time__{arm}"] = outcomes.times[:, a]
    truth = pd.DataFrame(truth)
    z = drivers[list(ASSIGNMENT_DRIVERS)].add_prefix("z__")
    others = drivers.drop(columns=list(ASSIGNMENT_DRIVERS)).add_prefix("z__")
    return pd.concat([truth, z, others], axis=1)


def _effective_sample_size(assignment: Assignment, lines: np.ndarray) -> list[float]:
    """ESS/n of the inverse-propensity weights of the factual arm, per line."""
    p = assignment.pi[np.arange(len(lines)), assignment.arm_idx]
    out = []
    for line in np.unique(lines):
        w = 1.0 / p[lines == line]
        out.append(float(w.sum() ** 2 / (w**2).sum() / len(w)))
    return out


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest(
    cfg: DGPConfig,
    replicate: int,
    cohort: pd.DataFrame,
    expanded: pd.DataFrame,
    assignment: Assignment,
    outcomes: Outcomes,
    censoring: Censoring,
    scaling: dict[str, tuple[float, float]],
) -> dict:
    lines = cohort[LINE].to_numpy().astype(int)
    n_lines = cfg.cohort.n_lines
    arm_counts = [
        np.bincount(assignment.arm_idx[lines == line], minlength=len(cfg.arms)).tolist()
        for line in range(1, n_lines + 1)
    ]
    fit = outcomes.fit
    return {
        "seed": cfg.seed,
        "replicate": replicate,
        "git_hash": _git_hash(),
        "config": dataclasses.asdict(cfg),
        "arms": list(cfg.arms),
        "n_samples": len(cohort),
        "n_expanded_rows": len(expanded),
        "samples_per_line": np.bincount(lines, minlength=n_lines + 1)[1:].tolist(),
        "driver_scaling": {k: list(v) for k, v in scaling.items()},
        "assignment": {
            "alpha": assignment.alpha.tolist(),
            "arm_counts_per_line": arm_counts,
            "ess_over_n_per_line": _effective_sample_size(assignment, lines),
        },
        "outcome": {
            "weibull_shape": fit.shape.tolist(),
            "weibull_scale": fit.scale.tolist(),
            "median_fitted": fit.achieved_median.tolist(),
            "median_real_km": fit.real_median.tolist(),
        },
        "censoring": {
            "dropout_rate_per_month": censoring.rate.tolist(),
            "event_rate_achieved": censoring.achieved_rate.tolist(),
            "event_rate_real": censoring.target_rate.tolist(),
            "event_rate_cutoff_only": censoring.cutoff_only_rate.tolist(),
        },
    }


def generate(cfg: DGPConfig, out_dir: str | Path, replicate: int = 0) -> dict:
    """Simulate one replicate into `out_dir` and return its manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hidden_rng, assign_rng, outcome_rng, censor_rng = (
        np.random.default_rng(child)
        for child in np.random.SeedSequence([cfg.seed, replicate]).spawn(4)
    )

    cohort = load_cohort(cfg.cohort)
    drivers, scaling = build_drivers(cohort, cfg.cohort.cohort_start_year)
    u = sample_hidden_confounder(
        drivers, cfg.hidden_confounder.liver_correlation, hidden_rng
    )

    assignment = assign(cohort, drivers, u, cfg, assign_rng)
    outcomes = simulate_outcomes(
        cohort, drivers, u, assignment.arm_idx, cfg, outcome_rng
    )
    factual = outcomes.times[np.arange(len(cohort)), assignment.arm_idx]
    censoring = simulate_censoring(cohort, factual, cfg, censor_rng)

    expanded = expand_prefixes(
        cohort, assignment.arm_name, censoring.time, censoring.event
    )
    static = pd.read_parquet(Path(cfg.cohort.data_dir) / STATIC_FILE)

    expanded.to_parquet(out_dir / dynamic_file(cfg.cohort.subtype))
    expand_static(static, expanded).to_parquet(out_dir / STATIC_FILE)
    _truth_frame(cohort, drivers, u, assignment, outcomes, censoring, cfg).to_parquet(
        out_dir / "truth.parquet"
    )

    manifest = _manifest(
        cfg, replicate, cohort, expanded, assignment, outcomes, censoring, scaling
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
