"""Typed, strict configuration for the semi-synthetic data-generating process.

Same contract as `CausalSurv.config`: an unknown key raises and names itself. On
top of that, `DGPConfig` checks the *causal roles* of the drivers -- an instrument
given an outcome coefficient, or a prognostic factor given an assignment
coefficient, silently changes what the benchmark measures, so it is rejected at
load time rather than discovered in a flat sweep curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import tomllib

from CausalSurv.config import StrictConfig

# Driver name -> role. Confounders drive both assignment and outcome, instruments
# only assignment, prognostic factors only outcome. Columns are built in drivers.py.
CONFOUNDERS = (
    "liver",
    "visceral",
    "bone_only",
    "mpps",
    "log_prev_line",
    "prev_cdk",
    "prev_et",
    "prev_ct",
    "cum_new_sites",
    "age",
    "menopause",
)
INSTRUMENTS = ("calendar",)
PROGNOSTIC = ("brain", "lobular", "brca", "old_site_progression")

ASSIGNMENT_DRIVERS = CONFOUNDERS + INSTRUMENTS
OUTCOME_DRIVERS = CONFOUNDERS + PROGNOSTIC
ALL_DRIVERS = CONFOUNDERS + INSTRUMENTS + PROGNOSTIC


@dataclass(frozen=True)
class CohortConfig(StrictConfig):
    data_dir: str
    subtype: str
    n_lines: int
    cohort_start_year: int
    data_cutoff: str
    # Order is irrelevant here; arm indices are always taken from sorted(arms),
    # which matches the datamodule's alphabetical one-hot.
    arms: tuple[str, ...]


@dataclass(frozen=True)
class AssignmentConfig(StrictConfig):
    """pi_a = softmax_a(alpha_{line,a} + gamma * sum_d B[a][d] * z_d)."""

    gamma: float
    coefficients: dict[str, dict[str, float]]


@dataclass(frozen=True)
class OutcomeConfig(StrictConfig):
    """eta_a = f(z) + tau_a + heterogeneity * g_a(z); positive eta = higher hazard."""

    heterogeneity: float
    arm_effects: dict[str, float]
    prognostic: dict[str, float]
    interactions: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class HiddenConfounderConfig(StrictConfig):
    """Latent u, never written to the model's input.

    `outcome` is fixed across a sweep and `strength` scales only the u ->
    assignment coupling, so strength 0 is an unconfounded reference that carries
    the same outcome heterogeneity; any error growth above it is confounding bias.
    """

    strength: float = 0.0
    liver_correlation: float = 0.3
    assignment: dict[str, float] = field(default_factory=dict)
    outcome: float = 0.0


@dataclass(frozen=True)
class CensoringConfig(StrictConfig):
    # Add exponential dropout on top of administrative censoring when the cutoff
    # alone leaves the event rate above the real one.
    dropout: bool = True


_SECTIONS = {
    "cohort": CohortConfig,
    "assignment": AssignmentConfig,
    "outcome": OutcomeConfig,
    "hidden_confounder": HiddenConfounderConfig,
    "censoring": CensoringConfig,
}


@dataclass(frozen=True)
class DGPConfig:
    seed: int
    cohort: CohortConfig
    assignment: AssignmentConfig
    outcome: OutcomeConfig
    hidden_confounder: HiddenConfounderConfig
    censoring: CensoringConfig

    def __post_init__(self) -> None:
        arms = set(self.cohort.arms)
        if len(arms) != len(self.cohort.arms):
            raise ValueError(f"[cohort] arms has duplicates: {self.cohort.arms}")

        _check_arms("[assignment] coefficients", self.assignment.coefficients, arms)
        _check_arms("[outcome] interactions", self.outcome.interactions, arms)
        _check_arms(
            "[hidden_confounder] assignment", self.hidden_confounder.assignment, arms
        )
        _check_arms("[outcome] arm_effects", self.outcome.arm_effects, arms)
        missing = sorted(arms - set(self.outcome.arm_effects))
        if missing:
            raise ValueError(f"[outcome] arm_effects missing arm(s) {missing}")

        for arm, coefs in self.assignment.coefficients.items():
            _check_drivers(
                f"[assignment] coefficients.{arm}", coefs, ASSIGNMENT_DRIVERS
            )
        _check_drivers("[outcome] prognostic", self.outcome.prognostic, OUTCOME_DRIVERS)
        for arm, coefs in self.outcome.interactions.items():
            _check_drivers(f"[outcome] interactions.{arm}", coefs, OUTCOME_DRIVERS)

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted(self.cohort.arms))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DGPConfig":
        unknown = sorted(set(data) - set(_SECTIONS) - {"seed"})
        if unknown:
            raise ValueError(
                f"DGPConfig: unknown key(s) {unknown}. "
                f"Known: {sorted(set(_SECTIONS) | {'seed'})}"
            )
        if "seed" not in data:
            raise ValueError("DGPConfig: missing required key 'seed'")
        sections = {
            name: section.from_dict(data.get(name, {}))
            for name, section in _SECTIONS.items()
        }
        return cls(seed=data["seed"], **sections)

    @classmethod
    def from_file(cls, path: str | Path) -> "DGPConfig":
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))


def _check_arms(where: str, mapping: Mapping[str, Any], arms: set[str]) -> None:
    unknown = sorted(set(mapping) - arms)
    if unknown:
        raise ValueError(f"{where}: unknown arm(s) {unknown}; arms are {sorted(arms)}")


def _check_drivers(
    where: str, coefs: Mapping[str, float], allowed: tuple[str, ...]
) -> None:
    bad = sorted(set(coefs) - set(allowed))
    if not bad:
        return
    wrong_role = [d for d in bad if d in ALL_DRIVERS]
    if wrong_role:
        raise ValueError(
            f"{where}: driver(s) {wrong_role} are not allowed in this equation "
            f"(role violation). Allowed: {list(allowed)}"
        )
    raise ValueError(f"{where}: unknown driver(s) {bad}. Known: {list(ALL_DRIVERS)}")
