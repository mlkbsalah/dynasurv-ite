"""Typed configuration for DynaSurv.

Every value that used to be typed by hand at each construction site lives here
instead. Six scripts each spelled out their own subset of the model's 40
constructor arguments, and the subsets had drifted: `best_config.toml` advertised
six dropout values that `TrainDynasurvCausal.py` never passed, so every run
trained at dropout 0 while the file said otherwise.

The loaders are therefore **strict** -- an unrecognised key raises and names
itself rather than being ignored. A permissive loader cannot tell a typo from a
deliberate default, which is precisely how the dropouts were lost.

Two files feed a run:

- `configs/config.toml`  -- hand-authored, stays TOML because its comments
  justifying the identifiability settings are load-bearing.
- `configs/best_config.json` -- machine-written by the Optuna study. JSON so it
  round-trips through `dataclasses.asdict` and cannot drift from the schema.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

import tomllib

T = TypeVar("T", bound="StrictConfig")


class StrictConfig:
    """Frozen-dataclass mixin providing a strict loader and list->tuple coercion.

    `from_dict` rejects unknown keys and reports missing required ones by name.
    `__post_init__` turns lists into tuples so the frozen dataclasses are
    genuinely immutable -- TOML and JSON both hand back lists.
    """

    def __post_init__(self) -> None:
        for f in fields(self):  # type: ignore[arg-type]
            value = getattr(self, f.name)
            if isinstance(value, list):
                object.__setattr__(self, f.name, tuple(value))

    @classmethod
    def from_dict(cls: type[T], data: Mapping[str, Any]) -> T:
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unknown key(s) {unknown}. Known keys: {sorted(known)}"
            )
        required = {
            f.name
            for f in fields(cls)  # type: ignore[arg-type]
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{cls.__name__}: missing required key(s) {missing}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


# --------------------------------------------------------------------------- #
# Model configuration (configs/best_config.json)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArchConfig(StrictConfig):
    """Layer widths and regularisation -- everything that shapes the weights.

    Deliberately excludes the data-derived dimensions: those stay explicit
    constructor arguments on the model, because they are dictated by the
    datamodule rather than chosen here.

    Note on what is absent. `init_h_hidden`, `init_p_hidden`, `init_h_dropout`,
    `init_p_dropout`, `mlpp_hidden_units` and `mlpp_dropout` used to be accepted
    and reach nothing: the `init_h_mlp`/`init_p_mlp` projections are commented
    out in the model, and no `MLPp` is ever built (P goes through `nn.Embedding`).
    They are omitted here so the strict loader rejects them rather than letting
    a tuned-looking value sit in the config doing nothing.
    """

    lstm_hidden_length: int
    lstm_num_layers: int
    x_embed_dim: int
    p_embed_dim: int
    mlpx_hidden_units: tuple[int, ...]
    mlpsa_hidden_units: tuple[int, ...]
    mlpprop_hidden_units: tuple[int, ...]
    mlpx_dropout: float
    mlpsa_dropout: float
    mlpprop_dropout: float
    attention: bool


@dataclass(frozen=True)
class TrainingConfig(StrictConfig):
    """Optimisation and causal-balancing weights.

    Nothing here changes the shape of the weights, so a checkpoint stays loadable
    across any change to this object -- which is what makes a lambda ablation or
    an LR sweep safe to run against an existing architecture.
    """

    lr: float
    weight_decay: float
    lr_scheduler_stepsize: int
    lr_scheduler_gamma: float
    lambda_prop_loss: float = 0.0
    lambda_ipm_mmd: float = 0.0
    lambda_ipm_emd2: float = 0.0
    # Minimum per-batch samples in EACH treatment group for a pair to contribute
    # to the IPM penalty. Deliberately not derived from lstm_hidden_length: that
    # coupling made the threshold equal the batch size, so no pair ever qualified
    # and both IPM terms returned 0.0 at every lambda.
    min_ipm_group_size: int = 16


@dataclass(frozen=True)
class ModelConfigFile(StrictConfig):
    """The whole of `configs/best_config.json`.

    `n_intervals` and `batch_size` sit at the root rather than inside
    `ArchConfig` because both are Optuna-tuned but consumed by the *datamodule*.
    `n_intervals` comes back to the model as `output_length`; `batch_size` never
    reaches the model at all. Keeping them out of `ArchConfig` stops the model's
    config object carrying values the model never reads.
    """

    n_intervals: int
    batch_size: int
    arch: ArchConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfigFile":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unknown key(s) {unknown}. Known keys: {sorted(known)}"
            )
        for section in ("arch", "training"):
            if section not in data:
                raise ValueError(
                    f"{cls.__name__}: missing required section '{section}'"
                )
        return cls(
            n_intervals=data["n_intervals"],
            batch_size=data["batch_size"],
            arch=ArchConfig.from_dict(data["arch"]),
            training=TrainingConfig.from_dict(data["training"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_intervals": self.n_intervals,
            "batch_size": self.batch_size,
            "arch": _jsonable(self.arch.to_dict()),
            "training": _jsonable(self.training.to_dict()),
        }

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfigFile":
        with open(path, "rb") as f:
            return cls.from_dict(json.load(f))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _jsonable(value: Any) -> Any:
    """Tuples back to lists so the emitted JSON reloads through `from_dict`."""
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


# --------------------------------------------------------------------------- #
# Run configuration (configs/config.toml)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EvalConfig(StrictConfig):
    """Evaluation reporting grid.

    Consumed by the epoch-metric logging, not by the network. `horizon_times` is
    per line and its length must equal `n_lines`; `calibration_times` is applied
    to every line, and a landmark past a line's last observation is skipped
    rather than reported, so later lines simply log fewer of them.
    """

    horizon_times: tuple[float, ...]
    integration_step: int = 100
    calibration_times: tuple[float, ...] = (6.0, 12.0, 24.0, 36.0)


@dataclass(frozen=True)
class DataConfig(StrictConfig):
    """Cohort construction. Mirrors `ESMEOnlineDataModuleCV`'s keyword arguments.

    `batch_size` is absent on purpose -- it lives in the model config file next
    to `n_intervals`, because the Optuna study tunes it and both are datamodule
    inputs. Keeping one tuned value here and one there is what let trial 33's
    winning `batch_size = 64` be silently replaced by this file's 128.
    """

    data_dir: str
    subtype: str
    n_lines: int
    cohort_start_year: int | None = None
    temporal_split_year: int | None = None
    add_calendar_feature: bool = False
    excluded_treatment_arms: tuple[str, ...] | None = None
    excluded_x_columns: tuple[str, ...] | None = None
    min_samples_per_treatment: int = 200
    min_events_per_treatment: int = 0
    min_followup_samples_per_treatment: int = 0
    propensity_min_probability: float = 0.0
    propensity_cv_folds: int = 5


@dataclass(frozen=True)
class TrainerConfig(StrictConfig):
    max_epochs: int = 100
    accelerator: str = "cpu"
    gradient_clip_val: float = 0.0


@dataclass(frozen=True)
class EarlyStoppingConfig(StrictConfig):
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 10
    enabled: bool = True


@dataclass(frozen=True)
class TrainConfig(StrictConfig):
    trainer: TrainerConfig
    early_stopping: EarlyStoppingConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"{cls.__name__}: unknown key(s) {unknown}. Known keys: {sorted(known)}"
            )
        return cls(
            trainer=TrainerConfig.from_dict(data.get("trainer", {})),
            early_stopping=EarlyStoppingConfig.from_dict(
                data.get("early_stopping", {})
            ),
        )


# --------------------------------------------------------------------------- #
# Recommendation policy (configs/config.toml, [recommendation])
# --------------------------------------------------------------------------- #
# Which of a run's ModelCheckpoint files an ensemble member is taken from. Names
# follow the callbacks in scripts/TrainDynasurvCausal.py; `last.ckpt` is absent on
# purpose -- Lightning writes it as a copy of the val_loss winner, not the final
# epoch, which is `final_epoch`.
CHECKPOINT_KINDS = ("val_loss", "bestCI", "bestIBS", "bestCALIB", "final_epoch")
LEADER_RULES = ("lcb", "mean")


@dataclass(frozen=True)
class RecommendationConfig(StrictConfig):
    """Decision rule of the ensemble recommender (`recommendation.ensemble`).

    Read at recommendation time only, never by training. Two thresholds of
    different nature: `p_best_min` is statistical (how often the leader is the
    per-member winner), `margin_months` is clinical (how many months of RMST must
    separate two arms before the difference is worth acting on). `p_best` takes
    values in multiples of 1/M, so with three members 0.7 already means unanimity.
    """

    p_best_min: float = 0.7
    margin_months: float = 1.0
    pessimism_c: float = 1.0
    leader_rule: str = "lcb"
    checkpoint_kind: str = "val_loss"
    min_members: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.p_best_min <= 1.0:
            raise ValueError(f"p_best_min must lie in [0, 1], got {self.p_best_min}")
        if self.margin_months < 0:
            raise ValueError(f"margin_months must be >= 0, got {self.margin_months}")
        if self.pessimism_c < 0:
            raise ValueError(f"pessimism_c must be >= 0, got {self.pessimism_c}")
        if self.leader_rule not in LEADER_RULES:
            raise ValueError(
                f"leader_rule must be one of {LEADER_RULES}, got {self.leader_rule!r}"
            )
        if self.checkpoint_kind not in CHECKPOINT_KINDS:
            raise ValueError(
                f"checkpoint_kind must be one of {CHECKPOINT_KINDS}, "
                f"got {self.checkpoint_kind!r}"
            )
        if self.min_members < 1:
            raise ValueError(f"min_members must be >= 1, got {self.min_members}")


# --------------------------------------------------------------------------- #
# The composed experiment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExperimentConfig:
    """Everything a run needs, loaded from the two config files."""

    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    model: ModelConfigFile
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)

    def __post_init__(self) -> None:
        # The model raised this from its own constructor; it belongs here, where
        # both halves of the comparison are in scope at load time rather than
        # after the datamodule has already been built.
        if len(self.eval.horizon_times) != self.data.n_lines:
            raise ValueError(
                f"[eval] horizon_times has length {len(self.eval.horizon_times)} "
                f"but [data] n_lines={self.data.n_lines}; they must match, since "
                "each line is evaluated at its own horizon."
            )

    @classmethod
    def from_files(
        cls, run_config: str | Path, model_config: str | Path
    ) -> "ExperimentConfig":
        with open(run_config, "rb") as f:
            run = tomllib.load(f)

        known_tables = ("data", "eval", "recommendation", "train")
        unknown = sorted(set(run) - set(known_tables))
        if unknown:
            raise ValueError(
                f"{Path(run_config).name}: unknown top-level table(s) {unknown}. "
                f"Known tables: {list(known_tables)}"
            )

        return cls(
            data=DataConfig.from_dict(run["data"]),
            train=TrainConfig.from_dict(run.get("train", {})),
            eval=EvalConfig.from_dict(run["eval"]),
            model=ModelConfigFile.from_json(model_config),
            recommendation=RecommendationConfig.from_dict(
                run.get("recommendation", {})
            ),
        )

    # -- convenience accessors, so call sites stop reaching through three dots --
    @property
    def arch(self) -> ArchConfig:
        return self.model.arch

    @property
    def training(self) -> TrainingConfig:
        return self.model.training

    def datamodule_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `ESMEOnlineDataModuleCV`.

        Assembled in one place so the training script, the Optuna objective and
        the validation notebook cannot build three different cohorts -- which
        they currently do.
        """
        kwargs = self.data.to_dict()
        kwargs["excluded_treatment_arms"] = _as_list(self.data.excluded_treatment_arms)
        kwargs["excluded_x_columns"] = _as_list(self.data.excluded_x_columns)
        kwargs["n_intervals"] = self.model.n_intervals
        kwargs["batch_size"] = self.model.batch_size
        kwargs["evaluation_horizon_times"] = list(self.eval.horizon_times)
        return kwargs


def _as_list(value: tuple[Any, ...] | None) -> list[Any] | None:
    return None if value is None else list(value)
