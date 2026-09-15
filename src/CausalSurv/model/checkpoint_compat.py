"""Load `DynaSurvCausalOnline` checkpoints written by either constructor.

`save_hyperparameters()` records the constructor signature, so a checkpoint
written before the config refactor carries 40 flat keys while the constructor now
expects grouped config objects. `Trainer`-agnostic loading therefore has to
translate rather than hand the saved dict straight back.

Checkpoints also carry a pickled `PropensityOverlapModel` holding an sklearn
estimator, so `weights_only=False` is mandatory and
`CausalSurv.evaluation.propensity_overlap` must stay importable at that exact
module path -- moving that file invalidates every existing checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import ArchConfig, EvalConfig, TrainingConfig
from .dynasurv_causal_online import DynaSurvCausalOnline

# Accepted by the old constructor and reaching nothing: the init_h_mlp/init_p_mlp
# projections were commented out, and no MLPp is ever built (P goes through an
# nn.Embedding). Dropped on load rather than carried forward.
_INERT_KEYS = (
    "init_h_hidden",
    "init_p_hidden",
    "init_h_dropout",
    "init_p_dropout",
    "mlpp_hidden_units",
    "mlpp_dropout",
)

_DIM_KEYS = (
    "x_input_dim",
    "x_static_dim",
    "p_input_dim",
    "p_static_dim",
    "n_treatments",
    "output_length",
    "interval_bounds",
    "n_lines",
)


def _is_legacy(hparams: dict[str, Any]) -> bool:
    return "arch" not in hparams


def _regroup(hparams: dict[str, Any]) -> dict[str, Any]:
    """Translate flat pre-refactor hyperparameters into constructor arguments."""
    kwargs = {k: hparams[k] for k in _DIM_KEYS if k in hparams}
    kwargs.setdefault("n_lines", len(hparams.get("evaluation_horizon_times", [])) or 4)

    kwargs["arch"] = ArchConfig(
        lstm_hidden_length=hparams["lstm_hidden_length"],
        lstm_num_layers=hparams.get("lstm_num_layers", 4),
        x_embed_dim=hparams["x_embed_dim"],
        p_embed_dim=hparams["p_embed_dim"],
        mlpx_hidden_units=tuple(hparams["mlpx_hidden_units"]),
        mlpsa_hidden_units=tuple(hparams["mlpsa_hidden_units"]),
        mlpprop_hidden_units=tuple(hparams["mlpprop_hidden_units"]),
        mlpx_dropout=hparams.get("mlpx_dropout", 0.0),
        mlpsa_dropout=hparams.get("mlpsa_dropout", 0.0),
        mlpprop_dropout=hparams.get("mlpprop_dropout", 0.0),
        attention=hparams.get("attention", True),
    )
    kwargs["training"] = TrainingConfig(
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        lr_scheduler_stepsize=hparams["lr_scheduler_stepsize"],
        lr_scheduler_gamma=hparams["lr_scheduler_gamma"],
        lambda_prop_loss=hparams.get("lambda_prop_loss", 0.0),
        lambda_ipm_mmd=hparams.get("lambda_ipm_mmd", 0.0),
        lambda_ipm_emd2=hparams.get("lambda_ipm_emd2", 0.0),
        min_ipm_group_size=hparams.get("min_ipm_group_size", 16),
    )
    kwargs["evaluation"] = EvalConfig(
        horizon_times=tuple(hparams["evaluation_horizon_times"]),
        integration_step=hparams.get("brier_integration_step", 100),
        calibration_times=tuple(hparams.get("calibration_times", (6, 12, 24, 36))),
    )
    return kwargs


def load_dynasurv_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> DynaSurvCausalOnline:
    """Rebuild a `DynaSurvCausalOnline` from a checkpoint of either vintage.

    Built directly and given its state dict rather than routed through
    `LightningModule.load_from_checkpoint`, which would feed the saved
    hyperparameters back as constructor arguments unchanged and fail on the
    legacy layout.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    if not hparams:
        raise ValueError(
            f"{path}: checkpoint carries no hyper_parameters; it cannot be rebuilt."
        )

    if _is_legacy(hparams):
        dropped = [k for k in _INERT_KEYS if k in hparams]
        kwargs = _regroup(hparams)
        if dropped:
            print(
                f"Loaded a pre-refactor checkpoint; dropped inert hyperparameters "
                f"{dropped} (they reached nothing in the model)."
            )
    else:
        kwargs = hparams

    model = DynaSurvCausalOnline(**kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)
    # Restores the pickled recommendation propensity model, which lives outside
    # the state dict.
    model.on_load_checkpoint(checkpoint)
    return model
