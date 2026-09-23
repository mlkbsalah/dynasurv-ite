"""Anything that predicts every arm's survival curve, on one common time grid.

The evaluator (`evaluate.py`) never sees a model: it is handed a `Prediction`, a
(n_samples, n_arms, n_grid) array of S_a(t) on `t_grid`. Three producers live here --
the oracle (the true Weibull curves), the naive per-arm Kaplan-Meier (no
covariates; a comparator, not an error ceiling) and a trained-model adapter.

Arms are always in sorted order, as everywhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.utils.data as TorchData

from CausalSurv.evaluation.discrete_survival import kaplan_meier, survival_at
from CausalSurv.recommendation.recommender import TreatmentRecommender
from CausalSurv.semisynthetic.outcome import survival

ID = "usubjid"


@dataclass(frozen=True)
class Prediction:
    survival: np.ndarray  # (n, n_arms, n_grid), S(0) = 1
    # (n, n_arms) arms the predictor considers supported for that patient, or None when
    # it has no notion of patient-level support (then only line-level support applies).
    patient_support: np.ndarray | None = None


def truth_arrays(truth: pd.DataFrame, arms: tuple[str, ...]):
    """(eta (n, A), k (n,), lambda (n,)) of the rows of a truth frame."""
    eta = truth[[f"eta__{a}" for a in arms]].to_numpy()
    return eta, truth["weibull_k"].to_numpy(), truth["weibull_lambda"].to_numpy()


def oracle_prediction(
    truth: pd.DataFrame, arms: tuple[str, ...], t_grid: np.ndarray
) -> Prediction:
    """The true S_a(t | z, u), including the hidden confounder: nothing can beat it."""
    return Prediction(survival(t_grid, *truth_arrays(truth, arms)))


def _step(uniq: np.ndarray, surv: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.concatenate([[1.0], surv])[np.searchsorted(uniq, t_grid, side="right")]


def _km_curve(rows: pd.DataFrame, t_grid: np.ndarray) -> np.ndarray:
    times, events = rows["obs_time"].to_numpy(), rows["event"].to_numpy()
    return _step(*kaplan_meier(times, events), t_grid)


def km_prediction(
    train: pd.DataFrame,
    evaluated: pd.DataFrame,
    arms: tuple[str, ...],
    t_grid: np.ndarray,
) -> Prediction:
    """Per (line, arm) Kaplan-Meier of the training samples, the same curve for every
    patient. An arm unseen at a line falls back to the line's pooled Kaplan-Meier."""
    out = np.empty((len(evaluated), len(arms), len(t_grid)))
    for line in np.unique(evaluated["lineid"]):
        rows = train[train["lineid"] == line]
        pooled = _km_curve(rows, t_grid)
        target = (evaluated["lineid"] == line).to_numpy()
        for a, arm in enumerate(arms):
            r = rows[rows["arm"] == arm]
            out[target, a] = _km_curve(r, t_grid) if len(r) else pooled
    return Prediction(out)


def dynasurv_prediction(
    model,
    dataset: TorchData.Dataset,
    treatment_names: dict[int, str],
    arms: tuple[str, ...],
    t_grid: np.ndarray,
    batch_size: int = 512,
) -> tuple[Prediction, np.ndarray, np.ndarray]:
    """Counterfactual survival of the four arms at each sample's own (last) line.

    `treatment_names` is the datamodule's `treatment_dict` (model arm index -> name);
    the model was trained on more arms than the DGP has, and only the DGP's are read.

    Returns the prediction, the sample ids in dataset order, and the (n_lines, n_arms)
    line-level recommendable mask the model carries from training.
    """
    by_name = {name: k for k, name in treatment_names.items()}
    missing = [a for a in arms if a not in by_name]
    if missing:
        raise ValueError(f"arms {missing} are not in the datamodule's treatment_dict")
    arm_index = torch.as_tensor([by_name[a] for a in arms])

    recommender = TreatmentRecommender.from_model(model)
    line_support = recommender.action_mask()[:, arm_index].numpy()
    grid = torch.as_tensor(t_grid, dtype=model.interval_bounds.dtype)

    model.eval()
    curves, supports, ids = [], [], []
    with torch.no_grad():
        for batch in TorchData.DataLoader(dataset, batch_size=batch_size):
            xpd, (x_static, p_static), _, treatment_idx, _, _, mask, patient_id = batch
            rows = torch.arange(len(xpd))
            line = mask.long().argmax(
                dim=1
            )  # the one unmasked line of an expanded sample
            survival_all, support = recommender.arm_survival(
                xpd, (x_static, p_static), treatment_idx
            )
            s = survival_all[rows, line][:, arm_index]  # (B, A, G + 1)
            b, a, g = s.shape
            on_grid = survival_at(
                s.reshape(b * a, g), grid, model.interval_bounds
            ).reshape(b, a, -1)
            curves.append(on_grid.numpy())
            supports.append(support[rows, line][:, arm_index].numpy())
            ids.append(patient_id.numpy())
    prediction = Prediction(np.concatenate(curves), np.concatenate(supports))
    return prediction, np.concatenate(ids), line_support
