"""RMST-based treatment recommendation over a trained model.

Lived on the model as `recommendable_mask`, `patient_support_mask` and
`recommend_treatment`. Recommendation is a policy built *on top of* the model's
counterfactual survival predictions -- it ranks arms, applies support masks and
decides when to abstain -- none of which is part of what the network computes,
so it moves out.

What stays on the model: `recommendation_propensity_model` and the checkpoint
hooks that save and restore it. The fitted propensity model is checkpoint
state, and `_setup_valid_treatments` is what populates it during fit/test.
`TreatmentRecommender.from_model` reads it from there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..evaluation.discrete_survival import rmst

if TYPE_CHECKING:
    from ..model.dynasurv_causal_online import DynaSurvCausalOnline

# Returned instead of an arm index when every arm at a line is unsupported for a
# patient. An abstention, never a treatment index: without it, argmax over an
# all -inf row would silently pick arm 0.
NO_SUPPORTED_ARM = -1


class TreatmentRecommender:
    """Ranks treatment arms by RMST and returns the best supported arm per line.

    Args:
        model: a trained `DynaSurvCausalOnline`.
        recommendable_treatments_per_line: `{line: [arm, ...]}` of arms eligible to
            be recommended at each line. `None` or empty means every arm is
            eligible, so the recommender still behaves sensibly for a model
            loaded outside a Lightning fit/test loop.
        propensity_model: a fitted `PropensityOverlapModel` giving patient-specific
            support, or `None` to rely on the per-line eligibility alone.
        horizon_times: default per-line RMST horizon (months), used when
            `recommend` is not given one.
    """

    def __init__(
        self,
        model: "DynaSurvCausalOnline",
        recommendable_treatments_per_line: dict[int, list[int]] | None = None,
        propensity_model=None,
        horizon_times: list[float] | None = None,
    ):
        self.model = model
        self.recommendable_treatments_per_line = recommendable_treatments_per_line
        self.propensity_model = propensity_model
        self.horizon_times = horizon_times

    @classmethod
    def from_model(cls, model: "DynaSurvCausalOnline") -> "TreatmentRecommender":
        """Build from the support state the model carries after fit/test or checkpoint load."""
        return cls(
            model,
            recommendable_treatments_per_line=getattr(
                model, "recommendable_treatments_per_line", None
            ),
            propensity_model=getattr(model, "recommendation_propensity_model", None),
            horizon_times=model.evaluation_horizon_times,
        )

    def action_mask(self, device: torch.device | None = None) -> torch.Tensor:
        """(n_lines, n_treatments) boolean mask of arms eligible at each line."""
        model = self.model
        mask = torch.zeros(
            model.n_lines, model.n_treatments, dtype=torch.bool, device=device
        )
        per_line = self.recommendable_treatments_per_line
        if not per_line:
            return torch.ones_like(mask)
        for line, arms in per_line.items():
            if line < model.n_lines:
                for k in arms:
                    mask[line, k] = True
        return mask

    def patient_support_mask(self, XPd, X_static) -> torch.Tensor:
        """(batch, n_lines, n_treatments) patient-specific arm eligibility.

        A false entry means the patient's observed pre-treatment history gives that
        arm too little estimated probability. Lines without a fitted multi-arm
        propensity model keep their per-line eligibility, which keeps single-arm
        lines and legacy checkpoints usable.
        """
        model = self.model
        batch_size, n_lines = XPd.shape[:2]
        global_mask = self.action_mask(device=XPd.device)[:n_lines]
        result = global_mask.unsqueeze(0).expand(batch_size, -1, -1).clone()
        overlap = self.propensity_model
        if overlap is None:
            return result

        x_end = model.x_input_dim
        p_end = x_end + model.p_input_dim
        x = XPd[:, :, :x_end]
        p = XPd[:, :, x_end:p_end]
        d = XPd[:, :, p_end:]
        x_static, _ = X_static
        propensity_mask = overlap.predict_mask(
            X=x,
            X_static=x_static,
            P=p,
            d=d,
            n_treatments=model.n_treatments,
        ).to(XPd.device)
        fitted_lines = torch.zeros(n_lines, dtype=torch.bool, device=XPd.device)
        for line in overlap.line_results:
            if line < n_lines:
                fitted_lines[line] = True
        result[:, fitted_lines] &= propensity_mask[:, fitted_lines]
        return result

    def arm_rmst(
        self,
        XPd,
        X_static,
        factual_idx,
        horizon_times: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RMST of every arm at every line plus the support mask; nothing decided yet.

        Split out of `recommend` so an ensemble can pool the finite RMST values of
        several members before any arm is masked: once an arm is -inf it can no
        longer be averaged or given a spread.

        Returns:
            arm_rmst: (batch, n_lines, n_treatments) finite RMST per arm, unsupported
                arms included.
            mask: (batch, n_lines, n_treatments) True where the arm is supported for
                that patient and line.
        """
        model = self.model
        survival = model.predict_discrete_survival(
            XPd=XPd, X_static=X_static, gather=False, factual_idx=factual_idx
        )  # (batch, n_lines, n_treatments, n_intervals + 1)
        n_lines = survival.shape[1]

        horizons = self.horizon_times if horizon_times is None else horizon_times
        if horizons is None or len(horizons) < n_lines:
            raise ValueError(
                f"need one RMST horizon per line ({n_lines} lines), got {horizons!r}; "
                "pass horizon_times or build the recommender with a default"
            )
        arm_rmst = torch.stack(
            [
                rmst(survival[:, line], horizons[line], model.interval_bounds)
                for line in range(n_lines)
            ],
            dim=1,
        )  # (batch, n_lines, n_treatments)

        mask = self.patient_support_mask(XPd, X_static)[:, :n_lines]
        return arm_rmst, mask

    def recommend(
        self,
        XPd,
        X_static,
        factual_idx,
        horizon_times: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank arms by RMST and return the best *supported* arm per line.

        Unsupported arms are set to -inf before the argmax rather than merely being
        reported alongside the rest: an arm with no empirical support at that line
        has a counterfactual the data cannot identify, and letting it win the
        comparison would surface an extrapolation as advice.

        Returns:
            best_idx: (batch, n_lines) recommended arm, or `NO_SUPPORTED_ARM` where
                every arm is unsupported for that patient and line.
            arm_rmst: (batch, n_lines, n_treatments) RMST per arm, -inf where masked.
        """
        arm_rmst, mask = self.arm_rmst(XPd, X_static, factual_idx, horizon_times)
        arm_rmst = arm_rmst.masked_fill(~mask, float("-inf"))
        best_idx = arm_rmst.argmax(dim=-1)
        # Never let argmax convert an all-masked row into an arbitrary arm 0.
        best_idx = best_idx.masked_fill(~mask.any(dim=-1), NO_SUPPORTED_ARM)
        return best_idx, arm_rmst
