"""Post-hoc refit of the per-line hazard temperature and bias.

Lived on the model as `fit_hazard_calibration`. It is a procedure run once
against a trained model, not part of the model's forward or training behaviour,
and it had no callers anywhere in the repository -- so it moves out whole.

The parameters it adjusts, `hazard_line_log_temperature` and `hazard_line_bias`,
stay on the model: they sit on the logits inside `forward` and are saved in the
checkpoint. This class only fits them.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch

from .discrete_survival import hazards_to_survival, kaplan_meier, km_at

if TYPE_CHECKING:
    from ..model.dynasurv_causal_online import DynaSurvCausalOnline


class HazardCalibrator:
    """Refits a model's per-line hazard temperature and bias against observed calibration.

    `hazard_line_log_temperature` and `hazard_line_bias` already sit on the logits
    (see `DynaSurvCausalOnline.forward`), but they are trained jointly with the
    survival NLL -- an individual-likelihood objective that is indifferent to
    whether the spread of predictions matches the spread of outcomes. Left to the
    NLL they come out over-confident: on this cohort line 1 spans predicted S(24)
    from 0.00 to 0.99 against an observed range of 0.44 to 0.89.

    This refits those 2 numbers per line and nothing else, against a
    *conditional* target: patients are binned by predicted survival and each
    bin's mean prediction is pulled toward that bin's Kaplan-Meier estimate. A
    marginal target would not do -- the mean predicted curve can already sit on
    the KM curve while every individual bin is wrong, because the too-low and
    too-high bins cancel.
    """

    def __init__(self, model: "DynaSurvCausalOnline"):
        self.model = model

    def fit(
        self,
        XPd,
        X_static,
        treatment_idx,
        time,
        event,
        mask,
        landmarks,
        n_bins: int = 10,
        bin_min_samples: int = 40,
        steps: int = 400,
        lr: float = 0.05,
    ) -> dict[int, dict[str, float]]:
        """Refit the temperature and bias in place on `self.model`.

        Fit this on the TRAINING partition. With `final_training=True` the
        holdout is also the validation set, so fitting on the holdout would be
        scoring the correction on the data that produced it.

        Bin membership is stable during the fit: `logit / T + b` is monotone in
        the logit, so the induced ordering of predicted survival never changes
        and the KM targets can be computed once up front.

        Note that every other model parameter is left with `requires_grad=False`
        afterwards, as it always was.

        Returns:
            Per-line (1-indexed) diagnostics: bins kept, the weighted calibration
            error before and after, and the fitted temperature and bias.
        """
        model = self.model
        device = next(model.parameters()).device
        bounds = model.interval_bounds.to(device)
        landmarks = [float(t) for t in landmarks]

        # Recover the logits as they were before the current temperature/bias, so
        # the fit starts from the encoder's raw output rather than compounding onto
        # a previous correction.
        with torch.no_grad():
            scaled, _ = model.forward_factual(XPd, X_static, treatment_idx)
            n_lines_obs = scaled.shape[1]
            t_old = torch.exp(model.hazard_line_log_temperature[:n_lines_obs]).view(
                1, -1, 1
            )
            b_old = model.hazard_line_bias[:n_lines_obs].view(1, -1, 1)
            raw = (scaled - b_old) * t_old  # (batch, n_lines, n_intervals)

        def survival_on_grid(logits, lo, hi, w):
            s = hazards_to_survival(torch.sigmoid(logits))
            return s[..., lo] * (1 - w) + s[..., hi] * w

        # Fixed grid weights for linear interpolation at each landmark.
        grid = {}
        for lm in landmarks:
            j = int(
                torch.clamp(
                    torch.searchsorted(bounds, torch.tensor(lm, device=device)),
                    1,
                    len(bounds) - 1,
                )
            )
            lo, hi = j - 1, j
            w = float((lm - bounds[lo]) / (bounds[hi] - bounds[lo]).clamp(min=1e-9))
            grid[lm] = (lo, hi, min(max(w, 0.0), 1.0))

        # Bin assignments and KM targets, computed once.
        targets = []
        report: dict[int, dict[str, float]] = {}
        for line in range(n_lines_obs):
            m = mask[:, line].bool()
            if not m.any():
                continue
            t_line = (
                time.reshape(time.shape[0], time.shape[1], -1)[m, line, 0].cpu().numpy()
            )
            e_line = (
                event.reshape(event.shape[0], event.shape[1], -1)[m, line, 0]
                .cpu()
                .numpy()
            )
            kept = 0
            for lm in landmarks:
                lo, hi, w = grid[lm]
                with torch.no_grad():
                    p = survival_on_grid(raw[m, line], lo, hi, w).cpu().numpy()
                edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
                for i in range(len(edges) - 1):
                    last = i == len(edges) - 2
                    sel = (p >= edges[i]) & (
                        p <= edges[i + 1] if last else p < edges[i + 1]
                    )
                    if sel.sum() < bin_min_samples:
                        continue
                    uniq, surv = kaplan_meier(t_line[sel], e_line[sel])
                    obs = km_at(uniq, surv, lm)
                    if not np.isfinite(obs):
                        continue
                    targets.append(
                        (
                            line,
                            lo,
                            hi,
                            w,
                            torch.as_tensor(np.flatnonzero(sel), device=device),
                            float(obs),
                            float(sel.sum()),
                        )
                    )
                    kept += 1
            report[line + 1] = {"bins": kept}

        if not targets:
            raise ValueError(
                "no bins had enough patients and follow-up to calibrate against"
            )

        def objective():
            t_new = torch.exp(model.hazard_line_log_temperature[:n_lines_obs]).view(
                1, -1, 1
            )
            b_new = model.hazard_line_bias[:n_lines_obs].view(1, -1, 1)
            logits = raw / t_new + b_new
            loss = 0.0
            per_line = defaultdict(lambda: [0.0, 0.0])
            for line, lo, hi, w, idx, obs, n in targets:
                m = mask[:, line].bool()
                pred = survival_on_grid(logits[m, line][idx], lo, hi, w).mean()
                gap = pred - obs
                loss = loss + n * gap.pow(2)
                per_line[line][0] += n * abs(float(gap.detach()))
                per_line[line][1] += n
            return loss / sum(t[6] for t in targets), per_line

        with torch.no_grad():
            _, before = objective()

        for param in model.parameters():
            param.requires_grad_(False)
        model.hazard_line_log_temperature.requires_grad_(True)
        model.hazard_line_bias.requires_grad_(True)

        opt = torch.optim.Adam(
            [model.hazard_line_log_temperature, model.hazard_line_bias], lr=lr
        )
        for _ in range(steps):
            opt.zero_grad()
            loss, _ = objective()
            loss.backward()
            opt.step()

        with torch.no_grad():
            _, after = objective()
        for line in report:
            i = line - 1
            report[line]["err_before"] = before[i][0] / max(before[i][1], 1)
            report[line]["err_after"] = after[i][0] / max(after[i][1], 1)
            report[line]["temperature"] = float(
                torch.exp(model.hazard_line_log_temperature[i].detach())
            )
            report[line]["bias"] = float(model.hazard_line_bias[i].detach())
        return report
