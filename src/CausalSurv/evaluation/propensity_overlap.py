"""Training-only propensity and common-support checks for recommendations.

This module is intentionally separate from the model's gradient-reversal
propensity head.  The latter is an adversary used to learn a balanced latent
representation, whereas these probabilities estimate the observed treatment
assignment mechanism and must remain calibrated enough for overlap diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class LinePropensityResult:
    """Fitted treatment model and training-only diagnostics for one line."""

    arms: np.ndarray
    estimator: LogisticRegression | DummyClassifier
    threshold: float
    factual_propensity: np.ndarray
    effective_sample_size: dict[int, float]
    low_propensity_rate: dict[int, float]


class PropensityOverlapModel:
    """Per-line multinomial propensity models with out-of-fold diagnostics."""

    def __init__(
        self,
        n_lines: int,
        min_probability: float = 0.0,
        n_splits: int = 5,
        random_state: int = 0,
    ) -> None:
        if not 0.0 <= min_probability < 1.0:
            raise ValueError("min_probability must be in [0, 1)")
        self.n_lines = n_lines
        self.min_probability = float(min_probability)
        self.n_splits = n_splits
        self.random_state = random_state
        self.assignment_scope = "all_observed"
        self.line_results: dict[int, LinePropensityResult] = {}

    @staticmethod
    def features_from_tensors(
        X: torch.Tensor,
        X_static: torch.Tensor,
        P: torch.Tensor,
        d: torch.Tensor,
        line: int,
    ) -> np.ndarray:
        """Return only information available immediately before treatment ``line``.

        ``X`` is included from baseline through the current line. Current-line
        treatment ``P[:, line]`` is deliberately excluded. Previous treatment
        assignments and elapsed durations are part of the observed treatment
        history and are valid inputs for a sequential propensity.
        """
        # Sequential treatment assignment may depend on the complete observed
        # covariate history, not only the most recent line's measurements.
        # At line 0 this is exactly X[:, 0, :]; at later lines it appends all
        # preceding X states through the current decision point.
        pieces = [X[:, : line + 1, :].flatten(start_dim=1), X_static]
        if line:
            pieces.extend(
                [
                    P[:, :line, :].flatten(start_dim=1),
                    d[:, :line, :].flatten(start_dim=1),
                ]
            )
        return torch.cat(pieces, dim=1).detach().cpu().numpy()

    @staticmethod
    def _effective_sample_size(weights: np.ndarray) -> float:
        return (
            float(weights.sum() ** 2 / np.square(weights).sum())
            if len(weights)
            else 0.0
        )

    def fit(
        self,
        X: torch.Tensor,
        X_static: torch.Tensor,
        P: torch.Tensor,
        d: torch.Tensor,
        treatment_idx: torch.Tensor,
        mask: torch.Tensor,
        eligible_arms_per_line: dict[int, list[int]],
    ) -> "PropensityOverlapModel":
        """Estimate P(A=a | H) over ALL observed assignment classes.

        Eligibility is applied after probability estimation. Renormalizing over
        recommendable classes would conceal lack of support for that entire set.
        """
        self.line_results = {}
        for line in range(self.n_lines):
            observed = mask[:, line].bool().cpu().numpy()
            labels = treatment_idx[:, line].cpu().numpy()
            keep = observed
            y = labels[keep]
            if len(y) < 2:
                continue
            arms, counts = np.unique(y, return_counts=True)

            features = self.features_from_tensors(X, X_static, P, d, line)[keep]
            min_class_count = int(counts.min())
            n_splits = min(
                self.n_splits, min_class_count if min_class_count >= 2 else len(y)
            )
            if n_splits < 2:
                continue

            oof = np.zeros((len(y), len(arms)), dtype=float)
            splitter_type = StratifiedKFold if min_class_count >= 2 else KFold
            splitter = splitter_type(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )

            def fit_assignment(x, labels):
                estimator = (
                    LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
                    if len(np.unique(labels)) > 1
                    else DummyClassifier(strategy="prior")
                )
                return estimator.fit(x, labels)

            for train, test in splitter.split(features, y):
                estimator = fit_assignment(features[train], y[train])
                columns = np.searchsorted(arms, estimator.classes_)
                oof[np.ix_(test, columns)] = estimator.predict_proba(features[test])

            estimator = fit_assignment(features, y)
            factual = oof[np.arange(len(y)), np.searchsorted(arms, y)]
            marginal = {arm: float((y == arm).mean()) for arm in arms}
            ess, low_rate = {}, {}
            for arm in arms:
                arm_factual = factual[y == arm]
                weights = marginal[arm] / np.clip(arm_factual, 1e-8, None)
                ess[int(arm)] = self._effective_sample_size(weights)
                low_rate[int(arm)] = float((arm_factual < self.min_probability).mean())

            self.line_results[line] = LinePropensityResult(
                arms=arms,
                estimator=estimator,
                threshold=self.min_probability,
                factual_propensity=factual,
                effective_sample_size=ess,
                low_propensity_rate=low_rate,
            )
        return self

    def predict_mask(
        self,
        X: torch.Tensor,
        X_static: torch.Tensor,
        P: torch.Tensor,
        d: torch.Tensor,
        n_treatments: int,
    ) -> torch.Tensor:
        """Patient-specific arm mask; false means insufficient propensity support."""
        batch_size, observed_lines = X.shape[:2]
        if not 0 < observed_lines <= self.n_lines:
            raise ValueError("Input history must contain between 1 and n_lines steps")
        result = torch.zeros(batch_size, observed_lines, n_treatments, dtype=torch.bool)
        for line, fitted in self.line_results.items():
            if line >= observed_lines:
                continue
            features = self.features_from_tensors(X, X_static, P, d, line)
            probabilities = fitted.estimator.predict_proba(features)
            supported = probabilities >= fitted.threshold
            result[:, line, torch.as_tensor(fitted.arms)] = torch.from_numpy(supported)
        return result

    def summary(self) -> dict[int, dict[str, object]]:
        """Compact diagnostics suitable for logging or an audit report."""
        return {
            line: {
                "arms": result.arms.tolist(),
                "threshold": result.threshold,
                "effective_sample_size": result.effective_sample_size,
                "low_propensity_rate": result.low_propensity_rate,
            }
            for line, result in self.line_results.items()
        }
