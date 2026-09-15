"""Evaluation utilities.

Keep the evaluator import lazy: data preparation also owns lightweight
evaluation helpers (such as propensity-overlap fitting), and eagerly importing
the evaluator would create a data <-> evaluation import cycle.
"""

__all__ = ["DynasurvEvaluator"]


def __getattr__(name):
    if name == "DynasurvEvaluator":
        from .evaluator import DynasurvEvaluator

        return DynasurvEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
