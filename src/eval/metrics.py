"""Metric calculations for evaluation results.

Computes accuracy, coverage, selective accuracy, calibration error,
and per-model comparison summaries from lists of EvalResult objects.
"""

from __future__ import annotations

from collections import defaultdict

from src.observer.uncertainty import expected_calibration_error
from src.pipeline.schemas import EvalResult


def accuracy(results: list[EvalResult]) -> float:
    """Compute overall accuracy across all results.

    Args:
        results: List of evaluation results.

    Returns:
        Mean score across all results, or 0.0 if empty.
    """
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


def coverage(results: list[EvalResult]) -> float:
    """Compute coverage: percentage where observer_gate is ACT or None.

    Coverage measures what fraction of tasks the system chose to answer
    (either because the observer approved, or because there was no observer).

    Args:
        results: List of evaluation results.

    Returns:
        Fraction of results that were acted on, or 0.0 if empty.
    """
    if not results:
        return 0.0
    acted = sum(
        1 for r in results
        if r.observer_gate is None or r.observer_gate == "act"
    )
    return acted / len(results)


def selective_accuracy(results: list[EvalResult]) -> float:
    """Compute accuracy only on tasks where the observer approved action.

    This is the core metric: does the observer improve accuracy by
    selectively approving only confident predictions?

    Args:
        results: List of evaluation results.

    Returns:
        Mean score among approved results, or 0.0 if none approved.
    """
    approved = [
        r for r in results
        if r.observer_gate is None or r.observer_gate == "act"
    ]
    if not approved:
        return 0.0
    return sum(r.score for r in approved) / len(approved)


def calibration_error(results: list[EvalResult]) -> float:
    """Compute Expected Calibration Error from observer confidence vs correctness.

    Args:
        results: List of evaluation results with observer_confidence populated.

    Returns:
        The ECE value. Lower is better. Returns 0.0 if no confidence data.
    """
    confidences = []
    correct = []

    for r in results:
        if r.observer_confidence is not None:
            confidences.append(r.observer_confidence)
            correct.append(r.score >= 0.5)

    if not confidences:
        return 0.0

    return expected_calibration_error(confidences, correct)


def model_comparison(results: list[EvalResult]) -> dict:
    """Generate per-model metrics summary.

    Groups results by model_name and computes accuracy, coverage,
    selective accuracy, and calibration error for each.

    Args:
        results: List of evaluation results from potentially multiple models.

    Returns:
        A dict mapping model_name to a dict of metric values.
    """
    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_name].append(r)

    summary: dict[str, dict] = {}
    for model_name, model_results in by_model.items():
        summary[model_name] = {
            "count": len(model_results),
            "accuracy": accuracy(model_results),
            "coverage": coverage(model_results),
            "selective_accuracy": selective_accuracy(model_results),
            "calibration_error": calibration_error(model_results),
        }

    return summary
