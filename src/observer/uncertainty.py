"""Confidence calibration utilities.

Provides Platt scaling for raw confidence scores, expected calibration error
computation, and reliability diagram data for visualization.
"""

from __future__ import annotations

import math


def platt_scale(
    raw_confidences: list[float],
    correctness: list[bool],
) -> tuple[float, float]:
    """Fit Platt scaling parameters via logistic regression.

    Uses gradient descent to fit a logistic model: P(correct) = 1 / (1 + exp(a*f + b))
    where f is the raw confidence score.

    Args:
        raw_confidences: List of raw confidence scores from the model.
        correctness: List of boolean ground-truth correctness labels.

    Returns:
        A tuple (a, b) of Platt scaling parameters.
    """
    if not raw_confidences or len(raw_confidences) != len(correctness):
        return 0.0, 0.0

    targets = [1.0 if c else 0.0 for c in correctness]

    # Initialize parameters
    a = 0.0
    b = 0.0
    learning_rate = 0.01
    n_iterations = 1000

    n = len(raw_confidences)

    for _ in range(n_iterations):
        grad_a = 0.0
        grad_b = 0.0

        for i in range(n):
            z = a * raw_confidences[i] + b
            # Numerically stable sigmoid
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                p = exp_z / (1.0 + exp_z)

            error = p - targets[i]
            grad_a += error * raw_confidences[i]
            grad_b += error

        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n

    return a, b


def apply_platt(raw_confidence: float, a: float, b: float) -> float:
    """Apply Platt scaling to a single raw confidence value.

    Args:
        raw_confidence: The raw confidence score to calibrate.
        a: The Platt scaling slope parameter.
        b: The Platt scaling intercept parameter.

    Returns:
        The calibrated confidence score, clamped to [0.01, 0.99].
    """
    z = a * raw_confidence + b
    # Numerically stable sigmoid
    if z >= 0:
        calibrated = 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        calibrated = exp_z / (1.0 + exp_z)

    return max(0.01, min(0.99, calibrated))


def expected_calibration_error(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the weighted average gap between predicted confidence and
    actual accuracy across confidence bins.

    Args:
        confidences: List of predicted confidence scores.
        accuracies: List of boolean correctness labels.
        n_bins: Number of bins to divide the confidence range into.

    Returns:
        The ECE value (lower is better, 0.0 is perfectly calibrated).
    """
    if not confidences:
        return 0.0

    n = len(confidences)
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    ece = 0.0
    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]

        # Collect samples in this bin
        bin_confs = []
        bin_accs = []
        for j in range(n):
            if low <= confidences[j] < high or (i == n_bins - 1 and confidences[j] == high):
                bin_confs.append(confidences[j])
                bin_accs.append(1.0 if accuracies[j] else 0.0)

        if not bin_confs:
            continue

        avg_conf = sum(bin_confs) / len(bin_confs)
        avg_acc = sum(bin_accs) / len(bin_accs)
        ece += (len(bin_confs) / n) * abs(avg_conf - avg_acc)

    return ece


def reliability_diagram_data(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 10,
) -> dict:
    """Compute binned data for a reliability diagram.

    Args:
        confidences: List of predicted confidence scores.
        accuracies: List of boolean correctness labels.
        n_bins: Number of bins to divide the confidence range into.

    Returns:
        A dict with keys:
            - bin_centers: list of bin center values
            - bin_accuracies: list of mean accuracy per bin
            - bin_counts: list of sample counts per bin
    """
    bin_centers: list[float] = []
    bin_accuracies: list[float] = []
    bin_counts: list[int] = []

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

    for i in range(n_bins):
        low = bin_boundaries[i]
        high = bin_boundaries[i + 1]
        center = (low + high) / 2.0

        bin_accs = []
        for j in range(len(confidences)):
            if low <= confidences[j] < high or (i == n_bins - 1 and confidences[j] == high):
                bin_accs.append(1.0 if accuracies[j] else 0.0)

        bin_centers.append(center)
        bin_accuracies.append(sum(bin_accs) / len(bin_accs) if bin_accs else 0.0)
        bin_counts.append(len(bin_accs))

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }
