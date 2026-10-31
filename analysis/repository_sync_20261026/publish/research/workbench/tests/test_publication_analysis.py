from __future__ import annotations

import numpy as np

from ghxtox.publication_analysis import (
    PredictionData,
    _calibration_bins,
    _paired_bootstrap,
)


def test_calibration_bins_cover_all_samples() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.05, 0.25, 0.75, 1.0], dtype=np.float64)
    bins = _calibration_bins(labels, probabilities)
    assert sum(row["count"] for row in bins) == 4


def test_paired_bootstrap_uses_positive_advantage_for_lower_calibration_error() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    first = PredictionData(
        labels=labels,
        probabilities=np.asarray([0.1, 0.2, 0.8, 0.9]),
        sequences=["A", "B", "C", "D"],
        threshold=0.5,
    )
    second = PredictionData(
        labels=labels,
        probabilities=np.asarray([0.3, 0.4, 0.6, 0.7]),
        sequences=["A", "B", "C", "D"],
        threshold=0.5,
    )
    result = _paired_bootstrap(first, second, iterations=100, seed=7)
    assert result["metrics"]["brier"]["advantage"] > 0.0
    assert result["mcnemar"]["exact_two_sided_p"] == 1.0
