import numpy as np

from ghxtox.bacc_recall_residual import (
    _decision_metrics,
    apply_minimal_decision_residual,
)


def test_minimal_residual_only_changes_selected_disagreements() -> None:
    base = np.asarray([0.4, 0.7, 0.2, 0.8])
    teacher = np.asarray([0.9, 0.1, 0.3, 0.9])
    selected = np.asarray([True, True, True, True])
    result = apply_minimal_decision_residual(
        base, teacher, selected, threshold=0.6
    )
    assert result[0] > 0.6
    assert result[1] < 0.6
    assert result[2] == base[2]
    assert result[3] == base[3]


def test_minimal_residual_respects_selection_mask() -> None:
    base = np.asarray([0.4, 0.7])
    teacher = np.asarray([0.9, 0.1])
    result = apply_minimal_decision_residual(
        base, teacher, np.asarray([False, True]), threshold=0.6
    )
    assert result[0] == base[0]
    assert result[1] < 0.6


def test_minimal_residual_supports_distinct_teacher_threshold() -> None:
    base = np.asarray([0.55, 0.65])
    teacher = np.asarray([0.50, 0.40])
    result = apply_minimal_decision_residual(
        base,
        teacher,
        np.asarray([True, True]),
        threshold=0.60,
        teacher_threshold=0.45,
    )
    assert result[0] > 0.60
    assert result[1] < 0.60


def test_decision_metrics_preserve_probability_metrics() -> None:
    labels = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.7, 0.4, 0.9])
    baseline = _decision_metrics(labels, probability, probability >= 0.6, 0.6)
    corrected = _decision_metrics(
        labels,
        probability,
        np.asarray([False, False, True, True]),
        0.6,
    )

    assert corrected["balanced_accuracy"] > baseline["balanced_accuracy"]
    for key in ("auroc", "auprc", "brier", "ece_10"):
        assert corrected[key] == baseline[key]
