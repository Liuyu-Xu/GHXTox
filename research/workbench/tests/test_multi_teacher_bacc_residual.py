import numpy as np

from ghxtox.multi_teacher_bacc_residual import (
    _apply_positive_residual,
    _candidate_mask,
    _passes_stability,
)


def test_candidate_mask_only_selects_current_negatives_supported_by_a_teacher():
    current = np.array([False, False, False, True])
    prott5 = np.array([0.46, 0.10, 0.10, 0.90])
    esm2 = np.array([0.10, 0.36, 0.10, 0.90])

    result = _candidate_mask(current, prott5, esm2)

    assert result.tolist() == [True, True, False, False]


def test_positive_residual_never_turns_a_positive_negative():
    reference = np.array([False, True, False])
    active = np.array([True, True, False])

    result = _apply_positive_residual(reference, active)

    assert result.tolist() == [True, True, False]


def test_stability_gate_accepts_small_allowed_tradeoff_with_higher_bacc():
    reference = {
        "accuracy": 0.95,
        "balanced_accuracy": 0.90,
        "f1": 0.88,
        "mcc": 0.85,
        "specificity": 0.97,
        "auroc": 0.98,
        "auprc": 0.96,
        "brier": 0.04,
        "ece_10": 0.02,
    }
    candidate = dict(reference)
    candidate.update(
        accuracy=0.9495,
        balanced_accuracy=0.905,
        f1=0.8795,
        mcc=0.8495,
        specificity=0.9685,
    )

    assert _passes_stability(reference, candidate)


def test_stability_gate_rejects_material_mcc_regression():
    reference = {
        "accuracy": 0.95,
        "balanced_accuracy": 0.90,
        "f1": 0.88,
        "mcc": 0.85,
        "specificity": 0.97,
        "auroc": 0.98,
        "auprc": 0.96,
        "brier": 0.04,
        "ece_10": 0.02,
    }
    candidate = dict(reference)
    candidate.update(balanced_accuracy=0.91, mcc=0.848)

    assert not _passes_stability(reference, candidate)


def test_stability_gate_rejects_equal_bacc():
    reference = {
        "accuracy": 0.95,
        "balanced_accuracy": 0.90,
        "f1": 0.88,
        "mcc": 0.85,
        "specificity": 0.97,
        "auroc": 0.98,
        "auprc": 0.96,
        "brier": 0.04,
        "ece_10": 0.02,
    }

    assert not _passes_stability(reference, dict(reference))
