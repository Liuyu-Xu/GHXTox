import numpy as np
import torch

from ghxtox.cross_dimension_attention import (
    CrossDimensionAttention,
    apply_attention_residual,
)


def test_attention_model_shapes_and_weights() -> None:
    model = CrossDimensionAttention(8, 8, hidden_dim=16, num_heads=2)
    raw_delta, weights = model(torch.randn(4, 8), torch.randn(4, 3, 8))
    assert raw_delta.shape == (4,)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-6)


def test_attention_residual_is_bounded_and_preserves_fallback() -> None:
    baseline = np.asarray([0.2, 0.6, 0.8])
    delta = np.asarray([100.0, -100.0, 100.0])
    active = np.asarray([True, True, False])
    candidate = apply_attention_residual(
        baseline, delta, active, residual_bound=0.1
    )
    baseline_logit = np.log(baseline / (1.0 - baseline))
    candidate_logit = np.log(candidate / (1.0 - candidate))
    assert np.all(np.abs(candidate_logit[active] - baseline_logit[active]) <= 0.100001)
    assert candidate[2] == baseline[2]


def test_attention_residual_rejects_mismatched_inputs() -> None:
    try:
        apply_attention_residual(
            np.asarray([0.2]),
            np.asarray([0.1, 0.2]),
            np.asarray([True]),
            residual_bound=0.1,
        )
    except ValueError as exc:
        assert "identical shapes" in str(exc)
    else:
        raise AssertionError("mismatched inputs must be rejected")


def test_decision_protection_preserves_threshold_side() -> None:
    baseline = np.asarray([0.59, 0.61])
    candidate = apply_attention_residual(
        baseline,
        np.asarray([100.0, -100.0]),
        np.asarray([True, True]),
        residual_bound=0.2,
        decision_threshold=0.6,
    )
    assert candidate[0] < 0.6
    assert candidate[1] >= 0.6
