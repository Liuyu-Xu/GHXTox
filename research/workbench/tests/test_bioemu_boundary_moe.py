from __future__ import annotations

import numpy as np

from ghxtox.bioemu_boundary_moe import (
    apply_boundary_mixture,
    apply_consensus_residual,
)


def test_consensus_residual_preserves_inactive_rows() -> None:
    default = np.asarray([0.58, 0.62, 0.4])
    specialist = np.asarray([0.9, 0.1, 0.8])
    active = np.asarray([True, True, False])
    candidate = apply_consensus_residual(default, specialist, active, residual_bound=0.15)
    assert candidate[0] > default[0]
    assert candidate[1] < default[1]
    assert candidate[2] == default[2]


def test_consensus_residual_is_bounded_in_logit_space() -> None:
    default = np.asarray([0.5, 0.5])
    specialist = np.asarray([1.0, 0.0])
    candidate = apply_consensus_residual(
        default, specialist, np.asarray([True, True]), residual_bound=0.15
    )
    candidate_logit = np.log(candidate / (1.0 - candidate))
    assert np.all(np.abs(candidate_logit) <= 0.150001)


def test_boundary_mixture_uses_mutually_exclusive_routes() -> None:
    default = np.asarray([0.5, 0.5, 0.5])
    specialist = np.asarray([0.9, 0.9, 0.9])
    bioemu = np.asarray([0.1, 0.1, 0.1])
    gate = np.asarray([1.0, 1.0, 1.0])
    boundary = np.asarray([True, True, False])
    consensus = np.asarray([True, False, False])
    candidate, evidence = apply_boundary_mixture(
        default, specialist, bioemu, gate, boundary, consensus, residual_bound=0.15
    )
    assert candidate[0] > default[0]
    assert candidate[1] < default[1]
    assert candidate[2] == default[2]
    assert evidence[0] > 0.0
    assert evidence[1] < 0.0
    assert evidence[2] == 0.0


def test_boundary_mixture_rejects_consensus_outside_boundary() -> None:
    with np.testing.assert_raises(ValueError):
        apply_boundary_mixture(
            np.asarray([0.5]),
            np.asarray([0.9]),
            np.asarray([0.9]),
            np.asarray([1.0]),
            np.asarray([False]),
            np.asarray([True]),
        )
