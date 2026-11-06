from __future__ import annotations

import numpy as np

from ghxtox.physical_3d_descriptors import (
    DESCRIPTOR_NAMES,
    evaluate_candidate,
    physical_descriptor_vector,
)


def test_physical_descriptor_contract_is_finite_and_fixed() -> None:
    sequence = "AKDECFIL"
    base = np.zeros((len(sequence), 3), dtype=np.float64)
    base[:, 0] = np.arange(len(sequence)) * 3.8
    bent = base.copy()
    bent[4:, 1] = np.arange(1, 5) * 1.5
    descriptors = physical_descriptor_vector(sequence, np.stack([base, bent]))
    assert len(DESCRIPTOR_NAMES) == 20
    assert descriptors.shape == (20,)
    assert np.isfinite(descriptors).all()
    assert descriptors[0] > 0.0
    assert 0.0 <= descriptors[4] <= 1.0
    assert 0.0 <= descriptors[7] <= 1.0


def test_spatial_patch_descriptor_responds_to_geometry() -> None:
    sequence = "AIAIAKDE"
    extended = np.zeros((len(sequence), 3), dtype=np.float64)
    extended[:, 0] = np.arange(len(sequence)) * 5.0
    compact = extended.copy()
    compact[:, 0] = np.arange(len(sequence)) % 4 * 3.0
    compact[:, 1] = np.arange(len(sequence)) // 4 * 3.0
    first = physical_descriptor_vector(sequence, extended[None])
    second = physical_descriptor_vector(sequence, compact[None])
    patch_index = DESCRIPTOR_NAMES.index("largest_hydrophobic_patch_fraction")
    assert second[patch_index] != first[patch_index]


def test_fold_local_candidate_preserves_missing_rows_exactly() -> None:
    reference_rows = []
    descriptor_rows = []
    for index in range(40):
        fold = index % 5
        label = index % 2
        probability = 0.75 if label else 0.25
        source = "PepFlow" if index % 3 == 0 else "AF2-16-30"
        reference_rows.append(
            {
                "source_index": str(index),
                "sample_id": f"peptide_{index + 1}",
                "fold": str(fold),
                "label": str(label),
                "sequence": "AKDECFIL",
                "length": "8",
                "length_stratum": "11-15",
                "generator_source": source if index < 30 else "not_generated",
                "toxicity_probability": str(probability),
            }
        )
        if index < 30:
            values = {
                name: float(label) + descriptor_index * 0.001
                for descriptor_index, name in enumerate(DESCRIPTOR_NAMES)
            }
            descriptor_rows.append(
                {
                    "source_index": index,
                    "sample_id": f"peptide_{index + 1}",
                    "fold": fold,
                    "label": label,
                    "sequence": "AKDECFIL",
                    "length": 8,
                    "length_stratum": "11-15",
                    "generator_source": source,
                    **values,
                }
            )
    report, rows = evaluate_candidate(
        reference_rows,
        descriptor_rows,
        threshold=0.677819,
        residual_weight=0.15,
        regularization_c=0.1,
        random_seed=20260801,
    )
    assert report["external_test_labels_read"] is False
    assert len(rows) == 40
    missing = rows[30:]
    assert all(row["conformer_available"] == 0 for row in missing)
    assert all(
        row["candidate_probability"] == row["reference_probability"]
        for row in missing
    )
    assert all(row["residual_logit_delta"] == 0.0 for row in missing)
