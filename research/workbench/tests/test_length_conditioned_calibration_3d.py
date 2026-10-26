import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ghxtox.bootstrap_ci import _metrics
from ghxtox.length_conditioned_calibration_3d import (
    cross_fit_probabilities,
    run_phase31,
)


MODEL = {
    "alpha_bounds": [-3.0, 3.0],
    "probability_clip": 1e-8,
}
OPTIMIZATION = {
    "slope_log_l2": 0.01,
    "length_offset_l2": 0.25,
    "maximum_iterations": 500,
    "gradient_tolerance": 1e-9,
    "function_tolerance": 1e-12,
}


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_fit_does_not_use_held_fold_labels() -> None:
    lengths = np.tile(np.asarray([12, 12, 20, 20, 40, 40]), 3)
    labels = np.tile(np.asarray([0, 1, 0, 1, 0, 1]), 3)
    probabilities = np.where(labels == 1, 0.75, 0.25).astype(np.float64)
    folds = np.repeat(np.arange(3), 6)

    original, reports = cross_fit_probabilities(
        probabilities, labels, lengths, folds, MODEL, OPTIMIZATION
    )
    changed_labels = labels.copy()
    changed_labels[folds == 0] = 1 - changed_labels[folds == 0]
    changed, _ = cross_fit_probabilities(
        probabilities, changed_labels, lengths, folds, MODEL, OPTIMIZATION
    )

    np.testing.assert_array_equal(original[folds == 0], changed[folds == 0])
    assert len(reports) == 3
    assert all(report["success"] for report in reports)
    assert all(report["slope"] > 0.0 for report in reports)


def test_phase31_writes_two_score_exploratory_handoff(tmp_path: Path) -> None:
    oof_rows = []
    fold_rows = []
    source_index = 0
    for fold in (0, 1):
        for length in (12, 20, 40):
            for label in (0, 1):
                sequence = ("ACDEFGHIKLMNPQRSTVWY" * 2)[:length]
                stratum = (
                    "11-15" if length == 12 else "16-30" if length == 20 else "31-50"
                )
                common = {
                    "source_index": source_index,
                    "sample_id": f"peptide_{source_index + 1}",
                    "fold": fold,
                    "label": label,
                    "sequence": sequence,
                }
                fold_rows.append(common)
                oof_rows.append(
                    {
                        **common,
                        "toxicity_probability": 0.8 if label else 0.2,
                        "length": length,
                        "length_stratum": stratum,
                    }
                )
                source_index += 1

    oof_path = tmp_path / "oof.csv"
    fold_path = tmp_path / "folds.csv"
    contract_path = tmp_path / "contract.json"
    _write_csv(oof_path, oof_rows)
    _write_csv(fold_path, fold_rows)
    contract_path.write_text("{}", encoding="utf-8")
    labels = np.asarray([int(row["label"]) for row in oof_rows])
    probabilities = np.asarray([float(row["toxicity_probability"]) for row in oof_rows])
    lengths = np.asarray([int(row["length"]) for row in oof_rows])
    reference = _metrics(labels, probabilities, 0.5)
    length_mcc = {}
    length_auprc = {}
    for name, selected in {
        "11-15": lengths == 12,
        "16-30": lengths == 20,
        "31-50": lengths == 40,
    }.items():
        current = _metrics(labels[selected], probabilities[selected], 0.5)
        length_mcc[name] = current["mcc"]
        length_auprc[name] = current["auprc"]

    paths = [oof_path, contract_path, fold_path]
    protocol = {
        "status": "frozen_before_candidate_oof",
        "evaluation_tier": "cross_fitted_screen_not_strict_nested_upstream",
        "direct_handoff_promotion_allowed": False,
        "external_test_labels_read": False,
        "provenance_limitation": "synthetic upstream limitation",
        "inputs": {
            "reference_oof": str(oof_path),
            "handoff_contract": str(contract_path),
            "fold_manifest": str(fold_path),
            "sha256": {str(path): _sha256(path) for path in paths},
        },
        "cohort": {
            "rows": 12,
            "positive": 6,
            "negative": 6,
            "folds": 2,
            "length_strata": {"11-15": 4, "16-30": 4, "31-50": 4},
        },
        "model": MODEL,
        "optimization": OPTIMIZATION,
        "evaluation": {
            "reference_threshold": 0.5,
            "calibrated_threshold": 0.5,
            "reference": {
                name: reference[name] for name in ("mcc", "auprc", "brier", "ece_10")
            },
            "length_reference_mcc": length_mcc,
            "length_reference_auprc": length_auprc,
        },
        "advance_gate": {
            "global_mcc_strictly_above": 1.0,
            "global_auprc_at_least": 0.0,
            "maximum_brier": 1.0,
            "maximum_ece_10": 1.0,
            "minimum_fold_mcc_non_decreases": 1,
            "short_mcc_at_least": -1.0,
            "maximum_other_length_mcc_regression": 1.0,
            "maximum_length_auprc_regression": 1.0,
            "minimum_net_decision_corrections": 0,
            "minimum_probability_pearson": -1.0,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    output = tmp_path / "output"
    report, decision = run_phase31(protocol_path, output)

    assert report["rows"] == 12
    assert report["strict_nested_upstream"] is False
    assert decision["promotion_authorized"] is False
    assert decision["official_phase25_handoff_modified"] is False
    assert decision["external_test_labels_read"] is False
    assert len(report["fold_coefficients"]) == 2
    with (output / "fusion_handoff.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        handoff = list(csv.DictReader(handle))
    assert len(handoff) == 12
    assert {"phase25_probability", "length_calibrated_probability"} <= set(handoff[0])
