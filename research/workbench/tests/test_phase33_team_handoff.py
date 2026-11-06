from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from ghxtox.phase33_team_handoff import build_phase33_team_handoff


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_inputs(tmp_path: Path) -> tuple[Path, Path]:
    reference_root = tmp_path / "reference"
    phase33_root = tmp_path / "phase33"
    reference_root.mkdir()
    phase33_root.mkdir()
    threshold = 0.677819
    reference_rows = []
    quality_rows = []
    phase33_rows = []
    for source_index in range(601):
        label = source_index % 2
        probability = 0.62 if label else 0.42
        boundary = int(source_index < 600)
        delta = (0.05 if label else -0.05) if boundary else 0.0
        reference_logit = math.log(probability / (1.0 - probability))
        candidate = (
            1.0 / (1.0 + math.exp(-(reference_logit + delta)))
            if boundary
            else probability
        )
        sequence = "ACDEFGHIKLM"
        reference_rows.append(
            {
                "source_index": source_index,
                "sample_id": f"peptide_{source_index + 1}",
                "fold": source_index % 5,
                "label": label,
                "sequence": sequence,
                "toxicity_probability": probability,
                "logit": reference_logit,
                "prediction": int(probability >= threshold),
                "decision_threshold": threshold,
                "probability_margin": 2 * abs(probability - 0.5),
                "probability_entropy": 0.5,
                "mc_mean_probability": probability,
                "mc_std_probability": 0.0,
                "generator_source": "not_generated",
            }
        )
        quality_rows.append(
            {
                "source_index": source_index,
                "fold": source_index % 5,
                "generator_source": "not_generated",
                "conformer_available": 0,
                "missing_conformer_mask": 1,
            }
        )
        phase33_rows.append(
            {
                "source_index": source_index,
                "sample_id": f"peptide_{source_index + 1}",
                "fold": source_index % 5,
                "label": label,
                "sequence": sequence,
                "phase33_boundary_member": boundary,
                "bioemu_boundary_available": boundary,
                "reference_probability": probability,
                "standalone_3d_probability": 0.75 if boundary else "",
                "bounded_logit_delta": delta,
                "candidate_probability": candidate,
                "decision_threshold": threshold,
                "reference_prediction": int(probability >= threshold),
                "candidate_prediction": int(candidate >= threshold),
            }
        )
    _write_csv(reference_root / "oof_3d_predictions.csv", reference_rows)
    _write_csv(reference_root / "oof_3d_quality.csv", quality_rows)
    (reference_root / "handoff_contract.json").write_text(
        json.dumps({"fixed_threshold": threshold, "rows": 601}), encoding="utf-8"
    )
    _write_csv(phase33_root / "oof_predictions.csv", phase33_rows)
    (phase33_root / "summary.json").write_text("{}", encoding="utf-8")
    (phase33_root / "decision.json").write_text(
        json.dumps(
            {
                "decision": "promote_phase33_3d_handoff",
                "promotion_authorized": True,
                "handoff_regeneration_authorized": True,
                "external_test_labels_read": False,
                "decision_transitions": {
                    "changed": 0,
                    "corrected": 0,
                    "regressed": 0,
                    "net": 0,
                },
                "gate": {"all_conditions_passed": True},
            }
        ),
        encoding="utf-8",
    )
    return reference_root, phase33_root


def test_phase33_handoff_is_reproducible_and_exact_outside_boundary(tmp_path) -> None:
    reference_root, phase33_root = _prepare_inputs(tmp_path)
    output_root = tmp_path / "output"
    report = build_phase33_team_handoff(reference_root, phase33_root, output_root)
    assert report["rows"] == 601
    assert report["phase33_boundary_rows"] == 600
    assert report["phase33_exact_fallback_rows"] == 1
    assert report["production_unseen_inference_ready"] is False
    assert report["provenance"]["output_predictions_sha256"]
    with (output_root / "oof_3d_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["toxicity_probability"] == rows[-1]["phase33_reference_probability"]
    assert rows[-1]["phase33_boundary_member"] == "0"


def test_phase33_handoff_rejects_unauthorized_decision(tmp_path) -> None:
    reference_root, phase33_root = _prepare_inputs(tmp_path)
    decision_path = phase33_root / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["promotion_authorized"] = False
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(PermissionError, match="does not authorize"):
        build_phase33_team_handoff(reference_root, phase33_root, tmp_path / "output")


def test_phase33_handoff_rejects_non_boundary_changes(tmp_path) -> None:
    reference_root, phase33_root = _prepare_inputs(tmp_path)
    path = phase33_root / "oof_predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[-1]["candidate_probability"] = "0.5"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="residual mismatch|exact fallback"):
        build_phase33_team_handoff(reference_root, phase33_root, tmp_path / "output")
