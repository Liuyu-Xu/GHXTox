from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from ghxtox.short_margin_3d import aggregate_candidate, verify_protocol
from ghxtox.oof import _resolve_evaluation_threshold
from ghxtox.train import _decision_margin_penalty


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_decision_margin_penalty_uses_fixed_logit_boundary() -> None:
    threshold = 0.7
    boundary = torch.logit(torch.tensor(threshold))
    penalties = _decision_margin_penalty(
        torch.tensor([boundary, boundary, boundary + 1.0, boundary - 1.0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        threshold=threshold,
        margin=0.3,
        positive_multiplier=2.0,
    )
    assert penalties.tolist() == pytest.approx([0.6, 0.3, 0.0, 0.0])


def test_oof_fixed_threshold_disables_threshold_search() -> None:
    logits = torch.tensor([-1.0, 0.2, 1.5, 2.0])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    threshold, metrics, protocol = _resolve_evaluation_threshold(
        logits, labels, "mcc", 0.677819
    )
    assert threshold == pytest.approx(0.677819)
    assert protocol == "externally frozen; OOF threshold search disabled"
    assert metrics["mcc"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="strictly between"):
        _resolve_evaluation_threshold(logits, labels, "mcc", 1.0)


def test_verify_protocol_rejects_changed_frozen_input(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    processed = tmp_path / "processed.pt"
    folds = tmp_path / "folds.csv"
    checkpoint = tmp_path / "fold0.pt"
    for path, content in (
        (reference, b"reference"),
        (processed, b"processed"),
        (folds, b"folds"),
        (checkpoint, b"checkpoint"),
    ):
        path.write_bytes(content)
    protocol = {
        "status": "frozen_before_candidate_oof",
        "external_test_labels_read": False,
        "reference": {"oof_path": str(reference), "oof_sha256": _digest(reference)},
        "training": {
            "processed_path": str(processed),
            "processed_sha256": _digest(processed),
            "fold_manifest": str(folds),
            "fold_manifest_sha256": _digest(folds),
            "initial_checkpoint_pattern": str(tmp_path / "fold{fold}.pt"),
            "initial_checkpoint_sha256": {"0": _digest(checkpoint)},
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    assert verify_protocol(protocol_path)["status"] == "verified"

    processed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_protocol(protocol_path)


def test_aggregate_replaces_only_frozen_pilot_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    pilot_path = tmp_path / "pilot.csv"
    output_dir = tmp_path / "output"
    threshold = 0.7
    labels = [1, 0, 1, 0, 1, 0]
    reference_probabilities = [0.60, 0.80, 0.65, 0.75, 0.91, 0.09]
    candidate_probabilities = [0.82, 0.20, 0.85, 0.15, 0.10, 0.90]
    reference_rows = []
    candidate_rows = []
    for index, (label, reference_probability, candidate_probability) in enumerate(
        zip(labels, reference_probabilities, candidate_probabilities, strict=True)
    ):
        fold = index % 2
        common = {
            "source_index": index,
            "fold": fold,
            "sample_id": f"sample_{index}",
            "sequence": "ACDE",
            "label": label,
        }
        reference_rows.append(
            {
                **common,
                "toxicity_probability": f"{reference_probability:.9g}",
                "logit": f"{math.log(reference_probability / (1-reference_probability)):.9g}",
                "prediction": int(reference_probability >= threshold),
                "decision_threshold": threshold,
                "probability_margin": abs(2 * reference_probability - 1),
                "probability_entropy": 0.0,
            }
        )
        candidate_rows.append(
            {**common, "toxicity_probability": f"{candidate_probability:.9g}"}
        )
    _write_csv(reference_path, reference_rows)
    _write_csv(candidate_path, candidate_rows)
    _write_csv(
        pilot_path,
        [
            {
                "source_index": index,
                "original_source_index": index,
                "sample_id": f"sample_{index}",
                "label": labels[index],
                "sequence": "ACDE",
                "group_id": f"group_{index}",
                "fold": index % 2,
            }
            for index in range(4)
        ],
    )
    protocol = {
        "status": "frozen_before_candidate_oof",
        "external_test_labels_read": False,
        "reference": {"oof_path": str(reference_path), "rows": 6},
        "training": {"fold_manifest": str(pilot_path), "rows": 4, "num_folds": 2},
        "evaluation": {"fixed_threshold": threshold},
        "promotion_gate": {
            "pilot250_mcc_gain_at_least": 0.005,
            "global_auprc_regression_at_most": 0.0005,
            "pilot250_auprc_regression_at_most": 0.005,
            "worst_pilot_fold_mcc_regression_at_most": 0.02,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr("ghxtox.short_margin_3d.verify_protocol", lambda _: {})

    summary = aggregate_candidate(
        protocol_path=protocol_path,
        candidate_oof=candidate_path,
        output_dir=output_dir,
    )
    with (output_dir / "oof_predictions.csv").open(encoding="utf-8", newline="") as handle:
        merged = list(csv.DictReader(handle))
    assert [float(row["toxicity_probability"]) for row in merged[:4]] == pytest.approx(
        candidate_probabilities[:4]
    )
    assert [row["toxicity_probability"] for row in merged[4:]] == [
        row["toxicity_probability"] for row in reference_rows[4:]
    ]
    assert summary["pilot_rows_replaced"] == 4
    assert summary["fallback_rows"] == 2
    assert summary["fallback_probability_exact"] is True
