"""Regenerate the promoted Phase 33 train-only 3D OOF team handoff."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.bootstrap_ci import _metrics
from ghxtox.oof_source_diagnostics import _read_csv
from ghxtox.three_d_team_handoff import _sha256, _write_csv
from ghxtox.utils import save_json


def _by_source_index(
    rows: list[dict[str, str]], name: str
) -> dict[int, dict[str, str]]:
    if not rows or "source_index" not in rows[0]:
        raise ValueError(f"{name} is empty or lacks source_index.")
    indexed = {int(row["source_index"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{name} contains duplicate source_index rows.")
    return indexed


def _stable_logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-7), 1.0 - 1e-7)
    return math.log(clipped / (1.0 - clipped))


def _binary_entropy(probability: float) -> float:
    clipped = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return -(clipped * math.log2(clipped) + (1.0 - clipped) * math.log2(1.0 - clipped))


def build_phase33_team_handoff(
    reference_root: str | Path,
    phase33_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Promote an authorized Phase33 OOF stream without changing its protocol."""

    reference_dir = Path(reference_root)
    phase33_dir = Path(phase33_root)
    output_dir = Path(output_root)
    reference_predictions_path = reference_dir / "oof_3d_predictions.csv"
    reference_quality_path = reference_dir / "oof_3d_quality.csv"
    reference_contract_path = reference_dir / "handoff_contract.json"
    phase33_predictions_path = phase33_dir / "oof_predictions.csv"
    phase33_summary_path = phase33_dir / "summary.json"
    phase33_decision_path = phase33_dir / "decision.json"

    reference_contract = json.loads(reference_contract_path.read_text(encoding="utf-8"))
    decision = json.loads(phase33_decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("decision") != "promote_phase33_3d_handoff"
        or not bool(decision.get("promotion_authorized"))
        or not bool(decision.get("handoff_regeneration_authorized"))
        or not bool(decision.get("gate", {}).get("all_conditions_passed"))
    ):
        raise PermissionError("Phase33 promotion gate does not authorize a handoff.")
    if bool(decision.get("external_test_labels_read", True)):
        raise PermissionError("Phase33 decision does not keep external labels sealed.")

    threshold = float(reference_contract["fixed_threshold"])
    reference_predictions = _by_source_index(
        _read_csv(reference_predictions_path), "Phase25 reference predictions"
    )
    reference_quality = _by_source_index(
        _read_csv(reference_quality_path), "Phase25 reference quality"
    )
    phase33_predictions = _by_source_index(
        _read_csv(phase33_predictions_path), "Phase33 predictions"
    )
    expected = set(reference_predictions)
    if expected != set(reference_quality) or expected != set(phase33_predictions):
        raise ValueError(
            "Phase25 and Phase33 handoff inputs do not cover identical rows."
        )
    if len(expected) != int(reference_contract["rows"]):
        raise ValueError("Phase25 reference row count differs from its contract.")

    prediction_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    labels: list[int] = []
    reference_probabilities: list[float] = []
    candidate_probabilities: list[float] = []
    boundary_rows = 0
    fallback_rows = 0
    bioemu_rows = 0
    maximum_delta = 0.0
    for source_index in sorted(expected):
        reference = reference_predictions[source_index]
        quality = reference_quality[source_index]
        phase33 = phase33_predictions[source_index]
        for key in ("sample_id", "fold", "label", "sequence"):
            if str(reference[key]) != str(phase33[key]):
                raise ValueError(
                    f"Phase33 alignment mismatch at source_index {source_index}: {key}."
                )
        if int(reference["fold"]) != int(quality["fold"]):
            raise ValueError(f"Quality fold mismatch at source_index {source_index}.")

        reference_probability = float(reference["toxicity_probability"])
        reported_reference = float(phase33["reference_probability"])
        candidate_probability = float(phase33["candidate_probability"])
        delta = float(phase33["bounded_logit_delta"])
        boundary = int(phase33["phase33_boundary_member"])
        available = int(phase33["bioemu_boundary_available"])
        if boundary not in {0, 1} or available not in {0, 1}:
            raise ValueError(f"Invalid Phase33 masks at source_index {source_index}.")
        if not all(
            np.isfinite(value)
            for value in (
                reference_probability,
                reported_reference,
                candidate_probability,
                delta,
            )
        ):
            raise ValueError(
                f"Non-finite Phase33 value at source_index {source_index}."
            )
        if not 0.0 <= candidate_probability <= 1.0:
            raise ValueError(
                f"Invalid Phase33 probability at source_index {source_index}."
            )
        if not np.isclose(
            reference_probability, reported_reference, atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                f"Phase33 reference mismatch at source_index {source_index}."
            )
        maximum_delta = max(maximum_delta, abs(delta))
        if abs(delta) > 0.15 + 1e-12:
            raise ValueError(
                f"Phase33 residual exceeds 0.15 at source_index {source_index}."
            )
        if boundary:
            recombined = 1.0 / (
                1.0 + math.exp(-(_stable_logit(reference_probability) + delta))
            )
            if not np.isclose(
                candidate_probability, recombined, atol=1e-10, rtol=1e-10
            ):
                raise ValueError(
                    f"Phase33 residual mismatch at source_index {source_index}."
                )
            boundary_rows += 1
            bioemu_rows += available
            if not available:
                raise ValueError(
                    f"Boundary row lacks BioEmu evidence at source_index {source_index}."
                )
        else:
            fallback_rows += 1
            if (
                available
                or delta != 0.0
                or candidate_probability != reference_probability
            ):
                raise ValueError(
                    f"Non-boundary row is not an exact fallback at source_index {source_index}."
                )

        candidate_prediction = int(candidate_probability >= threshold)
        if int(phase33["candidate_prediction"]) != candidate_prediction:
            raise ValueError(
                f"Phase33 fixed-threshold prediction mismatch at source_index {source_index}."
            )
        prediction_row = dict(reference)
        prediction_row.update(
            {
                "toxicity_probability": candidate_probability,
                "logit": _stable_logit(candidate_probability),
                "prediction": candidate_prediction,
                "decision_threshold": threshold,
                "probability_margin": 2.0 * abs(candidate_probability - 0.5),
                "probability_entropy": _binary_entropy(candidate_probability),
                "mc_mean_probability": candidate_probability,
                "phase33_reference_probability": reference_probability,
                "phase33_boundary_member": boundary,
                "phase33_bioemu_available": available,
                "phase33_standalone_3d_probability": phase33[
                    "standalone_3d_probability"
                ],
                "phase33_bounded_logit_delta": delta,
                "phase33_candidate_probability": candidate_probability,
                "phase33_candidate_prediction": candidate_prediction,
            }
        )
        prediction_rows.append(prediction_row)

        quality_row = dict(quality)
        quality_row.update(
            {
                "phase33_boundary_member": boundary,
                "phase33_bioemu_available": available,
                "phase33_bounded_logit_delta": delta,
                "phase33_standalone_3d_probability": phase33[
                    "standalone_3d_probability"
                ],
            }
        )
        quality_rows.append(quality_row)
        labels.append(int(reference["label"]))
        reference_probabilities.append(reference_probability)
        candidate_probabilities.append(candidate_probability)

    if boundary_rows != 600 or fallback_rows != len(expected) - 600:
        raise ValueError(
            f"Expected 600 boundary rows, found {boundary_rows}; fallback={fallback_rows}."
        )
    if bioemu_rows != boundary_rows:
        raise ValueError("Not every frozen boundary row has BioEmu evidence.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "oof_3d_predictions.csv"
    quality_path = output_dir / "oof_3d_quality.csv"
    _write_csv(prediction_rows, prediction_path)
    _write_csv(quality_rows, quality_path)
    labels_array = np.asarray(labels, dtype=np.int64)
    reference_array = np.asarray(reference_probabilities, dtype=np.float64)
    candidate_array = np.asarray(candidate_probabilities, dtype=np.float64)
    source_counts = Counter(row["generator_source"] for row in quality_rows)
    report = {
        "status": "complete",
        "artifact": "promoted_phase33_boundary_bioemu_3d_oof_team_handoff",
        "artifact_scope": "train_oof_only",
        "production_unseen_inference_ready": False,
        "external_test_labels_read": False,
        "fusion_weights_included": False,
        "probabilities_recalibrated": False,
        "threshold_retuned": False,
        "fixed_threshold": threshold,
        "rows": len(prediction_rows),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "phase33_boundary_rows": boundary_rows,
        "phase33_exact_fallback_rows": fallback_rows,
        "phase33_bioemu_available_rows": bioemu_rows,
        "maximum_absolute_logit_delta": maximum_delta,
        "generator_source_counts": dict(sorted(source_counts.items())),
        "phase33_decision": decision,
        "metrics_at_fixed_threshold": _metrics(
            labels_array, candidate_array, threshold
        ),
        "reference_metrics_at_fixed_threshold": _metrics(
            labels_array, reference_array, threshold
        ),
        "decision_transitions": decision["decision_transitions"],
        "gate": decision["gate"],
        "provenance": {
            "reference_handoff_predictions": str(reference_predictions_path),
            "reference_handoff_predictions_sha256": _sha256(reference_predictions_path),
            "reference_handoff_quality": str(reference_quality_path),
            "reference_handoff_quality_sha256": _sha256(reference_quality_path),
            "reference_handoff_contract": str(reference_contract_path),
            "reference_handoff_contract_sha256": _sha256(reference_contract_path),
            "phase33_summary": str(phase33_summary_path),
            "phase33_summary_sha256": _sha256(phase33_summary_path),
            "phase33_decision": str(phase33_decision_path),
            "phase33_decision_sha256": _sha256(phase33_decision_path),
            "phase33_oof": str(phase33_predictions_path),
            "phase33_oof_sha256": _sha256(phase33_predictions_path),
            "output_predictions_sha256": _sha256(prediction_path),
            "output_quality_sha256": _sha256(quality_path),
        },
        "handoff_rules": [
            "Join only by source_index and verify sample_id, sequence, label, and fold alignment.",
            "Use toxicity_probability as the promoted Phase33 standalone 3D OOF score.",
            "Rows with phase33_boundary_member=0 are exact Phase25 fallback scores.",
            "This artifact is train OOF only and does not authorize unseen/Test inference.",
            "A strict fusion claim requires outer-nested upstream 1D/2D/3D streams; ordinary second-level OOF cross-fitting is exploratory.",
            "Never fit fusion parameters, calibration, thresholds, or model selection on Test1/Test2.",
        ],
        "schema": {
            "oof_3d_predictions.csv": list(prediction_rows[0]),
            "oof_3d_quality.csv": list(quality_rows[0]),
        },
    }
    save_json(report, output_dir / "handoff_contract.json")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Phase33 Promoted 3D OOF Team Handoff",
                "",
                "This directory is a train-only OOF input contract for downstream 1D/2D/3D fusion.",
                "It is not a production/Test inference artifact.",
                "",
                "Use `oof_3d_predictions.csv::toxicity_probability` as the standalone 3D OOF score and join only by `source_index` after verifying sample/fold/label alignment.",
                f"The fixed 3D threshold remains `{threshold}`.",
                "",
                "Phase33 changes only the frozen 600 BioEmu decision-boundary rows; all other rows are exact Phase25 fallbacks.",
                "For a formal fusion claim, regenerate outer-nested upstream 1D/2D/3D streams. A meta-model fitted to ordinary OOF tables is exploratory only.",
                "",
                "See `handoff_contract.json` for hashes, schema, provenance, and usage rules.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="runs/three_d_team_handoff")
    parser.add_argument("--phase33-root", default="runs/bioemu_boundary_specialist_3d")
    parser.add_argument("--output-root", default="runs/three_d_team_handoff_phase33")
    args = parser.parse_args()
    report = build_phase33_team_handoff(
        args.reference_root, args.phase33_root, args.output_root
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "rows": report["rows"],
                "boundary_rows": report["phase33_boundary_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
