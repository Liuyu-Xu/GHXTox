"""JSON-stdin web runner for MPCE-3DKD evidence predictions.

Single-request mode runs detailed evidence by default.  Batch mode uses the
frozen base policy: prediction + prototype/cross-view evidence only, without
the perturbation budget screen.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from ghxtox.local_web_api import _validate_sequence
from ghxtox.mpce_3dkd_evidence import predict_batch_base, predict_evidence


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_BATCH = 200


def _env_path() -> tuple[str, str, str]:
    manifest = os.environ.get("GHXTOX_MPCE_3DKD_MANIFEST", "").strip()
    model_path = os.environ.get("GHXTOX_MPCE_3DKD_MODEL_PATH", "").strip() or None
    device = os.environ.get("GHXTOX_MPCE_3DKD_DEVICE", "cuda")
    if not manifest:
        raise RuntimeError("Set GHXTOX_MPCE_3DKD_MANIFEST before MPCE-3DKD web prediction.")
    return manifest, model_path, device


def _single(request: dict[str, Any]) -> dict[str, Any]:
    sequence = _validate_sequence(str(request["sequence"]))
    analysis_level = str(request.get("analysis_level", "detailed"))
    if analysis_level not in {"base", "detailed"}:
        raise ValueError("analysis_level must be 'base' or 'detailed'")
    manifest, model_path, device = _env_path()
    evidence = predict_evidence(
        sequence,
        manifest,
        analysis_level=analysis_level,
        model_path=model_path,
        device=device,
    )
    probability = float(evidence["probability"])
    margin = abs(probability - 0.5)
    confidence = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
    phys = evidence.get("physchem", {})
    return {
        "id": f"mpce_3dkd_{uuid.uuid4().hex}",
        "name": str(request.get("name") or "sequence"),
        "sequence": sequence,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "modelVersion": f"MPCE-3DKD {evidence.get('model_version')} · evidence schema {evidence.get('evidence_schema_version')}",
        "toxicity_probability": probability,
        "display_probability": float(evidence["calibrated_probability"]),
        "prediction": evidence["prediction"],
        "label": evidence["prediction"],
        "confidence": confidence,
        "decisionThreshold": 0.5,
        "route": "mpce-3dkd",
        "analysisLevel": analysis_level,
        "routeReason": "MPCE-3DKD sequence-only inference with evidence layers; no ESMFold/3D at prediction time.",
        "sequence_evidence": {
            "net_charge_ph7": phys.get("netCharge"),
            "basic_fraction": phys.get("basicFraction"),
            "hydrophobic_fraction": phys.get("hydrophobicFraction"),
            "aromatic_fraction": phys.get("aromaticFraction"),
            "cysteine_count": phys.get("cysteineCount"),
        },
        "evidence_schema_version": evidence.get("evidence_schema_version"),
        "model_architecture": evidence.get("model_architecture"),
        "model_version": evidence.get("model_version"),
        "prototype_profile_version": evidence.get("prototype_profile_version"),
        "prototype_profile_sha256": evidence.get("prototype_profile_sha256"),
        "prototype_evidence": evidence["prototype_evidence"],
        "cross_view_evidence": evidence["cross_view_evidence"],
        "perturbation_stability": evidence["perturbation_stability"],
        "evidence_record": evidence["evidence_record"],
        "provenance": evidence["provenance"],
        "reviewNotes": [
            "Research prediction only; not an experimental toxicity conclusion.",
            "Evidence fields are model-associated, not experimental causality.",
        ],
        "isMock": False,
        "recommendFullAnalysis": False,
    }


def _clean_sequence(sequence: str) -> str:
    return "".join(sequence.upper().split())


def _batch(request: dict[str, Any]) -> dict[str, Any]:
    raw_items = request.get("sequences") or []
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Batch request must contain a non-empty 'sequences' list.")
    if len(raw_items) > MAX_BATCH:
        raise ValueError(f"Batch limit is {MAX_BATCH}.")
    errors_by_index: dict[int, dict[str, Any]] = {}
    valid: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        item_id = str(raw_item.get("id") or raw_item.get("name") or f"item_{index + 1}")
        sequence = _clean_sequence(str(raw_item.get("sequence", "")))
        if not sequence:
            errors_by_index[index] = {"id": item_id, "sequence": "", "status": "error", "error": "Sequence is empty."}
            continue
        invalid = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
        if invalid:
            errors_by_index[index] = {"id": item_id, "sequence": sequence, "status": "error", "error": f"Non-standard residue(s): {', '.join(invalid)}."}
            continue
        if not 3 <= len(sequence) <= 50:
            errors_by_index[index] = {"id": item_id, "sequence": sequence, "status": "error", "error": "Sequence must be 3-50 residues in the audited context."}
            continue
        valid.append({"id": item_id, "sequence": sequence})
    manifest, model_path, device = _env_path()
    results_by_id: dict[str, dict[str, Any]] = {}
    if valid:
        predicted = predict_batch_base(
            [item["sequence"] for item in valid],
            manifest,
            model_path=model_path,
            device=device,
        )
        for item, result in zip(valid, predicted, strict=True):
            probability = float(result["probability"])
            margin = abs(probability - 0.5)
            confidence = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
            results_by_id[item["id"]] = {
                "id": item["id"],
                "sequence": item["sequence"],
                "toxicity_probability": probability,
                "prediction": result["label"],
                "confidence": confidence,
                "sequence_length": len(item["sequence"]),
                "status": "success",
                "evidence_schema_version": result.get("evidence_schema_version"),
                "prototype_evidence": result.get("prototype_evidence"),
                "cross_view_evidence": result.get("cross_view_evidence"),
                "provenance": result.get("provenance"),
            }
    results: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        item_id = str(raw_item.get("id") or raw_item.get("name") or f"item_{index + 1}")
        if index in errors_by_index:
            results.append(errors_by_index[index])
        else:
            results.append(results_by_id[item_id])
    return {"model_version": "MPCE-3DKD-v1.0-evidence", "total": len(raw_items), "results": results}


def run(request: dict[str, Any]) -> dict[str, Any]:
    if "sequences" in request:
        return _batch(request)
    return _single(request)


def main() -> None:
    try:
        print(json.dumps(run(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
