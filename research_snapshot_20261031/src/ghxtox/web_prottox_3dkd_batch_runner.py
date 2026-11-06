"""JSON-stdin batch runner for frozen ProtTox-3DKD v1."""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from ghxtox.prottox_3dkd_predict import predict_batch


STANDARD_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_BATCH = 200
MAX_SEQUENCE = 200


def _clean_sequence(sequence: str) -> str:
    return "".join(sequence.upper().split())


def _validate(item: dict[str, Any]) -> tuple[str, str, list[str], str | None]:
    sequence = _clean_sequence(str(item.get("sequence", "")))
    warnings: list[str] = []
    if not sequence:
        return "", sequence, warnings, "Sequence is empty."
    invalid = sorted(set(sequence) - STANDARD_AMINO_ACIDS)
    if invalid:
        return "", sequence, warnings, f"Non-standard residue(s): {', '.join(invalid)}."
    if len(sequence) < 3:
        return "", sequence, warnings, "Sequence is shorter than 3 residues."
    if len(sequence) > MAX_SEQUENCE:
        return "", sequence, warnings, f"Sequence exceeds the {MAX_SEQUENCE}-residue batch limit."
    if len(sequence) > 50:
        warnings.append("Length exceeds the audited 3-50 aa context; prediction is informational.")
    return sequence, sequence, warnings, None


def run(request: dict[str, Any]) -> dict[str, Any]:
    raw_items = request.get("sequences") or []
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Request must contain a non-empty 'sequences' list.")
    if len(raw_items) > MAX_BATCH:
        raise ValueError(f"Batch limit is {MAX_BATCH} sequences per request.")
    manifest = os.environ.get("GHXTOX_PROTTOX_3DKD_MANIFEST", "").strip()
    if not manifest:
        raise RuntimeError("Set GHXTOX_PROTTOX_3DKD_MANIFEST before batch prediction.")
    errors_by_index: dict[int, dict[str, Any]] = {}
    valid_items: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        item_id = str(raw_item.get("id") or raw_item.get("name") or f"item_{index + 1}")
        cleaned, original_sequence, warnings, error = _validate(raw_item)
        if error is not None:
            errors_by_index[index] = {"id": item_id, "sequence": original_sequence, "status": "error", "error": error}
            continue
        valid_items.append({"id": item_id, "sequence": cleaned, "_warnings": warnings, "_raw": raw_item})
    predicted_by_id: dict[str, dict[str, Any]] = {}
    if valid_items:
        predicted = predict_batch(
            [{"id": item["id"], "sequence": item["sequence"]} for item in valid_items],
            manifest,
            model_path=os.environ.get("GHXTOX_PROTTOX_3DKD_MODEL_PATH") or None,
            device=os.environ.get("GHXTOX_PROTTOX_3DKD_DEVICE", "cuda"),
        )
        for item, result in zip(valid_items, predicted, strict=True):
            probability = float(result["probability"])
            margin = abs(probability - 0.5)
            confidence = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
            predicted_by_id[item["id"]] = {
                "id": item["id"],
                "sequence": item["sequence"],
                "toxicity_probability": probability,
                "prediction": result["label"],
                "confidence": confidence,
                "sequence_length": len(item["sequence"]),
                "status": "success",
                "warning": "; ".join(item["_warnings"]) if item["_warnings"] else "",
            }
    results: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        item_id = str(raw_item.get("id") or raw_item.get("name") or f"item_{index + 1}")
        if index in errors_by_index:
            results.append(errors_by_index[index])
        else:
            results.append(predicted_by_id[item_id])
    return {"model_version": "ProtTox-3DKD-v1", "total": len(raw_items), "results": results}


def main() -> None:
    try:
        print(json.dumps(run(json.load(sys.stdin)), ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
