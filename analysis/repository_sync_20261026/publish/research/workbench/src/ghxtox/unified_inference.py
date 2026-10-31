"""Build an auditable, training-OOF-only sequence/structure inference contract."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


CONTRACT_FIELDS = (
    "source_index", "sample_id", "sequence", "label", "fold", "group_id",
    "p_seq", "p_struct", "p_fusion", "sequence_length", "cys_count",
    "cys_fraction", "cys_span_fraction", "cys_min_gap_fraction",
    "cys_adjacent_pairs", "cys_terminal_count", "sequence_seed_dispersion",
    "sequence_seed_dispersion_available",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def sequence_features(sequence: str) -> dict[str, float | int]:
    """Features computable before any structure prediction."""
    if not sequence:
        raise ValueError("Sequence must be non-empty")
    positions = [index for index, residue in enumerate(sequence) if residue == "C"]
    length, count = len(sequence), len(positions)
    gaps = [right - left for left, right in zip(positions, positions[1:])]
    return {
        "sequence_length": length,
        "cys_count": count,
        "cys_fraction": count / length,
        "cys_span_fraction": ((positions[-1] - positions[0]) / max(length - 1, 1)) if count >= 2 else 0.0,
        "cys_min_gap_fraction": (min(gaps) / max(length - 1, 1)) if gaps else 0.0,
        "cys_adjacent_pairs": sum(gap == 1 for gap in gaps),
        "cys_terminal_count": int(sequence[0] == "C") + int(sequence[-1] == "C"),
    }


def _load_seed_probabilities(paths: Iterable[str | Path], size: int) -> list[list[float]]:
    streams: list[list[float]] = []
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        values = document.get("oof")
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"Seed OOF stream must contain {size} ordered probabilities: {path}")
        stream = [float(value) for value in values]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in stream):
            raise ValueError(f"Invalid sequence-seed probability: {path}")
        streams.append(stream)
    return streams


def build_unified_oof_contract(
    prediction_csv: str | Path,
    processed_train: str | Path,
    fold_manifest: str | Path,
    output_csv: str | Path,
    seed_oof_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Join the retained OOF streams with raw sequence information by stable identity."""
    prediction_path, processed_path, manifest_path, output_path = map(Path, (prediction_csv, processed_train, fold_manifest, output_csv))
    predictions = read_csv(prediction_path)
    manifest = read_csv(manifest_path)
    records = torch.load(processed_path, map_location="cpu", weights_only=False)["records"]
    size = len(records)
    if len(predictions) != size or len(manifest) != size:
        raise ValueError("Prediction, manifest, and processed-training row counts must agree")
    seed_paths = [Path(path) for path in seed_oof_paths]
    seeds = _load_seed_probabilities(seed_paths, size)
    rows: list[dict[str, Any]] = []
    for expected_index, (prediction, fold_row, record) in enumerate(zip(predictions, manifest, records)):
        for source in (prediction, fold_row):
            if int(source["source_index"]) != expected_index:
                raise ValueError(f"Source indices must be ordered and complete at row {expected_index}")
        expected = (str(record["sample_id"]), record["sequence"], str(int(record["label"])))
        observed = (prediction["sample_id"], fold_row["sequence"], prediction["label"])
        manifest_identity = (fold_row["sample_id"], fold_row["sequence"], fold_row["label"])
        if observed != expected or manifest_identity != expected:
            raise ValueError(f"OOF identity mismatch at source_index={expected_index}")
        if prediction["fold"] != fold_row["fold"]:
            raise ValueError(f"OOF fold mismatch at source_index={expected_index}")
        probabilities = {"p_seq": float(prediction["p_1d"]), "p_struct": float(prediction["p_3d_current_default"]), "p_fusion": float(prediction["p_fusion_current_default"])}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise ValueError(f"Invalid probability at source_index={expected_index}")
        features = sequence_features(record["sequence"])
        dispersion = 0.0 if not seeds else math.sqrt(sum((stream[expected_index] - sum(s[expected_index] for s in seeds) / len(seeds)) ** 2 for stream in seeds) / len(seeds))
        rows.append({
            "source_index": expected_index, "sample_id": record["sample_id"], "sequence": record["sequence"],
            "label": int(record["label"]), "fold": int(fold_row["fold"]),
            "group_id": fold_row.get("group_id", fold_row.get("group", "")), **probabilities, **features,
            "sequence_seed_dispersion": dispersion, "sequence_seed_dispersion_available": int(bool(seeds)),
        })
    if any(not row["group_id"] for row in rows):
        raise ValueError("Each contract row requires a non-empty group_id")
    groups = {row["group_id"]: row["fold"] for row in rows}
    if any(groups[row["group_id"]] != row["fold"] for row in rows):
        raise ValueError("Sequence group crosses OOF folds")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTRACT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "artifact_type": "ghxtox_unified_inference_oof_contract", "version": 1, "rows": size,
        "contract": str(output_path.resolve()), "contract_sha256": sha256(output_path),
        "prediction_source": str(prediction_path.resolve()), "prediction_source_sha256": sha256(prediction_path),
        "processed_train": str(processed_path.resolve()), "processed_train_sha256": sha256(processed_path),
        "fold_manifest": str(manifest_path.resolve()), "fold_manifest_sha256": sha256(manifest_path),
        "sequence_seed_oof": [str(path.resolve()) for path in seed_paths],
        "sequence_seed_oof_sha256": [sha256(path) for path in seed_paths],
        "pre_esmfold_feature_columns": list(CONTRACT_FIELDS[9:]),
        "post_structure_columns_excluded_from_router_features": ["p_struct", "p_fusion"],
        "external_test_labels_used": False,
    }

