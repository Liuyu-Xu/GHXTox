"""Audited cache and checkpoint contracts for structure-guided ProtTox students.

Only sequence-derived ProtTox features reach the deployed student.  The three
structure-quality fields are offline training targets from an aligned OOF file;
they are never model inputs and never required at inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ghxtox.models.structure_quality_proxy import (
    StructureQualityProxyConfig,
    StructureQualityProxyStudent,
)
from ghxtox.prottox_tsk import (
    CACHE_FORMAT_VERSION,
    MODEL_NAME as BASE_MODEL_NAME,
    MODEL_VERSION as BASE_MODEL_VERSION,
    file_sha256,
    write_manifest,
)


MODEL_NAME = "ProtTox-StructGuide"
MODEL_VERSION = "prottox-struct-guide-v1"
CACHE_FORMAT_VERSION = 1
AUXILIARY_TARGET_NAMES = ("mean_plddt", "min_plddt", "low_plddt_fraction")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def build_cache(
    base_cache_path: str | Path,
    fold_manifest_path: str | Path,
    structure_oof_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Attach full-cohort, label-independent structure-quality targets safely."""

    base_path, folds_path, structure_path, output = map(
        Path, (base_cache_path, fold_manifest_path, structure_oof_path, output_path)
    )
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    if (
        base.get("format_version") != CACHE_FORMAT_VERSION
        or base.get("model_name") != BASE_MODEL_NAME
        or base.get("model_version") != BASE_MODEL_VERSION
    ):
        raise ValueError("Base cache is not a compatible ProtTox-TSK feature cache")
    records = list(base.get("records", []))
    if not records:
        raise ValueError("Base cache has no records")

    fold_rows = _read_csv(folds_path)
    required_fold = {"sample_id", "historical_sample_id", "sequence", "label", "fold"}
    if not required_fold <= set(fold_rows[0]):
        raise ValueError(f"Fold manifest must contain {sorted(required_fold)}")
    folds = {str(row["sample_id"]): row for row in fold_rows}
    if len(folds) != len(fold_rows):
        raise ValueError("Fold manifest has duplicate independent sample IDs")

    structure_rows = _read_csv(structure_path)
    required_structure = {"sample_id", "sequence", "label", "fold", *AUXILIARY_TARGET_NAMES}
    if not required_structure <= set(structure_rows[0]):
        raise ValueError(f"Structure OOF must contain {sorted(required_structure)}")
    structure = {str(row["sample_id"]): row for row in structure_rows}
    if len(structure) != len(structure_rows):
        raise ValueError("Structure OOF has duplicate historical sample IDs")

    joined: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        fold_row = folds.get(sample_id)
        if fold_row is None:
            raise ValueError(f"Base-cache sample is absent from fold manifest: {sample_id}")
        structure_row = structure.get(str(fold_row["historical_sample_id"]))
        if structure_row is None:
            raise ValueError(f"No structure OOF row for {fold_row['historical_sample_id']}")
        sequence = str(record["sequence"]).strip().upper()
        if sequence != str(fold_row["sequence"]).strip().upper() or sequence != str(structure_row["sequence"]).strip().upper():
            raise ValueError(f"Sequence mismatch while joining structure target for {sample_id}")
        if int(record["label"]) != int(fold_row["label"]) or int(record["label"]) != int(structure_row["label"]):
            raise ValueError(f"Label mismatch while joining structure target for {sample_id}")
        if int(fold_row["fold"]) != int(structure_row["fold"]):
            raise ValueError(f"Fold mismatch while joining structure target for {sample_id}")
        targets = [float(structure_row[name]) for name in AUXILIARY_TARGET_NAMES]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in targets):
            raise ValueError(f"Invalid structure-quality target for {sample_id}")
        joined.append({**record, "fold": int(fold_row["fold"]), "historical_sample_id": str(fold_row["historical_sample_id"]), "structure_quality_targets": torch.tensor(targets, dtype=torch.float32)})

    payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "base_cache": {"path": str(base_path), "sha256": file_sha256(base_path), "model_name": BASE_MODEL_NAME, "model_version": BASE_MODEL_VERSION},
        "fold_manifest": {"path": str(folds_path), "sha256": file_sha256(folds_path)},
        "structure_quality_source": {"path": str(structure_path), "sha256": file_sha256(structure_path), "fields": list(AUXILIARY_TARGET_NAMES), "scope": "training-only offline structure-quality targets; not inference inputs"},
        "encoder": dict(base["encoder"]),
        "records": joined,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "cache": {"path": str(output), "sha256": file_sha256(output), "records": len(joined)},
        "base_cache": payload["base_cache"],
        "fold_manifest": payload["fold_manifest"],
        "structure_quality_source": payload["structure_quality_source"],
        "checks": {"exact_sequence_mapping": True, "exact_label_mapping": True, "exact_fold_mapping": True, "all_targets_finite": True, "test_data_used": False},
    }
    write_manifest(str(output) + ".manifest.json", manifest)
    return manifest


def checkpoint_payload(
    model: StructureQualityProxyStudent,
    *,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    auxiliary_mean: torch.Tensor,
    auxiliary_std: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "feature_mean": feature_mean.detach().cpu().float(),
        "feature_std": feature_std.detach().cpu().float(),
        "auxiliary_mean": auxiliary_mean.detach().cpu().float(),
        "auxiliary_std": auxiliary_std.detach().cpu().float(),
        "metadata": metadata,
    }


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[StructureQualityProxyStudent, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Checkpoint is not a compatible ProtTox-StructGuide v1 checkpoint")
    model = StructureQualityProxyStudent(StructureQualityProxyConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    feature_mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    auxiliary_mean = torch.as_tensor(payload["auxiliary_mean"], dtype=torch.float32, device=device)
    auxiliary_std = torch.as_tensor(payload["auxiliary_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    if feature_mean.numel() != model.config.input_dim or feature_std.numel() != model.config.input_dim:
        raise ValueError("Checkpoint feature normalization does not match model input")
    if auxiliary_mean.numel() != len(AUXILIARY_TARGET_NAMES) or auxiliary_std.numel() != len(AUXILIARY_TARGET_NAMES):
        raise ValueError("Checkpoint auxiliary normalization does not match target contract")
    return model, feature_mean, feature_std, auxiliary_mean, auxiliary_std, dict(payload.get("metadata", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training-only structure-quality target cache for ProtTox-StructGuide.")
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--structure-oof", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_cache(args.base_cache, args.fold_manifest, args.structure_oof, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
