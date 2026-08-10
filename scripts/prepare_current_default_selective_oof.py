#!/usr/bin/env python
"""Build a provenance-checked canonical OOF table for selective prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = labels == 0
    tp = int(np.sum(predictions & positive))
    tn = int(np.sum(~predictions & negative))
    fp = int(np.sum(predictions & negative))
    fn = int(np.sum(~predictions & positive))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "mcc": float((tp * tn - fp * fn) / denominator),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def parse_args() -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--one-d-oof",
        type=Path,
        default=root / "runs" / "lora_mamba_full" / "oof_calibrated.json",
    )
    parser.add_argument(
        "--three-d-oof",
        type=Path,
        default=root
        / "runs"
        / "conformer_ensemble_residual_oof"
        / "full_calibrated_predictions.csv",
    )
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=root / "data" / "folds" / "train_cdhit080_fallback_folds.csv",
    )
    parser.add_argument(
        "--fusion-refreeze",
        type=Path,
        default=root / "runs" / "lora_mamba_full" / "fusion_refreeze.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "selective_prediction_current_default_input_20260810_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _root()
    inputs = {
        "one_d_oof": args.one_d_oof.resolve(),
        "three_d_oof": args.three_d_oof.resolve(),
        "fold_manifest": args.fold_manifest.resolve(),
        "fusion_refreeze": args.fusion_refreeze.resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    one_d = json.loads(inputs["one_d_oof"].read_text(encoding="utf-8-sig"))
    three_d = _read_csv(inputs["three_d_oof"])
    manifest = _read_csv(inputs["fold_manifest"])
    refreeze = json.loads(inputs["fusion_refreeze"].read_text(encoding="utf-8-sig"))
    n = len(manifest)
    one_d_fields = ("oof", "labels", "sample_ids", "folds")
    if any(len(one_d[field]) != n for field in one_d_fields) or len(three_d) != n:
        raise ValueError("Canonical 1D, 3D, and fold-manifest row counts differ.")

    one_d_probability = np.asarray(one_d["oof"], dtype=np.float64)
    three_d_probability = np.asarray(
        [float(row["toxicity_probability"]) for row in three_d], dtype=np.float64
    )
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in manifest], dtype=np.int64)
    sample_ids = [row["sample_id"] for row in manifest]

    output_rows: list[dict[str, Any]] = []
    for index, (fold_row, three_d_row) in enumerate(zip(manifest, three_d)):
        expected_index = str(index)
        identity_values = {
            "source_index": (fold_row["source_index"], three_d_row["source_index"], expected_index),
            "sample_id": (fold_row["sample_id"], three_d_row["sample_id"], str(one_d["sample_ids"][index])),
            "label": (fold_row["label"], three_d_row["label"], str(one_d["labels"][index])),
            "fold": (fold_row["fold"], three_d_row["fold"], str(one_d["folds"][index])),
        }
        for field, values in identity_values.items():
            if len(set(values)) != 1:
                raise ValueError(f"Canonical {field} mismatch at row {index}: {values}")
        if fold_row["sequence"] != three_d_row["sequence"]:
            raise ValueError(f"Canonical sequence mismatch at row {index}.")

    weight_1d = float(refreeze["best_weight_1d"])
    threshold = float(refreeze["best_threshold"])
    fused = weight_1d * one_d_probability + (1.0 - weight_1d) * three_d_probability
    if not np.all(np.isfinite(fused)) or np.any((fused < 0.0) | (fused > 1.0)):
        raise ValueError("Reconstructed canonical fusion probabilities are invalid.")
    reproduced = _metrics(labels, fused, threshold)
    expected_metrics = refreeze["fusion_oof_metrics"]
    differences = {
        name: abs(float(reproduced[name]) - float(expected_metrics[name]))
        for name in expected_metrics
    }
    if any(value > 1e-12 for value in differences.values()):
        raise ValueError(f"Canonical refreeze metric mismatch: {differences}")

    for index in range(n):
        output_rows.append(
            {
                "source_index": index,
                "sample_id": sample_ids[index],
                "fold": int(folds[index]),
                "label": int(labels[index]),
                "p_1d": repr(float(one_d_probability[index])),
                "p_3d_current_default": repr(float(three_d_probability[index])),
                "p_fusion_current_default": repr(float(fused[index])),
            }
        )

    prediction_path = output / "oof_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    manifest_rows = []
    for role, path in inputs.items():
        manifest_rows.append(
            {
                "role": role,
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
        )
    input_manifest_path = output / "input_manifest.csv"
    with input_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("role", "path", "sha256"))
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "artifact_type": "ghxtox_current_default_selective_oof_input",
        "schema_version": 1,
        "n": n,
        "weight_1d": weight_1d,
        "weight_3d": 1.0 - weight_1d,
        "decision_threshold": threshold,
        "probability_columns": {
            "one_d": "p_1d",
            "three_d": "p_3d_current_default",
            "fused": "p_fusion_current_default",
        },
        "reproduced_fusion_oof_metrics": reproduced,
        "absolute_metric_differences_vs_refreeze": differences,
        "test1_or_test2_used": False,
        "canonical_binary_model_changed": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Current-default selective-prediction OOF input\n\n"
        "This derived table joins the frozen calibrated 1D and current canonical "
        "single-structure 3D OOF streams after strict row-identity checks. It exactly "
        "reproduces `fusion_refreeze.json` and does not use Test1 or Test2.\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.name}")
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
