"""Training-OOF audit for order-aware local sequence motifs in ProtTox-TSK errors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.models import PHYS_CHEM_FEATURE_NAMES
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.prottox_tsk_error_audit import _cache, _effect, _folds, _oof, _sign


AUDIT_VERSION = "prottox-local-motif-audit-v1"
RIDGE_LAMBDA = 1e-3
HYDROPHOBIC_RESIDUES = frozenset("AVILMFWY")
POSITIVE_RESIDUES = frozenset("KRH")
LOCAL_MOTIF_FEATURE_NAMES = (
    "max_hydrophobic_fraction_w5",
    "max_hydrophobic_fraction_w7",
    "max_positive_fraction_w5",
    "max_positive_fraction_w7",
    "max_hydrophobic_positive_mix_w5",
    "max_hydrophobic_positive_mix_w7",
    "longest_hydrophobic_run_fraction",
    "n_terminal_hydrophobic_fraction_w5",
    "c_terminal_hydrophobic_fraction_w5",
    "n_terminal_positive_fraction_w5",
    "c_terminal_positive_fraction_w5",
)


def local_motif_feature_vector(sequence: str) -> torch.Tensor:
    """Return fixed local descriptors; they are audit features, never label rules."""
    residues = "".join(sequence.upper().split())
    if not residues:
        raise ValueError("Local motif descriptors require a non-empty amino-acid sequence")
    hydrophobic = np.asarray([residue in HYDROPHOBIC_RESIDUES for residue in residues], dtype=np.float64)
    positive = np.asarray([residue in POSITIVE_RESIDUES for residue in residues], dtype=np.float64)

    def window_max(values: np.ndarray, width: int) -> float:
        actual_width = min(width, len(values))
        return max(float(values[start : start + actual_width].mean()) for start in range(len(values) - actual_width + 1))

    def terminal(values: np.ndarray, *, from_end: bool) -> float:
        actual_width = min(5, len(values))
        selected = values[-actual_width:] if from_end else values[:actual_width]
        return float(selected.mean())

    longest, current = 0, 0
    for value in hydrophobic:
        current = current + 1 if value else 0
        longest = max(longest, current)
    features = (
        window_max(hydrophobic, 5),
        window_max(hydrophobic, 7),
        window_max(positive, 5),
        window_max(positive, 7),
        window_max(hydrophobic * positive, 5),
        window_max(hydrophobic * positive, 7),
        longest / len(residues),
        terminal(hydrophobic, from_end=False),
        terminal(hydrophobic, from_end=True),
        terminal(positive, from_end=False),
        terminal(positive, from_end=True),
    )
    return torch.tensor(features, dtype=torch.float32)


def _residualize_fold_purely(features: np.ndarray, global_features: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Remove fold-pure linear predictions from existing global PhysChem features."""
    result = np.empty_like(features, dtype=np.float64)
    for fold in range(5):
        train_mask, held_mask = folds != fold, folds == fold
        if not held_mask.any() or train_mask.sum() < len(PHYS_CHEM_FEATURE_NAMES) + 1:
            raise ValueError("Each fold must have sufficient other-fold records for local motif residualization")
        train_global = global_features[train_mask]
        mean = train_global.mean(axis=0)
        std = train_global.std(axis=0)
        std[std <= 1e-12] = 1.0
        train_design = np.column_stack((np.ones(train_mask.sum()), (train_global - mean) / std))
        held_design = np.column_stack((np.ones(held_mask.sum()), (global_features[held_mask] - mean) / std))
        penalty = np.diag(np.concatenate(([0.0], np.full(train_design.shape[1] - 1, RIDGE_LAMBDA))))
        coefficients = np.linalg.solve(train_design.T @ train_design + penalty, train_design.T @ features[train_mask])
        result[held_mask] = features[held_mask] - held_design @ coefficients
    return result


def _summary(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> dict[str, Any]:
    errors = [row for row in rows if row["error_type"] == "false_negative"]
    correct = [row for row in rows if row["error_type"] == "true_positive"]
    features: list[dict[str, Any]] = []
    for feature in feature_names:
        pooled = _effect([row[feature] for row in errors], [row[feature] for row in correct])
        fold_effects: dict[str, float | None] = {}
        for fold in range(5):
            value = _effect(
                [row[feature] for row in errors if row["fold"] == fold],
                [row[feature] for row in correct if row["fold"] == fold],
            )
            fold_effects[str(fold)] = None if not math.isfinite(value) else value
        pooled_sign = _sign(pooled)
        matching_folds = sum(_sign(value) == pooled_sign for value in fold_effects.values() if value is not None and pooled_sign != 0)
        eligible_folds = sum(value is not None for value in fold_effects.values())
        features.append({
            "feature": feature,
            "false_negative_mean": float(np.mean([row[feature] for row in errors])) if errors else float("nan"),
            "true_positive_mean": float(np.mean([row[feature] for row in correct])) if correct else float("nan"),
            "standardized_mean_difference": pooled,
            "fold_effects": fold_effects,
            "eligible_folds": eligible_folds,
            "matching_sign_folds": matching_folds,
            "stable": bool(math.isfinite(pooled) and abs(pooled) >= 0.20 and matching_folds >= 4),
        })
    features.sort(key=lambda item: abs(float(item["standardized_mean_difference"])) if math.isfinite(float(item["standardized_mean_difference"])) else -1.0, reverse=True)
    return {
        "error_kind": "false_negative",
        "correct_kind": "true_positive",
        "error_records": len(errors),
        "correct_records": len(correct),
        "features": features,
        "stable_features": [item for item in features if item["stable"]],
    }


def audit(cache_path: str | Path, oof_path: str | Path, fold_path: str | Path, output_dir: str | Path, *, threshold: float = 0.5) -> dict[str, Any]:
    """Audit the retained model with exact cache/OOF/fold identity validation."""
    cache_file, oof_file, folds_file = Path(cache_path), Path(oof_path), Path(fold_path)
    cache, oof_rows, folds = _cache(cache_file), _oof(oof_file), _folds(folds_file)
    records = {str(record["sample_id"]): record for record in cache["records"]}
    oof_ids = {row["sample_id"] for row in oof_rows}
    if len(records) != len(cache["records"]) or set(records) != oof_ids or set(records) != set(folds):
        raise ValueError("Cache, OOF and fold manifests must contain the exact same unique sample IDs")

    ordered = sorted(oof_rows, key=lambda row: row["sample_id"])
    raw_features = np.asarray([local_motif_feature_vector(str(records[row["sample_id"]]["sequence"])).tolist() for row in ordered], dtype=np.float64)
    global_features = np.asarray([torch.as_tensor(records[row["sample_id"]]["physchem"], dtype=torch.float32).tolist() for row in ordered], dtype=np.float64)
    if global_features.shape[1] != len(PHYS_CHEM_FEATURE_NAMES):
        raise ValueError("Local motif audit requires the exact existing nine-dimensional PhysChem cache branch")
    fold_values = np.asarray([row["fold"] for row in ordered], dtype=np.int64)
    residual_features = _residualize_fold_purely(raw_features, global_features, fold_values)
    residual_names = tuple(f"residual_{name}" for name in LOCAL_MOTIF_FEATURE_NAMES)

    ledger: list[dict[str, Any]] = []
    cys_index = PHYS_CHEM_FEATURE_NAMES.index("cysteine_count")
    for index, row in enumerate(ordered):
        record = records[row["sample_id"]]
        if str(record["sequence_sha256"]) != row["sequence_sha256"] or int(record["label"]) != row["label"] or folds[row["sample_id"]] != row["fold"]:
            raise ValueError(f"Identity, label, or fold mismatch for {row['sample_id']}")
        prediction = int(row["probability"] >= threshold)
        error_type = "true_positive" if row["label"] == prediction == 1 else "true_negative" if row["label"] == prediction == 0 else "false_positive" if prediction else "false_negative"
        values = dict(zip(LOCAL_MOTIF_FEATURE_NAMES, raw_features[index].tolist(), strict=True))
        values.update(dict(zip(residual_names, residual_features[index].tolist(), strict=True)))
        ledger.append({
            "sample_id": row["sample_id"], "sequence_sha256": row["sequence_sha256"], "label": row["label"], "fold": row["fold"],
            "probability": row["probability"], "prediction": prediction, "error_type": error_type,
            "cysteine_count": float(global_features[index, cys_index]), **values,
        })

    low_cys_toxic = [row for row in ledger if row["label"] == 1 and row["cysteine_count"] <= 1.0]
    raw_summary, residual_summary = _summary(low_cys_toxic, LOCAL_MOTIF_FEATURE_NAMES), _summary(low_cys_toxic, residual_names)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "local_motif_error_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    report = {
        "artifact_type": AUDIT_VERSION,
        "scope": "training OOF only; Test1/Test2 and website excluded",
        "threshold": threshold,
        "source": {
            "cache": {"path": str(cache_file), "sha256": file_sha256(cache_file), "records": len(ledger)},
            "oof": {"path": str(oof_file), "sha256": file_sha256(oof_file)},
            "fold_manifest": {"path": str(folds_file), "sha256": file_sha256(folds_file)},
        },
        "raw_feature_contract": list(LOCAL_MOTIF_FEATURE_NAMES),
        "residual_feature_contract": list(residual_names),
        "residualization": {"global_features": list(PHYS_CHEM_FEATURE_NAMES), "fold_pure": True, "ridge_lambda": RIDGE_LAMBDA, "labels_used": False},
        "stability_rule": "absolute pooled standardized mean difference >= 0.20 and matching pooled-sign effect in at least four eligible folds",
        "low_cys_toxic_records": len(low_cys_toxic),
        "low_cys_false_negative_vs_true_positive": {"raw": raw_summary, "global_residualized": residual_summary},
        "architecture_premise_supported": bool(residual_summary["stable_features"]),
    }
    write_manifest(output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    report = audit(args.cache, args.oof, args.fold_manifest, args.output_dir, threshold=args.threshold)
    print(json.dumps({"architecture_premise_supported": report["architecture_premise_supported"], "stable_residual_features": report["low_cys_false_negative_vs_true_positive"]["global_residualized"]["stable_features"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
