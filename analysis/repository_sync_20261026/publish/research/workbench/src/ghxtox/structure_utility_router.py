"""Cross-fitted, pre-ESMFold expected-utility router over training OOF records."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             matthews_corrcoef, roc_auc_score)

from ghxtox.unified_inference import read_csv, sha256


PROHIBITED_FEATURE_COLUMNS = {"label", "p_struct", "p_fusion", "realized_brier_gain"}


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    return {
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def _read_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    required = {"artifact_type", "status", "input_contract", "input_contract_sha256", "pre_esmfold_feature_columns", "target", "model", "route_policy", "classification_threshold"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Router config missing fields: {sorted(missing)}")
    if config["artifact_type"] != "ghxtox_structure_utility_router_protocol" or config["status"] != "frozen_before_oof":
        raise ValueError("Router protocol must be frozen before fitting")
    if set(config["pre_esmfold_feature_columns"]) & PROHIBITED_FEATURE_COLUMNS:
        raise ValueError("Post-structure probability, labels, or target cannot be router features")
    if config["target"] != {"loss": "brier", "definition": "(p_seq-label)^2 - (p_struct-label)^2"}:
        raise ValueError("Only the prespecified OOF Brier-gain target is supported")
    if config["model"].get("type") != "ridge" or float(config["model"].get("alpha", 0)) <= 0:
        raise ValueError("Router requires a positive-alpha ridge model")
    if config["route_policy"].get("trigger") != "predicted_brier_gain_gt_structure_cost":
        raise ValueError("Router trigger must be frozen expected-gain-versus-cost rule")
    if not 0 < float(config["classification_threshold"]) < 1 or float(config["route_policy"].get("structure_cost", -1)) < 0:
        raise ValueError("Invalid frozen threshold or structure cost")
    return config


def _array(rows: list[dict[str, str]], columns: list[str]) -> np.ndarray:
    try:
        values = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"Contract is missing router feature {error.args[0]}") from error
    if not np.isfinite(values).all():
        raise ValueError("Router features must be finite")
    return values


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> tuple[Ridge, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    model = Ridge(alpha=alpha)
    model.fit((features - mean) / scale, target)
    return model, mean, scale


def run_cross_fitted_router(protocol_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    protocol_path, output_dir = Path(protocol_path), Path(output_dir)
    config = _read_config(protocol_path)
    contract_path = Path(config["input_contract"])
    if sha256(contract_path).lower() != str(config["input_contract_sha256"]).lower():
        raise ValueError("Unified inference contract hash differs from frozen router protocol")
    rows = read_csv(contract_path)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite router output: {output_dir}")
    feature_columns = list(config["pre_esmfold_feature_columns"])
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Unified contract must contain all five OOF folds")
    groups: dict[str, int] = {}
    for row, fold in zip(rows, folds):
        group = row["group_id"]
        if not group:
            raise ValueError("Contract row has empty group_id")
        if group in groups and groups[group] != fold:
            raise ValueError("Group leaks across OOF folds")
        groups[group] = int(fold)
    features = _array(rows, feature_columns)
    p_seq = np.asarray([float(row["p_seq"]) for row in rows])
    p_struct = np.asarray([float(row["p_struct"]) for row in rows])
    if not np.all((0 <= p_seq) & (p_seq <= 1) & (0 <= p_struct) & (p_struct <= 1)):
        raise ValueError("Contract probabilities must lie in [0, 1]")
    target = (p_seq - labels) ** 2 - (p_struct - labels) ** 2
    predicted_gain = np.empty(len(rows), dtype=np.float64)
    states: dict[str, Any] = {"feature_columns": feature_columns, "fold_models": {}}
    alpha = float(config["model"]["alpha"])
    for held_fold in range(5):
        train_mask, held_mask = folds != held_fold, folds == held_fold
        model, mean, scale = _fit_ridge(features[train_mask], target[train_mask], alpha)
        predicted_gain[held_mask] = model.predict((features[held_mask] - mean) / scale)
        states["fold_models"][str(held_fold)] = {
            "training_folds": [int(fold) for fold in range(5) if fold != held_fold],
            "mean": mean.tolist(), "scale": scale.tolist(), "coef": model.coef_.tolist(), "intercept": float(model.intercept_),
        }
    full_model, full_mean, full_scale = _fit_ridge(features, target, alpha)
    states["full_training_model"] = {"mean": full_mean.tolist(), "scale": full_scale.tolist(), "coef": full_model.coef_.tolist(), "intercept": float(full_model.intercept_)}
    cost = float(config["route_policy"]["structure_cost"])
    structure_selected = predicted_gain > cost
    deployed = np.where(structure_selected, p_struct, p_seq)
    threshold = float(config["classification_threshold"])
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "cross_fitted_predictions.csv"
    fields = ["source_index", "sample_id", "sequence", "label", "fold", "group_id", "p_seq", "p_struct", "realized_brier_gain", "predicted_brier_gain", "route", "structure_computed", "deployed_probability"]
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow({
                **{key: row[key] for key in fields[:6]}, "p_seq": repr(float(p_seq[index])), "p_struct": repr(float(p_struct[index])),
                "realized_brier_gain": repr(float(target[index])), "predicted_brier_gain": repr(float(predicted_gain[index])),
                "route": "structure" if structure_selected[index] else "fast", "structure_computed": int(structure_selected[index]),
                "deployed_probability": repr(float(deployed[index])),
            })
    summary = {
        "artifact_type": "ghxtox_structure_utility_router_oof", "version": 1,
        "protocol": str(protocol_path.resolve()), "protocol_sha256": sha256(protocol_path),
        "contract": str(contract_path.resolve()), "contract_sha256": sha256(contract_path),
        "external_test_labels_used": False, "cross_fitted": True, "target": config["target"],
        "feature_columns": feature_columns, "post_structure_features_used_by_router": False,
        "route_policy": config["route_policy"], "classification_threshold": threshold,
        "structure_rate": float(structure_selected.mean()), "mean_realized_brier_gain": float(target.mean()),
        "mean_realized_brier_gain_when_selected": float(target[structure_selected].mean()) if structure_selected.any() else None,
        "metrics": {"fast_sequence_only": _metrics(labels, p_seq, threshold), "always_structure": _metrics(labels, p_struct, threshold), "cross_fitted_router": _metrics(labels, deployed, threshold)},
        "utility_brier_plus_structure_cost": {"fast_sequence_only": float(np.mean((p_seq - labels) ** 2)), "always_structure": float(np.mean((p_struct - labels) ** 2) + cost), "cross_fitted_router": float(np.mean((deployed - labels) ** 2) + cost * structure_selected.mean())},
        "folds": {str(fold): {"samples": int((folds == fold).sum()), "structure_rate": float(structure_selected[folds == fold].mean()), "metrics": _metrics(labels[folds == fold], deployed[folds == fold], threshold)} for fold in range(5)},
    }
    (output_dir / "router_models.json").write_text(json.dumps(states, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

