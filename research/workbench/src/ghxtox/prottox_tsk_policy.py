"""Training-OOF-only calibration and operating policies for ProtTox-TSK.

This module is deliberately a post-head deployment layer.  It never changes
the Frozen ProtT5/TriStat/PhysChem model and rejects held-out data as a policy
source by requiring exact alignment with the audited *training* feature cache.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ghxtox.calibrate_predictions import apply_calibration, fit_platt
from ghxtox.metrics import expected_calibration_error
from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, file_sha256, write_manifest


POLICY_ARTIFACT_TYPE = "prottox_tsk_deployment_policy"
POLICY_VERSION = "prottox-tsk-policy-v1"
POLICY_OBJECTIVES = {"screening": 2.0, "balanced": None, "strict": 0.5}


def _read_training_cache(path: str | Path) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("format_version") != CACHE_FORMAT_VERSION or cache.get("model_name") != MODEL_NAME or cache.get("model_version") != MODEL_VERSION:
        raise ValueError("Policy source must be a compatible ProtTox-TSK training feature cache")
    records = cache.get("records", [])
    if not records or any(record.get("label") is None for record in records):
        raise ValueError("Policy source cache must contain training labels")
    return cache


def _read_oof(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "sequence_sha256", "label", "fold", "probability"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"OOF CSV must contain {sorted(required)}")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        probability = float(row["probability"])
        label, fold = int(row["label"]), int(row["fold"])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0 or label not in {0, 1}:
            raise ValueError("OOF CSV contains an invalid probability or non-binary label")
        parsed.append({"sample_id": str(row["sample_id"]), "sequence_sha256": str(row["sequence_sha256"]), "label": label, "fold": fold, "probability": probability})
    if len({row["sample_id"] for row in parsed}) != len(parsed):
        raise ValueError("OOF CSV contains duplicate sample IDs")
    folds = sorted({row["fold"] for row in parsed})
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Policy requires the complete audited five-fold OOF set, got folds {folds}")
    return parsed


def _validate_alignment(cache: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    records = {str(record["sample_id"]): record for record in cache["records"]}
    source_ids, oof_ids = set(records), {row["sample_id"] for row in rows}
    if source_ids != oof_ids:
        raise ValueError("OOF sample IDs do not exactly match the training feature cache; held-out data is not accepted")
    for row in rows:
        record = records[row["sample_id"]]
        if str(record["sequence_sha256"]) != row["sequence_sha256"] or int(record["label"]) != row["label"]:
            raise ValueError(f"OOF identity/label mismatch for {row['sample_id']}")


def _checkpoint_contract(path: str | Path, cache_sha256: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != MODEL_NAME or checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Policy checkpoint is not a compatible ProtTox-TSK checkpoint")
    metadata = dict(checkpoint.get("metadata", {}))
    if metadata.get("cache_sha256") != cache_sha256:
        raise ValueError("Policy checkpoint was not trained from the supplied audited training cache")
    return {"path": str(path), "sha256": file_sha256(path), "head_kind": checkpoint["config"]["head_kind"], "training_cache_sha256": cache_sha256}


def _decision_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predicted, truth = probabilities >= threshold, labels.astype(bool)
    tp, tn = int(np.sum(predicted & truth)), int(np.sum(~predicted & ~truth))
    fp, fn = int(np.sum(predicted & ~truth)), int(np.sum(~predicted & truth))
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(labels), 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return {"accuracy": accuracy, "balanced_accuracy": (sensitivity + specificity) / 2.0, "precision": precision, "sensitivity": sensitivity, "specificity": specificity, "f1": f1, "mcc": (tp * tn - fp * fn) / denominator, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _f_beta(metrics: dict[str, float], beta: float) -> float:
    beta_squared = beta * beta
    return (1.0 + beta_squared) * metrics["precision"] * metrics["sensitivity"] / max(beta_squared * metrics["precision"] + metrics["sensitivity"], 1e-12)


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray, mode: str) -> dict[str, Any]:
    beta = POLICY_OBJECTIVES[mode]
    candidates = sorted({float(value) for value in probabilities.tolist()} | {0.01, 0.5, 0.99})
    evaluated: list[tuple[tuple[float, ...], float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = _decision_metrics(labels, probabilities, threshold)
        if beta is None:
            score = (metrics["mcc"], metrics["balanced_accuracy"], metrics["f1"], -abs(threshold - 0.5))
        elif beta > 1:
            score = (_f_beta(metrics, beta), metrics["sensitivity"], metrics["precision"], -threshold)
        else:
            score = (_f_beta(metrics, beta), metrics["precision"], metrics["sensitivity"], threshold)
        evaluated.append((score, threshold, metrics))
    _, threshold, metrics = max(evaluated, key=lambda item: item[0])
    return {"objective": "maximize_mcc" if beta is None else f"maximize_f{beta:g}", "threshold": threshold, "metrics": metrics}


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    result = {"brier": float(np.mean((probabilities - labels) ** 2)), "ece_10": expected_calibration_error(torch.as_tensor(probabilities), torch.as_tensor(labels))}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        result.update({"auroc": float(roc_auc_score(labels, probabilities)), "auprc": float(average_precision_score(labels, probabilities))})
    except (ImportError, ValueError):
        result.update({"auroc": float("nan"), "auprc": float("nan")})
    return result


def build_policy(cache_path: str | Path, oof_path: str | Path, checkpoint_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Create a checkpoint-bound policy from complete training OOF predictions only."""

    cache_file, oof_file, checkpoint_file = Path(cache_path), Path(oof_path), Path(checkpoint_path)
    cache, rows = _read_training_cache(cache_file), _read_oof(oof_file)
    _validate_alignment(cache, rows)
    cache_sha = file_sha256(cache_file)
    checkpoint = _checkpoint_contract(checkpoint_file, cache_sha)
    labels = np.asarray([row["label"] for row in rows], dtype=np.float64)
    raw = torch.as_tensor([row["probability"] for row in rows], dtype=torch.float32)
    folds = np.asarray([row["fold"] for row in rows], dtype=np.int64)
    cross_fitted = torch.empty_like(raw)
    fold_calibrators: dict[str, dict[str, Any]] = {}
    for fold in range(5):
        train_mask = torch.as_tensor(folds != fold)
        held_mask = ~train_mask
        train_labels = torch.as_tensor(labels[folds != fold], dtype=torch.float32)
        if len(set(train_labels.tolist())) != 2 or int(held_mask.sum()) == 0:
            raise ValueError(f"Fold {fold} cannot support cross-fitted Platt calibration")
        calibration = fit_platt(raw[train_mask], train_labels)
        cross_fitted[held_mask] = apply_calibration(raw[held_mask], calibration)
        fold_calibrators[str(fold)] = calibration
    calibrated_oof = np.asarray(cross_fitted.tolist(), dtype=np.float64)
    deployment_calibrator = fit_platt(raw, torch.as_tensor(labels, dtype=torch.float32))
    policies = {mode: _select_threshold(labels, calibrated_oof, mode) for mode in POLICY_OBJECTIVES}
    artifact = {"artifact_type": POLICY_ARTIFACT_TYPE, "policy_version": POLICY_VERSION, "model": {"name": MODEL_NAME, "version": MODEL_VERSION, **checkpoint}, "source": {"scope": "training OOF only; held-out tests are excluded", "training_cache": {"path": str(cache_file), "sha256": cache_sha, "records": len(rows)}, "oof_predictions": {"path": str(oof_file), "sha256": file_sha256(oof_file), "records": len(rows), "folds": sorted(set(folds.tolist()))}}, "calibration": {"method": "cross_fitted_platt_for_policy_selection", "fold_calibrators": fold_calibrators, "deployment_calibrator_fit_on_full_training_oof": deployment_calibrator, "raw_oof_probability_metrics": _probability_metrics(labels, np.asarray(raw.tolist(), dtype=np.float64)), "cross_fitted_calibrated_oof_probability_metrics": _probability_metrics(labels, calibrated_oof)}, "policies": policies, "limitations": ["Policy objectives are operational preferences, not clinical guarantees.", "Thresholds were selected on training OOF only.", "Test1 and Test2 are excluded from calibration and threshold selection.", "Test2 is a historical Test1 subset and is not an independent validation cohort."]}
    write_manifest(output_path, artifact)
    return artifact


def load_policy(path: str | Path, checkpoint_path: str | Path, *, mode: str = "balanced") -> dict[str, Any]:
    policy = __import__("json").loads(Path(path).read_text(encoding="utf-8"))
    if policy.get("artifact_type") != POLICY_ARTIFACT_TYPE or policy.get("policy_version") != POLICY_VERSION:
        raise ValueError("Policy is not a compatible ProtTox-TSK deployment policy")
    if mode not in policy.get("policies", {}):
        raise ValueError(f"Unknown ProtTox-TSK policy mode: {mode}")
    model = policy.get("model", {})
    if model.get("name") != MODEL_NAME or model.get("version") != MODEL_VERSION or model.get("sha256") != file_sha256(checkpoint_path):
        raise ValueError("Policy does not belong to the selected checkpoint")
    return policy


def apply_policy(probabilities: Sequence[float] | torch.Tensor, policy: dict[str, Any], *, mode: str = "balanced") -> tuple[torch.Tensor, float]:
    if mode not in policy.get("policies", {}):
        raise ValueError(f"Unknown ProtTox-TSK policy mode: {mode}")
    calibrated = apply_calibration(torch.as_tensor(probabilities, dtype=torch.float32), policy["calibration"]["deployment_calibrator_fit_on_full_training_oof"])
    return calibrated, float(policy["policies"][mode]["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Audited labeled training feature cache only.")
    parser.add_argument("--oof", required=True, help="Complete five-fold training OOF CSV only.")
    parser.add_argument("--checkpoint", required=True, help="Full-data-refit checkpoint bound to the training cache.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    artifact = build_policy(args.cache, args.oof, args.checkpoint, args.output)
    for mode, selection in artifact["policies"].items():
        print(f"{mode}: threshold={selection['threshold']:.6f} {selection['objective']}")


if __name__ == "__main__":
    main()
