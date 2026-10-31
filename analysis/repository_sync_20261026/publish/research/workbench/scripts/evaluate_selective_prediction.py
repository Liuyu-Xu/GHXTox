#!/usr/bin/env python
"""Fit and evaluate GHXTox selective prediction using training OOF data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ghxtox.selective_prediction import (
    POLICY_ARTIFACT_TYPE,
    build_uncertainty_features,
    score_uncertainty,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = _root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "selective_prediction_current_default_oof_v2.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "selective_prediction_current_default_oof_20260810_v2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _verify_sources(root: Path, protocol: dict[str, Any]) -> list[dict[str, str]]:
    verified = []
    for source in protocol["sources"]:
        path = root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        expected = str(source["sha256"])
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path}: {actual} != {expected}")
        verified.append({"path": str(source["path"]), "sha256": actual, "role": "frozen_input"})
    return verified


def _load_inputs(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = {Path(item["path"]).name: root / item["path"] for item in protocol["sources"]}
    prediction_rows = _read_csv(paths["oof_predictions.csv"])
    quality_rows = _read_csv(paths["oof_3d_quality.csv"])
    manifest_rows = _read_csv(paths["train_cdhit080_fallback_folds.csv"])
    n = len(prediction_rows)
    if len(quality_rows) != n or len(manifest_rows) != n:
        raise ValueError("OOF prediction, quality, and fold-manifest row counts differ.")

    for index, (prediction, quality, manifest) in enumerate(
        zip(prediction_rows, quality_rows, manifest_rows)
    ):
        source_index = str(index)
        if prediction["source_index"] != source_index:
            raise ValueError(f"Prediction source_index is not contiguous at row {index}.")
        if quality["source_index"] != source_index or manifest["source_index"] != source_index:
            raise ValueError(f"OOF inputs are not source-index aligned at row {index}.")
        if prediction["sample_id"] != manifest["sample_id"]:
            raise ValueError(f"OOF sample_id mismatch at row {index}.")
        if prediction["fold"] != quality["fold"] or prediction["fold"] != manifest["fold"]:
            raise ValueError(f"OOF fold mismatch at row {index}.")
        if prediction["label"] != manifest["label"]:
            raise ValueError(f"OOF label mismatch at row {index}.")

    sample_ids = [row["sample_id"] for row in prediction_rows]
    labels = np.asarray([int(row["label"]) for row in prediction_rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in prediction_rows], dtype=np.int64)
    operating_point = protocol["canonical_operating_point"]
    one_d_column = str(operating_point["one_d_probability_column"])
    three_d_column = str(operating_point["three_d_probability_column"])
    fused_column = str(operating_point["probability_column"])
    required_columns = {one_d_column, three_d_column, fused_column}
    missing_columns = required_columns - set(prediction_rows[0])
    if missing_columns:
        raise ValueError(
            "Canonical OOF prediction table is missing protocol-declared columns: "
            f"{sorted(missing_columns)}"
        )
    p_1d = np.asarray([float(row[one_d_column]) for row in prediction_rows])
    p_3d = np.asarray([float(row[three_d_column]) for row in prediction_rows])
    fused = np.asarray([float(row[fused_column]) for row in prediction_rows])

    seed_vectors = []
    seed_ids = []
    for seed in (42, 123, 2025):
        payload = json.loads(paths[f"oof_seed{seed}.json"].read_text(encoding="utf-8"))
        if int(payload["seed"]) != seed:
            raise ValueError(f"Incorrect seed payload in oof_seed{seed}.json")
        if [str(value) for value in payload["sample_ids"]] != sample_ids:
            raise ValueError(f"Seed {seed} sample IDs/order differ from canonical OOF rows.")
        if not np.array_equal(np.asarray(payload["labels"], dtype=np.int64), labels):
            raise ValueError(f"Seed {seed} labels differ from canonical OOF rows.")
        if not np.array_equal(np.asarray(payload["folds"], dtype=np.int64), folds):
            raise ValueError(f"Seed {seed} folds differ from canonical OOF rows.")
        seed_vectors.append(np.asarray(payload["oof"], dtype=np.float64))
        seed_ids.append(seed)
    seed_std = np.std(np.stack(seed_vectors), axis=0, ddof=1)

    def quality(name: str) -> np.ndarray:
        return np.asarray(
            [float(row[name]) if row[name].strip() else np.nan for row in quality_rows],
            dtype=np.float64,
        )

    threshold = float(protocol["canonical_operating_point"]["decision_threshold"])
    feature_map = build_uncertainty_features(
        fused,
        p_1d,
        p_3d,
        threshold,
        one_d_seed_std=seed_std,
        mean_plddt=quality("mean_plddt"),
        mean_residue_rmsf=quality("mean_residue_rmsf"),
        mean_conformer_rmsd=quality("mean_conformer_rmsd"),
        contact_uncertainty_8=quality("contact_uncertainty_8"),
        conformer_missing=quality("missing_conformer_mask"),
    )
    base = {
        "sample_ids": sample_ids,
        "source_index": np.arange(n, dtype=np.int64),
        "folds": folds,
        "labels": labels,
        "p_1d": p_1d,
        "p_3d": p_3d,
        "fused": fused,
        "prediction": (fused >= threshold).astype(np.int64),
        "seed_ids": seed_ids,
    }
    return feature_map, base


def _matrix(features: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.column_stack([np.asarray(features[name], dtype=np.float64) for name in names])


def _fit_preprocessing(matrix: np.ndarray, quantiles: list[float], minimum_scale: float) -> dict[str, list[float]]:
    median = np.nanmedian(matrix, axis=0)
    if not np.all(np.isfinite(median)):
        raise ValueError("At least one selective-prediction feature is entirely missing.")
    filled = np.where(np.isfinite(matrix), matrix, median[None, :])
    lower = np.quantile(filled, float(quantiles[0]), axis=0)
    upper = np.quantile(filled, float(quantiles[1]), axis=0)
    clipped = np.clip(filled, lower[None, :], upper[None, :])
    mean = np.mean(clipped, axis=0)
    scale = np.std(clipped, axis=0, ddof=0)
    scale = np.where(scale < float(minimum_scale), 1.0, scale)
    return {
        "median": median.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _transform(matrix: np.ndarray, preprocessing: dict[str, list[float]]) -> np.ndarray:
    median = np.asarray(preprocessing["median"])
    filled = np.where(np.isfinite(matrix), matrix, median[None, :])
    clipped = np.clip(
        filled,
        np.asarray(preprocessing["lower"])[None, :],
        np.asarray(preprocessing["upper"])[None, :],
    )
    return (clipped - np.asarray(preprocessing["mean"])[None, :]) / np.asarray(
        preprocessing["scale"]
    )[None, :]


def _fit_logistic(matrix: np.ndarray, target: np.ndarray, spec: dict[str, Any]) -> LogisticRegression:
    model = LogisticRegression(
        C=float(spec["C"]),
        class_weight=spec["class_weight"],
        solver=str(spec["solver"]),
        max_iter=int(spec["max_iter"]),
        random_state=int(spec["random_state"]),
    )
    model.fit(matrix, target)
    return model


def _make_policy(
    profile: str,
    feature_names: list[str],
    preprocessing: dict[str, list[float]],
    model: LogisticRegression,
    protocol: dict[str, Any],
    abstention_threshold: float,
    realized_coverage: float,
) -> dict[str, Any]:
    return {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "version": int(protocol.get("version", 1)),
        "profile": profile,
        "scope": "post_hoc_selective_prediction_only",
        "training_source": "training OOF only",
        "test_labels_used": False,
        "feature_names": feature_names,
        "decision_threshold": float(protocol["canonical_operating_point"]["decision_threshold"]),
        "target_coverage": float(protocol["evaluation"]["reported_operating_coverage"]),
        "realized_training_oof_coverage": float(realized_coverage),
        "abstention_threshold": float(abstention_threshold),
        "abstain_when": "uncertainty_score > abstention_threshold",
        "preprocessing": preprocessing,
        "model": {
            "type": "class_balanced_logistic_error_ranking_score",
            "intercept": float(model.intercept_[0]),
            "coefficients": model.coef_[0].astype(float).tolist(),
        },
        "selective_labels": {"0": "Non-toxic", "1": "Toxic", "abstain": "Uncertain"},
        "preserves": [
            "toxicity_probability",
            "binary_prediction",
            "fusion_weight",
            "binary_decision_threshold",
        ],
        "protocol": str(
            protocol.get("artifact_path", "configs/selective_prediction_oof_v1.json")
        ),
        "source_checksums": {item["path"]: item["sha256"] for item in protocol["sources"]},
    }


def _make_direct_policy(
    profile: str,
    feature_name: str,
    protocol: dict[str, Any],
    abstention_threshold: float,
    realized_coverage: float,
) -> dict[str, Any]:
    """Freeze a raw, label-independent feature as the uncertainty score."""

    return {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "version": int(protocol.get("version", 1)),
        "profile": profile,
        "scope": "post_hoc_selective_prediction_only",
        "training_source": "training OOF only",
        "test_labels_used": False,
        "feature_names": [feature_name],
        "decision_threshold": float(protocol["canonical_operating_point"]["decision_threshold"]),
        "target_coverage": float(protocol["evaluation"]["reported_operating_coverage"]),
        "realized_training_oof_coverage": float(realized_coverage),
        "abstention_threshold": float(abstention_threshold),
        "abstain_when": "uncertainty_score > abstention_threshold",
        "preprocessing": {},
        "model": {
            "type": "direct_feature_score",
            "feature_name": feature_name,
        },
        "selective_labels": {"0": "Non-toxic", "1": "Toxic", "abstain": "Uncertain"},
        "preserves": [
            "toxicity_probability",
            "binary_prediction",
            "fusion_weight",
            "binary_decision_threshold",
        ],
        "protocol": str(
            protocol.get("artifact_path", "configs/selective_prediction_oof_v1.json")
        ),
        "source_checksums": {item["path"]: item["sha256"] for item in protocol["sources"]},
    }


def _threshold_for_coverage(scores: np.ndarray, coverage: float) -> tuple[float, float]:
    n = scores.shape[0]
    accepted = max(1, min(n, int(round(float(coverage) * n))))
    threshold = float(np.sort(scores, kind="mergesort")[accepted - 1])
    realized = float(np.mean(scores <= threshold))
    return threshold, realized


def _cross_fit_profile(
    profile: str,
    features: dict[str, np.ndarray],
    names: list[str],
    folds: np.ndarray,
    error_target: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _matrix(features, names)
    scoring = protocol.get("profile_scoring", {}).get(
        profile, {"type": "cross_fitted_logistic"}
    )
    scoring_type = str(scoring["type"])
    if scoring_type == "direct_feature":
        feature_name = str(scoring.get("feature_name", ""))
        if names != [feature_name]:
            raise ValueError(
                f"Direct profile {profile!r} must contain only feature {feature_name!r}."
            )
        scores = raw[:, 0].astype(np.float64, copy=True)
        if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"Direct profile {profile!r} must be finite and lie in [0, 1].")
        threshold, realized = _threshold_for_coverage(
            scores, float(protocol["evaluation"]["reported_operating_coverage"])
        )
        return scores, _make_direct_policy(
            profile, feature_name, protocol, threshold, realized
        )
    if scoring_type != "cross_fitted_logistic":
        raise ValueError(
            f"Unsupported profile scoring type for {profile!r}: {scoring_type!r}."
        )

    scores = np.full(raw.shape[0], np.nan, dtype=np.float64)
    prep_spec = protocol["preprocessing"]
    model_spec = protocol["uncertainty_model"]
    for fold in sorted(np.unique(folds).tolist()):
        train = folds != fold
        held = folds == fold
        preprocessing = _fit_preprocessing(
            raw[train], prep_spec["winsorization_quantiles"], prep_spec["minimum_scale"]
        )
        model = _fit_logistic(_transform(raw[train], preprocessing), error_target[train], model_spec)
        probability = model.predict_proba(_transform(raw[held], preprocessing))[:, 1]
        scores[held] = probability
    if not np.all(np.isfinite(scores)):
        raise ValueError("Cross-fitted uncertainty scores are incomplete.")

    preprocessing = _fit_preprocessing(
        raw, prep_spec["winsorization_quantiles"], prep_spec["minimum_scale"]
    )
    model = _fit_logistic(_transform(raw, preprocessing), error_target, model_spec)
    full_policy_stub = {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "feature_names": names,
        "decision_threshold": float(protocol["canonical_operating_point"]["decision_threshold"]),
        "abstention_threshold": 0.5,
        "abstain_when": "uncertainty_score > abstention_threshold",
        "preprocessing": preprocessing,
        "model": {
            "intercept": float(model.intercept_[0]),
            "coefficients": model.coef_[0].astype(float).tolist(),
        },
    }
    full_scores = score_uncertainty(features, full_policy_stub)
    threshold, realized = _threshold_for_coverage(
        full_scores, float(protocol["evaluation"]["reported_operating_coverage"])
    )
    policy = _make_policy(profile, names, preprocessing, model, protocol, threshold, realized)
    return scores, policy


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "mcc": float(mcc),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _expected_calibration_error(
    target: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(target[mask])) - float(np.mean(probabilities[mask]))
            )
    return float(error)


def _accepted_mask(scores: np.ndarray, coverage: float) -> np.ndarray:
    n = scores.shape[0]
    k = max(1, min(n, int(round(float(coverage) * n))))
    order = np.argsort(scores, kind="mergesort")
    accepted = np.zeros(n, dtype=bool)
    accepted[order[:k]] = True
    return accepted


def _risk_row(
    profile: str,
    scores: np.ndarray,
    coverage: float,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    accepted = _accepted_mask(scores, coverage)
    rejected = ~accepted
    errors = predictions != labels
    metrics = _classification_metrics(labels[accepted], predictions[accepted])
    toxic = labels == 1
    non_toxic = labels == 0
    pred_toxic = predictions == 1
    pred_non_toxic = predictions == 0
    full_error = float(np.mean(errors))
    rejected_error = float(np.mean(errors[rejected])) if np.any(rejected) else float("nan")
    return {
        "profile": profile,
        "coverage_target": float(coverage),
        "coverage": float(np.mean(accepted)),
        "accepted_n": int(np.sum(accepted)),
        "rejected_n": int(np.sum(rejected)),
        "accepted_error_rate": float(np.mean(errors[accepted])),
        "accepted_mcc": metrics["mcc"],
        "accepted_balanced_accuracy": metrics["balanced_accuracy"],
        "accepted_sensitivity": metrics["sensitivity"],
        "accepted_specificity": metrics["specificity"],
        "toxic_class_coverage": float(np.sum(accepted & toxic) / np.sum(toxic)),
        "non_toxic_class_coverage": float(np.sum(accepted & non_toxic) / np.sum(non_toxic)),
        "predicted_toxic_coverage": float(np.sum(accepted & pred_toxic) / max(np.sum(pred_toxic), 1)),
        "predicted_non_toxic_coverage": float(
            np.sum(accepted & pred_non_toxic) / max(np.sum(pred_non_toxic), 1)
        ),
        "rejected_error_rate": rejected_error,
        "rejected_error_enrichment": rejected_error / full_error if rejected.any() and full_error else float("nan"),
    }


def _aurc(
    scores: np.ndarray,
    errors: np.ndarray,
    minimum_coverage: float = 0.5,
    integration: str = "trapezoid",
) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_errors = errors[order].astype(np.float64)
    cumulative = np.cumsum(sorted_errors)
    n = scores.shape[0]
    start = max(1, int(math.ceil(float(minimum_coverage) * n)))
    counts = np.arange(start, n + 1)
    coverage = counts / n
    risk = cumulative[start - 1 :] / counts
    if integration == "discrete_mean":
        return float(np.mean(risk))
    if integration != "trapezoid":
        raise ValueError(f"Unsupported AURC integration method: {integration!r}.")
    width = 1.0 - float(minimum_coverage)
    if width <= 0.0:
        return float(risk[-1])
    return float(np.trapezoid(risk, coverage) / width)


def _bootstrap_intervals(
    profiles: dict[str, np.ndarray],
    labels: np.ndarray,
    predictions: np.ndarray,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = protocol["evaluation"]["bootstrap"]
    aurc_spec = protocol["evaluation"].get("aurc", {})
    aurc_minimum_coverage = float(aurc_spec.get("minimum_coverage", 0.5))
    aurc_integration = str(aurc_spec.get("integration", "trapezoid"))
    aurc_metric_name = str(aurc_spec.get("summary_field", "aurc_50_100"))
    replicates = int(spec["replicates"])
    rng = np.random.default_rng(int(spec["seed"]))
    coverages = (0.8, 0.9, 0.95)
    indices_by_class = [np.flatnonzero(labels == value) for value in (0, 1)]
    samples: dict[tuple[str, str, float], list[float]] = {}
    for profile in profiles:
        for coverage in coverages:
            for metric in ("accepted_error_rate", "accepted_mcc", "toxic_class_coverage", "non_toxic_class_coverage"):
                samples[(profile, metric, coverage)] = []
        samples[(profile, aurc_metric_name, 1.0)] = []
    for _ in range(replicates):
        drawn = np.concatenate(
            [rng.choice(indices, size=indices.shape[0], replace=True) for indices in indices_by_class]
        )
        boot_labels = labels[drawn]
        boot_predictions = predictions[drawn]
        boot_errors = boot_predictions != boot_labels
        for profile, values in profiles.items():
            boot_scores = values[drawn]
            for coverage in coverages:
                row = _risk_row(profile, boot_scores, coverage, boot_labels, boot_predictions)
                for metric in ("accepted_error_rate", "accepted_mcc", "toxic_class_coverage", "non_toxic_class_coverage"):
                    samples[(profile, metric, coverage)].append(float(row[metric]))
            samples[(profile, aurc_metric_name, 1.0)].append(
                _aurc(
                    boot_scores,
                    boot_errors,
                    aurc_minimum_coverage,
                    aurc_integration,
                )
            )
    rows = []
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    for (profile, metric, coverage), values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "profile": profile,
                "metric": metric,
                "coverage_target": coverage,
                "bootstrap_replicates": replicates,
                "ci_lower": float(np.quantile(array, alpha)),
                "ci_median": float(np.quantile(array, 0.5)),
                "ci_upper": float(np.quantile(array, 1.0 - alpha)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    root = _root()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8-sig"))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    verified_sources = _verify_sources(root, protocol)
    features, base = _load_inputs(root, protocol)
    labels = base["labels"]
    predictions = base["prediction"]
    error_target = (predictions != labels).astype(np.int64)
    profiles: dict[str, np.ndarray] = {}
    policies: dict[str, dict[str, Any]] = {}
    for profile, names in protocol["features"].items():
        scores, policy = _cross_fit_profile(
            profile, features, list(names), base["folds"], error_target, protocol
        )
        profiles[profile] = scores
        policies[profile] = policy

    risk_rows: list[dict[str, Any]] = []
    grid = [float(value) for value in protocol["evaluation"]["coverage_grid"]]
    for profile, scores in profiles.items():
        for coverage in grid:
            risk_rows.append(_risk_row(profile, scores, coverage, labels, predictions))

    full_metrics = _classification_metrics(labels, predictions)
    full_error = float(np.mean(error_target))
    aurc_spec = protocol["evaluation"].get("aurc", {})
    aurc_minimum_coverage = float(aurc_spec.get("minimum_coverage", 0.5))
    aurc_integration = str(aurc_spec.get("integration", "trapezoid"))
    aurc_field = str(aurc_spec.get("summary_field", "aurc_50_100"))
    summaries = []
    rows_at_90 = {
        row["profile"]: row
        for row in risk_rows
        if abs(float(row["coverage_target"]) - 0.9) < 1e-12
    }
    for profile, scores in profiles.items():
        row = rows_at_90[profile]
        summary_row = {
                "profile": profile,
                "deployable_with_public_single_structure_inference": profile != "conformer_extended",
                "error_detection_auroc": float(roc_auc_score(error_target, scores)),
                "error_detection_auprc": float(average_precision_score(error_target, scores)),
                "uncertainty_score_brier_diagnostic": float(
                    brier_score_loss(error_target, scores)
                ),
                "uncertainty_score_ece_10_diagnostic": _expected_calibration_error(
                    error_target, scores
                ),
                "full_coverage_error_rate": full_error,
                "full_coverage_mcc": full_metrics["mcc"],
                "error_rate_at_90_coverage": row["accepted_error_rate"],
                "relative_error_reduction_at_90": (
                    (full_error - float(row["accepted_error_rate"])) / full_error if full_error else 0.0
                ),
                "mcc_at_90_coverage": row["accepted_mcc"],
                "toxic_class_coverage_at_90": row["toxic_class_coverage"],
                "non_toxic_class_coverage_at_90": row["non_toxic_class_coverage"],
                "selected_public_policy": False,
                "promotion_passed": False,
            }
        summary_row[aurc_field] = _aurc(
            scores, error_target, aurc_minimum_coverage, aurc_integration
        )
        summaries.append(summary_row)

    by_profile = {row["profile"]: row for row in summaries}
    margin = by_profile["margin_only"]
    core = by_profile["deployable_core"]
    core_checks = {
        "aurc_no_higher_than_margin": core[aurc_field] <= margin[aurc_field] + 1e-12,
        "risk_at_90_no_higher_than_margin": core["error_rate_at_90_coverage"]
        <= margin["error_rate_at_90_coverage"] + 1e-12,
        "risk_at_90_lower_than_full": core["error_rate_at_90_coverage"] < full_error,
        "minimum_true_class_coverage": min(
            core["toxic_class_coverage_at_90"], core["non_toxic_class_coverage_at_90"]
        )
        >= 0.85,
        "mcc_at_90_no_lower_than_full": core["mcc_at_90_coverage"] >= full_metrics["mcc"] - 1e-12,
    }
    selected = "deployable_core" if all(core_checks.values()) else "margin_only"
    by_profile[selected]["selected_public_policy"] = True
    by_profile["deployable_core"]["promotion_passed"] = bool(all(core_checks.values()))
    by_profile["margin_only"]["promotion_passed"] = True

    for profile, policy in policies.items():
        (output / f"policy_{profile}.json").write_text(
            json.dumps(policy, indent=2), encoding="utf-8"
        )
    shutil.copyfile(output / f"policy_{selected}.json", output / "selected_policy.json")

    prediction_rows = []
    feature_names = list(features)
    for index, sample_id in enumerate(base["sample_ids"]):
        row: dict[str, Any] = {
            "source_index": int(base["source_index"][index]),
            "sample_id": sample_id,
            "fold": int(base["folds"][index]),
            "label": int(labels[index]),
            "p_1d": float(base["p_1d"][index]),
            "p_3d": float(base["p_3d"][index]),
            "toxicity_probability": float(base["fused"][index]),
            "binary_prediction": int(predictions[index]),
            "binary_error": int(error_target[index]),
        }
        row.update(
            {
                name: float(features[name][index]) if np.isfinite(features[name][index]) else ""
                for name in feature_names
            }
        )
        row.update({f"uncertainty_{name}": float(scores[index]) for name, scores in profiles.items()})
        prediction_rows.append(row)
    _write_csv(output / "oof_selective_predictions.csv", prediction_rows)
    _write_csv(output / "risk_coverage.csv", risk_rows)
    _write_csv(output / "profile_summary.csv", summaries)
    bootstrap_rows = _bootstrap_intervals(
        {name: profiles[name] for name in ("margin_only", selected)},
        labels,
        predictions,
        protocol,
    )
    _write_csv(output / "bootstrap_intervals.csv", bootstrap_rows)

    selection = {
        "artifact_type": "ghxtox_selective_prediction_selection",
        "selected_public_policy": selected,
        "selection_basis": "prespecified training-OOF promotion rule",
        "core_promotion_checks": core_checks,
        "test1_or_test2_labels_used": False,
        "canonical_binary_model_changed": False,
        "full_coverage": {
            "n": int(labels.shape[0]),
            "error_rate": full_error,
            **full_metrics,
        },
        "profiles": summaries,
    }
    (output / "summary.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    shutil.copyfile(args.protocol, output / "frozen_protocol.json")
    _write_csv(output / "input_manifest.csv", verified_sources)

    selected_model_type = str(policies[selected]["model"]["type"])
    score_description = (
        "The selected score is the direct, normalized distance-to-decision-boundary "
        "feature and is not a calibrated per-sample error probability."
        if selected_model_type == "direct_feature_score"
        else "The selected class-balanced logistic score is used for ranking and is not "
        "a calibrated per-sample error probability."
    )
    readme = f"""# Selective prediction OOF analysis\n\nThis directory contains the frozen, training-OOF-only selective-prediction analysis for GHXTox. No Test1 or Test2 labels were loaded. The canonical fused probability, binary threshold, and binary prediction are retained unchanged.\n\nSelected opt-in policy: `{selected}` at nominal 90% coverage. Larger uncertainty scores are less reliable; samples above the frozen policy threshold are reported as `Uncertain`. {score_description}\n\nThe `conformer_extended` profile is diagnostic only because conformer-variability fields are not available in canonical single-structure public inference.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    checksum_lines = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
