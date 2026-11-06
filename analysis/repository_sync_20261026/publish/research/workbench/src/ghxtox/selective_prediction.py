"""Post-hoc selective prediction for GHXTox.

The module is deliberately NumPy-only.  It consumes a frozen policy learned
from training OOF predictions and never changes the underlying model
probabilities, fusion weight, binary threshold, or binary prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


POLICY_ARTIFACT_TYPE = "ghxtox_selective_prediction_policy"
SUPPORTED_FEATURES = {
    "decision_closeness",
    "expert_disagreement",
    "one_d_seed_std",
    "structure_uncertainty",
    "mean_residue_rmsf",
    "mean_conformer_rmsd",
    "contact_uncertainty_8",
    "conformer_missing",
}


def _as_vector(value: Any, name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if n is not None and array.shape[0] != n:
        raise ValueError(f"{name} has {array.shape[0]} rows; expected {n}.")
    return array


def _optional_vector(value: Any | None, name: str, n: int) -> np.ndarray:
    if value is None:
        return np.full(n, np.nan, dtype=np.float64)
    return _as_vector(value, name, n)


def build_uncertainty_features(
    fused_probability: Any,
    one_d_probability: Any,
    three_d_probability: Any,
    decision_threshold: float,
    *,
    one_d_seed_std: Any | None = None,
    mean_plddt: Any | None = None,
    mean_residue_rmsf: Any | None = None,
    mean_conformer_rmsd: Any | None = None,
    contact_uncertainty_8: Any | None = None,
    conformer_missing: Any | None = None,
) -> dict[str, np.ndarray]:
    """Build raw uncertainty features without fitting or changing model outputs."""

    fused = _as_vector(fused_probability, "fused_probability")
    n = fused.shape[0]
    p_1d = _as_vector(one_d_probability, "one_d_probability", n)
    p_3d = _as_vector(three_d_probability, "three_d_probability", n)
    for name, values in (
        ("fused_probability", fused),
        ("one_d_probability", p_1d),
        ("three_d_probability", p_3d),
    ):
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1].")
    threshold = float(decision_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("decision_threshold must lie strictly between 0 and 1.")

    seed_std = _optional_vector(one_d_seed_std, "one_d_seed_std", n)
    plddt = _optional_vector(mean_plddt, "mean_plddt", n)
    rmsf = _optional_vector(mean_residue_rmsf, "mean_residue_rmsf", n)
    rmsd = _optional_vector(mean_conformer_rmsd, "mean_conformer_rmsd", n)
    contact = _optional_vector(contact_uncertainty_8, "contact_uncertainty_8", n)
    if conformer_missing is None:
        conformer_missing_values = (
            ~np.isfinite(rmsf) | ~np.isfinite(rmsd) | ~np.isfinite(contact)
        ).astype(np.float64)
    else:
        conformer_missing_values = _as_vector(conformer_missing, "conformer_missing", n)

    denominator = max(threshold, 1.0 - threshold)
    decision_closeness = 1.0 - np.abs(fused - threshold) / denominator
    decision_closeness = np.clip(decision_closeness, 0.0, 1.0)
    structure_uncertainty = np.where(np.isfinite(plddt), 1.0 - plddt, np.nan)
    return {
        "decision_closeness": decision_closeness,
        "expert_disagreement": np.abs(p_1d - p_3d),
        "one_d_seed_std": seed_std,
        "structure_uncertainty": structure_uncertainty,
        "mean_residue_rmsf": rmsf,
        "mean_conformer_rmsd": rmsd,
        "contact_uncertainty_8": contact,
        "conformer_missing": conformer_missing_values,
    }


def validate_selective_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("artifact_type") != POLICY_ARTIFACT_TYPE:
        raise ValueError("Not a GHXTox selective-prediction policy.")
    feature_names = list(policy.get("feature_names", []))
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Policy feature_names must be non-empty and unique.")
    unknown = sorted(set(feature_names) - SUPPORTED_FEATURES)
    if unknown:
        raise ValueError(f"Policy uses unsupported features: {unknown}.")
    model = policy.get("model", {})
    model_type = str(model.get("type", "class_balanced_logistic_error_ranking_score"))
    expected = len(feature_names)
    if model_type == "direct_feature_score":
        if expected != 1 or model.get("feature_name") != feature_names[0]:
            raise ValueError(
                "A direct-feature policy must name its single policy feature."
            )
    elif model_type != "class_balanced_logistic_error_ranking_score":
        raise ValueError(f"Unsupported selective-prediction model type: {model_type!r}.")

    preprocessing = policy.get("preprocessing", {})
    if model_type == "direct_feature_score":
        if preprocessing not in ({}, None):
            raise ValueError("A direct-feature policy must not define preprocessing.")
    else:
        for key in ("median", "lower", "upper", "mean", "scale"):
            values = np.asarray(preprocessing.get(key, []), dtype=np.float64)
            if values.shape != (expected,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Policy preprocessing.{key} must contain {expected} finite values.")
        scale = np.asarray(preprocessing["scale"], dtype=np.float64)
        if np.any(scale <= 0.0):
            raise ValueError("Policy preprocessing scales must be positive.")
        coefficients = np.asarray(model.get("coefficients", []), dtype=np.float64)
        if coefficients.shape != (expected,) or not np.all(np.isfinite(coefficients)):
            raise ValueError(f"Policy model coefficients must contain {expected} finite values.")
        intercept = float(model.get("intercept", np.nan))
        if not np.isfinite(intercept):
            raise ValueError("Policy model intercept must be finite.")
    abstention_threshold = float(policy.get("abstention_threshold", np.nan))
    decision_threshold = float(policy.get("decision_threshold", np.nan))
    if not 0.0 <= abstention_threshold <= 1.0:
        raise ValueError("Policy abstention_threshold must lie in [0, 1].")
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("Policy decision_threshold must lie strictly between 0 and 1.")
    if policy.get("abstain_when") != "uncertainty_score > abstention_threshold":
        raise ValueError("Policy abstention direction is missing or unsupported.")


def load_selective_policy(path: str | Path) -> dict[str, Any]:
    policy = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_selective_policy(policy)
    return policy


def _transform_features(
    features: Mapping[str, Any], policy: Mapping[str, Any]
) -> np.ndarray:
    validate_selective_policy(policy)
    names = list(policy["feature_names"])
    arrays = [_as_vector(features[name], name) for name in names]
    n = arrays[0].shape[0]
    if any(values.shape[0] != n for values in arrays):
        raise ValueError("Selective-prediction feature vectors are not row-aligned.")
    matrix = np.column_stack(arrays).astype(np.float64, copy=False)
    prep = policy["preprocessing"]
    median = np.asarray(prep["median"], dtype=np.float64)
    lower = np.asarray(prep["lower"], dtype=np.float64)
    upper = np.asarray(prep["upper"], dtype=np.float64)
    mean = np.asarray(prep["mean"], dtype=np.float64)
    scale = np.asarray(prep["scale"], dtype=np.float64)
    matrix = np.where(np.isfinite(matrix), matrix, median[None, :])
    matrix = np.clip(matrix, lower[None, :], upper[None, :])
    return (matrix - mean[None, :]) / scale[None, :]


def score_uncertainty(
    features: Mapping[str, Any], policy: Mapping[str, Any]
) -> np.ndarray:
    """Return a learned error-ranking score, not a calibrated error probability."""

    validate_selective_policy(policy)
    model_type = str(
        policy.get("model", {}).get(
            "type", "class_balanced_logistic_error_ranking_score"
        )
    )
    if model_type == "direct_feature_score":
        feature_name = str(policy["model"]["feature_name"])
        score = _as_vector(features[feature_name], feature_name)
        if not np.all(np.isfinite(score)):
            raise ValueError("A direct uncertainty feature must be finite for every sample.")
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError("A direct uncertainty feature must lie in [0, 1].")
        return score.astype(np.float64, copy=True)

    matrix = _transform_features(features, policy)
    coefficients = np.asarray(policy["model"]["coefficients"], dtype=np.float64)
    logits = matrix @ coefficients + float(policy["model"]["intercept"])
    logits = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def apply_selective_policy(
    fused_probability: Any,
    one_d_probability: Any,
    three_d_probability: Any,
    policy: Mapping[str, Any],
    *,
    one_d_seed_std: Any | None = None,
    mean_plddt: Any | None = None,
    mean_residue_rmsf: Any | None = None,
    mean_conformer_rmsd: Any | None = None,
    contact_uncertainty_8: Any | None = None,
    conformer_missing: Any | None = None,
) -> dict[str, np.ndarray]:
    """Apply a frozen policy while retaining the original binary prediction."""

    validate_selective_policy(policy)
    fused = _as_vector(fused_probability, "fused_probability")
    policy_threshold = float(policy["decision_threshold"])
    features = build_uncertainty_features(
        fused,
        one_d_probability,
        three_d_probability,
        policy_threshold,
        one_d_seed_std=one_d_seed_std,
        mean_plddt=mean_plddt,
        mean_residue_rmsf=mean_residue_rmsf,
        mean_conformer_rmsd=mean_conformer_rmsd,
        contact_uncertainty_8=contact_uncertainty_8,
        conformer_missing=conformer_missing,
    )
    uncertainty = score_uncertainty(features, policy)
    abstained = uncertainty > float(policy["abstention_threshold"])
    binary = (fused >= policy_threshold).astype(np.int64)
    selective = np.where(abstained, "Uncertain", np.where(binary == 1, "Toxic", "Non-toxic"))
    return {
        "uncertainty_score": uncertainty,
        "abstained": abstained,
        "selective_prediction": selective,
        "binary_prediction": binary,
    }
