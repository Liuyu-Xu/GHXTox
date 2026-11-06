#!/usr/bin/env python
"""Evaluate a frozen OOF-derived abstention cutoff on Test1 and Test2.

The policy, feature transform, binary threshold, and abstention cutoff are
immutable. Test labels are read only after label-blind decisions have been
computed; they are used solely for post-selection metrics and resampling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ghxtox.selective_prediction import (  # noqa: E402
    apply_selective_policy,
    build_uncertainty_features,
    load_selective_policy,
)


PERFORMANCE_METRICS = (
    "accepted_error_rate",
    "accepted_mcc",
    "accepted_balanced_accuracy",
    "accepted_sensitivity",
    "accepted_specificity",
)
ALL_SUMMARY_METRICS = ("coverage",) + PERFORMANCE_METRICS
METHODS = (
    "full_binary_predictor",
    "frozen_threshold_accepted_set",
    "equal_count_random_rejection",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Empty input table: {path}")
    return fields, rows


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: Iterable[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fieldnames = list(fields or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or predictions.shape != labels.shape or labels.size == 0:
        raise ValueError("Labels and predictions must be aligned non-empty vectors.")
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    bacc = (sensitivity + specificity) / 2.0 if np.isfinite(sensitivity + specificity) else math.nan
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else math.nan
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accepted_error_rate": (fp + fn) / labels.size,
        "accepted_mcc": mcc,
        "accepted_balanced_accuracy": bacc,
        "accepted_sensitivity": sensitivity,
        "accepted_specificity": specificity,
    }


def _method_row(
    method: str,
    split_name: str,
    accepted_mask: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(accepted_mask, dtype=bool)
    if mask.shape != labels.shape or not np.any(mask):
        raise ValueError(f"{split_name}/{method} has no valid accepted set.")
    result = {
        "method": method,
        "split_name": split_name,
        "coverage": float(np.mean(mask)),
        "accepted_n": int(np.sum(mask)),
        "rejected_n": int(mask.size - np.sum(mask)),
    }
    result.update(_classification_metrics(labels[mask], predictions[mask]))
    return result


def _finite_summary(values: np.ndarray, confidence_level: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "median": math.nan,
            "ci95_lower": math.nan,
            "ci95_upper": math.nan,
            "valid_replicates": 0,
        }
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "ci95_lower": float(np.quantile(finite, alpha)),
        "ci95_upper": float(np.quantile(finite, 1.0 - alpha)),
        "valid_replicates": int(finite.size),
    }


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("artifact_type") != (
        "ghxtox_selective_prediction_frozen_threshold_test_transfer_protocol"
    ):
        raise ValueError("Unexpected transfer protocol artifact type.")
    scope = protocol.get("scope", {})
    prohibited = (
        "allow_policy_refit",
        "allow_feature_reselection",
        "allow_test_specific_preprocessing",
        "allow_test_specific_cutoff",
        "allow_test_specific_coverage_target",
        "allow_binary_model_change",
        "allow_fusion_or_binary_threshold_change",
        "allow_dataset_identity_in_selection",
        "allow_test_labels_in_selection",
    )
    if any(scope.get(key) is not False for key in prohibited):
        raise ValueError("The protocol does not preserve the no-tuning selection firewall.")
    if scope.get("test_labels_used_for_post_selection_evaluation_only") is not True:
        raise ValueError("The evaluation-only label boundary is not explicit.")
    contract = protocol.get("frozen_policy_contract", {})
    if contract.get("required_profile") != "margin_only":
        raise ValueError("Only the frozen margin-only policy is authorized.")
    if contract.get("required_feature_names") != ["decision_closeness"]:
        raise ValueError("The frozen transfer policy must use decision_closeness only.")
    if contract.get("accept_when") != "uncertainty_score <= uncertainty_score_cutoff":
        raise ValueError("Unexpected acceptance inequality.")
    if protocol.get("reported_methods") != list(METHODS):
        raise ValueError("Unexpected method set in frozen protocol.")
    if protocol.get("reported_metrics") != [
        "coverage",
        "accepted_n",
        "rejected_n",
        *PERFORMANCE_METRICS,
    ]:
        raise ValueError("Unexpected reporting endpoint set.")


def _validate_inputs_and_policy(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    inputs = protocol["inputs"]
    for item in inputs:
        path = _resolve(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
        manifest_rows.append(
            {
                "role": item["role"],
                "split_name": item.get("split_name", "training OOF policy"),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    policy_item = next(item for item in inputs if item["role"] == "frozen_training_oof_selective_policy")
    policy = load_selective_policy(_resolve(policy_item["path"]))
    contract = protocol["frozen_policy_contract"]
    if policy.get("profile") != contract["required_profile"]:
        raise ValueError("Policy profile differs from frozen transfer contract.")
    if policy.get("feature_names") != contract["required_feature_names"]:
        raise ValueError("Policy features differ from frozen transfer contract.")
    required_model_type = contract.get("required_model_type")
    if required_model_type and policy.get("model", {}).get("type") != required_model_type:
        raise ValueError("Policy model type differs from frozen transfer contract.")
    if float(policy["decision_threshold"]) != float(contract["binary_decision_threshold"]):
        raise ValueError("Binary threshold differs from frozen transfer contract.")
    if float(policy["abstention_threshold"]) != float(contract["uncertainty_score_cutoff"]):
        raise ValueError("Abstention cutoff differs from frozen transfer contract.")
    if policy.get("training_source") != "training OOF only" or policy.get("test_labels_used") is not False:
        raise ValueError("Policy provenance is not training-OOF-only.")
    return policy, manifest_rows


def _compute_label_blind_split(
    split_name: str,
    path: Path,
    protocol: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], np.ndarray, np.ndarray]:
    fields, raw_rows = _read_csv(path)
    required = protocol["validation_guards"]["required_prediction_columns"]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")

    allowed = protocol["selection_firewall"]["columns_permitted_for_score_and_mask"]
    selection_rows = [{name: row[name] for name in allowed} for row in raw_rows]
    source_index = np.asarray([int(row["source_index"]) for row in selection_rows], dtype=np.int64)
    sample_id = [row["sample_id"] for row in selection_rows]
    if np.unique(source_index).size != source_index.size:
        raise ValueError(f"{split_name} source_index values are not unique.")
    if len(set(sample_id)) != len(sample_id):
        raise ValueError(f"{split_name} sample_id values are not unique.")

    p_1d = np.asarray([float(row["p_1d"]) for row in selection_rows], dtype=np.float64)
    p_3d = np.asarray([float(row["p_3d"]) for row in selection_rows], dtype=np.float64)
    fused = np.asarray(
        [float(row["toxicity_probability"]) for row in selection_rows], dtype=np.float64
    )
    stored_prediction = np.asarray(
        [int(row["prediction"]) for row in selection_rows], dtype=np.int64
    )
    stored_threshold = np.asarray(
        [float(row["decision_threshold"]) for row in selection_rows], dtype=np.float64
    )
    for name, values in (("p_1d", p_1d), ("p_3d", p_3d), ("toxicity_probability", fused)):
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{split_name} {name} contains invalid probabilities.")
    if not np.all(np.isin(stored_prediction, [0, 1])):
        raise ValueError(f"{split_name} prediction is not binary.")
    decision_threshold = float(policy["decision_threshold"])
    threshold_tolerance = float(
        protocol["validation_guards"]["row_binary_threshold_absolute_tolerance"]
    )
    if not np.allclose(
        stored_threshold, decision_threshold, atol=threshold_tolerance, rtol=0.0
    ):
        raise ValueError(f"{split_name} row thresholds differ from the frozen policy.")
    reconstructed_prediction = (fused >= decision_threshold).astype(np.int64)
    if not np.array_equal(stored_prediction, reconstructed_prediction):
        raise ValueError(f"{split_name} binary prediction reconstruction failed.")

    features = build_uncertainty_features(fused, p_1d, p_3d, decision_threshold)
    result = apply_selective_policy(fused, p_1d, p_3d, policy)
    uncertainty = np.asarray(result["uncertainty_score"], dtype=np.float64)
    abstained = np.asarray(result["abstained"], dtype=bool)
    accepted = ~abstained
    cutoff = float(policy["abstention_threshold"])
    if not np.array_equal(accepted, uncertainty <= cutoff):
        raise ValueError(f"{split_name} fixed score-cutoff reconstruction failed.")
    raw_cutoff = float(protocol["frozen_policy_contract"]["equivalent_raw_decision_closeness_cutoff"])
    if not np.array_equal(accepted, features["decision_closeness"] <= raw_cutoff):
        raise ValueError(f"{split_name} raw cutoff equivalence failed.")

    decisions: list[dict[str, Any]] = []
    for index, row in enumerate(selection_rows):
        decisions.append(
            {
                "split_name": split_name,
                "source_index": int(source_index[index]),
                "sample_id": sample_id[index],
                "p_1d": float(p_1d[index]),
                "p_3d": float(p_3d[index]),
                "toxicity_probability": float(fused[index]),
                "binary_prediction": int(reconstructed_prediction[index]),
                "binary_decision_threshold": decision_threshold,
                "decision_closeness": float(features["decision_closeness"][index]),
                "uncertainty_score": float(uncertainty[index]),
                "uncertainty_score_cutoff": cutoff,
                "equivalent_raw_decision_closeness_cutoff": raw_cutoff,
                "accepted": bool(accepted[index]),
                "abstained": bool(abstained[index]),
                "selective_prediction": str(result["selective_prediction"][index]),
            }
        )
    return decisions, raw_rows, accepted, reconstructed_prediction


def _random_control(
    split_name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    accepted_n: int,
    replicates: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    n = labels.size
    raw: list[dict[str, Any]] = []
    values = {metric: np.empty(replicates, dtype=np.float64) for metric in ALL_SUMMARY_METRICS}
    for replicate in range(replicates):
        selected = rng.choice(n, size=accepted_n, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[selected] = True
        row = _method_row(
            "equal_count_random_rejection", split_name, mask, labels, predictions
        )
        raw.append({"replicate": replicate, **row})
        for metric in ALL_SUMMARY_METRICS:
            values[metric][replicate] = float(row[metric])
    summary_rows: list[dict[str, Any]] = []
    means: dict[str, float] = {}
    for metric in ALL_SUMMARY_METRICS:
        summary = _finite_summary(values[metric], confidence_level)
        means[metric] = float(summary["mean"])
        summary_rows.append(
            {
                "method": "equal_count_random_rejection",
                "split_name": split_name,
                "metric": metric,
                **summary,
                "replicates": replicates,
                "uncertainty_type": "distribution across uniform equal-count random masks on the fixed test cohort",
            }
        )
    return raw, summary_rows, means


def _paired_bootstrap(
    split_name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    accepted_mask: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
    confidence_level: float,
    point_rows: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    class_indices = {
        label: np.flatnonzero(labels == label) for label in (0, 1)
    }
    if any(indices.size == 0 for indices in class_indices.values()):
        raise ValueError(f"{split_name} must contain both classes for stratified bootstrap.")
    method_values = {
        (method, metric): np.full(replicates, np.nan, dtype=np.float64)
        for method in METHODS
        for metric in ALL_SUMMARY_METRICS
    }
    raw_rows: list[dict[str, Any]] = []
    for replicate in range(replicates):
        resampled = np.concatenate(
            [
                rng.choice(class_indices[label], size=class_indices[label].size, replace=True)
                for label in (0, 1)
            ]
        )
        boot_labels = labels[resampled]
        boot_predictions = predictions[resampled]
        frozen_mask = accepted_mask[resampled]
        accepted_n = int(np.sum(frozen_mask))
        if accepted_n <= 0 or accepted_n >= resampled.size:
            raise ValueError(f"{split_name} bootstrap replicate has invalid realized acceptance.")
        random_positions = rng.choice(resampled.size, size=accepted_n, replace=False)
        random_mask = np.zeros(resampled.size, dtype=bool)
        random_mask[random_positions] = True
        masks = {
            "full_binary_predictor": np.ones(resampled.size, dtype=bool),
            "frozen_threshold_accepted_set": frozen_mask,
            "equal_count_random_rejection": random_mask,
        }
        for method in METHODS:
            row = _method_row(method, split_name, masks[method], boot_labels, boot_predictions)
            raw_rows.append({"replicate": replicate, **row})
            for metric in ALL_SUMMARY_METRICS:
                method_values[(method, metric)][replicate] = float(row[metric])

    interval_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for metric in ALL_SUMMARY_METRICS:
            summary = _finite_summary(method_values[(method, metric)], confidence_level)
            interval_rows.append(
                {
                    "method": method,
                    "split_name": split_name,
                    "metric": metric,
                    "point_estimate": float(point_rows[method][metric]),
                    **summary,
                    "bootstrap_replicates": replicates,
                    "uncertainty_type": "class-stratified fixed-mask paired bootstrap; random method uses one equal-count random mask per replicate",
                }
            )

    contrast_rows: list[dict[str, Any]] = []
    for comparator, public_definition in (
        ("full_binary_predictor", "full versus frozen accepted"),
        ("equal_count_random_rejection", "random versus frozen accepted"),
    ):
        for metric in PERFORMANCE_METRICS:
            frozen = method_values[("frozen_threshold_accepted_set", metric)]
            reference = method_values[(comparator, metric)]
            if metric == "accepted_error_rate":
                values = reference - frozen
                point = float(point_rows[comparator][metric]) - float(
                    point_rows["frozen_threshold_accepted_set"][metric]
                )
                definition = f"{comparator} minus frozen accepted"
            else:
                values = frozen - reference
                point = float(point_rows["frozen_threshold_accepted_set"][metric]) - float(
                    point_rows[comparator][metric]
                )
                definition = f"frozen accepted minus {comparator}"
            summary = _finite_summary(values, confidence_level)
            finite = values[np.isfinite(values)]
            contrast_rows.append(
                {
                    "contrast": public_definition,
                    "comparator": comparator,
                    "split_name": split_name,
                    "metric": metric,
                    "positive_improvement_definition": definition,
                    "point_estimate": point,
                    **summary,
                    "bootstrap_probability_gt_0": (
                        float(np.mean(finite > 0.0)) if finite.size else math.nan
                    ),
                    "bootstrap_replicates": replicates,
                }
            )
    return raw_rows, interval_rows, contrast_rows


def _point_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Split | Method | Coverage | Accepted | Error | MCC | BACC | SN | SP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    public_names = {
        "full_binary_predictor": "Full binary predictor",
        "frozen_threshold_accepted_set": "Frozen-threshold accepted set",
        "equal_count_random_rejection": "Equal-count random rejection",
    }
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split_name"]),
                    public_names[str(row["method"])],
                    f"{100 * float(row['coverage']):.2f}%",
                    f"{int(row['accepted_n']):,}",
                    f"{100 * float(row['accepted_error_rate']):.2f}%",
                    f"{float(row['accepted_mcc']):.4f}",
                    f"{float(row['accepted_balanced_accuracy']):.4f}",
                    f"{float(row['accepted_sensitivity']):.4f}",
                    f"{float(row['accepted_specificity']):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _classify_split(
    point_rows: dict[str, dict[str, Any]], contrast_rows: list[dict[str, Any]]
) -> str:
    selective_error = float(point_rows["frozen_threshold_accepted_set"]["accepted_error_rate"])
    full_error = float(point_rows["full_binary_predictor"]["accepted_error_rate"])
    random_error = float(point_rows["equal_count_random_rejection"]["accepted_error_rate"])
    point_pass = selective_error < full_error and selective_error < random_error
    full_error_contrast = next(
        row
        for row in contrast_rows
        if row["comparator"] == "full_binary_predictor"
        and row["metric"] == "accepted_error_rate"
    )
    if point_pass and float(full_error_contrast["ci95_lower"]) > 0.0:
        return "strong_support"
    if point_pass:
        return "descriptive_support"
    return "not_supported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "selective_prediction_frozen_threshold_test_transfer_v1.json",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    policy, input_manifest = _validate_inputs_and_policy(protocol)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _resolve(protocol["output_dir"])
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty result directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    split_inputs = [item for item in protocol["inputs"] if item.get("split_name")]
    label_blind_rows: list[dict[str, Any]] = []
    labeled_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    split_state: dict[str, dict[str, Any]] = {}

    # Phase 1: compute and freeze masks without accessing the label field.
    deferred: dict[str, tuple[list[dict[str, str]], np.ndarray, np.ndarray]] = {}
    for item in split_inputs:
        split_name = item["split_name"]
        decisions, raw_rows, accepted, predictions = _compute_label_blind_split(
            split_name, _resolve(item["path"]), protocol, policy
        )
        label_blind_rows.extend(decisions)
        deferred[split_name] = (raw_rows, accepted.copy(), predictions.copy())

    _write_csv(output / "label_blind_decisions.csv", label_blind_rows)
    label_blind_sha256 = _sha256(output / "label_blind_decisions.csv")

    # Phase 2: labels are now read only for post-selection outcome evaluation.
    for item in split_inputs:
        split_name = item["split_name"]
        raw_rows, accepted, predictions = deferred[split_name]
        labels = np.asarray([int(row["label"]) for row in raw_rows], dtype=np.int64)
        if not np.all(np.isin(labels, [0, 1])):
            raise ValueError(f"{split_name} labels are not binary.")
        split_decisions = [row for row in label_blind_rows if row["split_name"] == split_name]
        if len(split_decisions) != labels.size:
            raise ValueError(f"{split_name} decision/label row alignment failed.")
        for index, decision in enumerate(split_decisions):
            labeled_rows.append(
                {
                    **decision,
                    "label": int(labels[index]),
                    "binary_error": int(predictions[index] != labels[index]),
                    "accepted_error": (
                        int(predictions[index] != labels[index]) if accepted[index] else ""
                    ),
                }
            )
        full_row = _method_row(
            "full_binary_predictor",
            split_name,
            np.ones(labels.size, dtype=bool),
            labels,
            predictions,
        )
        selective_row = _method_row(
            "frozen_threshold_accepted_set", split_name, accepted, labels, predictions
        )
        exact_rows.extend([full_row, selective_row])
        split_state[split_name] = {
            "labels": labels,
            "predictions": predictions,
            "accepted": accepted,
            "full_row": full_row,
            "selective_row": selective_row,
        }

    _write_csv(output / "selective_predictions_with_labels.csv", labeled_rows)

    random_spec = protocol["random_rejection_control"]
    confidence_level = float(protocol["paired_bootstrap"]["confidence_level"])
    random_rng = np.random.default_rng(int(random_spec["seed"]))
    random_raw: list[dict[str, Any]] = []
    random_summary: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    point_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for item in split_inputs:
        split_name = item["split_name"]
        state = split_state[split_name]
        accepted_n = int(np.sum(state["accepted"]))
        raw, summary, means = _random_control(
            split_name,
            state["labels"],
            state["predictions"],
            accepted_n,
            int(random_spec["replicates"]),
            random_rng,
            confidence_level,
        )
        random_raw.extend(raw)
        random_summary.extend(summary)
        random_point = {
            "method": "equal_count_random_rejection",
            "split_name": split_name,
            "coverage": means["coverage"],
            "accepted_n": accepted_n,
            "rejected_n": int(state["labels"].size - accepted_n),
            "tp": "",
            "tn": "",
            "fp": "",
            "fn": "",
            **{metric: means[metric] for metric in PERFORMANCE_METRICS},
        }
        rows_for_split = {
            "full_binary_predictor": state["full_row"],
            "frozen_threshold_accepted_set": state["selective_row"],
            "equal_count_random_rejection": random_point,
        }
        point_by_split[split_name] = rows_for_split
        point_rows.extend(rows_for_split[method] for method in METHODS)

    bootstrap_spec = protocol["paired_bootstrap"]
    bootstrap_rng = np.random.default_rng(int(bootstrap_spec["seed"]))
    bootstrap_raw: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    split_decisions: dict[str, str] = {}
    for item in split_inputs:
        split_name = item["split_name"]
        state = split_state[split_name]
        raw, intervals, contrasts = _paired_bootstrap(
            split_name,
            state["labels"],
            state["predictions"],
            state["accepted"],
            int(bootstrap_spec["replicates"]),
            bootstrap_rng,
            confidence_level,
            point_by_split[split_name],
        )
        bootstrap_raw.extend(raw)
        interval_rows.extend(intervals)
        contrast_rows.extend(contrasts)
        split_decisions[split_name] = _classify_split(
            point_by_split[split_name], contrasts
        )

    if all(value == "strong_support" for value in split_decisions.values()):
        overall_decision = "strong_support_on_both_benchmarks"
    elif all(value in {"strong_support", "descriptive_support"} for value in split_decisions.values()):
        overall_decision = "consistent_descriptive_support_on_both_benchmarks"
    else:
        overall_decision = "mixed_or_not_supported_across_benchmarks"

    split_summary: list[dict[str, Any]] = []
    for item in split_inputs:
        split_name = item["split_name"]
        state = split_state[split_name]
        full = point_by_split[split_name]["full_binary_predictor"]
        selective = point_by_split[split_name]["frozen_threshold_accepted_set"]
        random = point_by_split[split_name]["equal_count_random_rejection"]
        full_contrast = next(
            row
            for row in contrast_rows
            if row["split_name"] == split_name
            and row["comparator"] == "full_binary_predictor"
            and row["metric"] == "accepted_error_rate"
        )
        random_contrast = next(
            row
            for row in contrast_rows
            if row["split_name"] == split_name
            and row["comparator"] == "equal_count_random_rejection"
            and row["metric"] == "accepted_error_rate"
        )
        split_summary.append(
            {
                "split_name": split_name,
                "sample_count": int(state["labels"].size),
                "positive_count": int(np.sum(state["labels"] == 1)),
                "negative_count": int(np.sum(state["labels"] == 0)),
                "uncertainty_score_cutoff": float(policy["abstention_threshold"]),
                "oof_target_coverage_reference": float(policy["target_coverage"]),
                "realized_coverage": float(selective["coverage"]),
                "accepted_n": int(selective["accepted_n"]),
                "rejected_n": int(selective["rejected_n"]),
                "full_error_rate": float(full["accepted_error_rate"]),
                "frozen_accepted_error_rate": float(selective["accepted_error_rate"]),
                "random_mean_error_rate": float(random["accepted_error_rate"]),
                "full_minus_frozen_error": float(full_contrast["point_estimate"]),
                "full_minus_frozen_error_ci95_lower": float(full_contrast["ci95_lower"]),
                "full_minus_frozen_error_ci95_upper": float(full_contrast["ci95_upper"]),
                "random_minus_frozen_error": float(random_contrast["point_estimate"]),
                "random_minus_frozen_error_ci95_lower": float(random_contrast["ci95_lower"]),
                "random_minus_frozen_error_ci95_upper": float(random_contrast["ci95_upper"]),
                "transfer_decision": split_decisions[split_name],
            }
        )

    decision = {
        "artifact_type": "ghxtox_selective_prediction_frozen_threshold_test_transfer_decision",
        "decision": overall_decision,
        "per_split_decision": split_decisions,
        "policy_reselected": False,
        "test_specific_cutoff_used": False,
        "test_specific_coverage_forced": False,
        "test_labels_used_for_selection": False,
        "test_labels_used_for_evaluation_only": True,
        "canonical_binary_model_changed": False,
        "fusion_or_binary_threshold_changed": False,
        "external_benchmark_transfer_evaluated": True,
        "new_independent_dataset_claimed": False,
        "label_blind_decision_table_sha256": label_blind_sha256,
    }
    summary = {
        "artifact_type": "ghxtox_selective_prediction_frozen_threshold_test_transfer",
        "protocol": str(protocol_path),
        "frozen_policy": protocol["frozen_policy_contract"],
        "selection_firewall": protocol["selection_firewall"],
        "split_summary": split_summary,
        "decision": decision,
        "random_rejection_control": random_spec,
        "paired_bootstrap": bootstrap_spec,
    }

    _write_csv(output / "method_point_estimates.csv", point_rows)
    _write_csv(output / "split_summary.csv", split_summary)
    _write_csv(output / "random_rejection_replicates.csv", random_raw)
    _write_csv(output / "random_rejection_summary.csv", random_summary)
    _write_csv(output / "paired_bootstrap_method_replicates.csv", bootstrap_raw)
    _write_csv(output / "method_bootstrap_intervals.csv", interval_rows)
    _write_csv(output / "paired_bootstrap_contrasts.csv", contrast_rows)
    _write_csv(output / "input_manifest.csv", input_manifest)
    (output / "results_table.md").write_text(_point_markdown(point_rows), encoding="utf-8")
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    result_lines = []
    for row in split_summary:
        result_lines.append(
            f"- {row['split_name']}: realized coverage {100 * row['realized_coverage']:.2f}% "
            f"({row['accepted_n']}/{row['sample_count']} accepted); accepted error "
            f"{100 * row['frozen_accepted_error_rate']:.2f}% versus full "
            f"{100 * row['full_error_rate']:.2f}% and equal-count random mean "
            f"{100 * row['random_mean_error_rate']:.2f}%; decision "
            f"`{row['transfer_decision']}`."
        )
    readme = f"""# Frozen-threshold Test1/Test2 selective-prediction transfer

This package applies the already selected training-OOF `margin_only` policy to the canonical single-structure Test1 and Test2 prediction files. The policy uses only normalized closeness of the frozen GHXTox fusion probability to the fixed binary threshold. Its uncertainty-score cutoff is `{float(policy['abstention_threshold']):.15f}`; samples at or below this value are accepted, and samples above it are labeled `Uncertain`.

The score transform, feature, binary threshold, abstention cutoff, and acceptance inequality were frozen before Test outcomes were inspected. Test-specific quantiles and target coverages are not used. `label_blind_decisions.csv` was written and hashed before labels were read for metrics. Test labels enter only the post-selection metric, random-control, and bootstrap stages.

Random rejection uses {int(random_spec['replicates']):,} uniform masks with exactly the realized accepted count for each split (seed {int(random_spec['seed'])}). Paired uncertainty uses {int(bootstrap_spec['replicates']):,} label-stratified bootstrap replicates (seed {int(bootstrap_spec['seed'])}); the original frozen acceptance mask is indexed onto resampled observations and is never recalculated from Test quantiles. No p-values or significance marks are reported.

## Results

{chr(10).join(result_lines)}

Overall decision: `{overall_decision}`.

These are frozen-policy transfer results on established Test1/Test2 benchmark sets. They are stronger than test-specific abstention tuning, but they are not evidence from a newly collected independent dataset and do not authorize any change to the selected policy.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    checksum_lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output), "decision": decision, "splits": split_summary}, indent=2))


if __name__ == "__main__":
    main()
