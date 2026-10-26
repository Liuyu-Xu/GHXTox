#!/usr/bin/env python
"""Complete the frozen GHXTox selective-prediction OOF experiment.

The selected uncertainty policy is not refitted.  This script adds five exact
coverage operating points, an equal-coverage random-rejection control, and
paired class-stratified bootstrap intervals using training OOF rows only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "accepted_error_rate",
    "accepted_mcc",
    "accepted_balanced_accuracy",
    "accepted_sensitivity",
    "accepted_specificity",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


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
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields is not None else list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _accepted_count(coverage: float, n: int) -> int:
    return max(1, min(n, int(round(float(coverage) * n))))


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accepted_error_rate": (fp + fn) / max(labels.size, 1),
        "accepted_mcc": float(mcc),
        "accepted_balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "accepted_sensitivity": float(sensitivity),
        "accepted_specificity": float(specificity),
    }


def _operating_row(
    method: str,
    coverage: float,
    accepted_indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    metrics = _classification_metrics(labels[accepted_indices], predictions[accepted_indices])
    accepted_n = int(accepted_indices.size)
    return {
        "method": method,
        "split_name": "training OOF",
        "coverage_target": float(coverage),
        "coverage": float(accepted_n / labels.size),
        "accepted_n": accepted_n,
        "rejected_n": int(labels.size - accepted_n),
        **metrics,
    }


def _stable_uncertainty_order(scores: np.ndarray, source_index: np.ndarray) -> np.ndarray:
    if scores.shape != source_index.shape:
        raise ValueError("Uncertainty and source-index vectors differ in shape.")
    return np.lexsort((source_index, scores))


def _quantiles(values: np.ndarray, confidence_level: float) -> tuple[float, float, float, float]:
    alpha = (1.0 - float(confidence_level)) / 2.0
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _load_and_validate(protocol: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    verified: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for source in protocol["inputs"]:
        path = _resolve(source["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != str(source["sha256"]):
            raise ValueError(f"Checksum mismatch for {path}: {actual} != {source['sha256']}")
        paths[str(source["role"])] = path
        verified.append(
            {
                "role": source["role"],
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": actual,
            }
        )

    policy = json.loads(paths["frozen_selected_policy"].read_text(encoding="utf-8"))
    selection = json.loads(paths["frozen_policy_selection_audit"].read_text(encoding="utf-8"))
    expected = protocol["selected_uncertainty_policy"]
    if policy.get("profile") != expected["required_profile"]:
        raise ValueError("Frozen selected policy profile differs from the formal protocol.")
    if policy.get("feature_names") != expected["required_feature_names"]:
        raise ValueError("Frozen selected policy features differ from the formal protocol.")
    required_model_type = expected.get("required_model_type")
    if required_model_type and policy.get("model", {}).get("type") != required_model_type:
        raise ValueError("Frozen selected policy model type differs from the formal protocol.")
    if selection.get("selected_public_policy") != expected["required_profile"]:
        raise ValueError("Policy-selection audit does not identify the required public policy.")
    if bool(policy.get("test_labels_used")) or bool(selection.get("test1_or_test2_labels_used")):
        raise ValueError("Frozen source violates the Test1/Test2 data firewall.")
    if bool(selection.get("canonical_binary_model_changed")):
        raise ValueError("Frozen source reports a changed canonical binary model.")

    rows = _read_csv(paths["frozen_cross_fitted_oof_predictions"])
    required = {
        "source_index",
        "sample_id",
        "fold",
        "label",
        "toxicity_probability",
        "binary_prediction",
        "binary_error",
        "decision_closeness",
        expected["oof_uncertainty_column"],
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"OOF table is missing columns: {sorted(missing)}")
    n = len(rows)
    source_index = np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64)
    if not np.array_equal(source_index, np.arange(n, dtype=np.int64)):
        raise ValueError("OOF source_index is not exactly contiguous in source order.")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != n:
        raise ValueError("OOF sample IDs are not unique.")
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != {0, 1, 2, 3, 4}:
        raise ValueError("OOF rows do not contain exactly folds 0--4.")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    predictions = np.asarray([int(row["binary_prediction"]) for row in rows], dtype=np.int64)
    binary_error = np.asarray([int(row["binary_error"]) for row in rows], dtype=np.int64)
    if not np.array_equal(binary_error, (predictions != labels).astype(np.int64)):
        raise ValueError("OOF binary_error does not match prediction versus label.")
    probability = np.asarray([float(row["toxicity_probability"]) for row in rows], dtype=np.float64)
    uncertainty = np.asarray(
        [float(row[expected["oof_uncertainty_column"]]) for row in rows], dtype=np.float64
    )
    decision_closeness = np.asarray([float(row["decision_closeness"]) for row in rows])
    if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(uncertainty)):
        raise ValueError("OOF probability or uncertainty contains non-finite values.")
    threshold = float(policy["decision_threshold"])
    expected_closeness = 1.0 - np.abs(probability - threshold) / max(threshold, 1.0 - threshold)
    if float(np.max(np.abs(expected_closeness - decision_closeness))) > 1e-10:
        raise ValueError("Stored decision_closeness differs from the frozen definition.")
    if policy.get("model", {}).get("type") == "direct_feature_score" and float(
        np.max(np.abs(uncertainty - decision_closeness))
    ) > 1e-12:
        raise ValueError("Direct margin uncertainty differs from decision_closeness.")
    if not np.array_equal(predictions, (probability >= threshold).astype(np.int64)):
        raise ValueError("OOF binary predictions differ from the frozen threshold.")
    return (
        {
            "source_index": source_index,
            "folds": folds,
            "labels": labels,
            "predictions": predictions,
            "probability": probability,
            "uncertainty": uncertainty,
            "decision_closeness": decision_closeness,
            "sample_ids": np.asarray(sample_ids, dtype=object),
            "threshold": np.asarray([threshold], dtype=np.float64),
        },
        verified,
    )


def _randomization_experiment(
    labels: np.ndarray,
    predictions: np.ndarray,
    coverages: list[float],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    n = labels.size
    raw_rows: list[dict[str, Any]] = []
    store = {
        (coverage, metric): np.empty(replicates, dtype=np.float64)
        for coverage in coverages
        for metric in METRICS
    }
    for replicate in range(replicates):
        order = rng.permutation(n)
        for coverage in coverages:
            accepted = order[: _accepted_count(coverage, n)]
            row = _operating_row("random_rejection", coverage, accepted, labels, predictions)
            raw_rows.append({"replicate": replicate, **row})
            for metric in METRICS:
                store[(coverage, metric)][replicate] = float(row[metric])
    summary_rows: list[dict[str, Any]] = []
    for coverage in coverages:
        for metric in METRICS:
            mean, median, lower, upper = _quantiles(store[(coverage, metric)], 0.95)
            summary_rows.append(
                {
                    "method": "random_rejection",
                    "split_name": "training OOF",
                    "coverage_target": coverage,
                    "accepted_n": _accepted_count(coverage, n),
                    "rejected_n": n - _accepted_count(coverage, n),
                    "metric": metric,
                    "point_estimate": mean,
                    "median": median,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "uncertainty_type": "uniform-random-ranking distribution on frozen OOF rows",
                    "replicates": replicates,
                    "seed": seed,
                }
            )
    return raw_rows, summary_rows


def _paired_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    uncertainty: np.ndarray,
    source_index: np.ndarray,
    coverages: list[float],
    replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == value) for value in (0, 1)]
    n = labels.size
    raw_rows: list[dict[str, Any]] = []
    method_values = {
        (method, coverage, metric): np.empty(replicates, dtype=np.float64)
        for method in ("frozen_uncertainty_ranking", "random_rejection")
        for coverage in coverages
        for metric in METRICS
    }
    improvement_values = {
        (coverage, metric): np.empty(replicates, dtype=np.float64)
        for coverage in coverages
        for metric in METRICS
    }
    for replicate in range(replicates):
        draw = np.concatenate(
            [rng.choice(indices, size=indices.size, replace=True) for indices in class_indices]
        )
        boot_labels = labels[draw]
        boot_predictions = predictions[draw]
        boot_uncertainty = uncertainty[draw]
        # Draw position is the final tie-breaker when bootstrap duplicates share source_index.
        boot_tie_index = np.arange(n, dtype=np.int64)
        selective_order = _stable_uncertainty_order(boot_uncertainty, boot_tie_index)
        random_order = rng.permutation(n)
        for coverage in coverages:
            k = _accepted_count(coverage, n)
            selective = _operating_row(
                "frozen_uncertainty_ranking",
                coverage,
                selective_order[:k],
                boot_labels,
                boot_predictions,
            )
            random = _operating_row(
                "random_rejection",
                coverage,
                random_order[:k],
                boot_labels,
                boot_predictions,
            )
            raw: dict[str, Any] = {"replicate": replicate, "coverage_target": coverage}
            for metric in METRICS:
                selective_value = float(selective[metric])
                random_value = float(random[metric])
                raw[f"selective_{metric}"] = selective_value
                raw[f"random_{metric}"] = random_value
                raw[f"selective_minus_random_{metric}"] = selective_value - random_value
                improvement = (
                    random_value - selective_value
                    if metric == "accepted_error_rate"
                    else selective_value - random_value
                )
                raw[f"positive_improvement_{metric}"] = improvement
                method_values[("frozen_uncertainty_ranking", coverage, metric)][replicate] = selective_value
                method_values[("random_rejection", coverage, metric)][replicate] = random_value
                improvement_values[(coverage, metric)][replicate] = improvement
            raw_rows.append(raw)

    method_rows: list[dict[str, Any]] = []
    for (method, coverage, metric), values in method_values.items():
        mean, median, lower, upper = _quantiles(values, confidence_level)
        method_rows.append(
            {
                "method": method,
                "split_name": "training OOF",
                "coverage_target": coverage,
                "accepted_n": _accepted_count(coverage, n),
                "rejected_n": n - _accepted_count(coverage, n),
                "metric": metric,
                "bootstrap_mean": mean,
                "bootstrap_median": median,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "uncertainty_type": "class-stratified paired bootstrap; random method includes a uniform nested random ranking per replicate",
                "bootstrap_replicates": replicates,
                "seed": seed,
            }
        )

    contrast_rows: list[dict[str, Any]] = []
    for (coverage, metric), values in improvement_values.items():
        mean, median, lower, upper = _quantiles(values, confidence_level)
        contrast_rows.append(
            {
                "contrast": "positive values favor frozen uncertainty ranking",
                "split_name": "training OOF",
                "coverage_target": coverage,
                "accepted_n": _accepted_count(coverage, n),
                "rejected_n": n - _accepted_count(coverage, n),
                "metric": metric,
                "positive_improvement_definition": (
                    "random minus selective" if metric == "accepted_error_rate" else "selective minus random"
                ),
                "point_estimate": mean,
                "median": median,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_probability_gt_0": float(np.mean(values > 0.0)),
                "bootstrap_replicates": replicates,
                "seed": seed,
            }
        )
    return raw_rows, method_rows, contrast_rows


def _markdown_table(comparison_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Selective prediction operating points",
        "",
        "All values are from 6,387 training out-of-fold predictions. Random values are means over 5,000 equal-coverage uniform random rankings.",
        "",
        "| Method | Coverage | Accepted n | Error | MCC | BACC | SN | SP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        name = "Frozen uncertainty ranking" if row["method"] == "frozen_uncertainty_ranking" else "Random rejection"
        lines.append(
            f"| {name} | {100 * float(row['coverage_target']):.0f}% | {int(row['accepted_n'])} | "
            f"{float(row['accepted_error_rate']):.4f} | {float(row['accepted_mcc']):.4f} | "
            f"{float(row['accepted_balanced_accuracy']):.4f} | {float(row['accepted_sensitivity']):.4f} | "
            f"{float(row['accepted_specificity']):.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "selective_prediction_formal_experiment_v1.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    protocol_path = _resolve(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if any(
        bool(protocol["scope"][key])
        for key in (
            "allow_policy_refit",
            "allow_feature_reselection",
            "allow_binary_model_change",
            "allow_fusion_or_threshold_change",
            "allow_test1_or_test2_labels",
            "external_validation_claimed",
        )
    ):
        raise ValueError("Formal selective experiment violates its frozen scope.")
    output = _resolve(args.output_dir or protocol["output_dir"]).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data, input_manifest = _load_and_validate(protocol)
    labels = data["labels"]
    predictions = data["predictions"]
    source_index = data["source_index"]
    uncertainty = data["uncertainty"]
    coverages = [float(value) for value in protocol["operating_points"]["coverage_targets"]]
    selective_order = _stable_uncertainty_order(uncertainty, source_index)
    selective_rows = [
        _operating_row(
            "frozen_uncertainty_ranking",
            coverage,
            selective_order[: _accepted_count(coverage, labels.size)],
            labels,
            predictions,
        )
        for coverage in coverages
    ]

    random_spec = protocol["random_rejection_control"]
    random_raw, random_summary = _randomization_experiment(
        labels,
        predictions,
        coverages,
        int(random_spec["replicates"]),
        int(random_spec["seed"]),
    )
    bootstrap_spec = protocol["paired_bootstrap"]
    paired_raw, method_intervals, contrast_rows = _paired_bootstrap(
        labels,
        predictions,
        uncertainty,
        source_index,
        coverages,
        int(bootstrap_spec["replicates"]),
        int(bootstrap_spec["seed"]),
        float(bootstrap_spec["confidence_level"]),
    )

    random_lookup = {
        (float(row["coverage_target"]), row["metric"]): row for row in random_summary
    }
    comparison_rows: list[dict[str, Any]] = []
    for selective in selective_rows:
        comparison_rows.append(dict(selective))
        coverage = float(selective["coverage_target"])
        random_row = {
            "method": "random_rejection",
            "split_name": "training OOF",
            "coverage_target": coverage,
            "coverage": float(selective["coverage"]),
            "accepted_n": int(selective["accepted_n"]),
            "rejected_n": int(selective["rejected_n"]),
        }
        for metric in METRICS:
            random_row[metric] = float(random_lookup[(coverage, metric)]["point_estimate"])
        comparison_rows.append(random_row)

    primary_coverage = float(protocol["interpretation_rule"]["primary_control_point"])
    primary = next(
        row
        for row in contrast_rows
        if abs(float(row["coverage_target"]) - primary_coverage) < 1e-12
        and row["metric"] == "accepted_error_rate"
    )
    partial_coverages = [coverage for coverage in coverages if coverage < 1.0]
    selective_lookup = {
        float(row["coverage_target"]): float(row["accepted_error_rate"]) for row in selective_rows
    }
    random_error_lookup = {
        coverage: float(random_lookup[(coverage, "accepted_error_rate")]["point_estimate"])
        for coverage in coverages
    }
    selective_below_random_all_partial = all(
        selective_lookup[coverage] < random_error_lookup[coverage] for coverage in partial_coverages
    )
    support_passed = float(primary["ci95_lower"]) > 0.0 and selective_below_random_all_partial

    _write_csv(output / "selective_operating_points.csv", selective_rows)
    _write_csv(output / "random_rejection_replicates.csv", random_raw)
    _write_csv(output / "random_rejection_summary.csv", random_summary)
    _write_csv(output / "paired_bootstrap_replicates.csv", paired_raw)
    _write_csv(output / "method_bootstrap_intervals.csv", method_intervals)
    _write_csv(output / "paired_bootstrap_contrasts.csv", contrast_rows)
    _write_csv(output / "five_coverage_comparison.csv", comparison_rows)
    (output / "five_coverage_table.md").write_text(
        _markdown_table(comparison_rows), encoding="utf-8"
    )

    decision = {
        "artifact_type": "ghxtox_selective_prediction_formal_decision",
        "decision": (
            "supports_difficult_sample_identification_on_training_oof"
            if support_passed
            else "does_not_confirm_difficult_sample_identification"
        ),
        "selected_policy_remains": protocol["selected_uncertainty_policy"]["required_profile"],
        "policy_reselected": False,
        "test1_or_test2_labels_used": False,
        "canonical_binary_model_changed": False,
        "primary_coverage": primary_coverage,
        "primary_metric": "random minus selective accepted error",
        "primary_effect": primary,
        "selective_error_below_random_mean_at_all_partial_coverages": selective_below_random_all_partial,
        "external_generalization_claimed": False,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    summary = {
        "artifact_type": "ghxtox_selective_prediction_formal_experiment",
        "protocol": str(protocol_path),
        "sample_count": int(labels.size),
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
        "fold_counts": {
            str(fold): int(np.sum(data["folds"] == fold)) for fold in sorted(set(data["folds"].tolist()))
        },
        "binary_threshold": float(data["threshold"][0]),
        "selected_uncertainty_policy": protocol["selected_uncertainty_policy"],
        "coverages": coverages,
        "selective_operating_points": selective_rows,
        "decision": decision,
        "scope": protocol["scope"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    input_manifest.extend(
        [
            {
                "role": "formal_protocol",
                "path": str(protocol_path),
                "size_bytes": protocol_path.stat().st_size,
                "sha256": _sha256(protocol_path),
            },
            {
                "role": "formal_evaluator",
                "path": str(Path(__file__).resolve()),
                "size_bytes": Path(__file__).stat().st_size,
                "sha256": _sha256(Path(__file__).resolve()),
            },
        ]
    )
    _write_csv(output / "input_manifest.csv", input_manifest)
    readme = f"""# Formal selective-prediction experiment

This package completes the frozen GHXTox selective-prediction experiment on 6,387 training out-of-fold predictions. The selected policy remains `margin_only`; no feature was reselected, no model/fusion/threshold was changed, and no Test1 or Test2 label was loaded.

The uncertainty rank is a monotone function of closeness to the frozen binary threshold. It is an error-ranking score, not a calibrated error probability. Exact results are reported at 100%, 95%, 90%, 85%, and 80% coverage.

Random rejection uses {int(random_spec['replicates']):,} uniform nested random rankings with exactly the same accepted count as selective prediction at every coverage. Paired uncertainty uses {int(bootstrap_spec['replicates']):,} true-label-stratified bootstrap replicates; each replicate evaluates both the frozen uncertainty ranking and an equal-coverage uniform random ranking on the same resampled observations.

Primary control result at 90% coverage: random-minus-selective accepted error = {float(primary['point_estimate']):.6f}, 95% paired-bootstrap CI [{float(primary['ci95_lower']):.6f}, {float(primary['ci95_upper']):.6f}]. Decision: `{decision['decision']}`.

This is a training-OOF reliability analysis, not external validation. The original binary prediction remains available for every abstained sample.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    checksum_lines = []
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.name}")
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
