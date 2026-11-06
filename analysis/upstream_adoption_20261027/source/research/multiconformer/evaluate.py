"""Evaluate the exploratory multi-conformer residual against its frozen base."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ghxtox.structure_inference import predict_structure
from ghxtox.utils import resolve_device


METRIC_NAMES = (
    "balanced_accuracy",
    "f1",
    "mcc",
    "auroc",
    "auprc",
    "brier",
)
LOWER_IS_BETTER = {"brier"}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no prediction rows")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate sample identifiers")
    return rows


def _indexed_rows(path: str | Path) -> dict[str, dict[str, str]]:
    return {row["sample_id"]: row for row in _read_rows(path)}


def _probability(row: dict[str, str], column: str, path: str | Path) -> float:
    if column not in row:
        raise ValueError(f"{path}: missing probability column {column!r}")
    value = float(row[column])
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{path}: invalid probability for {row['sample_id']}")
    return value


def _metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(bool, copy=False)
    positives = labels == 1
    negatives = ~positives
    tp = float(np.sum(predictions & positives))
    tn = float(np.sum(~predictions & negatives))
    fp = float(np.sum(predictions & negatives))
    fn = float(np.sum(~predictions & positives))
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    denominator = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0)
    )
    values = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "mcc": (tp * tn - fp * fn) / denominator,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(np.mean((scores - labels) ** 2)),
    }
    return values


def _stratified_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(group.size == 0 for group in groups):
        raise ValueError("Both classes are required for paired resampling.")
    return np.concatenate(
        [rng.choice(group, size=group.size, replace=True) for group in groups]
    )


def _paired_comparison(
    labels: np.ndarray,
    base_scores: np.ndarray,
    residual_scores: np.ndarray,
    threshold: float,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, Any] | None:
    if bootstrap_iterations == 0 and permutation_iterations == 0:
        return None
    if bootstrap_iterations < 100 or permutation_iterations < 100:
        raise ValueError("Paired resampling requires at least 100 iterations of each type.")
    base_predictions = base_scores >= threshold
    residual_predictions = residual_scores >= threshold
    base_values = _metric_values(labels, base_scores, base_predictions)
    residual_values = _metric_values(labels, residual_scores, residual_predictions)
    observed = {
        name: residual_values[name] - base_values[name] for name in METRIC_NAMES
    }

    rng = np.random.default_rng(seed)
    bootstrap = {
        name: np.empty(bootstrap_iterations, dtype=np.float64)
        for name in METRIC_NAMES
    }
    for iteration in range(bootstrap_iterations):
        indices = _stratified_indices(labels, rng)
        first = _metric_values(
            labels[indices], base_scores[indices], base_predictions[indices]
        )
        second = _metric_values(
            labels[indices], residual_scores[indices], residual_predictions[indices]
        )
        for name in METRIC_NAMES:
            bootstrap[name][iteration] = second[name] - first[name]

    randomized = {
        name: np.empty(permutation_iterations, dtype=np.float64)
        for name in METRIC_NAMES
    }
    for iteration in range(permutation_iterations):
        swap = rng.random(labels.size) < 0.5
        first_scores = np.where(swap, residual_scores, base_scores)
        second_scores = np.where(swap, base_scores, residual_scores)
        first_predictions = np.where(swap, residual_predictions, base_predictions)
        second_predictions = np.where(swap, base_predictions, residual_predictions)
        first = _metric_values(labels, first_scores, first_predictions)
        second = _metric_values(labels, second_scores, second_predictions)
        for name in METRIC_NAMES:
            randomized[name][iteration] = second[name] - first[name]

    metrics: dict[str, Any] = {}
    for name in METRIC_NAMES:
        sign = -1.0 if name in LOWER_IS_BETTER else 1.0
        metrics[name] = {
            "base": base_values[name],
            "residual": residual_values[name],
            "delta_residual_minus_base": observed[name],
            "paired_bootstrap_95_ci": {
                "lower": float(np.quantile(bootstrap[name], 0.025)),
                "upper": float(np.quantile(bootstrap[name], 0.975)),
            },
            "bootstrap_probability_residual_better": float(
                np.mean(sign * bootstrap[name] > 0.0)
            ),
            "paired_randomization_two_sided_p": float(
                (np.sum(np.abs(randomized[name]) >= abs(observed[name])) + 1)
                / (permutation_iterations + 1)
            ),
        }
    return {
        "bootstrap": "class-stratified paired percentile bootstrap",
        "bootstrap_iterations": bootstrap_iterations,
        "randomization": "within-sample exchange of paired model outputs",
        "permutation_iterations": permutation_iterations,
        "seed": seed,
        "metrics": metrics,
    }


def _assemble_rows(
    sequence_rows: list[dict[str, str]],
    base_structure: dict[str, float],
    residual_structure: dict[str, float],
    available: dict[str, bool],
    sequence_column: str,
    sequence_path: str | Path,
    sequence_weight: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = {row["sample_id"] for row in sequence_rows}
    if set(base_structure) != expected or set(residual_structure) != expected:
        raise ValueError("Sequence and structural predictions do not cover identical samples.")
    for row in sequence_rows:
        sample_id = row["sample_id"]
        label = int(row["label"])
        sequence_probability = _probability(row, sequence_column, sequence_path)
        base_probability = base_structure[sample_id]
        residual_probability = residual_structure[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "sequence_probability": sequence_probability,
                "base_structure_probability": base_probability,
                "residual_structure_probability": residual_probability,
                "base_fusion_probability": (
                    sequence_weight * sequence_probability
                    + (1.0 - sequence_weight) * base_probability
                ),
                "residual_fusion_probability": (
                    sequence_weight * sequence_probability
                    + (1.0 - sequence_weight) * residual_probability
                ),
                "conformer_available": int(available.get(sample_id, False)),
            }
        )
    return rows


def _write_outputs(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    threshold: float,
    sequence_weight: float,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    base = np.asarray([row["base_fusion_probability"] for row in rows])
    residual = np.asarray([row["residual_fusion_probability"] for row in rows])
    base_predictions = base >= threshold
    residual_predictions = residual >= threshold
    summary = {
        "protocol": {
            "records": len(rows),
            "records_with_conformers": int(
                sum(row["conformer_available"] for row in rows)
            ),
            "sequence_weight": sequence_weight,
            "structure_weight": 1.0 - sequence_weight,
            "fusion_threshold": threshold,
            "changed_binary_decisions": int(
                np.sum(base_predictions != residual_predictions)
            ),
        },
        "base": _metric_values(labels, base, base_predictions),
        "residual": _metric_values(labels, residual, residual_predictions),
        "paired_comparison": _paired_comparison(
            labels,
            base,
            residual,
            threshold,
            bootstrap_iterations,
            permutation_iterations,
            seed,
        ),
    }
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.10g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _checkpoint_command(args: argparse.Namespace) -> dict[str, Any]:
    sequence_rows = _read_rows(args.sequence_predictions)
    result = predict_structure(
        args.processed,
        args.checkpoint,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
    )
    if any(label is None for label in result["labels"]):
        raise ValueError("The processed dataset must contain labels.")
    base = {
        sample_id: float(result["base_probabilities"][index])
        for index, sample_id in enumerate(result["sample_ids"])
    }
    residual = {
        sample_id: float(result["probabilities"][index])
        for index, sample_id in enumerate(result["sample_ids"])
    }
    available = {
        sample_id: bool(result["conformer_available"][index])
        for index, sample_id in enumerate(result["sample_ids"])
    }
    labels = {
        sample_id: int(result["labels"][index])
        for index, sample_id in enumerate(result["sample_ids"])
    }
    for row in sequence_rows:
        sample_id = row["sample_id"]
        if sample_id not in labels or int(row["label"]) != labels[sample_id]:
            raise ValueError(f"Label or sample mismatch for {sample_id}.")
    rows = _assemble_rows(
        sequence_rows,
        base,
        residual,
        available,
        args.sequence_column,
        args.sequence_predictions,
        args.sequence_weight,
    )
    return _write_outputs(
        rows,
        args.output_dir,
        args.threshold,
        args.sequence_weight,
        args.bootstrap_iterations,
        args.permutation_iterations,
        args.seed,
    )


def _predictions_command(args: argparse.Namespace) -> dict[str, Any]:
    sequence_rows = _read_rows(args.sequence_predictions)
    base_rows = _indexed_rows(args.base_structure_predictions)
    residual_rows = _indexed_rows(args.residual_structure_predictions)
    expected = {row["sample_id"] for row in sequence_rows}
    if set(base_rows) != expected or set(residual_rows) != expected:
        raise ValueError("All prediction files must contain identical sample identifiers.")
    base: dict[str, float] = {}
    residual: dict[str, float] = {}
    available: dict[str, bool] = {}
    covered_ids: set[str] = set()
    if args.coverage_manifest:
        covered_ids = {
            row["sample_id"] for row in _read_rows(args.coverage_manifest)
        }
        unknown = covered_ids - expected
        if unknown:
            raise ValueError(
                "The coverage manifest contains samples absent from the predictions."
            )
    for row in sequence_rows:
        sample_id = row["sample_id"]
        for source in (base_rows[sample_id], residual_rows[sample_id]):
            if int(source["label"]) != int(row["label"]):
                raise ValueError(f"Label mismatch for {sample_id}.")
        base[sample_id] = _probability(
            base_rows[sample_id], args.base_structure_column, args.base_structure_predictions
        )
        residual[sample_id] = _probability(
            residual_rows[sample_id],
            args.residual_structure_column,
            args.residual_structure_predictions,
        )
        available[sample_id] = (
            sample_id in covered_ids
            if args.coverage_manifest
            else bool(int(residual_rows[sample_id].get("conformer_available", "0")))
        )
    rows = _assemble_rows(
        sequence_rows,
        base,
        residual,
        available,
        args.sequence_column,
        args.sequence_predictions,
        args.sequence_weight,
    )
    return _write_outputs(
        rows,
        args.output_dir,
        args.threshold,
        args.sequence_weight,
        args.bootstrap_iterations,
        args.permutation_iterations,
        args.seed,
    )


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sequence-predictions", required=True)
    parser.add_argument("--sequence-column", default="toxicity_probability")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-weight", type=float, default=0.72)
    parser.add_argument("--threshold", type=float, default=0.4702225653)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--permutation-iterations", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260808)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    checkpoint = commands.add_parser(
        "checkpoint", help="Evaluate both logits emitted by one residual checkpoint."
    )
    _common_arguments(checkpoint)
    checkpoint.add_argument("--processed", required=True)
    checkpoint.add_argument("--checkpoint", required=True)
    checkpoint.add_argument("--device", default="cuda")
    checkpoint.add_argument("--batch-size", type=int, default=16)
    checkpoint.set_defaults(handler=_checkpoint_command)

    predictions = commands.add_parser(
        "predictions", help="Compare aligned precomputed structural predictions."
    )
    _common_arguments(predictions)
    predictions.add_argument("--base-structure-predictions", required=True)
    predictions.add_argument("--residual-structure-predictions", required=True)
    predictions.add_argument(
        "--coverage-manifest",
        help="Optional CSV whose sample_id column defines the conformer-covered cohort.",
    )
    predictions.add_argument(
        "--base-structure-column", default="toxicity_probability"
    )
    predictions.add_argument(
        "--residual-structure-column", default="toxicity_probability"
    )
    predictions.set_defaults(handler=_predictions_command)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = args.handler(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
