"""Analyse where a conformer-residual candidate changes OOF predictions.

The report is deliberately cohort-level and training-only.  It links paired
baseline/candidate probabilities to ensemble confidence and flexibility
descriptors without using Test1/Test2 labels for model selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.metrics import binary_metrics
from ghxtox.utils import save_json


PROBABILITY_COLUMN = "toxicity_probability"
LENGTH_BUCKETS = (
    "1-15",
    "16-20",
    "21-25",
    "26-30",
    "31-35",
    "36-40",
    "41-45",
    "46-50",
    "51+",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    clipped = np.clip(probabilities.astype(np.float32), 1e-7, 1.0 - 1e-7)
    logits = torch.logit(torch.from_numpy(clipped))
    return binary_metrics(
        logits,
        torch.from_numpy(labels.astype(np.float32)),
        threshold=threshold,
    )


def _metric_deltas(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float]]:
    first = _metrics(labels, baseline, threshold)
    second = _metrics(labels, candidate, threshold)
    keys = ("balanced_accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece_10")
    return {
        key: {
            "baseline": float(first[key]),
            "candidate": float(second[key]),
            "delta_candidate_minus_baseline": float(second[key] - first[key]),
        }
        for key in keys
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranked[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranked


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 3 or np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _ensemble_features(cache_root: Path, sample_id: str) -> dict[str, float]:
    sample_root = cache_root / sample_id
    summary_path = sample_root / "ensemble_summary.npz"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing ensemble summary for {sample_id}: {summary_path}")
    summary = np.load(summary_path)
    conformers = sorted(sample_root.glob("conformer_*.npz"))
    if not conformers:
        raise FileNotFoundError(f"No conformer cache files for {sample_id}: {sample_root}")
    plddt = np.concatenate(
        [np.asarray(np.load(path)["plddt"], dtype=np.float32).reshape(-1) for path in conformers]
    )
    return {
        "num_conformers": float(len(conformers)),
        "conformer_mean_plddt": float(plddt.mean()),
        "conformer_min_plddt": float(plddt.min()),
        "conformer_low_plddt_fraction": float((plddt < 0.5).mean()),
        "mean_residue_rmsf": float(np.asarray(summary["residue_rmsf"]).mean()),
        "mean_conformer_rmsd": float(np.asarray(summary["conformer_rmsd"]).mean()),
        "std_conformer_rmsd": float(np.asarray(summary["conformer_rmsd"]).std()),
        "mean_radius_gyration": float(np.asarray(summary["radius_gyration"]).mean()),
        "std_radius_gyration": float(np.asarray(summary["radius_gyration"]).std()),
    }


def _length_bucket(length: int) -> str:
    if length <= 15:
        return "1-15"
    if length <= 20:
        return "16-20"
    if length <= 25:
        return "21-25"
    if length <= 30:
        return "26-30"
    if length <= 35:
        return "31-35"
    if length <= 40:
        return "36-40"
    if length <= 45:
        return "41-45"
    if length <= 50:
        return "46-50"
    return "51+"


def analyse_conformer_cohort(
    baseline_path: str | Path,
    candidate_path: str | Path,
    task_manifest_path: str | Path,
    cache_root: str | Path,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_rows = _read_csv(baseline_path)
    candidate_rows = _read_csv(candidate_path)
    task_rows = _read_csv(task_manifest_path)
    baseline = {row["sample_id"]: row for row in baseline_rows}
    candidate = {row["sample_id"]: row for row in candidate_rows}
    tasks = {row["sample_id"]: row for row in task_rows}
    if set(baseline) != set(candidate) or set(baseline) != set(tasks):
        raise ValueError(
            "Baseline, candidate, and task manifests must contain exactly the same sample IDs."
        )

    cache_root = Path(cache_root)
    sample_rows: list[dict[str, Any]] = []
    for sample_id in baseline:
        first = baseline[sample_id]
        second = candidate[sample_id]
        task = tasks[sample_id]
        label = int(first["label"])
        if label != int(second["label"]) or label != int(task["label"]):
            raise ValueError(f"Label mismatch for {sample_id}.")
        first_probability = float(first[PROBABILITY_COLUMN])
        second_probability = float(second[PROBABILITY_COLUMN])
        features = _ensemble_features(cache_root, sample_id)
        base_correct = int((first_probability >= threshold) == bool(label))
        candidate_correct = int((second_probability >= threshold) == bool(label))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "fold": int(first["fold"]),
                "label": label,
                "sequence": task["sequence"],
                "length": int(task.get("length") or len(task["sequence"])),
                "length_bucket": _length_bucket(
                    int(task.get("length") or len(task["sequence"]))
                ),
                "cysteines": int(task.get("cysteines") or task["sequence"].count("C")),
                "topology_risk": task.get("topology_risk", ""),
                "baseline_probability": first_probability,
                "candidate_probability": second_probability,
                "probability_delta": second_probability - first_probability,
                "absolute_probability_delta": abs(second_probability - first_probability),
                "per_sample_brier_improvement": (first_probability - label) ** 2
                - (second_probability - label) ** 2,
                "baseline_correct": base_correct,
                "candidate_correct": candidate_correct,
                **features,
            }
        )

    labels = np.asarray([row["label"] for row in sample_rows], dtype=np.int64)
    first_probabilities = np.asarray(
        [row["baseline_probability"] for row in sample_rows], dtype=np.float64
    )
    second_probabilities = np.asarray(
        [row["candidate_probability"] for row in sample_rows], dtype=np.float64
    )
    corrected = sum(
        row["baseline_correct"] == 0 and row["candidate_correct"] == 1 for row in sample_rows
    )
    worsened = sum(
        row["baseline_correct"] == 1 and row["candidate_correct"] == 0 for row in sample_rows
    )

    structural_keys = (
        "length",
        "cysteines",
        "conformer_mean_plddt",
        "conformer_low_plddt_fraction",
        "mean_residue_rmsf",
        "mean_conformer_rmsd",
        "mean_radius_gyration",
    )
    targets = (
        "absolute_probability_delta",
        "per_sample_brier_improvement",
    )
    correlations = {
        feature: {
            target: _spearman(
                np.asarray([row[feature] for row in sample_rows], dtype=np.float64),
                np.asarray([row[target] for row in sample_rows], dtype=np.float64),
            )
            for target in targets
        }
        for feature in structural_keys
    }

    strata: dict[str, Any] = {}
    observed_buckets = {str(row["length_bucket"]) for row in sample_rows}
    for bucket in LENGTH_BUCKETS:
        if bucket not in observed_buckets:
            continue
        selected = [index for index, row in enumerate(sample_rows) if row["length_bucket"] == bucket]
        selected_labels = labels[selected]
        strata[bucket] = {
            "samples": len(selected),
            "positive": int(selected_labels.sum()),
            "negative": int(len(selected) - selected_labels.sum()),
            "metrics": _metric_deltas(
                selected_labels,
                first_probabilities[selected],
                second_probabilities[selected],
                threshold,
            ),
        }

    report = {
        "protocol": {
            "scope": "training-only group-aware OOF exploratory cohort analysis",
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "task_manifest": str(task_manifest_path),
            "cache_root": str(cache_root),
            "fixed_threshold": float(threshold),
            "selection_warning": "Do not use these post-hoc strata to retune the pilot model.",
        },
        "cohort": {
            "samples": len(sample_rows),
            "positive": int(labels.sum()),
            "negative": int(len(labels) - labels.sum()),
            "conformer_caches": int(sum(row["num_conformers"] for row in sample_rows)),
        },
        "overall_metrics": _metric_deltas(
            labels,
            first_probabilities,
            second_probabilities,
            threshold,
        ),
        "classification_transitions": {
            "baseline_wrong_candidate_correct": int(corrected),
            "baseline_correct_candidate_wrong": int(worsened),
            "net_corrections": int(corrected - worsened),
            "unchanged": int(len(sample_rows) - corrected - worsened),
        },
        "spearman_correlations": correlations,
        "length_strata": strata,
    }
    return report, sample_rows


def _write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.677819)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-output")
    args = parser.parse_args()
    report, rows = analyse_conformer_cohort(
        args.baseline,
        args.candidate,
        args.task_manifest,
        args.cache_root,
        args.threshold,
    )
    save_json(report, args.output)
    if args.sample_output:
        _write_rows(rows, args.sample_output)


if __name__ == "__main__":
    main()
