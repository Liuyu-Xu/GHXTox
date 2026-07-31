"""Training-only OOF diagnostics for length and conformer-generator confounding."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.conformer_cohort_analysis import _metrics
from ghxtox.multisource_conformer_analysis import _summary_features
from ghxtox.utils import save_json


LENGTH_STRATA = ("11-15", "16-30", "31-50")
STRUCTURE_FEATURES = (
    "mean_plddt",
    "min_plddt",
    "low_plddt_fraction",
    "global_gate",
)
CONFORMER_FEATURES = (
    "mean_residue_rmsf",
    "mean_conformer_rmsd",
    "mean_radius_gyration",
    "contact_uncertainty_8",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _length_stratum(length: int) -> str:
    if 11 <= length <= 15:
        return "11-15"
    if 16 <= length <= 30:
        return "16-30"
    if 31 <= length <= 50:
        return "31-50"
    raise ValueError(
        f"Training sequence length {length} is outside the frozen 11-50 aa scope."
    )


def _confusion(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, int]:
    predictions = probabilities >= threshold
    positive = labels == 1
    return {
        "true_positive": int(np.sum(predictions & positive)),
        "true_negative": int(np.sum(~predictions & ~positive)),
        "false_positive": int(np.sum(predictions & ~positive)),
        "false_negative": int(np.sum(~predictions & positive)),
    }


def _model_summary(
    selected: list[dict[str, Any]],
    probability_key: str,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray([row["label"] for row in selected], dtype=np.int64)
    probabilities = np.asarray(
        [row[probability_key] for row in selected], dtype=np.float64
    )
    return {
        "samples": len(selected),
        "positive": int(labels.sum()),
        "negative": int(len(labels) - labels.sum()),
        "threshold": float(threshold),
        "metrics": _metrics(labels, probabilities, threshold),
        "confusion": _confusion(labels, probabilities, threshold),
    }


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = []
    for label in np.unique(labels):
        selected = labels == label
        recalls.append(float(np.mean(predictions[selected] == label)))
    return float(np.mean(recalls))


def _cross_fitted_nearest_centroid(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    target: str,
) -> dict[str, Any]:
    """Evaluate a fixed, parameter-free classifier on the existing five OOF folds."""

    features = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    labels_text = [str(row[target]) for row in rows]
    classes = sorted(set(labels_text))
    class_to_index = {value: index for index, value in enumerate(classes)}
    labels = np.asarray(
        [class_to_index[value] for value in labels_text], dtype=np.int64
    )
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    predictions = np.full(len(rows), -1, dtype=np.int64)
    for fold in sorted(set(folds.tolist())):
        train = folds != fold
        test = folds == fold
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        train_features = (features[train] - mean) / scale
        test_features = (features[test] - mean) / scale
        centroids = []
        for class_index in range(len(classes)):
            class_rows = train_features[labels[train] == class_index]
            if len(class_rows) == 0:
                raise ValueError(
                    f"Fold {fold} has no training rows for class {classes[class_index]}."
                )
            centroids.append(class_rows.mean(axis=0))
        distances = np.stack(
            [np.sum((test_features - centroid) ** 2, axis=1) for centroid in centroids],
            axis=1,
        )
        predictions[test] = np.argmin(distances, axis=1)

    confusion = {
        actual: {
            predicted: int(
                np.sum(
                    (labels == class_to_index[actual])
                    & (predictions == class_to_index[predicted])
                )
            )
            for predicted in classes
        }
        for actual in classes
    }
    majority_accuracy = max(Counter(labels_text).values()) / len(labels_text)
    return {
        "protocol": "fixed five-fold cross-fitted standardized nearest-centroid",
        "features": list(feature_names),
        "classes": classes,
        "samples": len(rows),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "majority_class_accuracy": float(majority_accuracy),
        "confusion": confusion,
    }


def _resolve_summary(row: dict[str, str], cache_root: Path) -> Path:
    indexed_id = str(row["sample_id"])
    fallback = cache_root / indexed_id / "ensemble_summary.npz"
    portable_text = str(row.get("cache_path") or "").strip()
    if portable_text:
        portable = Path(portable_text).parent / "ensemble_summary.npz"
        if portable.exists():
            return portable
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing ensemble summary for {indexed_id}: {fallback}")


def _source_entries(
    manifests: list[str | Path],
    cache_roots: list[str | Path],
    source_names: list[str],
) -> dict[str, dict[str, Any]]:
    if not (len(manifests) == len(cache_roots) == len(source_names)):
        raise ValueError(
            "Each source manifest requires one cache root and one source name."
        )
    entries: dict[str, dict[str, Any]] = {}
    for manifest, cache_root, source_name in zip(manifests, cache_roots, source_names):
        root = Path(cache_root)
        for row in _read_csv(manifest):
            sample_id = str(row.get("original_id") or row["sample_id"])
            if sample_id in entries:
                if entries[sample_id]["generator_source"] != source_name:
                    raise ValueError(f"Conflicting generator source for {sample_id}.")
                continue
            entries[sample_id] = {
                "generator_source": source_name,
                **_summary_features(_resolve_summary(row, root)),
            }
    return entries


def _cramers_v(rows: list[dict[str, Any]], first: str, second: str) -> float:
    first_values = sorted({str(row[first]) for row in rows})
    second_values = sorted({str(row[second]) for row in rows})
    table = np.zeros((len(first_values), len(second_values)), dtype=np.float64)
    for row in rows:
        table[
            first_values.index(str(row[first])), second_values.index(str(row[second]))
        ] += 1
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    chi_square = float(np.sum((table - expected) ** 2 / np.maximum(expected, 1e-12)))
    denominator = total * min(len(first_values) - 1, len(second_values) - 1)
    return float(np.sqrt(chi_square / denominator)) if denominator > 0 else 0.0


def diagnose_oof_sources(
    one_d_path: str | Path,
    three_d_path: str | Path,
    joint_path: str | Path,
    fold_manifest_path: str | Path,
    conformer_manifests: list[str | Path],
    cache_roots: list[str | Path],
    source_names: list[str],
    thresholds: tuple[float, float, float] = (0.45, 0.677819, 0.60),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows = _read_csv(fold_manifest_path)
    one_d = {int(row["source_index"]): row for row in _read_csv(one_d_path)}
    three_d = {int(row["source_index"]): row for row in _read_csv(three_d_path)}
    joint = {int(row["source_index"]): row for row in _read_csv(joint_path)}
    expected = set(range(len(fold_rows)))
    if set(one_d) != expected or set(three_d) != expected or set(joint) != expected:
        raise ValueError(
            "All OOF prediction files must cover every canonical source_index exactly once."
        )
    sources = _source_entries(conformer_manifests, cache_roots, source_names)

    rows: list[dict[str, Any]] = []
    for source_index, fold_row in enumerate(fold_rows):
        sequence = str(fold_row["sequence"])
        label = int(fold_row["label"])
        candidates = (one_d[source_index], three_d[source_index], joint[source_index])
        if any(
            str(row["sequence"]) != sequence or int(row["label"]) != label
            for row in candidates
        ):
            raise ValueError(
                f"OOF sequence/label mismatch at source_index={source_index}."
            )
        sample_id = str(fold_row["sample_id"])
        source_features = sources.get(sample_id, {})
        row: dict[str, Any] = {
            "source_index": source_index,
            "sample_id": sample_id,
            "fold": int(fold_row["fold"]),
            "group_id": str(fold_row["group_id"]),
            "label": label,
            "sequence": sequence,
            "length": len(sequence),
            "length_stratum": _length_stratum(len(sequence)),
            "generator_source": source_features.get(
                "generator_source", "not_generated"
            ),
            "one_d_probability": float(one_d[source_index]["toxicity_probability"]),
            "three_d_probability": float(three_d[source_index]["toxicity_probability"]),
            "joint_probability": float(joint[source_index]["toxicity_probability"]),
        }
        for name in STRUCTURE_FEATURES:
            row[name] = float(three_d[source_index][name])
        for name in CONFORMER_FEATURES:
            row[name] = source_features.get(name, "")
        for model, threshold in zip(("one_d", "three_d", "joint"), thresholds):
            prediction = int(float(row[f"{model}_probability"]) >= threshold)
            row[f"{model}_prediction"] = prediction
            row[f"{model}_error_type"] = (
                "TP"
                if label == 1 and prediction == 1
                else "FN"
                if label == 1
                else "FP"
                if prediction == 1
                else "TN"
            )
        rows.append(row)

    models = {
        "one_d": {"path": str(one_d_path), "threshold": thresholds[0]},
        "three_d": {"path": str(three_d_path), "threshold": thresholds[1]},
        "joint": {"path": str(joint_path), "threshold": thresholds[2]},
    }
    by_length = {}
    for stratum in LENGTH_STRATA:
        selected = [row for row in rows if row["length_stratum"] == stratum]
        by_length[stratum] = {
            "samples": len(selected),
            "positive": int(sum(row["label"] for row in selected)),
            "models": {
                model: _model_summary(
                    selected, f"{model}_probability", spec["threshold"]
                )
                for model, spec in models.items()
            },
        }

    by_source = {}
    for source in source_names:
        selected = [row for row in rows if row["generator_source"] == source]
        if not selected:
            continue
        by_source[source] = {
            "samples": len(selected),
            "positive": int(sum(row["label"] for row in selected)),
            "positive_fraction": float(np.mean([row["label"] for row in selected])),
            "mean_structure_features": {
                name: float(np.mean([row[name] for row in selected]))
                for name in STRUCTURE_FEATURES
            },
            "mean_conformer_features": {
                name: float(np.mean([float(row[name]) for row in selected]))
                for name in CONFORMER_FEATURES
            },
            "models": {
                model: _model_summary(
                    selected, f"{model}_probability", spec["threshold"]
                )
                for model, spec in models.items()
            },
        }

    covered = [row for row in rows if row["generator_source"] != "not_generated"]
    report = {
        "protocol": {
            "scope": "training-only fixed group-aware OOF diagnostic",
            "external_test_labels_read": False,
            "length_strata": list(LENGTH_STRATA),
            "models": models,
            "selection_warning": "Diagnostic only; do not use strata to retune weights or thresholds.",
        },
        "cohort": {
            "samples": len(rows),
            "positive": int(sum(row["label"] for row in rows)),
            "negative": int(len(rows) - sum(row["label"] for row in rows)),
            "conformer_covered": len(covered),
        },
        "by_length": by_length,
        "by_generator_source": by_source,
        "confounding": {
            "covered_generator_vs_length_cramers_v": _cramers_v(
                covered, "generator_source", "length_stratum"
            ),
            "covered_contingency": {
                source: {
                    stratum: int(
                        sum(
                            row["generator_source"] == source
                            and row["length_stratum"] == stratum
                            for row in covered
                        )
                    )
                    for stratum in LENGTH_STRATA
                }
                for source in source_names
            },
        },
        "feature_predictability": {
            "length_from_single_structure_quality": _cross_fitted_nearest_centroid(
                rows,
                STRUCTURE_FEATURES,
                "length_stratum",
            ),
            "generator_cohort_from_structure_and_conformer_features": (
                _cross_fitted_nearest_centroid(
                    covered,
                    STRUCTURE_FEATURES + CONFORMER_FEATURES,
                    "generator_source",
                )
                if covered
                else None
            ),
        },
    }
    return report, rows


def _write_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d", required=True)
    parser.add_argument("--three-d", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--conformer-manifest", nargs="+", required=True)
    parser.add_argument("--cache-root", nargs="+", required=True)
    parser.add_argument("--source-name", nargs="+", required=True)
    parser.add_argument("--one-d-threshold", type=float, default=0.45)
    parser.add_argument("--three-d-threshold", type=float, default=0.677819)
    parser.add_argument("--joint-threshold", type=float, default=0.60)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    report, rows = diagnose_oof_sources(
        args.one_d,
        args.three_d,
        args.joint,
        args.fold_manifest,
        args.conformer_manifest,
        args.cache_root,
        args.source_name,
        (args.one_d_threshold, args.three_d_threshold, args.joint_threshold),
    )
    save_json(report, args.output_json)
    _write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
