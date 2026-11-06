"""Paired train-only analysis of BioEmu and legacy conformer descriptors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.oof_source_diagnostics import (
    CONFORMER_FEATURES,
    _cross_fitted_nearest_centroid,
    _cross_fitted_nearest_centroid_predictions,
    _read_csv,
    _source_entries,
)
from ghxtox.utils import save_json


def _feature_mean(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[f"{prefix}{name}"]) for row in rows]))
        for name in CONFORMER_FEATURES
    }


def _paired_feature_shift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "samples": len(rows),
        "mean_delta_bioemu_minus_legacy": {
            name: float(np.mean([float(row[f"delta_{name}"]) for row in rows]))
            for name in CONFORMER_FEATURES
        },
        "mean_absolute_delta": {
            name: float(np.mean([abs(float(row[f"delta_{name}"])) for row in rows]))
            for name in CONFORMER_FEATURES
        },
    }


def _permutation_test(
    rows: list[dict[str, Any]],
    target: str,
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Test fixed-classifier BACC after permuting labels within fold and length."""

    observed = _cross_fitted_nearest_centroid(rows, CONFORMER_FEATURES, target)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (int(row["fold"]), str(row["length_stratum"]))
        groups.setdefault(key, []).append(index)

    null_scores = []
    labels = [str(row[target]) for row in rows]
    for _ in range(permutations):
        permuted_labels = labels.copy()
        for indices in groups.values():
            shuffled = rng.permutation([labels[index] for index in indices])
            for index, label in zip(indices, shuffled, strict=True):
                permuted_labels[index] = str(label)
        permuted_rows = [
            {**row, target: permuted_labels[index]} for index, row in enumerate(rows)
        ]
        null_scores.append(
            _cross_fitted_nearest_centroid(
                permuted_rows,
                CONFORMER_FEATURES,
                target,
            )["balanced_accuracy"]
        )

    null = np.asarray(null_scores, dtype=np.float64)
    observed_score = float(observed["balanced_accuracy"])
    return {
        **observed,
        "permutation_test": {
            "null": "labels permuted within canonical fold and length stratum",
            "permutations": permutations,
            "seed": seed,
            "null_mean_balanced_accuracy": float(null.mean()),
            "null_std_balanced_accuracy": float(null.std()),
            "null_95th_percentile_balanced_accuracy": float(np.quantile(null, 0.95)),
            "one_sided_p_value": float(
                (1 + np.sum(null >= observed_score)) / (permutations + 1)
            ),
        },
    }


def _prediction_summary(
    labels: np.ndarray,
    one_d_predictions: np.ndarray,
    bioemu_predictions: np.ndarray,
) -> dict[str, Any]:
    one_d_correct = one_d_predictions == labels
    bioemu_correct = bioemu_predictions == labels
    recalls = []
    for label in (0, 1):
        selected = labels == label
        recalls.append(float(np.mean(bioemu_predictions[selected] == label)))
    return {
        "samples": int(len(labels)),
        "one_d_accuracy": float(np.mean(one_d_correct)),
        "bioemu_accuracy": float(np.mean(bioemu_correct)),
        "bioemu_balanced_accuracy": float(np.mean(recalls)),
        "decision_disagreement_count": int(
            np.sum(one_d_predictions != bioemu_predictions)
        ),
        "decision_disagreement_rate": float(
            np.mean(one_d_predictions != bioemu_predictions)
        ),
        "bioemu_corrected_one_d": int(np.sum(~one_d_correct & bioemu_correct)),
        "bioemu_newly_incorrect": int(np.sum(one_d_correct & ~bioemu_correct)),
        "net_corrected": int(
            np.sum(~one_d_correct & bioemu_correct)
            - np.sum(one_d_correct & ~bioemu_correct)
        ),
    }


def analyze_bioemu_source_shift(
    frozen_manifest: str | Path,
    bioemu_manifest: str | Path,
    bioemu_cache_root: str | Path,
    legacy_manifests: list[str | Path],
    legacy_cache_roots: list[str | Path],
    legacy_source_names: list[str],
    *,
    permutations: int = 1000,
    permutation_seed: int = 20260731,
    one_d_oof: str | Path | None = None,
    one_d_threshold: float = 0.45,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare one unified generator with legacy sources on identical samples."""

    frozen_rows = _read_csv(frozen_manifest)
    bioemu = _source_entries([bioemu_manifest], [bioemu_cache_root], ["BioEmu-v1.2"])
    legacy = _source_entries(legacy_manifests, legacy_cache_roots, legacy_source_names)
    rows: list[dict[str, Any]] = []
    bioemu_classifier_rows: list[dict[str, Any]] = []
    paired_classifier_rows: list[dict[str, Any]] = []

    for frozen in frozen_rows:
        sample_id = str(frozen["sample_id"])
        if sample_id not in bioemu:
            raise ValueError(f"Missing BioEmu summary for frozen sample {sample_id}.")
        bio_features = bioemu[sample_id]
        row: dict[str, Any] = {
            "source_index": int(frozen["source_index"]),
            "sample_id": sample_id,
            "fold": int(frozen["fold"]),
            "label": int(frozen["label"]),
            "sequence": str(frozen["sequence"]),
            "length": int(frozen["length"]),
            "length_stratum": str(frozen["length_bin"]),
            "legacy_source": legacy.get(sample_id, {}).get(
                "generator_source", "not_generated"
            ),
        }
        classifier_row = {
            "fold": row["fold"],
            "length_stratum": row["length_stratum"],
            "label": str(row["label"]),
        }
        for name in CONFORMER_FEATURES:
            value = float(bio_features[name])
            row[f"bioemu_{name}"] = value
            classifier_row[name] = value
        bioemu_classifier_rows.append(classifier_row)

        legacy_features = legacy.get(sample_id)
        if legacy_features is not None:
            for name in CONFORMER_FEATURES:
                legacy_value = float(legacy_features[name])
                row[f"legacy_{name}"] = legacy_value
                row[f"delta_{name}"] = row[f"bioemu_{name}"] - legacy_value
            for generator_family, features in (
                ("BioEmu", bio_features),
                ("legacy", legacy_features),
            ):
                paired_classifier_rows.append(
                    {
                        "fold": row["fold"],
                        "generator_family": generator_family,
                        **{name: float(features[name]) for name in CONFORMER_FEATURES},
                    }
                )
        else:
            for name in CONFORMER_FEATURES:
                row[f"legacy_{name}"] = ""
                row[f"delta_{name}"] = ""
        rows.append(row)

    by_length = {}
    for length_stratum in ("11-15", "16-30", "31-50"):
        selected = [row for row in rows if row["length_stratum"] == length_stratum]
        paired = [row for row in selected if row["legacy_source"] != "not_generated"]
        by_length[length_stratum] = {
            "samples": len(selected),
            "positive": int(sum(row["label"] for row in selected)),
            "legacy_overlap": len(paired),
            "mean_bioemu_features": _feature_mean(selected, "bioemu_"),
            "paired_shift": _paired_feature_shift(paired) if paired else None,
        }

    by_legacy_source = {}
    for source in legacy_source_names:
        selected = [row for row in rows if row["legacy_source"] == source]
        if selected:
            by_legacy_source[source] = _paired_feature_shift(selected)

    paired_rows = [row for row in rows if row["legacy_source"] != "not_generated"]
    anchor_comparison = None
    if one_d_oof is not None:
        one_d = {int(row["source_index"]): row for row in _read_csv(one_d_oof)}
        labels, bioemu_predictions, classes = (
            _cross_fitted_nearest_centroid_predictions(
                bioemu_classifier_rows,
                CONFORMER_FEATURES,
                "label",
            )
        )
        if classes != ["0", "1"]:
            raise ValueError(f"Expected binary toxicity labels, received {classes}.")
        one_d_predictions = np.asarray(
            [
                int(
                    float(one_d[int(row["source_index"])]["toxicity_probability"])
                    >= one_d_threshold
                )
                for row in rows
            ],
            dtype=np.int64,
        )
        for index, row in enumerate(rows):
            source_index = int(row["source_index"])
            one_d_row = one_d.get(source_index)
            if one_d_row is None:
                raise ValueError(f"Missing 1D OOF prediction for {source_index}.")
            if int(one_d_row["label"]) != int(row["label"]):
                raise ValueError(f"1D OOF label mismatch for {source_index}.")
            row["one_d_probability"] = float(one_d_row["toxicity_probability"])
            row["one_d_prediction"] = int(one_d_predictions[index])
            row["bioemu_prediction"] = int(bioemu_predictions[index])

        anchor_comparison = {
            "protocol": (
                "fixed 1D OOF threshold versus fixed cross-fitted BioEmu "
                "nearest-centroid decisions; no fusion or selection"
            ),
            "one_d_threshold": one_d_threshold,
            "overall": _prediction_summary(
                labels,
                one_d_predictions,
                bioemu_predictions,
            ),
            "by_length": {
                length_stratum: _prediction_summary(
                    labels[selected],
                    one_d_predictions[selected],
                    bioemu_predictions[selected],
                )
                for length_stratum in ("11-15", "16-30", "31-50")
                for selected in [
                    np.asarray(
                        [row["length_stratum"] == length_stratum for row in rows],
                        dtype=bool,
                    )
                ]
            },
        }

    report = {
        "protocol": {
            "scope": "frozen 200-row training-only BioEmu descriptor comparison",
            "external_test_labels_read": False,
            "features": list(CONFORMER_FEATURES),
            "paired_split_rule": "both generators for one sample stay in its canonical OOF fold",
            "selection_warning": "Diagnostic only; no model or threshold selection.",
        },
        "cohort": {
            "bioemu_samples": len(rows),
            "legacy_overlap": len(paired_rows),
            "legacy_missing": len(rows) - len(paired_rows),
        },
        "by_length": by_length,
        "by_legacy_source": by_legacy_source,
        "one_d_anchor_comparison": anchor_comparison,
        "feature_predictability": {
            "length_from_bioemu_features": _cross_fitted_nearest_centroid(
                bioemu_classifier_rows,
                CONFORMER_FEATURES,
                "length_stratum",
            ),
            "toxicity_from_bioemu_features": _permutation_test(
                bioemu_classifier_rows,
                "label",
                permutations=permutations,
                seed=permutation_seed,
            ),
            "toxicity_from_bioemu_features_by_length": {
                length_stratum: _permutation_test(
                    [
                        row
                        for row in bioemu_classifier_rows
                        if row["length_stratum"] == length_stratum
                    ],
                    "label",
                    permutations=permutations,
                    seed=permutation_seed + index + 1,
                )
                for index, length_stratum in enumerate(("11-15", "16-30", "31-50"))
            },
            "bioemu_vs_legacy_on_paired_samples": (
                _cross_fitted_nearest_centroid(
                    paired_classifier_rows,
                    CONFORMER_FEATURES,
                    "generator_family",
                )
                if paired_classifier_rows
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
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--bioemu-manifest", required=True)
    parser.add_argument("--bioemu-cache-root", required=True)
    parser.add_argument("--legacy-manifest", nargs="+", required=True)
    parser.add_argument("--legacy-cache-root", nargs="+", required=True)
    parser.add_argument("--legacy-source-name", nargs="+", required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--permutation-seed", type=int, default=20260731)
    parser.add_argument("--one-d-oof")
    parser.add_argument("--one-d-threshold", type=float, default=0.45)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    report, rows = analyze_bioemu_source_shift(
        args.frozen_manifest,
        args.bioemu_manifest,
        args.bioemu_cache_root,
        args.legacy_manifest,
        args.legacy_cache_root,
        args.legacy_source_name,
        permutations=args.permutations,
        permutation_seed=args.permutation_seed,
        one_d_oof=args.one_d_oof,
        one_d_threshold=args.one_d_threshold,
    )
    save_json(report, args.output_json)
    _write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
