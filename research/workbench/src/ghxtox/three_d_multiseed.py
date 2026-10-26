"""Prepare, validate, and aggregate the preregistered 3D-v2 multi-seed run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.bootstrap_ci import _metrics


FROZEN_SEEDS = (42, 123, 2025)
FIXED_THRESHOLD = 0.677819
REFERENCE_TOLERANCES = {
    "minimum_probability_pearson": 0.995,
    "maximum_absolute_mcc_difference": 0.003,
    "maximum_absolute_auprc_difference": 0.001,
    "maximum_absolute_brier_difference": 0.001,
}
PROMOTION_TOLERANCES = {
    "maximum_brier_increase": 0.001,
    "maximum_ece_10_increase": 0.002,
    "maximum_length_stratum_mcc_decrease": 0.005,
    "maximum_length_stratum_auprc_decrease": 0.002,
}
ALIGNMENT_FIELDS = ("source_index", "fold", "sample_id", "sequence", "label")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    return {
        "path": artifact_path.as_posix(),
        "bytes": artifact_path.stat().st_size,
        "sha256": _sha256(artifact_path),
    }


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction table is empty: {path}")
    required = set(ALIGNMENT_FIELDS) | {"toxicity_probability"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            f"Prediction table {path} is missing columns: {sorted(missing)}"
        )
    rows.sort(key=lambda row: int(row["source_index"]))
    indices = [int(row["source_index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Prediction table {path} has duplicate source_index values.")
    return rows


def _assert_aligned(
    reference: list[dict[str, str]], candidate: list[dict[str, str]]
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(
            f"Prediction row-count mismatch: {len(reference)} != {len(candidate)}"
        )
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        for field in ALIGNMENT_FIELDS:
            if reference_row[field] != candidate_row[field]:
                raise ValueError(
                    f"Prediction alignment mismatch at source_index={reference_row['source_index']} "
                    f"for {field}: {reference_row[field]!r} != {candidate_row[field]!r}"
                )


def _probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    # Use the same float64 implementation as the retained 3D handoff. The
    # training helper intentionally casts to float32 and can merge close OOF
    # probabilities into ties, which changes average precision.
    return _metrics(
        np.asarray(labels, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        FIXED_THRESHOLD,
    )


def _metrics_for_rows(
    rows: list[dict[str, str]], probabilities: np.ndarray | None = None
) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    if probabilities is None:
        probabilities = np.asarray(
            [float(row["toxicity_probability"]) for row in rows], dtype=np.float64
        )
    return _probability_metrics(labels, probabilities)


def _length_stratum(sequence: str) -> str:
    length = len(sequence)
    if 11 <= length <= 15:
        return "11-15"
    if 16 <= length <= 30:
        return "16-30"
    if 31 <= length <= 50:
        return "31-50"
    return "outside-11-50"


def _group_metrics(
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    field: str,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        value = (
            _length_stratum(row["sequence"])
            if field == "length_stratum"
            else row[field]
        )
        groups.setdefault(str(value), []).append(index)
    output: dict[str, dict[str, float]] = {}
    for value, indices in sorted(groups.items()):
        subset_rows = [rows[index] for index in indices]
        subset_probabilities = probabilities[np.asarray(indices, dtype=np.int64)]
        output[value] = {
            "samples": len(indices),
            **_metrics_for_rows(subset_rows, subset_probabilities),
        }
    return output


def _attach_generator_source(
    reference_rows: list[dict[str, str]], quality_csv: str | Path
) -> list[dict[str, str]]:
    with Path(quality_csv).open(newline="", encoding="utf-8") as handle:
        quality_rows = list(csv.DictReader(handle))
    if not quality_rows or not {"source_index", "generator_source"}.issubset(
        quality_rows[0]
    ):
        raise ValueError(
            "Quality table must contain source_index and generator_source columns."
        )
    quality_by_index = {int(row["source_index"]): row for row in quality_rows}
    expected_indices = {int(row["source_index"]) for row in reference_rows}
    if set(quality_by_index) != expected_indices:
        raise ValueError(
            "Quality table must cover the exact prediction source_index set."
        )
    attached = []
    for row in reference_rows:
        quality = quality_by_index[int(row["source_index"])]
        if "fold" in quality and int(quality["fold"]) != int(row["fold"]):
            raise ValueError(
                f"Quality fold mismatch at source_index={row['source_index']}."
            )
        attached.append({**row, "generator_source": quality["generator_source"]})
    return attached


def _decision_comparison(
    labels: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
) -> dict[str, float | int]:
    reference_predictions = reference_probabilities >= FIXED_THRESHOLD
    candidate_predictions = candidate_probabilities >= FIXED_THRESHOLD
    disagreements = reference_predictions != candidate_predictions
    corrected = disagreements & (candidate_predictions == labels)
    regressed = disagreements & (reference_predictions == labels)
    return {
        "decision_disagreements": int(disagreements.sum()),
        "decision_disagreement_rate": float(disagreements.mean()),
        "corrected_reference_errors": int(corrected.sum()),
        "regressed_reference_correct": int(regressed.sum()),
        "net_corrections": int(corrected.sum() - regressed.sum()),
        "mean_absolute_probability_difference": float(
            np.abs(reference_probabilities - candidate_probabilities).mean()
        ),
    }


def _seed_artifact_summary(oof_path: str | Path, seed: int) -> dict[str, Any]:
    oof_path = Path(oof_path)
    seed_root = next(
        (parent for parent in oof_path.parents if parent.name == f"seed_{seed}"),
        oof_path.parent,
    )
    files = sorted(path for path in seed_root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Seed artifact root is empty: {seed_root}")
    modified = [path.stat().st_mtime for path in files]
    return {
        "root": seed_root.as_posix(),
        "files": len(files),
        "bytes": int(sum(path.stat().st_size for path in files)),
        "observed_artifact_time_span_seconds": float(max(modified) - min(modified)),
        "first_artifact_utc": datetime.fromtimestamp(
            min(modified), tz=timezone.utc
        ).isoformat(),
        "last_artifact_utc": datetime.fromtimestamp(
            max(modified), tz=timezone.utc
        ).isoformat(),
    }


def prepare_protocol(
    output_root: str | Path,
    base_config: str | Path,
    residual_config: str | Path,
    base_processed: str | Path,
    residual_train_processed: str | Path,
    residual_oof_processed: str | Path,
    base_fold_manifest: str | Path,
    residual_fold_manifest: str | Path,
    reference_oof: str | Path,
    base_control_pattern: str,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
) -> dict[str, Any]:
    if tuple(seeds) != FROZEN_SEEDS:
        raise ValueError(
            f"Phase 26 seeds are frozen to {FROZEN_SEEDS}, received {tuple(seeds)}."
        )
    output_root = Path(output_root)
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    base_payload = _load_json(base_config)
    residual_payload = _load_json(residual_config)
    generated_configs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_base = json.loads(json.dumps(base_payload))
        seed_base["seed"] = seed
        seed_base["experiment"] = {
            "family": "three_d_v2_multiseed",
            "stage": "chemical_site_3d_v2",
            "seed": seed,
            "protocol": "phase26_preregistered_v1",
        }
        base_output = config_root / f"base_seed{seed}.json"
        _write_json(seed_base, base_output)

        seed_residual = json.loads(json.dumps(residual_payload))
        seed_residual["seed"] = seed
        seed_residual["train"]["initial_checkpoint"] = (
            output_root / f"seed_{seed}" / "base" / "fold{fold}" / "best_model.pt"
        ).as_posix()
        seed_residual["experiment"] = {
            "family": "three_d_v2_multiseed",
            "stage": "capability_aware_conformer_residual",
            "seed": seed,
            "protocol": "phase26_preregistered_v1",
        }
        residual_output = config_root / f"residual_seed{seed}.json"
        _write_json(seed_residual, residual_output)
        generated_configs.extend([_artifact(base_output), _artifact(residual_output)])

    input_paths = {
        "base_config": base_config,
        "residual_config": residual_config,
        "base_processed": base_processed,
        "residual_train_processed": residual_train_processed,
        "residual_oof_processed": residual_oof_processed,
        "base_fold_manifest": base_fold_manifest,
        "residual_fold_manifest": residual_fold_manifest,
        "reference_oof": reference_oof,
    }
    input_artifacts = {name: _artifact(path) for name, path in input_paths.items()}
    control_checkpoints = [
        _artifact(base_control_pattern.format(fold=fold)) for fold in range(5)
    ]

    reference_rows = _read_rows(reference_oof)
    fold_counts = Counter(int(row["fold"]) for row in reference_rows)
    label_counts = Counter(int(row["label"]) for row in reference_rows)
    protocol = {
        "status": "prepared",
        "protocol": "phase26_preregistered_v1",
        "seeds": list(seeds),
        "fixed_threshold": FIXED_THRESHOLD,
        "ensemble_rule": "unweighted arithmetic mean of the three OOF toxicity probabilities",
        "external_test_labels_read": False,
        "reference": {
            "rows": len(reference_rows),
            "fold_counts": {
                str(key): value for key, value in sorted(fold_counts.items())
            },
            "label_counts": {
                str(key): value for key, value in sorted(label_counts.items())
            },
            "metrics_at_fixed_threshold": _metrics_for_rows(reference_rows),
        },
        "reference_reproduction_tolerances": REFERENCE_TOLERANCES,
        "promotion_tolerances": PROMOTION_TOLERANCES,
        "promotion_requirements": {
            "ensemble_mcc_strictly_higher_than_reference": True,
            "ensemble_auprc_at_least_reference": True,
            "fold_mcc_non_decrease_count_at_least": 3,
            "all_length_strata_within_frozen_tolerances": True,
        },
        "input_artifacts": input_artifacts,
        "base_control_checkpoints": control_checkpoints,
        "generated_configs": generated_configs,
        "output_layout": {
            "base_fold": (
                output_root / "seed_{seed}" / "base" / "fold{fold}"
            ).as_posix(),
            "base_oof": (
                output_root / "seed_{seed}" / "base" / "oof_predictions.csv"
            ).as_posix(),
            "residual_fold": (
                output_root / "seed_{seed}" / "residual" / "fold{fold}"
            ).as_posix(),
            "residual_oof": (
                output_root / "seed_{seed}" / "residual" / "oof_predictions.csv"
            ).as_posix(),
        },
    }
    _write_json(protocol, output_root / "protocol.json")
    return protocol


def compare_reference(
    reference_oof: str | Path,
    candidate_oof: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    reference_rows = _read_rows(reference_oof)
    candidate_rows = _read_rows(candidate_oof)
    _assert_aligned(reference_rows, candidate_rows)
    reference_probabilities = np.asarray(
        [float(row["toxicity_probability"]) for row in reference_rows], dtype=np.float64
    )
    candidate_probabilities = np.asarray(
        [float(row["toxicity_probability"]) for row in candidate_rows], dtype=np.float64
    )
    reference_metrics = _metrics_for_rows(reference_rows, reference_probabilities)
    candidate_metrics = _metrics_for_rows(candidate_rows, candidate_probabilities)
    correlation = float(
        np.corrcoef(reference_probabilities, candidate_probabilities)[0, 1]
    )
    differences = np.abs(reference_probabilities - candidate_probabilities)
    gates = {
        "probability_pearson_at_least_0_995": correlation
        >= REFERENCE_TOLERANCES["minimum_probability_pearson"],
        "absolute_mcc_difference_at_most_0_003": abs(
            candidate_metrics["mcc"] - reference_metrics["mcc"]
        )
        <= REFERENCE_TOLERANCES["maximum_absolute_mcc_difference"],
        "absolute_auprc_difference_at_most_0_001": abs(
            candidate_metrics["auprc"] - reference_metrics["auprc"]
        )
        <= REFERENCE_TOLERANCES["maximum_absolute_auprc_difference"],
        "absolute_brier_difference_at_most_0_001": abs(
            candidate_metrics["brier"] - reference_metrics["brier"]
        )
        <= REFERENCE_TOLERANCES["maximum_absolute_brier_difference"],
    }
    result = {
        "status": "complete",
        "decision": "reference_reproduction_passed"
        if all(gates.values())
        else "stop_before_new_seeds",
        "new_seeds_authorized": all(gates.values()),
        "external_test_labels_read": False,
        "rows": len(reference_rows),
        "alignment": "exact",
        "probability_comparison": {
            "pearson": correlation,
            "mean_absolute_difference": float(differences.mean()),
            "maximum_absolute_difference": float(differences.max()),
        },
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "absolute_metric_differences": {
            key: abs(candidate_metrics[key] - reference_metrics[key])
            for key in ("mcc", "auprc", "brier")
        },
        "gate": {**gates, "all_conditions_passed": all(gates.values())},
        "provenance": {
            "reference_oof": _artifact(reference_oof),
            "candidate_oof": _artifact(candidate_oof),
        },
    }
    _write_json(result, output_json)
    return result


def _parse_seed_oofs(values: list[str]) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SEED=PATH, received {value!r}.")
        seed = int(seed_text)
        if seed in parsed:
            raise ValueError(f"Duplicate seed {seed}.")
        parsed[seed] = Path(path_text)
    if tuple(sorted(parsed)) != tuple(sorted(FROZEN_SEEDS)):
        raise ValueError(
            f"Expected exactly seeds {FROZEN_SEEDS}, received {tuple(sorted(parsed))}."
        )
    return parsed


def aggregate_multiseed(
    reference_oof: str | Path,
    seed_oofs: dict[int, str | Path],
    output_csv: str | Path,
    output_summary: str | Path,
    output_decision: str | Path,
    quality_csv: str | Path | None = None,
) -> dict[str, Any]:
    if tuple(sorted(seed_oofs)) != tuple(sorted(FROZEN_SEEDS)):
        raise ValueError(
            f"Expected exactly seeds {FROZEN_SEEDS}, received {tuple(sorted(seed_oofs))}."
        )
    reference_rows = _read_rows(reference_oof)
    reference_probabilities = np.asarray(
        [float(row["toxicity_probability"]) for row in reference_rows], dtype=np.float64
    )
    seed_rows: dict[int, list[dict[str, str]]] = {}
    seed_probabilities: dict[int, np.ndarray] = {}
    for seed in FROZEN_SEEDS:
        rows = _read_rows(seed_oofs[seed])
        _assert_aligned(reference_rows, rows)
        seed_rows[seed] = rows
        seed_probabilities[seed] = np.asarray(
            [float(row["toxicity_probability"]) for row in rows], dtype=np.float64
        )
    stacked_probabilities = np.stack(
        [seed_probabilities[seed] for seed in FROZEN_SEEDS]
    )
    ensemble_probabilities = stacked_probabilities.mean(axis=0)
    ensemble_standard_deviation = stacked_probabilities.std(axis=0, ddof=0)
    labels = np.asarray([int(row["label"]) for row in reference_rows], dtype=np.int64)
    cohort_rows = (
        _attach_generator_source(reference_rows, quality_csv)
        if quality_csv is not None
        else None
    )

    reference_metrics = _metrics_for_rows(reference_rows, reference_probabilities)
    individual_metrics = {
        str(seed): _metrics_for_rows(seed_rows[seed], seed_probabilities[seed])
        for seed in FROZEN_SEEDS
    }
    ensemble_metrics = _metrics_for_rows(reference_rows, ensemble_probabilities)
    metric_names = ("mcc", "auprc", "auroc", "brier", "ece_10")
    seed_metric_distribution = {
        metric: {
            "mean": float(
                np.mean(
                    [individual_metrics[str(seed)][metric] for seed in FROZEN_SEEDS]
                )
            ),
            "standard_deviation": float(
                np.std(
                    [individual_metrics[str(seed)][metric] for seed in FROZEN_SEEDS],
                    ddof=0,
                )
            ),
            "minimum": float(
                min(individual_metrics[str(seed)][metric] for seed in FROZEN_SEEDS)
            ),
            "maximum": float(
                max(individual_metrics[str(seed)][metric] for seed in FROZEN_SEEDS)
            ),
        }
        for metric in metric_names
    }

    reference_by_fold = _group_metrics(reference_rows, reference_probabilities, "fold")
    ensemble_by_fold = _group_metrics(reference_rows, ensemble_probabilities, "fold")
    reference_by_length = _group_metrics(
        reference_rows, reference_probabilities, "length_stratum"
    )
    ensemble_by_length = _group_metrics(
        reference_rows, ensemble_probabilities, "length_stratum"
    )
    individual_by_fold = {
        str(seed): _group_metrics(reference_rows, seed_probabilities[seed], "fold")
        for seed in FROZEN_SEEDS
    }
    individual_by_length = {
        str(seed): _group_metrics(
            reference_rows, seed_probabilities[seed], "length_stratum"
        )
        for seed in FROZEN_SEEDS
    }
    reference_by_generator = (
        _group_metrics(cohort_rows, reference_probabilities, "generator_source")
        if cohort_rows is not None
        else None
    )
    ensemble_by_generator = (
        _group_metrics(cohort_rows, ensemble_probabilities, "generator_source")
        if cohort_rows is not None
        else None
    )
    individual_by_generator = (
        {
            str(seed): _group_metrics(
                cohort_rows, seed_probabilities[seed], "generator_source"
            )
            for seed in FROZEN_SEEDS
        }
        if cohort_rows is not None
        else None
    )
    fold_non_decrease = sum(
        ensemble_by_fold[fold]["mcc"] >= reference_by_fold[fold]["mcc"]
        for fold in reference_by_fold
    )
    length_gates: dict[str, dict[str, bool]] = {}
    for length, reference_values in reference_by_length.items():
        ensemble_values = ensemble_by_length[length]
        length_gates[length] = {
            "mcc_within_0_005": ensemble_values["mcc"]
            >= reference_values["mcc"]
            - PROMOTION_TOLERANCES["maximum_length_stratum_mcc_decrease"],
            "auprc_within_0_002": ensemble_values["auprc"]
            >= reference_values["auprc"]
            - PROMOTION_TOLERANCES["maximum_length_stratum_auprc_decrease"],
        }
        length_gates[length]["all_conditions_passed"] = all(
            length_gates[length].values()
        )

    gates = {
        "mcc_strictly_higher": ensemble_metrics["mcc"] > reference_metrics["mcc"],
        "auprc_at_least_reference": ensemble_metrics["auprc"]
        >= reference_metrics["auprc"],
        "at_least_three_fold_mcc_non_decreases": fold_non_decrease >= 3,
        "brier_within_0_001": ensemble_metrics["brier"]
        <= reference_metrics["brier"] + PROMOTION_TOLERANCES["maximum_brier_increase"],
        "ece_10_within_0_002": ensemble_metrics["ece_10"]
        <= reference_metrics["ece_10"]
        + PROMOTION_TOLERANCES["maximum_ece_10_increase"],
        "all_length_strata_within_tolerances": all(
            values["all_conditions_passed"] for values in length_gates.values()
        ),
    }
    all_conditions_passed = all(gates.values())

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        *ALIGNMENT_FIELDS,
        *(f"seed_{seed}_probability" for seed in FROZEN_SEEDS),
        "seed_probability_std",
        "toxicity_probability",
        "prediction",
        "decision_threshold",
        "length_stratum",
        "generator_source",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(reference_rows):
            probability = float(ensemble_probabilities[index])
            writer.writerow(
                {
                    **{field: row[field] for field in ALIGNMENT_FIELDS},
                    **{
                        f"seed_{seed}_probability": f"{float(seed_probabilities[seed][index]):.9g}"
                        for seed in FROZEN_SEEDS
                    },
                    "seed_probability_std": f"{float(ensemble_standard_deviation[index]):.9g}",
                    "toxicity_probability": f"{probability:.9g}",
                    "prediction": int(probability >= FIXED_THRESHOLD),
                    "decision_threshold": FIXED_THRESHOLD,
                    "length_stratum": _length_stratum(row["sequence"]),
                    "generator_source": cohort_rows[index]["generator_source"]
                    if cohort_rows is not None
                    else "",
                }
            )

    seed_vs_reference = {
        str(seed): _decision_comparison(
            labels, reference_probabilities, seed_probabilities[seed]
        )
        for seed in FROZEN_SEEDS
    }
    pairwise_seed_disagreements = {}
    for first_seed, second_seed in combinations(FROZEN_SEEDS, 2):
        first = seed_probabilities[first_seed]
        second = seed_probabilities[second_seed]
        key = f"{first_seed}_vs_{second_seed}"
        pairwise_seed_disagreements[key] = {
            "probability_pearson": float(np.corrcoef(first, second)[0, 1]),
            "mean_absolute_probability_difference": float(
                np.abs(first - second).mean()
            ),
            "decision_disagreements": int(
                ((first >= FIXED_THRESHOLD) != (second >= FIXED_THRESHOLD)).sum()
            ),
        }
    seed_decisions = stacked_probabilities >= FIXED_THRESHOLD
    all_seed_agreement = np.all(seed_decisions == seed_decisions[0], axis=0)

    summary = {
        "status": "complete",
        "protocol": "phase26_preregistered_v1",
        "seeds": list(FROZEN_SEEDS),
        "ensemble_rule": "unweighted arithmetic probability mean",
        "fixed_threshold": FIXED_THRESHOLD,
        "external_test_labels_read": False,
        "rows": len(reference_rows),
        "reference_metrics": reference_metrics,
        "individual_seed_metrics": individual_metrics,
        "seed_metric_distribution": seed_metric_distribution,
        "ensemble_metrics": ensemble_metrics,
        "reference_by_fold": reference_by_fold,
        "ensemble_by_fold": ensemble_by_fold,
        "individual_seed_by_fold": individual_by_fold,
        "fold_mcc_non_decrease_count": fold_non_decrease,
        "reference_by_length": reference_by_length,
        "ensemble_by_length": ensemble_by_length,
        "individual_seed_by_length": individual_by_length,
        "reference_by_generator": reference_by_generator,
        "ensemble_by_generator": ensemble_by_generator,
        "individual_seed_by_generator": individual_by_generator,
        "decision_disagreements": {
            "ensemble_vs_reference": _decision_comparison(
                labels, reference_probabilities, ensemble_probabilities
            ),
            "individual_seed_vs_reference": seed_vs_reference,
            "pairwise_seed": pairwise_seed_disagreements,
            "all_seed_decision_agreement_count": int(all_seed_agreement.sum()),
            "all_seed_decision_agreement_rate": float(all_seed_agreement.mean()),
        },
        "mean_seed_probability_standard_deviation": float(
            ensemble_standard_deviation.mean()
        ),
        "maximum_seed_probability_standard_deviation": float(
            ensemble_standard_deviation.max()
        ),
        "provenance": {
            "reference_oof": _artifact(reference_oof),
            "seed_oofs": {
                str(seed): _artifact(seed_oofs[seed]) for seed in FROZEN_SEEDS
            },
            "quality_csv": _artifact(quality_csv) if quality_csv is not None else None,
            "ensemble_oof": _artifact(output_csv),
        },
        "runtime_and_storage": {
            str(seed): _seed_artifact_summary(seed_oofs[seed], seed)
            for seed in FROZEN_SEEDS
        },
    }
    decision = {
        "status": "complete",
        "decision": "promote_three_seed_3d_ensemble"
        if all_conditions_passed
        else "retain_seed42_3d_handoff",
        "promotion_authorized": all_conditions_passed,
        "handoff_regeneration_authorized": all_conditions_passed,
        "best_single_seed_selection_authorized": False,
        "external_test_labels_read": False,
        "fold_mcc_non_decrease_count": fold_non_decrease,
        "length_gates": length_gates,
        "gate": {**gates, "all_conditions_passed": all_conditions_passed},
        "reference_metrics": reference_metrics,
        "ensemble_metrics": ensemble_metrics,
    }
    _write_json(summary, output_summary)
    _write_json(decision, output_decision)
    return decision


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Freeze configs, hashes, seeds, and gates."
    )
    prepare.add_argument("--output-root", default="runs/3d_v2_multiseed")
    prepare.add_argument(
        "--base-config",
        default="configs/chemical_site_interaction_pareto_same_residue.json",
    )
    prepare.add_argument(
        "--residual-config",
        default="configs/conformer_ensemble_residual_calibrated.json",
    )
    prepare.add_argument(
        "--base-processed", default="data/processed/train_chemical_sites_final_esm2.pt"
    )
    prepare.add_argument(
        "--residual-train-processed",
        default="data/processed/train_chemical_sites_final_esm2_pepflow_pilot250.pt",
    )
    prepare.add_argument(
        "--residual-oof-processed",
        default="data/processed/train_chemical_sites_final_esm2_pepflow.pt",
    )
    prepare.add_argument(
        "--base-fold-manifest", default="data/folds/train_cdhit080_fallback_folds.csv"
    )
    prepare.add_argument(
        "--residual-fold-manifest", default="data/folds/pepflow_pilot250_folds.csv"
    )
    prepare.add_argument(
        "--reference-oof",
        default="runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv",
    )
    prepare.add_argument(
        "--base-control-pattern",
        default="runs/chemical_site_final_control_fold{fold}/best_model.pt",
    )

    compare = subparsers.add_parser(
        "compare-reference", help="Apply the frozen seed-42 reproduction gate."
    )
    compare.add_argument("--reference-oof", required=True)
    compare.add_argument("--candidate-oof", required=True)
    compare.add_argument("--output-json", required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="Build and gate the fixed three-seed probability mean."
    )
    aggregate.add_argument("--reference-oof", required=True)
    aggregate.add_argument("--seed-oof", nargs="+", required=True, metavar="SEED=PATH")
    aggregate.add_argument("--output-csv", required=True)
    aggregate.add_argument("--output-summary", required=True)
    aggregate.add_argument("--output-decision", required=True)
    aggregate.add_argument(
        "--quality-csv", default="runs/three_d_team_handoff/oof_3d_quality.csv"
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "prepare":
        result = prepare_protocol(
            output_root=args.output_root,
            base_config=args.base_config,
            residual_config=args.residual_config,
            base_processed=args.base_processed,
            residual_train_processed=args.residual_train_processed,
            residual_oof_processed=args.residual_oof_processed,
            base_fold_manifest=args.base_fold_manifest,
            residual_fold_manifest=args.residual_fold_manifest,
            reference_oof=args.reference_oof,
            base_control_pattern=args.base_control_pattern,
        )
    elif args.command == "compare-reference":
        result = compare_reference(
            args.reference_oof, args.candidate_oof, args.output_json
        )
    else:
        result = aggregate_multiseed(
            reference_oof=args.reference_oof,
            seed_oofs=_parse_seed_oofs(args.seed_oof),
            output_csv=args.output_csv,
            output_summary=args.output_summary,
            output_decision=args.output_decision,
            quality_csv=args.quality_csv,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
