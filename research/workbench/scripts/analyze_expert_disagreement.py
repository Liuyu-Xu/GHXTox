"""Analyze frozen 1D--3D expert disagreement without model inference.

The script reads canonical Test1/Test2 expert probabilities and already frozen
processed metadata. It never loads a checkpoint, changes a probability, or fits
an expert threshold. All outcome definitions and resampling rules come from the
versioned protocol JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


GROUP_ORDER = (
    "both_correct",
    "one_d_only_correct",
    "three_d_only_correct",
    "both_wrong",
)
GROUP_PUBLIC_NAMES = {
    "both_correct": "Both experts correct",
    "one_d_only_correct": "1D expert only correct",
    "three_d_only_correct": "3D expert only correct",
    "both_wrong": "Both experts wrong",
}
CONTINUOUS_ATTRIBUTES = (
    "peptide_length",
    "cysteine_count",
    "mean_plddt",
    "chemical_site_coverage",
)
BINARY_ATTRIBUTES = ("conformer_available",)
ATTRIBUTE_PUBLIC_NAMES = {
    "peptide_length": "Peptide length",
    "cysteine_count": "Cys count",
    "mean_plddt": "Mean pLDDT",
    "chemical_site_coverage": "Chemical-site coverage",
    "conformer_available": "Conformer availability",
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=root / "configs" / "expert_disagreement_analysis_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "expert_disagreement_analysis_20260810_v1",
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields) if fields is not None else list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base: int, *parts: str) -> int:
    payload = "|".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _quantile(values: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, probability))


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return _quantile(finite, 0.025), _quantile(finite, 0.975)


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return float("nan")
    ordered = np.sort(second)
    less = np.searchsorted(ordered, first, side="left").sum()
    greater = (ordered.size - np.searchsorted(ordered, first, side="right")).sum()
    return float((less - greater) / (first.size * second.size))


def _bh_adjust(p_values: list[float | None]) -> list[float | None]:
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    adjusted: list[float | None] = [None] * len(p_values)
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda item: item[1])
    running = 1.0
    count = len(ordered)
    for rank_index in range(count - 1, -1, -1):
        original_index, probability = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, probability * count / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _group_code(one_d_correct: bool, three_d_correct: bool) -> str:
    if one_d_correct and three_d_correct:
        return "both_correct"
    if one_d_correct:
        return "one_d_only_correct"
    if three_d_correct:
        return "three_d_only_correct"
    return "both_wrong"


def _verify_protocol(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("artifact_type") != "ghxtox_1d_3d_expert_disagreement_protocol":
        raise ValueError("Unrecognized disagreement protocol")
    if protocol.get("protocol_status") != "frozen_before_outcome_group_computation":
        raise ValueError("Protocol is not frozen")
    for item in protocol["inputs"]:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Input size mismatch: {path}")
        digest = _sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}: {digest}")
    threshold_summary = json.loads(
        (root / "paper_figures/data/current_interpretability/frozen_results/summary.json").read_text(encoding="utf-8")
    )
    operating = threshold_summary["operating_point"]
    if float(operating["standalone_1d_threshold"]) != float(protocol["expert_operating_points"]["one_d_threshold"]):
        raise ValueError("1D threshold does not match frozen provenance")
    if float(operating["standalone_3d_threshold"]) != float(protocol["expert_operating_points"]["three_d_threshold"]):
        raise ValueError("3D threshold does not match frozen provenance")
    default_config = json.loads((root / "configs/default_multidimensional_fusion.json").read_text(encoding="utf-8"))
    if default_config["test_time_conformer_policy"]["status"] != "disabled":
        raise ValueError("Canonical Test-time conformer policy is not disabled")
    if default_config["test_time_conformer_policy"]["probability_source"] != "base_logits":
        raise ValueError("Canonical 3D probability is not single-structure base_logits")
    return protocol


def _load_samples(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    one_d_threshold = float(protocol["expert_operating_points"]["one_d_threshold"])
    three_d_threshold = float(protocol["expert_operating_points"]["three_d_threshold"])
    all_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}
    for split_key, split_name, expected_count in (
        ("test1", "Test1", 1126),
        ("test2", "Test2", 582),
    ):
        prediction_path = root / f"runs/default_multidimensional_fusion_single_structure/{split_key}_predictions.csv"
        component_path = root / f"runs/single_structure_test_components/{split_key}_structure_predictions.csv"
        tensor_path = root / f"paper_figures/data/current_interpretability/inputs/{split_key}_current_multiconformer.pt"
        prediction_rows = _read_csv(prediction_path)
        component_rows = _read_csv(component_path)
        payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
        records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"Processed tensor lacks a records list: {tensor_path}")
        if not (len(prediction_rows) == len(component_rows) == len(records) == expected_count):
            raise ValueError(f"Record-count mismatch for {split_name}")
        if len({row["sample_id"] for row in prediction_rows}) != expected_count:
            raise ValueError(f"Duplicate sample_id in {prediction_path}")
        mismatch_counts = Counter()
        for source_index, (prediction, component, record) in enumerate(zip(prediction_rows, component_rows, records)):
            sample_id = prediction["sample_id"]
            label = int(prediction["label"])
            sequence = str(record["sequence"]).strip().upper()
            if int(prediction["source_index"]) != source_index or int(component["source_index"]) != source_index:
                mismatch_counts["source_index"] += 1
            if sample_id != component["sample_id"] or sample_id != str(record["sample_id"]):
                mismatch_counts["sample_id"] += 1
            if label != int(component["label"]) or label != int(record["label"]):
                mismatch_counts["label"] += 1
            if sequence != component["sequence"].strip().upper():
                mismatch_counts["sequence"] += 1
            p_1d = float(prediction["p_1d"])
            p_3d = float(prediction["p_3d"])
            if abs(p_3d - float(component["structure_probability"])) > 5e-10:
                mismatch_counts["p_3d"] += 1
            if not (0.0 <= p_1d <= 1.0 and 0.0 <= p_3d <= 1.0):
                mismatch_counts["probability_range"] += 1
            length = len(sequence)
            plddt = record["plddt"][:length].float()
            site_mask = record["chemical_site_mask"][:length].bool()
            if plddt.ndim != 1 or plddt.shape[0] != length or not bool(torch.isfinite(plddt).all()):
                mismatch_counts["plddt"] += 1
            if site_mask.ndim != 2 or site_mask.shape[0] != length:
                mismatch_counts["chemical_site_mask"] += 1
            if mismatch_counts:
                continue
            one_d_prediction = int(p_1d >= one_d_threshold)
            three_d_prediction = int(p_3d >= three_d_threshold)
            one_d_correct = one_d_prediction == label
            three_d_correct = three_d_prediction == label
            coverage = float(site_mask.any(dim=-1).float().mean().item()) if length else float("nan")
            conformer_available = bool(record.get("conformer_available", False))
            all_rows.append(
                {
                    "split_name": split_name,
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "label": label,
                    "p_1d": p_1d,
                    "one_d_threshold": one_d_threshold,
                    "one_d_prediction": one_d_prediction,
                    "one_d_correct": int(one_d_correct),
                    "p_3d": p_3d,
                    "three_d_threshold": three_d_threshold,
                    "three_d_prediction": three_d_prediction,
                    "three_d_correct": int(three_d_correct),
                    "outcome_group": _group_code(one_d_correct, three_d_correct),
                    "peptide_length": length,
                    "cysteine_count": sequence.count("C"),
                    "mean_plddt": float(plddt.mean().item()),
                    "chemical_site_coverage": coverage,
                    "conformer_available": int(conformer_available),
                    "conformer_source": str(record.get("conformer_source", "none") or "none"),
                    "canonical_prediction_uses_conformer": 0,
                }
            )
        if mismatch_counts:
            raise ValueError(f"Integrity mismatch for {split_name}: {dict(mismatch_counts)}")
        integrity[split_name] = {
            "expected_records": expected_count,
            "joined_records": len([row for row in all_rows if row["split_name"] == split_name]),
            "mismatch_counts": dict(mismatch_counts),
            "prediction_sha256": _sha256(prediction_path),
            "component_sha256": _sha256(component_path),
            "metadata_sha256": _sha256(tensor_path),
        }
    return all_rows, integrity


def _rows_for(rows: list[dict[str, Any]], split_name: str, group: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split_name"] == split_name and row["outcome_group"] == group]


def _stratified_group_resample(
    rows: list[dict[str, Any]], attribute: str, rng: np.random.Generator
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for label in (0, 1):
        values = np.asarray(
            [float(row[attribute]) for row in rows if row["label"] == label and math.isfinite(float(row[attribute]))],
            dtype=np.float64,
        )
        if values.size:
            sampled.append(rng.choice(values, size=values.size, replace=True))
    return np.concatenate(sampled) if sampled else np.empty(0, dtype=np.float64)


def _group_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name in ("Test1", "Test2"):
        split_rows = [row for row in rows if row["split_name"] == split_name]
        for group in GROUP_ORDER:
            subset = [row for row in split_rows if row["outcome_group"] == group]
            toxic_count = sum(int(row["label"]) for row in subset)
            output.append(
                {
                    "split_name": split_name,
                    "outcome_group": group,
                    "group_public_name": GROUP_PUBLIC_NAMES[group],
                    "sample_count": len(subset),
                    "group_fraction": len(subset) / len(split_rows),
                    "toxic_count": toxic_count,
                    "non_toxic_count": len(subset) - toxic_count,
                    "toxic_fraction": toxic_count / len(subset) if subset else "",
                }
            )
    return output


def _group_summaries(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    replicates = int(protocol["resampling"]["bootstrap_replicates"])
    base_seed = int(protocol["resampling"]["bootstrap_seed"])
    output: list[dict[str, Any]] = []
    for split_name in ("Test1", "Test2"):
        for group in GROUP_ORDER:
            subset = _rows_for(rows, split_name, group)
            for attribute in CONTINUOUS_ATTRIBUTES:
                values = np.asarray([float(row[attribute]) for row in subset], dtype=np.float64)
                finite = values[np.isfinite(values)]
                seed = _stable_seed(base_seed, "group_summary", split_name, group, attribute)
                rng = np.random.default_rng(seed)
                bootstrap = np.empty(replicates, dtype=np.float64)
                for index in range(replicates):
                    sampled = _stratified_group_resample(subset, attribute, rng)
                    bootstrap[index] = np.median(sampled) if sampled.size else np.nan
                lower, upper = _percentile_interval(bootstrap)
                output.append(
                    {
                        "split_name": split_name,
                        "outcome_group": group,
                        "group_public_name": GROUP_PUBLIC_NAMES[group],
                        "attribute": attribute,
                        "attribute_public_name": ATTRIBUTE_PUBLIC_NAMES[attribute],
                        "attribute_type": "continuous",
                        "n": int(finite.size),
                        "missing_n": int(values.size - finite.size),
                        "mean": float(finite.mean()) if finite.size else "",
                        "standard_deviation": float(finite.std(ddof=1)) if finite.size > 1 else "",
                        "median": float(np.median(finite)) if finite.size else "",
                        "q1": _quantile(finite, 0.25) if finite.size else "",
                        "q3": _quantile(finite, 0.75) if finite.size else "",
                        "median_ci_lower": lower,
                        "median_ci_upper": upper,
                        "available_count": "",
                        "proportion": "",
                        "proportion_ci_lower": "",
                        "proportion_ci_upper": "",
                        "bootstrap_seed": seed,
                        "bootstrap_replicates": replicates,
                    }
                )
            successes = sum(int(row["conformer_available"]) for row in subset)
            lower, upper = _wilson_interval(successes, len(subset))
            output.append(
                {
                    "split_name": split_name,
                    "outcome_group": group,
                    "group_public_name": GROUP_PUBLIC_NAMES[group],
                    "attribute": "conformer_available",
                    "attribute_public_name": ATTRIBUTE_PUBLIC_NAMES["conformer_available"],
                    "attribute_type": "binary",
                    "n": len(subset),
                    "missing_n": 0,
                    "mean": "",
                    "standard_deviation": "",
                    "median": "",
                    "q1": "",
                    "q3": "",
                    "median_ci_lower": "",
                    "median_ci_upper": "",
                    "available_count": successes,
                    "proportion": successes / len(subset) if subset else "",
                    "proportion_ci_lower": lower,
                    "proportion_ci_upper": upper,
                    "bootstrap_seed": "",
                    "bootstrap_replicates": "",
                }
            )
    return output


def _rescue_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name in ("Test1", "Test2"):
        split_rows = [row for row in rows if row["split_name"] == split_name]
        for label_name, label_value in (("all", None), ("non_toxic", 0), ("toxic", 1)):
            subset = split_rows if label_value is None else [row for row in split_rows if row["label"] == label_value]
            counts = Counter(row["outcome_group"] for row in subset)
            definitions = (
                (
                    "three_d_rescue_among_one_d_errors",
                    counts["three_d_only_correct"],
                    counts["three_d_only_correct"] + counts["both_wrong"],
                ),
                (
                    "one_d_rescue_among_three_d_errors",
                    counts["one_d_only_correct"],
                    counts["one_d_only_correct"] + counts["both_wrong"],
                ),
            )
            for endpoint, numerator, denominator in definitions:
                lower, upper = _wilson_interval(numerator, denominator)
                output.append(
                    {
                        "split_name": split_name,
                        "label_stratum": label_name,
                        "endpoint": endpoint,
                        "numerator": numerator,
                        "denominator": denominator,
                        "rate": numerator / denominator if denominator else "",
                        "wilson_ci_lower": lower,
                        "wilson_ci_upper": upper,
                    }
                )
    return output


def _bootstrap_primary(
    three_d_rows: list[dict[str, Any]],
    one_d_rows: list[dict[str, Any]],
    attribute: str,
    attribute_type: str,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    primary = np.empty(replicates, dtype=np.float64)
    secondary = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        three_d = _stratified_group_resample(three_d_rows, attribute, rng)
        one_d = _stratified_group_resample(one_d_rows, attribute, rng)
        if attribute_type == "binary":
            primary[index] = three_d.mean() - one_d.mean()
            secondary[index] = np.nan
        else:
            primary[index] = _cliffs_delta(three_d, one_d)
            secondary[index] = np.median(three_d) - np.median(one_d)
    return primary, secondary


def _permutation_p_value(
    three_d_rows: list[dict[str, Any]],
    one_d_rows: list[dict[str, Any]],
    attribute: str,
    attribute_type: str,
    observed: float,
    replicates: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(replicates):
        permuted_three_d: list[np.ndarray] = []
        permuted_one_d: list[np.ndarray] = []
        for label in (0, 1):
            first = np.asarray(
                [float(row[attribute]) for row in three_d_rows if row["label"] == label],
                dtype=np.float64,
            )
            second = np.asarray(
                [float(row[attribute]) for row in one_d_rows if row["label"] == label],
                dtype=np.float64,
            )
            values = np.concatenate((first, second))
            if values.size == 0:
                continue
            shuffled = rng.permutation(values)
            permuted_three_d.append(shuffled[: first.size])
            permuted_one_d.append(shuffled[first.size :])
        first_values = np.concatenate(permuted_three_d) if permuted_three_d else np.empty(0)
        second_values = np.concatenate(permuted_one_d) if permuted_one_d else np.empty(0)
        statistic = (
            first_values.mean() - second_values.mean()
            if attribute_type == "binary"
            else _cliffs_delta(first_values, second_values)
        )
        if abs(statistic) >= abs(observed) - 1e-15:
            exceed += 1
    return (exceed + 1) / (replicates + 1)


def _primary_contrasts(
    rows: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    bootstrap_replicates = int(protocol["resampling"]["bootstrap_replicates"])
    bootstrap_base_seed = int(protocol["resampling"]["bootstrap_seed"])
    permutation_replicates = int(protocol["resampling"]["permutation_replicates"])
    permutation_base_seed = int(protocol["resampling"]["permutation_seed"])
    minimum_n = int(protocol["reporting"]["minimum_group_n_for_inference"])
    output: list[dict[str, Any]] = []
    for split_name in ("Test1", "Test2"):
        three_d_rows = _rows_for(rows, split_name, "three_d_only_correct")
        one_d_rows = _rows_for(rows, split_name, "one_d_only_correct")
        for attribute in (*CONTINUOUS_ATTRIBUTES, *BINARY_ATTRIBUTES):
            attribute_type = "binary" if attribute in BINARY_ATTRIBUTES else "continuous"
            three_d = np.asarray([float(row[attribute]) for row in three_d_rows], dtype=np.float64)
            one_d = np.asarray([float(row[attribute]) for row in one_d_rows], dtype=np.float64)
            inferential = three_d.size >= minimum_n and one_d.size >= minimum_n
            if attribute_type == "binary":
                primary_effect = float(three_d.mean() - one_d.mean()) if three_d.size and one_d.size else float("nan")
                median_difference = float("nan")
                cliffs_delta = float("nan")
            else:
                cliffs_delta = _cliffs_delta(three_d, one_d)
                primary_effect = cliffs_delta
                median_difference = float(np.median(three_d) - np.median(one_d)) if three_d.size and one_d.size else float("nan")
            bootstrap_seed = _stable_seed(bootstrap_base_seed, "primary", split_name, attribute)
            permutation_seed = _stable_seed(permutation_base_seed, "primary", split_name, attribute)
            if inferential:
                bootstrap_primary, bootstrap_secondary = _bootstrap_primary(
                    three_d_rows,
                    one_d_rows,
                    attribute,
                    attribute_type,
                    bootstrap_replicates,
                    bootstrap_seed,
                )
                effect_lower, effect_upper = _percentile_interval(bootstrap_primary)
                secondary_lower, secondary_upper = _percentile_interval(bootstrap_secondary)
                p_value = _permutation_p_value(
                    three_d_rows,
                    one_d_rows,
                    attribute,
                    attribute_type,
                    primary_effect,
                    permutation_replicates,
                    permutation_seed,
                )
            else:
                effect_lower = effect_upper = secondary_lower = secondary_upper = float("nan")
                p_value = None
            output.append(
                {
                    "split_name": split_name,
                    "attribute": attribute,
                    "attribute_public_name": ATTRIBUTE_PUBLIC_NAMES[attribute],
                    "attribute_type": attribute_type,
                    "contrast": "three_d_only_correct_minus_one_d_only_correct",
                    "three_d_only_n": int(three_d.size),
                    "one_d_only_n": int(one_d.size),
                    "three_d_only_median_or_proportion": float(three_d.mean()) if attribute_type == "binary" else float(np.median(three_d)),
                    "one_d_only_median_or_proportion": float(one_d.mean()) if attribute_type == "binary" else float(np.median(one_d)),
                    "primary_effect_name": "availability_proportion_difference" if attribute_type == "binary" else "cliffs_delta",
                    "primary_effect": primary_effect,
                    "primary_effect_ci_lower": effect_lower,
                    "primary_effect_ci_upper": effect_upper,
                    "median_difference": median_difference if attribute_type == "continuous" else "",
                    "median_difference_ci_lower": secondary_lower if attribute_type == "continuous" else "",
                    "median_difference_ci_upper": secondary_upper if attribute_type == "continuous" else "",
                    "permutation_p_value": p_value if p_value is not None else "",
                    "bh_q_value": "",
                    "inferential_status": "estimated" if inferential else "descriptive_only_small_group",
                    "effect_orientation": "positive means larger or more available in the 3D-only-correct group",
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_replicates": bootstrap_replicates if inferential else 0,
                    "permutation_seed": permutation_seed,
                    "permutation_replicates": permutation_replicates if inferential else 0,
                }
            )
    adjusted = _bh_adjust(
        [float(row["permutation_p_value"]) if row["permutation_p_value"] != "" else None for row in output]
    )
    for row, q_value in zip(output, adjusted):
        row["bh_q_value"] = q_value if q_value is not None else ""
    return output


def _length_bin(length: float) -> str:
    return "<=20" if length <= 20 else "21-30" if length <= 30 else "31-50" if length <= 50 else ">50"


def _cys_bin(count: float) -> str:
    return "0" if count == 0 else "1-2" if count <= 2 else ">=3"


def _plddt_bin(value: float) -> str:
    return "<0.55" if value < 0.55 else "0.55-0.70" if value < 0.70 else ">=0.70"


def _coverage_bin(value: float) -> str:
    return "<0.80" if value < 0.80 else "0.80-0.95" if value < 0.95 else ">=0.95"


def _sensitivity_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = (
        ("peptide_length", _length_bin),
        ("cysteine_count", _cys_bin),
        ("mean_plddt", _plddt_bin),
        ("chemical_site_coverage", _coverage_bin),
    )
    output: list[dict[str, Any]] = []
    for split_name in ("Test1", "Test2"):
        for group in GROUP_ORDER:
            subset = _rows_for(rows, split_name, group)
            for attribute, assign in definitions:
                counts = Counter(assign(float(row[attribute])) for row in subset)
                for bin_name, count in counts.items():
                    bin_rows = [row for row in subset if assign(float(row[attribute])) == bin_name]
                    available = sum(int(row["conformer_available"]) for row in bin_rows)
                    output.append(
                        {
                            "split_name": split_name,
                            "outcome_group": group,
                            "attribute": attribute,
                            "bin": bin_name,
                            "sample_count": count,
                            "fraction_within_group": count / len(subset) if subset else "",
                            "toxic_count": sum(int(row["label"]) for row in bin_rows),
                            "conformer_available_count": available,
                            "conformer_available_fraction": available / count if count else "",
                            "scope": "descriptive_prespecified_sensitivity_view",
                        }
                    )
    return output


def _conformer_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name in ("Test1", "Test2"):
        for group in GROUP_ORDER:
            subset = _rows_for(rows, split_name, group)
            counts = Counter(row["conformer_source"] for row in subset)
            for source, count in sorted(counts.items()):
                output.append(
                    {
                        "split_name": split_name,
                        "outcome_group": group,
                        "conformer_source": source,
                        "sample_count": count,
                        "fraction_within_group": count / len(subset) if subset else "",
                        "interpretation": "observational availability metadata; canonical probability is single-structure",
                    }
                )
    return output


def _format_number(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "NA"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _results_markdown(
    counts: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    rescue_rates: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
) -> str:
    lookup = {(row["split_name"], row["outcome_group"], row["attribute"]): row for row in summaries}
    lines = [
        "# Frozen 1D--3D expert disagreement results",
        "",
        "All groups use the training-derived 1D threshold 0.392 and 3D threshold 0.677819. Values are descriptive associations on established benchmarks.",
        "",
    ]
    for split_name in ("Test1", "Test2"):
        lines.extend(
            [
                f"## {split_name}",
                "",
                "| Group | n | Toxic n | Length median [IQR] | Cys median [IQR] | Mean pLDDT median [IQR] | Site coverage median [IQR] | Conformer available |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for count_row in [row for row in counts if row["split_name"] == split_name]:
            group = count_row["outcome_group"]
            cells = []
            for attribute in CONTINUOUS_ATTRIBUTES:
                item = lookup[(split_name, group, attribute)]
                cells.append(
                    f"{_format_number(item['median'])} [{_format_number(item['q1'])}, {_format_number(item['q3'])}]"
                )
            conformer = lookup[(split_name, group, "conformer_available")]
            lines.append(
                f"| {GROUP_PUBLIC_NAMES[group]} | {count_row['sample_count']} | {count_row['toxic_count']} | "
                + " | ".join(cells)
                + f" | {conformer['available_count']}/{conformer['n']} ({100.0 * float(conformer['proportion']):.1f}%) |"
            )
        lines.append("")
        lines.append("Direct complementarity:")
        lines.append("")
        for row in rescue_rates:
            if row["split_name"] == split_name and row["label_stratum"] == "all":
                lines.append(
                    f"- {row['endpoint']}: {row['numerator']}/{row['denominator']} = {100.0 * float(row['rate']):.2f}% "
                    f"(95% Wilson CI {100.0 * float(row['wilson_ci_lower']):.2f}--{100.0 * float(row['wilson_ci_upper']):.2f}%)."
                )
        lines.append("")
    lines.extend(
        [
            "## Primary contrast: 3D-only correct minus 1D-only correct",
            "",
            "| Split | Attribute | Effect | 95% CI | Permutation p | BH q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in contrasts:
        lines.append(
            f"| {row['split_name']} | {row['attribute_public_name']} | {_format_number(row['primary_effect'])} | "
            f"[{_format_number(row['primary_effect_ci_lower'])}, {_format_number(row['primary_effect_ci_upper'])}] | "
            f"{_format_number(row['permutation_p_value'], 4)} | {_format_number(row['bh_q_value'], 4)} |"
        )
    lines.extend(
        [
            "",
            "For continuous attributes the effect is Cliff's delta; for conformer availability it is a proportion difference. Positive values indicate larger values or higher availability in the 3D-only-correct group.",
            "",
        ]
    )
    return "\n".join(lines)


def _readme(protocol: dict[str, Any]) -> str:
    return f"""# Frozen 1D--3D expert disagreement analysis

This package classifies every canonical Test1/Test2 sample into four mutually exclusive groups using the pre-existing standalone thresholds: 1D = `{protocol['expert_operating_points']['one_d_threshold']}` and 3D = `{protocol['expert_operating_points']['three_d_threshold']}`. No checkpoint, probability, calibration, fusion rule, or threshold is changed.

## Outputs

- `sample_assignments.csv`: identity-bearing probabilities, decisions, correctness groups, and all five requested attributes.
- `group_counts.csv`: group size and true-label composition.
- `group_attribute_summaries.csv`: four-group descriptive distributions and confidence intervals.
- `rescue_rates.csv`: reciprocal expert rescue rates overall and by true label.
- `primary_contrasts.csv`: 3D-only-correct versus 1D-only-correct effects, bootstrap intervals, label-stratified permutation p-values, and BH q-values.
- `prespecified_bins.csv`: frozen descriptive length/Cys/pLDDT/site-coverage views, including availability within length bins.
- `conformer_source_summary.csv`: observational generator-availability composition.
- `results_table.md`, `summary.json`, `input_manifest.csv`, and `checksums.sha256`: human- and machine-readable provenance.

## Statistical contract

Continuous group summaries use medians/IQRs and label-stratified bootstrap intervals. The primary continuous effect is Cliff's delta; conformer availability uses a proportion difference. Tests shuffle the two discordant group labels within true-label strata. BH correction is applied once across the ten split-by-attribute primary tests.

## Interpretation boundary

These are post hoc descriptive associations on established benchmarks. pLDDT is prediction confidence, chemical sites are model-derived candidates, and conformer availability is confounded by generator/length assignment. The canonical public predictions use single-structure 3D base logits, so availability does not mean the prediction consumed a conformer ensemble. Test1 and Test2 must remain separate.
"""


def _write_checksums(output_dir: Path) -> None:
    paths = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = args.protocol.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(f"Non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _verify_protocol(root, protocol_path)
    samples, integrity = _load_samples(root, protocol)
    if len(samples) != 1708:
        raise ValueError(f"Expected 1,708 joined samples, found {len(samples)}")

    counts = _group_counts(samples)
    summaries = _group_summaries(samples, protocol)
    rescue_rates = _rescue_rates(samples)
    contrasts = _primary_contrasts(samples, protocol)
    bins = _sensitivity_bins(samples)
    conformer_sources = _conformer_sources(samples)

    _write_csv(output_dir / "sample_assignments.csv", samples)
    _write_csv(output_dir / "group_counts.csv", counts)
    _write_csv(output_dir / "group_attribute_summaries.csv", summaries)
    _write_csv(output_dir / "rescue_rates.csv", rescue_rates)
    _write_csv(output_dir / "primary_contrasts.csv", contrasts)
    _write_csv(output_dir / "prespecified_bins.csv", bins)
    _write_csv(output_dir / "conformer_source_summary.csv", conformer_sources)

    manifest_rows = [
        {
            "input_type": "protocol",
            "input_identifier": str(protocol_path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256(protocol_path),
            "size_bytes": protocol_path.stat().st_size,
            "role": "frozen analysis protocol",
        }
    ] + [
        {
            "input_type": "local_file",
            "input_identifier": item["path"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
            "role": item["role"],
        }
        for item in protocol["inputs"]
    ]
    _write_csv(output_dir / "input_manifest.csv", manifest_rows)

    summary = {
        "artifact_type": "ghxtox_1d_3d_expert_disagreement_results",
        "schema_version": 1,
        "protocol_sha256": _sha256(protocol_path),
        "decision": "descriptive_analysis_complete_no_model_change",
        "expert_operating_points": protocol["expert_operating_points"],
        "group_order": list(GROUP_ORDER),
        "group_counts": counts,
        "rescue_rates": rescue_rates,
        "primary_contrasts": contrasts,
        "integrity": integrity,
        "row_counts": {
            "sample_assignments": len(samples),
            "group_counts": len(counts),
            "group_attribute_summaries": len(summaries),
            "rescue_rates": len(rescue_rates),
            "primary_contrasts": len(contrasts),
            "prespecified_bins": len(bins),
            "conformer_source_summary": len(conformer_sources),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_used_for": "read-only deserialization and tensor reductions; no model or checkpoint loaded",
        },
        "interpretation_boundary": protocol["interpretation_boundary"],
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "results_table.md").write_text(
        _results_markdown(counts, summaries, rescue_rates, contrasts), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_readme(protocol), encoding="utf-8")
    _write_checksums(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "row_counts": summary["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
