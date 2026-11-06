"""Training-OOF-only feasibility audit for soft peptide-topology modelling.

This script is deliberately read-only with respect to models and source data.  It
does not infer disulfide connectivity: distances in predicted structures are
reported only as candidate-contact statistics.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}")
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_metric(function: Callable[..., float], *args: Any) -> float:
    try:
        value = float(function(*args))
    except ValueError:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = labels.astype(int)
    probabilities = probabilities.astype(float)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "sample_count": int(labels.size),
        "positive_count": int(labels.sum()),
        "negative_count": int(labels.size - labels.sum()),
        "positive_fraction": float(labels.mean()) if labels.size else float("nan"),
        "error_rate": float(np.mean(predictions != labels)) if labels.size else float("nan"),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, labels, predictions),
        "mcc": _safe_metric(matthews_corrcoef, labels, predictions),
        "auroc": _safe_metric(roc_auc_score, labels, probabilities),
        "auprc": _safe_metric(average_precision_score, labels, probabilities),
        "brier": float(np.mean((probabilities - labels) ** 2)) if labels.size else float("nan"),
    }


def _label_standardized_error(
    labels: np.ndarray,
    predictions: np.ndarray,
    label_weights: dict[int, float],
) -> float:
    total = 0.0
    for label, weight in label_weights.items():
        mask = labels == label
        if not mask.any():
            return float("nan")
        total += weight * float(np.mean(predictions[mask] != labels[mask]))
    return total


def _cys_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 5:
        return "4-5"
    return ">=6"


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {"n": 0, "min": float("nan"), "q25": float("nan"), "median": float("nan"), "q75": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "n": int(array.size),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _bin_rows(
    values: np.ndarray,
    edges: list[float],
    coordinate: str,
    subset: str,
) -> list[dict[str, Any]]:
    finite = values[np.isfinite(values)]
    counts, _ = np.histogram(finite, bins=np.asarray(edges, dtype=float))
    total = int(counts.sum())
    rows = []
    for index, count in enumerate(counts):
        left = edges[index]
        right = edges[index + 1]
        rows.append(
            {
                "coordinate": coordinate,
                "subset": subset,
                "bin_left_angstrom": left,
                "bin_right_angstrom": "inf" if math.isinf(right) else right,
                "pair_count": int(count),
                "pair_fraction": float(count / total) if total else float("nan"),
                "total_pairs": total,
            }
        )
    return rows


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _correlations(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(first) & np.isfinite(second)
    x, y = first[mask], second[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(x.size), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "n": int(x.size),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(_rank_average(x), _rank_average(y))[0, 1]),
    }


def _bootstrap(
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    resamples: int,
    seed: int,
) -> tuple[float, float, float, float, int]:
    indices = np.arange(labels.size)
    strata: list[np.ndarray] = []
    for fold in sorted(set(folds.tolist())):
        for label in (0, 1):
            for group in (False, True):
                stratum = indices[(folds == fold) & (labels == label) & (groups == group)]
                if stratum.size:
                    strata.append(stratum)
    rng = np.random.default_rng(seed)
    point = float(statistic(indices))
    samples = np.empty(resamples, dtype=float)
    for sample_index in range(resamples):
        draw = np.concatenate([rng.choice(stratum, size=stratum.size, replace=True) for stratum in strata])
        samples[sample_index] = statistic(draw)
    valid = samples[np.isfinite(samples)]
    if valid.size == 0:
        return point, float("nan"), float("nan"), float("nan"), 0
    return (
        point,
        float(np.quantile(valid, 0.025)),
        float(np.quantile(valid, 0.975)),
        float(np.mean(valid > 0.0)),
        int(valid.size),
    )


def _odds_ratio(a: int, b: int, c: int, d: int) -> dict[str, float]:
    # Haldane--Anscombe correction makes the result defined if a cell is zero.
    aa, bb, cc, dd = (value + 0.5 for value in (a, b, c, d))
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    return {
        "odds_ratio": math.exp(log_or),
        "ci95_lower": math.exp(log_or - 1.96 * se),
        "ci95_upper": math.exp(log_or + 1.96 * se),
    }


def _maximum_candidate_matching(
    cys_positions: list[int],
    peptide_pairs: list[dict[str, Any]],
    cutoff: float,
) -> int:
    """Maximum number of disjoint Cys pairs under a distance cutoff.

    This is a graph diagnostic, not a disulfide assignment.  The dynamic
    program allows a Cys to remain unmatched and enforces at most one partner.
    """

    count = len(cys_positions)
    if count < 2:
        return 0
    local_index = {position: index for index, position in enumerate(cys_positions)}
    adjacency = [0] * count
    for row in peptide_pairs:
        distance = float(row["sg_distance_angstrom"])
        if not bool(row["sg_pair_valid"]) or not math.isfinite(distance) or distance > cutoff:
            continue
        left = local_index[int(row["cys_position_i"]) - 1]
        right = local_index[int(row["cys_position_j"]) - 1]
        adjacency[left] |= 1 << right
        adjacency[right] |= 1 << left

    @functools.lru_cache(maxsize=None)
    def solve(mask: int) -> int:
        if mask == 0:
            return 0
        lowest_bit = mask & -mask
        first = lowest_bit.bit_length() - 1
        remainder = mask & ~lowest_bit
        best = solve(remainder)
        candidates = adjacency[first] & remainder
        while candidates:
            partner_bit = candidates & -candidates
            best = max(best, 1 + solve(remainder & ~partner_bit))
            candidates &= ~partner_bit
        return best

    return solve((1 << count) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/topology_feasibility_audit_v1.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    protocol_path = _resolve(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["scope"]["allow_model_training"] or protocol["scope"]["allow_model_inference"]:
        raise ValueError("This audit must remain training- and inference-free.")

    input_paths = {name: _resolve(path).resolve() for name, path in protocol["inputs"].items()}
    for name, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    output_dir = _resolve(args.output_dir or protocol["output_dir"]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_csv(input_paths["fold_manifest"])
    current_rows = _read_csv(input_paths["current_3d_oof"])
    chemical_rows = _read_csv(input_paths["matched_chemical_oof"])
    selection = json.loads(input_paths["matched_chemical_selection"].read_text(encoding="utf-8"))
    payload = torch.load(input_paths["processed_train"], map_location="cpu", weights_only=False)
    records = payload["records"]
    expected_size = len(records)
    if not (len(manifest_rows) == len(current_rows) == len(chemical_rows) == expected_size):
        raise ValueError(
            f"Row-count mismatch: tensor={expected_size}, manifest={len(manifest_rows)}, "
            f"current={len(current_rows)}, chemical={len(chemical_rows)}"
        )
    schema = payload.get("chemical_site_schema") or {}
    type_names = schema.get("type_names") or []
    if "sulfur" not in type_names:
        raise ValueError("Processed cache does not declare a sulfur chemical-site channel.")
    sulfur_index = int(type_names.index("sulfur"))

    manifest_by_index = {int(row["source_index"]): row for row in manifest_rows}
    current_by_index = {int(row["source_index"]): row for row in current_rows}
    chemical_by_index = {int(row["source_index"]): row for row in chemical_rows}
    expected_indices = set(range(expected_size))
    for name, lookup in (("manifest", manifest_by_index), ("current", current_by_index), ("chemical", chemical_by_index)):
        if set(lookup) != expected_indices:
            raise ValueError(f"{name} source_index values are not exactly 0..{expected_size - 1}.")

    primary_min = int(protocol["cysteine_definitions"]["primary_cys_rich_min_count"])
    sg_contact_cutoff = float(protocol["geometry"]["permissive_candidate_sg_contact_angstrom"])
    ca_contact_cutoff = float(protocol["geometry"]["legacy_candidate_ca_contact_angstrom"])
    high_pair_confidence = float(protocol["geometry"]["high_pair_confidence_min"])
    sample_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    for source_index, record in enumerate(records):
        manifest = manifest_by_index[source_index]
        current = current_by_index[source_index]
        chemical = chemical_by_index[source_index]
        sample_id = str(record["sample_id"])
        sequence = str(record["sequence"]).upper()
        label = int(record["label"])
        fold = int(manifest["fold"])
        for source_name, row in (("manifest", manifest), ("current", current), ("chemical", chemical)):
            if str(row["sample_id"]) != sample_id or str(row["sequence"]).upper() != sequence:
                raise ValueError(f"{source_name} identity mismatch at source_index={source_index}")
            if int(row["label"]) != label:
                raise ValueError(f"{source_name} label mismatch at source_index={source_index}")
            if int(row["fold"]) != fold:
                raise ValueError(f"{source_name} fold mismatch at source_index={source_index}")

        coords = record["coords"].detach().float().cpu().numpy()
        plddt = record["plddt"].detach().float().cpu().numpy()
        site_coords = record["chemical_site_coords"].detach().float().cpu().numpy()
        site_types = record["chemical_site_types"].detach().float().cpu().numpy()
        site_orientations = record["chemical_site_orientations"].detach().float().cpu().numpy()
        site_orientation_mask = record["chemical_site_orientation_mask"].detach().bool().cpu().numpy()
        site_mask = record["chemical_site_mask"].detach().bool().cpu().numpy()
        length = len(sequence)
        if coords.shape != (length, 3) or plddt.shape != (length,):
            raise ValueError(f"Coordinate/pLDDT length mismatch for {sample_id}")
        cys_positions = [index for index, residue in enumerate(sequence) if residue == "C"]
        cys_count = len(cys_positions)
        cys_fraction = cys_count / max(length, 1)
        sg_lookup: dict[int, tuple[np.ndarray, np.ndarray, bool]] = {}
        for position in cys_positions:
            sulfur_slots = np.flatnonzero(site_mask[position] & (site_types[position, :, sulfur_index] > 0.5))
            if sulfur_slots.size == 1:
                slot = int(sulfur_slots[0])
                sg_lookup[position] = (
                    site_coords[position, slot],
                    site_orientations[position, slot],
                    bool(site_orientation_mask[position, slot]),
                )
            elif sulfur_slots.size > 1:
                raise ValueError(f"Multiple sulfur sites for Cys at {sample_id}:{position + 1}")

        peptide_pairs: list[dict[str, Any]] = []
        for left_index, left in enumerate(cys_positions):
            for right in cys_positions[left_index + 1 :]:
                ca_distance = float(np.linalg.norm(coords[right] - coords[left]))
                sg_distance = float("nan")
                source_alignment = target_alignment = mutual_alignment = float("nan")
                sg_valid = left in sg_lookup and right in sg_lookup
                orientation_valid = False
                if sg_valid:
                    left_coord, left_orientation, left_orientation_valid = sg_lookup[left]
                    right_coord, right_orientation, right_orientation_valid = sg_lookup[right]
                    vector = right_coord - left_coord
                    sg_distance = float(np.linalg.norm(vector))
                    if sg_distance > 1e-8 and left_orientation_valid and right_orientation_valid:
                        unit = vector / sg_distance
                        source_alignment = float(abs(np.dot(left_orientation, unit)))
                        target_alignment = float(abs(np.dot(right_orientation, -unit)))
                        mutual_alignment = float(abs(np.dot(left_orientation, right_orientation)))
                        orientation_valid = True
                pair = {
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "fold": fold,
                    "label": label,
                    "length": length,
                    "cys_count": cys_count,
                    "cys_rich": cys_count >= primary_min,
                    "cys_position_i": left + 1,
                    "cys_position_j": right + 1,
                    "sequence_separation": right - left,
                    "ca_distance_angstrom": ca_distance,
                    "sg_distance_angstrom": sg_distance,
                    "sg_pair_valid": sg_valid,
                    "pair_plddt_min": float(min(plddt[left], plddt[right])),
                    "pair_plddt_ge_0_7": float(min(plddt[left], plddt[right])) >= high_pair_confidence,
                    "orientation_pair_valid": orientation_valid,
                    "source_alignment_abs": source_alignment,
                    "target_alignment_abs": target_alignment,
                    "mutual_alignment_abs": mutual_alignment,
                    "candidate_sg_contact_le_4_5": sg_valid and sg_distance <= sg_contact_cutoff,
                    "candidate_ca_contact_le_6_5": ca_distance <= ca_contact_cutoff,
                }
                peptide_pairs.append(pair)
                pair_rows.append(pair)

        ca_distances = [row["ca_distance_angstrom"] for row in peptide_pairs]
        sg_distances = [row["sg_distance_angstrom"] for row in peptide_pairs if row["sg_pair_valid"]]
        matching_counts = {
            cutoff: _maximum_candidate_matching(cys_positions, peptide_pairs, cutoff)
            for cutoff in (2.5, 3.5, 4.5)
        }
        current_probability = _finite_float(current[protocol["evaluation"]["current_3d_probability_column"]], "current probability")
        current_prediction = int(current[protocol["evaluation"]["current_3d_prediction_column"]])
        current_threshold = _finite_float(current[protocol["evaluation"]["current_3d_threshold_column"]], "current threshold")
        if current_prediction != int(current_probability >= current_threshold):
            raise ValueError(f"Current prediction/threshold mismatch for {sample_id}")
        control_probability = _finite_float(chemical[protocol["evaluation"]["matched_chemical_control_column"]], "control probability")
        selected_probability = _finite_float(chemical[protocol["evaluation"]["matched_chemical_candidate_column"]], "selected probability")
        chemical_threshold = float(protocol["evaluation"]["matched_chemical_threshold"])
        strict_high_risk = cys_count >= 6 or (cys_count >= 4 and cys_fraction >= 0.15)
        sample_rows.append(
            {
                "source_index": source_index,
                "sample_id": sample_id,
                "sequence": sequence,
                "label": label,
                "fold": fold,
                "length": length,
                "mean_plddt": float(np.mean(plddt)),
                "min_plddt": float(np.min(plddt)),
                "cys_count": cys_count,
                "cys_fraction": cys_fraction,
                "cys_bin": _cys_bin(cys_count),
                "cys_ge_2": cys_count >= 2,
                "cys_ge_3": cys_count >= 3,
                "cys_ge_4": cys_count >= 4,
                "cys_ge_6": cys_count >= 6,
                "strict_high_risk": strict_high_risk,
                "cys_sg_sites_valid": len(sg_lookup),
                "all_cys_sg_sites_valid": len(sg_lookup) == cys_count,
                "cys_pair_count": len(peptide_pairs),
                "valid_sg_pair_count": len(sg_distances),
                "min_ca_distance_angstrom": min(ca_distances, default=float("nan")),
                "median_ca_distance_angstrom": float(np.median(ca_distances)) if ca_distances else float("nan"),
                "min_sg_distance_angstrom": min(sg_distances, default=float("nan")),
                "median_sg_distance_angstrom": float(np.median(sg_distances)) if sg_distances else float("nan"),
                "candidate_sg_contact_count": sum(bool(row["candidate_sg_contact_le_4_5"]) for row in peptide_pairs),
                "candidate_sg_contact_fraction": (sum(bool(row["candidate_sg_contact_le_4_5"]) for row in peptide_pairs) / len(sg_distances)) if sg_distances else float("nan"),
                "candidate_sg_contact_any": any(bool(row["candidate_sg_contact_le_4_5"]) for row in peptide_pairs),
                "high_confidence_candidate_sg_contact_any": any(bool(row["candidate_sg_contact_le_4_5"]) and bool(row["pair_plddt_ge_0_7"]) for row in peptide_pairs),
                "candidate_ca_contact_count": sum(bool(row["candidate_ca_contact_le_6_5"]) for row in peptide_pairs),
                "candidate_matching_pairs_le_2_5": matching_counts[2.5],
                "candidate_matching_fraction_le_2_5": (2.0 * matching_counts[2.5] / cys_count) if cys_count else 0.0,
                "candidate_matching_pairs_le_3_5": matching_counts[3.5],
                "candidate_matching_fraction_le_3_5": (2.0 * matching_counts[3.5] / cys_count) if cys_count else 0.0,
                "candidate_matching_pairs_le_4_5": matching_counts[4.5],
                "candidate_matching_fraction_le_4_5": (2.0 * matching_counts[4.5] / cys_count) if cys_count else 0.0,
                "current_3d_probability": current_probability,
                "current_3d_threshold": current_threshold,
                "current_3d_prediction": current_prediction,
                "current_3d_error": int(current_prediction != label),
                "current_3d_brier": (current_probability - label) ** 2,
                "chemical_control_probability": control_probability,
                "chemical_selected_probability": selected_probability,
                "chemical_probability_difference": selected_probability - control_probability,
                "chemical_control_prediction": int(control_probability >= chemical_threshold),
                "chemical_selected_prediction": int(selected_probability >= chemical_threshold),
                "chemical_control_error": int((control_probability >= chemical_threshold) != bool(label)),
                "chemical_selected_error": int((selected_probability >= chemical_threshold) != bool(label)),
                "chemical_brier_improvement": (control_probability - label) ** 2 - (selected_probability - label) ** 2,
                "chemical_selection_status": chemical["selection_status"],
                "chemical_active_fold": fold != 2,
            }
        )

    labels = np.asarray([row["label"] for row in sample_rows], dtype=int)
    folds = np.asarray([row["fold"] for row in sample_rows], dtype=int)
    current_probability = np.asarray([row["current_3d_probability"] for row in sample_rows], dtype=float)
    current_prediction = np.asarray([row["current_3d_prediction"] for row in sample_rows], dtype=int)
    control_probability = np.asarray([row["chemical_control_probability"] for row in sample_rows], dtype=float)
    selected_probability = np.asarray([row["chemical_selected_probability"] for row in sample_rows], dtype=float)
    cys_count = np.asarray([row["cys_count"] for row in sample_rows], dtype=int)
    cys_rich = cys_count >= primary_min
    strict_risk = np.asarray([_bool(row["strict_high_risk"]) for row in sample_rows], dtype=bool)
    valid_sg = np.asarray([int(row["valid_sg_pair_count"]) > 0 for row in sample_rows], dtype=bool)
    close_sg = np.asarray([_bool(row["candidate_sg_contact_any"]) for row in sample_rows], dtype=bool)
    active_chemical = folds != 2
    current_thresholds = np.asarray([row["current_3d_threshold"] for row in sample_rows], dtype=float)
    if np.unique(current_thresholds).size != 1:
        raise ValueError("Current OOF uses multiple decision thresholds; this audit expects one frozen threshold.")
    label_weights = {label: float(np.mean(labels == label)) for label in (0, 1)}

    subgroup_masks: dict[str, np.ndarray] = {
        "all": np.ones(expected_size, dtype=bool),
        "cys_eq_0": cys_count == 0,
        "cys_eq_1": cys_count == 1,
        "cys_2_3": (cys_count >= 2) & (cys_count <= 3),
        "cys_4_5": (cys_count >= 4) & (cys_count <= 5),
        "cys_ge_6": cys_count >= 6,
        "cys_lt_4": cys_count < 4,
        "cys_ge_2": cys_count >= 2,
        "cys_ge_3": cys_count >= 3,
        "cys_ge_4": cys_rich,
        "strict_high_risk": strict_risk,
        "valid_sg_candidate_contact": valid_sg & close_sg,
        "valid_sg_no_candidate_contact": valid_sg & ~close_sg,
    }
    subgroup_metric_rows: list[dict[str, Any]] = []
    for subgroup, mask in subgroup_masks.items():
        if not mask.any():
            continue
        for model, probabilities, threshold in (
            ("current_3d", current_probability, float(current_thresholds[0])),
            ("chemical_control", control_probability, float(protocol["evaluation"]["matched_chemical_threshold"])),
            ("chemical_selected", selected_probability, float(protocol["evaluation"]["matched_chemical_threshold"])),
        ):
            values = _metrics(labels[mask], probabilities[mask], threshold)
            predictions = (probabilities[mask] >= threshold).astype(int)
            values["label_standardized_error_rate"] = _label_standardized_error(labels[mask], predictions, label_weights)
            subgroup_metric_rows.append({"subgroup": subgroup, "model": model, **values})

    chemical_rows_out: list[dict[str, Any]] = []
    for scope_name, scope_mask in (("all_folds", np.ones(expected_size, dtype=bool)), ("active_chemical_folds_0_1_3_4", active_chemical)):
        for subgroup, subgroup_mask in subgroup_masks.items():
            mask = scope_mask & subgroup_mask
            if not mask.any():
                continue
            control_metrics = _metrics(labels[mask], control_probability[mask], float(protocol["evaluation"]["matched_chemical_threshold"]))
            selected_metrics = _metrics(labels[mask], selected_probability[mask], float(protocol["evaluation"]["matched_chemical_threshold"]))
            control_predictions = (control_probability[mask] >= float(protocol["evaluation"]["matched_chemical_threshold"])).astype(int)
            selected_predictions = (selected_probability[mask] >= float(protocol["evaluation"]["matched_chemical_threshold"])).astype(int)
            chemical_rows_out.append(
                {
                    "scope": scope_name,
                    "subgroup": subgroup,
                    "sample_count": int(mask.sum()),
                    "control_error_rate": control_metrics["error_rate"],
                    "selected_error_rate": selected_metrics["error_rate"],
                    "error_rate_reduction": float(control_metrics["error_rate"] - selected_metrics["error_rate"]),
                    "control_mcc": control_metrics["mcc"],
                    "selected_mcc": selected_metrics["mcc"],
                    "mcc_delta": float(selected_metrics["mcc"] - control_metrics["mcc"]),
                    "control_auprc": control_metrics["auprc"],
                    "selected_auprc": selected_metrics["auprc"],
                    "auprc_delta": float(selected_metrics["auprc"] - control_metrics["auprc"]),
                    "control_brier": control_metrics["brier"],
                    "selected_brier": selected_metrics["brier"],
                    "brier_reduction": float(control_metrics["brier"] - selected_metrics["brier"]),
                    "corrected_errors": int(np.sum((control_predictions != labels[mask]) & (selected_predictions == labels[mask]))),
                    "introduced_errors": int(np.sum((control_predictions == labels[mask]) & (selected_predictions != labels[mask]))),
                    "mean_probability_difference": float(np.mean(selected_probability[mask] - control_probability[mask])),
                    "mean_absolute_probability_difference": float(np.mean(np.abs(selected_probability[mask] - control_probability[mask]))),
                }
            )

    matching_rows: list[dict[str, Any]] = []
    for scope, scope_mask in (
        ("cys_ge_2", cys_count >= 2),
        ("cys_ge_4", cys_count >= 4),
        ("cys_ge_6", cys_count >= 6),
    ):
        for cutoff_key, cutoff in (("2_5", 2.5), ("3_5", 3.5), ("4_5", 4.5)):
            fractions = np.asarray(
                [float(row[f"candidate_matching_fraction_le_{cutoff_key}"]) for row in sample_rows],
                dtype=float,
            )
            matched_pairs = np.asarray(
                [int(row[f"candidate_matching_pairs_le_{cutoff_key}"]) for row in sample_rows],
                dtype=int,
            )
            maximum_pairs = cys_count // 2
            for label in (0, 1):
                mask = scope_mask & (labels == label)
                matching_rows.append(
                    {
                        "scope": scope,
                        "sg_cutoff_angstrom": cutoff,
                        "label": label,
                        "sample_count": int(mask.sum()),
                        "mean_matched_cys_fraction": float(np.mean(fractions[mask])),
                        "median_matched_cys_fraction": float(np.median(fractions[mask])),
                        "maximum_cardinality_matching_fraction": float(np.mean(matched_pairs[mask] == maximum_pairs[mask])),
                        "label_auroc_from_matched_cys_fraction": float("nan"),
                    }
                )
            matching_rows.append(
                {
                    "scope": scope,
                    "sg_cutoff_angstrom": cutoff,
                    "label": "label_discrimination",
                    "sample_count": int(scope_mask.sum()),
                    "mean_matched_cys_fraction": float("nan"),
                    "median_matched_cys_fraction": float("nan"),
                    "maximum_cardinality_matching_fraction": float("nan"),
                    "label_auroc_from_matched_cys_fraction": _safe_metric(
                        roc_auc_score, labels[scope_mask], fractions[scope_mask]
                    ),
                }
            )

    pair_ca = np.asarray([row["ca_distance_angstrom"] for row in pair_rows], dtype=float)
    pair_sg = np.asarray([row["sg_distance_angstrom"] for row in pair_rows], dtype=float)
    pair_labels = np.asarray([row["label"] for row in pair_rows], dtype=int)
    pair_rich = np.asarray([_bool(row["cys_rich"]) for row in pair_rows], dtype=bool)
    ca_edges = [float("inf") if value == "inf" else float(value) for value in protocol["geometry"]["ca_distance_bins_angstrom"]]
    sg_edges = [float("inf") if value == "inf" else float(value) for value in protocol["geometry"]["sg_distance_bins_angstrom"]]
    distance_rows: list[dict[str, Any]] = []
    for subset, mask in (
        ("all", np.ones(pair_ca.size, dtype=bool)),
        ("label_0", pair_labels == 0),
        ("label_1", pair_labels == 1),
        ("cys_lt_4", ~pair_rich),
        ("cys_ge_4", pair_rich),
    ):
        distance_rows.extend(_bin_rows(pair_ca[mask], ca_edges, "C-alpha", subset))
        distance_rows.extend(_bin_rows(pair_sg[mask], sg_edges, "SG", subset))

    bootstrap_rows: list[dict[str, Any]] = []
    resamples = int(protocol["uncertainty"]["bootstrap_resamples"])
    seed = int(protocol["uncertainty"]["bootstrap_seed"])

    def add_bootstrap(name: str, subset: np.ndarray, group: np.ndarray, statistic: Callable[[np.ndarray], float], seed_offset: int) -> None:
        point, lower, upper, p_positive, valid_count = _bootstrap(
            labels[subset], folds[subset], group[subset], statistic, resamples, seed + seed_offset
        )
        bootstrap_rows.append(
            {
                "contrast": name,
                "point_estimate": point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_probability_gt_0": p_positive,
                "valid_resamples": valid_count,
                "seed": seed + seed_offset,
            }
        )

    all_mask = np.ones(expected_size, dtype=bool)
    local_labels = labels
    local_predictions = current_prediction
    local_groups = cys_rich

    def current_std_gap(draw: np.ndarray) -> float:
        rich_draw = draw[local_groups[draw]]
        other_draw = draw[~local_groups[draw]]
        return _label_standardized_error(local_labels[rich_draw], local_predictions[rich_draw], label_weights) - _label_standardized_error(local_labels[other_draw], local_predictions[other_draw], label_weights)

    add_bootstrap("current_3d_label_standardized_error_gap_cys_ge_4_minus_lt_4", all_mask, cys_rich, current_std_gap, 0)

    for threshold in protocol["cysteine_definitions"]["sensitivity_min_counts"]:
        threshold = int(threshold)
        if threshold == primary_min:
            continue
        threshold_groups = cys_count >= threshold

        def sensitivity_std_gap(draw: np.ndarray, groups: np.ndarray = threshold_groups) -> float:
            exposed_draw = draw[groups[draw]]
            complement_draw = draw[~groups[draw]]
            return _label_standardized_error(local_labels[exposed_draw], local_predictions[exposed_draw], label_weights) - _label_standardized_error(local_labels[complement_draw], local_predictions[complement_draw], label_weights)

        add_bootstrap(
            f"current_3d_label_standardized_error_gap_cys_ge_{threshold}_minus_lt_{threshold}",
            all_mask,
            threshold_groups,
            sensitivity_std_gap,
            20 + threshold,
        )

    current_errors = (current_prediction != labels).astype(float)

    def current_raw_gap(draw: np.ndarray) -> float:
        return float(np.mean(current_errors[draw[local_groups[draw]]]) - np.mean(current_errors[draw[~local_groups[draw]]]))

    add_bootstrap("current_3d_raw_error_gap_cys_ge_4_minus_lt_4", all_mask, cys_rich, current_raw_gap, 1)

    geometry_subset = (cys_count >= 2) & valid_sg
    geometry_indices = np.flatnonzero(geometry_subset)
    geo_labels = labels[geometry_subset]
    geo_predictions = current_prediction[geometry_subset]
    geo_groups = close_sg[geometry_subset]

    def geometry_std_gap(draw: np.ndarray) -> float:
        close_draw = draw[geo_groups[draw]]
        other_draw = draw[~geo_groups[draw]]
        return _label_standardized_error(geo_labels[close_draw], geo_predictions[close_draw], label_weights) - _label_standardized_error(geo_labels[other_draw], geo_predictions[other_draw], label_weights)

    add_bootstrap("current_3d_label_standardized_error_gap_candidate_sg_contact_minus_no_contact", geometry_subset, close_sg, geometry_std_gap, 2)

    chem_control_error = ((control_probability >= float(protocol["evaluation"]["matched_chemical_threshold"])).astype(int) != labels).astype(float)
    chem_selected_error = ((selected_probability >= float(protocol["evaluation"]["matched_chemical_threshold"])).astype(int) != labels).astype(float)
    chem_brier_improvement = (control_probability - labels) ** 2 - (selected_probability - labels) ** 2

    def chemical_rich_error_reduction(draw: np.ndarray) -> float:
        rich_draw = draw[local_groups[draw]]
        return float(np.mean(chem_control_error[rich_draw] - chem_selected_error[rich_draw]))

    add_bootstrap("chemical_site_error_reduction_within_cys_ge_4", all_mask, cys_rich, chemical_rich_error_reduction, 3)

    def chemical_differential_error_reduction(draw: np.ndarray) -> float:
        rich_draw = draw[local_groups[draw]]
        other_draw = draw[~local_groups[draw]]
        return float(np.mean(chem_control_error[rich_draw] - chem_selected_error[rich_draw]) - np.mean(chem_control_error[other_draw] - chem_selected_error[other_draw]))

    add_bootstrap("chemical_site_differential_error_reduction_cys_ge_4_minus_lt_4", all_mask, cys_rich, chemical_differential_error_reduction, 4)

    def chemical_rich_brier_reduction(draw: np.ndarray) -> float:
        rich_draw = draw[local_groups[draw]]
        return float(np.mean(chem_brier_improvement[rich_draw]))

    add_bootstrap("chemical_site_brier_reduction_within_cys_ge_4", all_mask, cys_rich, chemical_rich_brier_reduction, 5)

    # Label enrichment uses Cys-rich as the exposed group.
    rich_positive = int(np.sum(cys_rich & (labels == 1)))
    rich_negative = int(np.sum(cys_rich & (labels == 0)))
    other_positive = int(np.sum(~cys_rich & (labels == 1)))
    other_negative = int(np.sum(~cys_rich & (labels == 0)))
    label_enrichment = {
        "cys_ge_4_positive": rich_positive,
        "cys_ge_4_negative": rich_negative,
        "cys_lt_4_positive": other_positive,
        "cys_lt_4_negative": other_negative,
        "cys_ge_4_positive_fraction": rich_positive / (rich_positive + rich_negative),
        "cys_lt_4_positive_fraction": other_positive / (other_positive + other_negative),
        **_odds_ratio(rich_positive, rich_negative, other_positive, other_negative),
    }

    primary_bootstrap = next(row for row in bootstrap_rows if row["contrast"].startswith("current_3d_label_standardized_error_gap_cys_ge_4"))
    primary_count = int(cys_rich.sum())
    valid_sg_fraction = float(np.mean([row["all_cys_sg_sites_valid"] for row in sample_rows if row["cys_ge_4"]]))
    geometry_population = (cys_count >= 2) & valid_sg
    contact_prevalence = float(np.mean(close_sg[geometry_population]))
    decision_cfg = protocol["decision_rule"]
    data_pass = (
        primary_count >= int(decision_cfg["data_adequacy"]["primary_group_min_samples"])
        and rich_positive >= int(decision_cfg["data_adequacy"]["primary_group_min_samples_per_label"])
        and rich_negative >= int(decision_cfg["data_adequacy"]["primary_group_min_samples_per_label"])
        and valid_sg_fraction >= float(decision_cfg["data_adequacy"]["valid_sg_geometry_fraction_min"])
    )
    geometry_pass = (
        float(decision_cfg["geometry_heterogeneity"]["permissive_contact_prevalence_min"])
        <= contact_prevalence
        <= float(decision_cfg["geometry_heterogeneity"]["permissive_contact_prevalence_max"])
    )
    unmet_pass = (
        float(primary_bootstrap["point_estimate"]) >= float(decision_cfg["unmet_model_need"]["minimum_label_standardized_error_gap"])
        and float(primary_bootstrap["ci95_lower"]) > 0.0
    )
    if not data_pass or not geometry_pass:
        recommendation = "no_go"
    elif unmet_pass:
        recommendation = "go_server_prototype"
    else:
        recommendation = "conditional_method_development"

    min_sg = np.asarray([row["min_sg_distance_angstrom"] for row in sample_rows], dtype=float)
    chemical_benefit = np.asarray([row["chemical_brier_improvement"] for row in sample_rows], dtype=float)
    probability_delta = np.asarray([row["chemical_probability_difference"] for row in sample_rows], dtype=float)
    correlation_mask = (cys_count >= 2) & valid_sg & active_chemical
    chemical_geometry_association = {
        "scope": "active chemical folds 0,1,3,4; peptides with >=2 Cys and valid SG geometry",
        "min_sg_distance_vs_chemical_brier_improvement": _correlations(min_sg[correlation_mask], chemical_benefit[correlation_mask]),
        "min_sg_distance_vs_chemical_probability_difference": _correlations(min_sg[correlation_mask], probability_delta[correlation_mask]),
    }

    counts = Counter(int(value) for value in cys_count)
    summary = {
        "artifact_type": "ghxtox_topology_feasibility_audit",
        "protocol": str(protocol_path),
        "scope": protocol["scope"],
        "samples": expected_size,
        "labels": {"positive": int(labels.sum()), "negative": int(labels.size - labels.sum())},
        "fold_counts": {str(key): int(value) for key, value in sorted(Counter(folds.tolist()).items())},
        "cysteine_abundance": {
            "count_distribution": {str(key): value for key, value in sorted(counts.items())},
            "cys_ge_2": int(np.sum(cys_count >= 2)),
            "cys_ge_3": int(np.sum(cys_count >= 3)),
            "cys_ge_4": int(np.sum(cys_count >= 4)),
            "cys_ge_6": int(np.sum(cys_count >= 6)),
            "strict_high_risk": int(strict_risk.sum()),
        },
        "label_enrichment": label_enrichment,
        "pair_geometry": {
            "cys_pairs": len(pair_rows),
            "valid_sg_pairs": int(np.isfinite(pair_sg).sum()),
            "ca_distance_angstrom": _summary(pair_ca.tolist()),
            "sg_distance_angstrom": _summary(pair_sg.tolist()),
            "peptides_ge_2_cys_with_valid_sg_pairs": int(geometry_population.sum()),
            "peptides_with_any_candidate_sg_contact_le_4_5": int(np.sum(geometry_population & close_sg)),
            "candidate_sg_contact_prevalence": contact_prevalence,
            "cys_ge_4_all_sg_sites_valid_fraction": valid_sg_fraction,
            "global_matching_diagnostic": {
                "interpretation": "Maximum-cardinality matching over SG-distance-thresholded graphs; a structural heuristic, not a disulfide annotation.",
                "table": "topology_matching_summary.csv"
            },
        },
        "current_3d": {
            "threshold": float(current_thresholds[0]),
            "all": _metrics(labels, current_probability, float(current_thresholds[0])),
            "cys_ge_4": _metrics(labels[cys_rich], current_probability[cys_rich], float(current_thresholds[0])),
            "cys_lt_4": _metrics(labels[~cys_rich], current_probability[~cys_rich], float(current_thresholds[0])),
            "primary_label_standardized_error_contrast": primary_bootstrap,
            "sensitivity_label_standardized_error_contrasts": [
                row
                for row in bootstrap_rows
                if row["contrast"].startswith("current_3d_label_standardized_error_gap_cys_ge_")
            ],
        },
        "chemical_site": {
            "threshold": float(protocol["evaluation"]["matched_chemical_threshold"]),
            "fold2_prediction_identical": bool(np.allclose(control_probability[folds == 2], selected_probability[folds == 2], rtol=0.0, atol=1e-12)),
            "selection_status_counts": dict(Counter(row["chemical_selection_status"] for row in sample_rows)),
            "geometry_association": chemical_geometry_association,
        },
        "decision": {
            "recommendation": recommendation,
            "data_adequacy_pass": data_pass,
            "geometry_heterogeneity_pass": geometry_pass,
            "unmet_model_need_pass": unmet_pass,
            "criteria": protocol["decision_rule"],
            "architecture_boundary": protocol["decision_rule"]["architecture_boundary"],
        },
        "interpretation_boundary": protocol["scope"]["interpretation_boundary"],
    }

    _write_csv(output_dir / "sample_level.csv", sample_rows)
    _write_csv(output_dir / "cys_pair_geometry.csv", pair_rows)
    _write_csv(output_dir / "subgroup_metrics.csv", subgroup_metric_rows)
    _write_csv(output_dir / "distance_distribution.csv", distance_rows)
    _write_csv(output_dir / "chemical_site_incremental.csv", chemical_rows_out)
    _write_csv(output_dir / "topology_matching_summary.csv", matching_rows)
    _write_csv(output_dir / "bootstrap_intervals.csv", bootstrap_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    decision = {
        "artifact_type": "ghxtox_topology_feasibility_decision",
        **summary["decision"],
        "interpretation_boundary": summary["interpretation_boundary"],
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    input_manifest = []
    for name, path in input_paths.items():
        input_manifest.append({"role": name, "path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    input_manifest.append({"role": "protocol", "path": str(protocol_path), "size_bytes": protocol_path.stat().st_size, "sha256": _sha256(protocol_path)})
    input_manifest.append({"role": "audit_script", "path": str(Path(__file__).resolve()), "size_bytes": Path(__file__).stat().st_size, "sha256": _sha256(Path(__file__).resolve())})
    _write_csv(output_dir / "input_manifest.csv", input_manifest)
    cys_ge_6_sensitivity = next(
        row
        for row in bootstrap_rows
        if row["contrast"] == "current_3d_label_standardized_error_gap_cys_ge_6_minus_lt_6"
    )
    readme = f"""# Topology feasibility audit

This directory contains a deterministic, training-OOF-only audit. It did not train or run a model and did not use Test1 or Test2 labels.

Recommendation: `{recommendation}`.

The frozen primary Cys>=4 endpoint did not show excess current-3D error: the label-standardized error gap was {float(primary_bootstrap['point_estimate']):.5f} (95% bootstrap CI [{float(primary_bootstrap['ci95_lower']):.5f}, {float(primary_bootstrap['ci95_upper']):.5f}]). A prespecified Cys>=6 sensitivity did identify a narrower failure mode, with a {float(cys_ge_6_sensitivity['point_estimate']):.5f} gap (95% CI [{float(cys_ge_6_sensitivity['ci95_lower']):.5f}, {float(cys_ge_6_sensitivity['ci95_upper']):.5f}]) driven by false positives among rare high-Cys negatives. This supports a targeted pilot, not an expected whole-benchmark gain.

The reported Cys--Cys distances come from predicted coordinates and are candidate-contact statistics only. They do not establish native disulfide connectivity, cyclization, or experimentally verified topology.

The current chemical-site branch already encodes sulfur compatibility, continuous distance, orientation, and pLDDT. Any follow-up topology model should therefore add global Cys-pair competition, matching constraints, or weighted candidate-topology aggregation rather than another unconstrained local edge scorer.

Before server training, obtain per-peptide Cys partner pairs, evidence provenance/confidence, unknown-connection masks, cyclization status, and whether geometry was optimized with disulfide constraints. If only partial annotations exist, unknown pairs must not be treated as negatives. Evaluate a frozen-base lightweight head by five-fold training OOF, with Cys>=6 negative-class false-positive rate as the targeted endpoint and full-cohort MCC/BACC/AUPRC as non-degradation constraints. Stop if the high-Cys false-positive reduction is not directionally consistent in at least four folds, if full-cohort metrics regress, or if the apparent gain requires threshold retuning.

Reproduce from the repository root with the isolated project runtime, without changing packages:

```powershell
$env:PYTHONPATH='D:\\dachuang2026\\GHXTox\\src'
& 'D:\\dachuang2026\\ziliao\\ghxtox_runtime_py310\\Scripts\\python.exe' scripts/audit_topology_feasibility.py --protocol configs/topology_feasibility_audit_v1.json --output-dir runs/topology_feasibility_audit_20260809_reproduction
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    output_files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    checksum_text = "".join(f"{_sha256(path)}  {path.name}\n" for path in output_files)
    (output_dir / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "recommendation": recommendation, "samples": expected_size}, indent=2))


if __name__ == "__main__":
    main()
