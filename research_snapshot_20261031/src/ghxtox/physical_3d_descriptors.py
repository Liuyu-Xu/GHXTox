"""Frozen source-robust physical-chemical 3D descriptor OOF probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from ghxtox.bootstrap_ci import _metrics
from ghxtox.utils import save_json


DESCRIPTOR_NAMES = (
    "rg_per_sqrt_length",
    "shape_asphericity",
    "shape_acylindricity",
    "end_to_end_ratio",
    "contact_density_8",
    "long_range_contact_fraction_8",
    "relative_contact_order_8",
    "exposed_residue_fraction",
    "exposed_hydrophobic_fraction",
    "exposed_charged_fraction",
    "largest_hydrophobic_patch_fraction",
    "largest_positive_patch_fraction",
    "largest_negative_patch_fraction",
    "hydrophobic_spatial_moment",
    "charge_spatial_moment",
    "charge_centroid_separation",
    "opposite_charge_contact_fraction",
    "cystine_contact_fraction",
    "helix_geometry_proxy_fraction",
    "extended_geometry_proxy_fraction",
)

HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.1, "D": -1.0, "E": -1.0}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _length_stratum(length: int) -> str:
    if length <= 15:
        return "11-15"
    if length <= 30:
        return "16-30"
    return "31-50"


def _generator_name(source: str, length: int) -> str:
    normalized = source.lower()
    if "pepflow" in normalized:
        return "PepFlow"
    if "colabfold" in normalized or "alphafold" in normalized or "af2" in normalized:
        return "AF2-16-30" if length <= 30 else "AF2-31-50-pilot"
    return source or "unknown"


def _largest_patch_fraction(nodes: np.ndarray, distances: np.ndarray) -> float:
    indices = np.flatnonzero(nodes)
    if indices.size == 0:
        return 0.0
    adjacency = distances[np.ix_(indices, indices)] < 8.0
    np.fill_diagonal(adjacency, False)
    visited = np.zeros(indices.size, dtype=bool)
    largest = 0
    for start in range(indices.size):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = 0
        while stack:
            current = stack.pop()
            component += 1
            neighbors = np.flatnonzero(adjacency[current] & ~visited)
            visited[neighbors] = True
            stack.extend(int(value) for value in neighbors)
        largest = max(largest, component)
    return float(largest / indices.size)


def _safe_fraction(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _conformer_descriptors(sequence: str, coords: np.ndarray) -> np.ndarray:
    length = len(sequence)
    if coords.shape != (length, 3):
        raise ValueError(f"Coordinate/sequence mismatch: {coords.shape} versus {length}.")
    centered = coords.astype(np.float64) - coords.mean(axis=0, keepdims=True)
    squared_radius = np.sum(centered**2, axis=1)
    radius_gyration = float(np.sqrt(np.mean(squared_radius)))
    scale = max(radius_gyration, 1e-8)
    tensor = centered.T @ centered / max(length, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(tensor))
    eigen_sum = max(float(eigenvalues.sum()), 1e-8)
    asphericity = float((eigenvalues[2] - 0.5 * (eigenvalues[0] + eigenvalues[1])) / eigen_sum)
    acylindricity = float((eigenvalues[1] - eigenvalues[0]) / eigen_sum)

    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    positions = np.arange(length)
    separation = np.abs(positions[:, None] - positions[None, :])
    upper = np.triu(np.ones((length, length), dtype=bool), k=1)
    nonlocal_pairs = upper & (separation >= 3)
    contacts = nonlocal_pairs & (distances < 8.0)
    contact_count = int(contacts.sum())
    possible_contacts = int(nonlocal_pairs.sum())
    long_contacts = contacts & (separation >= 6)

    neighbor_mask = (distances < 10.0) & (separation >= 3)
    exposed = neighbor_mask.sum(axis=1) <= 2
    hydrophobicity = np.asarray([HYDROPHOBICITY.get(residue, 0.0) for residue in sequence]) / 4.5
    charges = np.asarray([CHARGE.get(residue, 0.0) for residue in sequence])
    hydrophobic = hydrophobicity > 0.0
    charged = charges != 0.0
    positive = charges > 0.0
    negative = charges < 0.0
    cysteine = np.asarray(list(sequence)) == "C"

    hydro_denominator = max(float(np.abs(hydrophobicity).sum()) * scale, 1e-8)
    charge_denominator = max(float(np.abs(charges).sum()) * scale, 1e-8)
    hydrophobic_moment = float(
        np.linalg.norm(np.sum(hydrophobicity[:, None] * centered, axis=0))
        / hydro_denominator
    )
    charge_moment = float(
        np.linalg.norm(np.sum(charges[:, None] * centered, axis=0))
        / charge_denominator
    )
    if positive.any() and negative.any():
        charge_separation = float(
            np.linalg.norm(coords[positive].mean(axis=0) - coords[negative].mean(axis=0))
            / (2.0 * scale)
        )
        opposite_distances = distances[np.ix_(positive, negative)]
        opposite_contact_fraction = float(np.mean(opposite_distances < 8.0))
    else:
        charge_separation = 0.0
        opposite_contact_fraction = 0.0

    cysteine_indices = np.flatnonzero(cysteine)
    if cysteine_indices.size >= 2:
        cysteine_distances = distances[np.ix_(cysteine_indices, cysteine_indices)]
        cysteine_upper = np.triu(np.ones_like(cysteine_distances, dtype=bool), k=1)
        cystine_fraction = float(
            np.mean(cysteine_distances[cysteine_upper] < 6.5)
        )
    else:
        cystine_fraction = 0.0

    if length >= 4:
        i_i3 = np.linalg.norm(coords[3:] - coords[:-3], axis=1)
        helix_fraction = float(np.mean((i_i3 >= 4.5) & (i_i3 <= 6.5)))
    else:
        helix_fraction = 0.0
    if length >= 3:
        left = coords[:-2] - coords[1:-1]
        right = coords[2:] - coords[1:-1]
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = np.divide(
            np.sum(left * right, axis=1),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > 1e-8,
        )
        angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        extended_fraction = float(np.mean(angles >= 120.0))
    else:
        extended_fraction = 0.0

    end_to_end = float(np.linalg.norm(coords[-1] - coords[0])) if length > 1 else 0.0
    return np.asarray(
        [
            radius_gyration / np.sqrt(max(length, 1)),
            asphericity,
            acylindricity,
            _safe_fraction(end_to_end, 3.8 * max(length - 1, 1)),
            _safe_fraction(contact_count, possible_contacts),
            _safe_fraction(int(long_contacts.sum()), contact_count),
            float(np.mean(separation[contacts] / max(length - 1, 1))) if contact_count else 0.0,
            float(np.mean(exposed)),
            _safe_fraction(int(np.sum(exposed & hydrophobic)), int(hydrophobic.sum())),
            _safe_fraction(int(np.sum(exposed & charged)), int(charged.sum())),
            _largest_patch_fraction(exposed & hydrophobic, distances),
            _largest_patch_fraction(exposed & positive, distances),
            _largest_patch_fraction(exposed & negative, distances),
            hydrophobic_moment,
            charge_moment,
            charge_separation,
            opposite_contact_fraction,
            cystine_fraction,
            helix_fraction,
            extended_fraction,
        ],
        dtype=np.float64,
    )


def physical_descriptor_vector(sequence: str, aligned_coords: np.ndarray) -> np.ndarray:
    """Return the frozen 20-feature mean across cached conformers."""

    coords = np.asarray(aligned_coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected conformer coordinates [K,L,3], got {coords.shape}.")
    if coords.shape[0] == 0:
        raise ValueError("At least one conformer is required.")
    values = np.stack([_conformer_descriptors(sequence, item) for item in coords])
    result = np.nan_to_num(values.mean(axis=0), nan=0.0, posinf=5.0, neginf=-5.0)
    if result.shape != (len(DESCRIPTOR_NAMES),):
        raise AssertionError("Descriptor contract dimension changed unexpectedly.")
    return result


def _resolve_summary(row: dict[str, str], cache_root: Path) -> Path:
    cache_text = str(row.get("cache_path") or "").strip()
    if cache_text:
        portable = Path(cache_text).parent / "ensemble_summary.npz"
        if portable.exists():
            return portable
    fallback = cache_root / str(row["sample_id"]) / "ensemble_summary.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing ensemble summary for {row['sample_id']}: {fallback}")


def build_descriptor_rows(
    fold_manifest: str | Path,
    manifests: list[str | Path],
    cache_roots: list[str | Path],
) -> list[dict[str, Any]]:
    """Align unique train-only conformer summaries to the canonical five folds."""

    if len(manifests) != len(cache_roots):
        raise ValueError("Each manifest requires one matching cache root.")
    folds = {row["sample_id"]: row for row in _read_csv(fold_manifest)}
    entries: dict[str, dict[str, Any]] = {}
    for manifest, root in zip(manifests, cache_roots):
        root_path = Path(root)
        for row in _read_csv(manifest):
            sample_id = str(row.get("original_id") or row["sample_id"])
            if sample_id in entries:
                continue
            if sample_id not in folds:
                raise ValueError(f"Conformer sample is absent from canonical folds: {sample_id}")
            fold = folds[sample_id]
            sequence = str(row["sequence"])
            label = int(row["label"])
            if sequence != str(fold["sequence"]) or label != int(fold["label"]):
                raise ValueError(f"Manifest/fold mismatch for {sample_id}.")
            summary_path = _resolve_summary(row, root_path)
            entries[sample_id] = {
                "source_index": int(fold["source_index"]),
                "sample_id": sample_id,
                "fold": int(fold["fold"]),
                "label": label,
                "sequence": sequence,
                "length": len(sequence),
                "length_stratum": _length_stratum(len(sequence)),
                "generator_source": _generator_name(str(row.get("source") or ""), len(sequence)),
                "summary_path": str(summary_path),
            }
    rows: list[dict[str, Any]] = []
    for entry in sorted(entries.values(), key=lambda item: int(item["source_index"])):
        with np.load(entry["summary_path"]) as payload:
            descriptors = physical_descriptor_vector(
                str(entry["sequence"]), payload["aligned_coords"]
            )
            conformers = int(payload["aligned_coords"].shape[0])
        rows.append(
            {
                **entry,
                "conformers": conformers,
                **{
                    name: float(value)
                    for name, value in zip(DESCRIPTOR_NAMES, descriptors)
                },
            }
        )
    return rows


def _source_standardize(
    train: np.ndarray,
    train_sources: np.ndarray,
    values: np.ndarray,
    value_sources: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    global_scaler = StandardScaler().fit(train)
    for source in np.unique(value_sources):
        target_mask = value_sources == source
        source_train = train[train_sources == source]
        scaler = StandardScaler().fit(source_train) if len(source_train) >= 2 else global_scaler
        output[target_mask] = scaler.transform(values[target_mask])
    return np.clip(output, -5.0, 5.0)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _cohort_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    reference = np.asarray([float(row["reference_probability"]) for row in rows])
    candidate = np.asarray([float(row["candidate_probability"]) for row in rows])
    return {
        "samples": len(rows),
        "reference": _metrics(labels, reference, threshold),
        "candidate": _metrics(labels, candidate, threshold),
    }


def evaluate_candidate(
    reference_rows: list[dict[str, str]],
    descriptor_rows: list[dict[str, Any]],
    *,
    threshold: float,
    residual_weight: float,
    regularization_c: float,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the frozen fold-local descriptor residual to all reference rows."""

    descriptors = {int(row["source_index"]): row for row in descriptor_rows}
    reference = {int(row["source_index"]): row for row in reference_rows}
    if set(descriptors) - set(reference):
        raise ValueError("At least one descriptor row is absent from the reference OOF.")
    ordered_indices = sorted(reference)
    candidate_probability = {
        index: float(reference[index]["toxicity_probability"]) for index in ordered_indices
    }
    descriptor_probability: dict[int, float] = {}
    residual_delta: dict[int, float] = {}
    source_true: list[str] = []
    source_raw_prediction: list[str] = []
    source_normalized_prediction: list[str] = []

    for held_fold in range(5):
        train_rows = [row for row in descriptor_rows if int(row["fold"]) != held_fold]
        held_rows = [row for row in descriptor_rows if int(row["fold"]) == held_fold]
        if not train_rows or not held_rows:
            raise ValueError(f"Missing descriptor rows for outer fold {held_fold}.")
        train_x = np.asarray(
            [[float(row[name]) for name in DESCRIPTOR_NAMES] for row in train_rows]
        )
        held_x = np.asarray(
            [[float(row[name]) for name in DESCRIPTOR_NAMES] for row in held_rows]
        )
        train_y = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
        train_sources = np.asarray([str(row["generator_source"]) for row in train_rows])
        held_sources = np.asarray([str(row["generator_source"]) for row in held_rows])
        train_normalized = _source_standardize(
            train_x, train_sources, train_x, train_sources
        )
        held_normalized = _source_standardize(
            train_x, train_sources, held_x, held_sources
        )
        classifier = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_seed,
        ).fit(train_normalized, train_y)
        physical_probability = classifier.predict_proba(held_normalized)[:, 1]
        delta = residual_weight * np.clip(_logit(physical_probability), -1.0, 1.0)
        held_reference = np.asarray(
            [float(reference[int(row["source_index"])]["toxicity_probability"]) for row in held_rows]
        )
        held_candidate = _sigmoid(_logit(held_reference) + delta)
        for row, phys, change, value in zip(
            held_rows, physical_probability, delta, held_candidate
        ):
            index = int(row["source_index"])
            descriptor_probability[index] = float(phys)
            residual_delta[index] = float(change)
            candidate_probability[index] = float(value)

        raw_scaler = StandardScaler().fit(train_x)
        raw_probe = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_seed,
        ).fit(raw_scaler.transform(train_x), train_sources)
        normalized_probe = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_seed,
        ).fit(train_normalized, train_sources)
        source_true.extend(held_sources.tolist())
        source_raw_prediction.extend(raw_probe.predict(raw_scaler.transform(held_x)).tolist())
        source_normalized_prediction.extend(
            normalized_probe.predict(held_normalized).tolist()
        )

    output_rows: list[dict[str, Any]] = []
    for index in ordered_indices:
        base = reference[index]
        descriptor = descriptors.get(index)
        probability = float(candidate_probability[index])
        output_rows.append(
            {
                "source_index": index,
                "sample_id": base["sample_id"],
                "fold": int(base["fold"]),
                "label": int(base["label"]),
                "sequence": base["sequence"],
                "length": int(base["length"]),
                "length_stratum": base["length_stratum"],
                "generator_source": base["generator_source"],
                "conformer_available": int(descriptor is not None),
                "reference_probability": float(base["toxicity_probability"]),
                "descriptor_probability": descriptor_probability.get(index, ""),
                "residual_logit_delta": residual_delta.get(index, 0.0),
                "candidate_probability": probability,
                "reference_prediction": int(float(base["toxicity_probability"]) >= threshold),
                "candidate_prediction": int(probability >= threshold),
            }
        )

    overall = _cohort_metrics(output_rows, threshold)
    by_fold = {
        str(fold): _cohort_metrics(
            [row for row in output_rows if int(row["fold"]) == fold], threshold
        )
        for fold in range(5)
    }
    by_length = {}
    for stratum in ("11-15", "16-30", "31-50"):
        selected = [
            row for row in output_rows if row["length_stratum"] == stratum
        ]
        if selected:
            by_length[stratum] = _cohort_metrics(selected, threshold)
    sources = sorted({str(row["generator_source"]) for row in output_rows})
    by_source = {
        source: _cohort_metrics(
            [row for row in output_rows if row["generator_source"] == source], threshold
        )
        for source in sources
    }
    reference_metrics = overall["reference"]
    candidate_metrics = overall["candidate"]
    fold_non_decrease = sum(
        values["candidate"]["mcc"] >= values["reference"]["mcc"]
        for values in by_fold.values()
    )
    cohort_tolerances = [*by_length.values(), *by_source.values()]
    gate = {
        "mcc_strictly_higher": candidate_metrics["mcc"] > reference_metrics["mcc"],
        "auprc_at_least_reference": candidate_metrics["auprc"] >= reference_metrics["auprc"],
        "at_least_three_fold_mcc_non_decreases": fold_non_decrease >= 3,
        "brier_within_tolerance": candidate_metrics["brier"] <= reference_metrics["brier"] + 0.001,
        "ece_10_within_tolerance": candidate_metrics["ece_10"] <= reference_metrics["ece_10"] + 0.002,
        "all_length_source_mcc_within_tolerance": all(
            value["candidate"]["mcc"] >= value["reference"]["mcc"] - 0.005
            for value in cohort_tolerances
        ),
        "all_length_source_auprc_within_tolerance": all(
            value["candidate"]["auprc"] >= value["reference"]["auprc"] - 0.002
            for value in cohort_tolerances
        ),
    }
    gate["all_conditions_passed"] = all(gate.values())
    transitions = {
        "corrected_reference_errors": sum(
            row["reference_prediction"] != row["label"]
            and row["candidate_prediction"] == row["label"]
            for row in output_rows
        ),
        "regressed_reference_correct": sum(
            row["reference_prediction"] == row["label"]
            and row["candidate_prediction"] != row["label"]
            for row in output_rows
        ),
        "decision_disagreements": sum(
            row["reference_prediction"] != row["candidate_prediction"]
            for row in output_rows
        ),
    }
    transitions["net_corrections"] = (
        transitions["corrected_reference_errors"]
        - transitions["regressed_reference_correct"]
    )
    report = {
        "status": "complete",
        "external_test_labels_read": False,
        "descriptor_count": len(DESCRIPTOR_NAMES),
        "overall": overall,
        "folds": by_fold,
        "length_strata": by_length,
        "generator_sources": by_source,
        "source_probe": {
            "raw_descriptor_balanced_accuracy": float(
                balanced_accuracy_score(source_true, source_raw_prediction)
            ),
            "source_standardized_balanced_accuracy": float(
                balanced_accuracy_score(source_true, source_normalized_prediction)
            ),
        },
        "fold_mcc_non_decrease_count": int(fold_non_decrease),
        "decision_transitions": transitions,
        "gate": gate,
        "decision": (
            "promote_physical_3d_descriptor_residual"
            if gate["all_conditions_passed"]
            else "retain_phase25_3d_handoff"
        ),
    }
    return report, output_rows


def run_protocol(protocol_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Execute the immutable Phase 27 protocol and write compact evidence."""

    protocol_file = Path(protocol_path)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 27 protocol must be frozen before evaluation.")
    inputs = protocol["inputs"]
    candidate = protocol["outer_oof_candidate"]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    descriptor_rows = build_descriptor_rows(
        inputs["fold_manifest"], inputs["manifests"], inputs["cache_roots"]
    )
    if len(descriptor_rows) != int(protocol["reference"]["conformer_available"]):
        raise ValueError("Descriptor coverage differs from the frozen protocol.")
    descriptor_path = output / "descriptors.csv"
    _write_csv(descriptor_rows, descriptor_path)
    report, rows = evaluate_candidate(
        _read_csv(protocol["reference"]["artifact"]),
        descriptor_rows,
        threshold=float(candidate["threshold"]),
        residual_weight=float(candidate["residual_weight"]),
        regularization_c=float(candidate["C"]),
        random_seed=int(candidate["random_seed"]),
    )
    prediction_path = output / "oof_predictions.csv"
    _write_csv(rows, prediction_path)
    report["provenance"] = {
        "protocol": str(protocol_file),
        "protocol_sha256": _sha256(protocol_file),
        "reference_oof": protocol["reference"]["artifact"],
        "reference_oof_sha256": _sha256(protocol["reference"]["artifact"]),
        "descriptors_sha256": _sha256(descriptor_path),
        "oof_predictions_sha256": _sha256(prediction_path),
    }
    save_json(report, output / "summary.json")
    save_json(
        {
            "status": "complete",
            "external_test_labels_read": False,
            "decision": report["decision"],
            "promotion_authorized": bool(report["gate"]["all_conditions_passed"]),
            "handoff_regeneration_authorized": bool(
                report["gate"]["all_conditions_passed"]
            ),
            "gate": report["gate"],
        },
        output / "decision.json",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", default="runs/physical_3d_descriptors/protocol.json"
    )
    parser.add_argument(
        "--output-root", default="runs/physical_3d_descriptors"
    )
    args = parser.parse_args()
    report = run_protocol(args.protocol, args.output_root)
    print(json.dumps({"decision": report["decision"], "gate": report["gate"]}))


if __name__ == "__main__":
    main()
