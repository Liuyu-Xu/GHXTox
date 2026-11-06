"""Frozen all-atom surface-chemistry specialist for PepFlow short peptides."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ghxtox.bootstrap_ci import _metrics
from ghxtox.utils import save_json


SURFACE_DESCRIPTOR_NAMES = (
    "total_sasa_per_residue_mean",
    "hydrophobic_sasa_fraction_mean",
    "charged_sasa_fraction_mean",
    "charge_sasa_imbalance_mean",
    "sulfur_sasa_fraction_mean",
    "backbone_sasa_fraction_mean",
    "largest_hydrophobic_surface_patch_fraction_mean",
    "largest_positive_surface_patch_fraction_mean",
    "largest_negative_surface_patch_fraction_mean",
    "sasa_weighted_hydrophobic_moment_mean",
    "sasa_weighted_charge_moment_mean",
    "total_sasa_coefficient_of_variation",
)

MAX_RESIDUE_SASA = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "Q": 225.0,
    "E": 223.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}
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
HYDROPHOBIC_RESIDUES = set("AVILMFWYC")
POSITIVE_RESIDUES = set("KRH")
NEGATIVE_RESIDUES = set("DE")
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


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


def _one_letter(resname: str) -> str:
    key = resname.strip().upper()
    value = protein_letters_3to1.get(key)
    if value is None:
        value = protein_letters_3to1.get(key.title())
    if value is None:
        raise ValueError(f"Unsupported residue in surface PDB: {resname}")
    return value.upper()


def _remove_hydrogens(model: Any) -> int:
    hydrogens = [
        atom
        for atom in model.get_atoms()
        if str(atom.element or "").strip().upper() == "H"
    ]
    for atom in hydrogens:
        atom.get_parent().detach_child(atom.id)
    return len(hydrogens)


def _largest_weighted_patch(
    nodes: np.ndarray,
    centroids: np.ndarray,
    weights: np.ndarray,
    total_sasa: float,
) -> float:
    indices = np.flatnonzero(nodes)
    if indices.size == 0 or total_sasa <= 0.0:
        return 0.0
    distances = np.linalg.norm(
        centroids[indices, None, :] - centroids[None, indices, :], axis=-1
    )
    adjacency = distances < 8.0
    np.fill_diagonal(adjacency, False)
    visited = np.zeros(indices.size, dtype=bool)
    largest_weight = 0.0
    for start in range(indices.size):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component_weight = 0.0
        while stack:
            current = stack.pop()
            component_weight += float(weights[indices[current]])
            neighbors = np.flatnonzero(adjacency[current] & ~visited)
            visited[neighbors] = True
            stack.extend(int(value) for value in neighbors)
        largest_weight = max(largest_weight, component_weight)
    return float(largest_weight / total_sasa)


def _weighted_spatial_moment(
    centroids: np.ndarray,
    residue_sasa: np.ndarray,
    property_values: np.ndarray,
) -> float:
    total_sasa = max(float(residue_sasa.sum()), 1e-8)
    center = np.sum(residue_sasa[:, None] * centroids, axis=0) / total_sasa
    centered = centroids - center
    radius = np.sqrt(
        np.sum(residue_sasa * np.sum(centered**2, axis=1)) / total_sasa
    )
    weights = residue_sasa * property_values
    denominator = max(float(np.abs(weights).sum()) * float(radius), 1e-8)
    return float(np.linalg.norm(np.sum(weights[:, None] * centered, axis=0)) / denominator)


def _model_surface_descriptors(
    model: Any,
    expected_sequence: str,
    *,
    sphere_points: int,
    probe_radius: float,
    relative_exposure_threshold: float,
) -> tuple[np.ndarray, int]:
    hydrogens_removed = _remove_hydrogens(model)
    ShrakeRupley(
        probe_radius=float(probe_radius), n_points=int(sphere_points)
    ).compute(model, level="A")
    residues = [residue for residue in model.get_residues() if residue.id[0] == " "]
    sequence = "".join(_one_letter(residue.resname) for residue in residues)
    if sequence != expected_sequence:
        raise ValueError(
            f"Surface PDB sequence mismatch: expected={expected_sequence}, parsed={sequence}"
        )

    residue_sasa = []
    centroids = []
    sulfur_sasa = 0.0
    backbone_sasa = 0.0
    for residue in residues:
        atoms = list(residue.get_atoms())
        if not atoms:
            raise ValueError(f"Residue without heavy atoms: {residue}")
        residue_sasa.append(sum(float(getattr(atom, "sasa", 0.0)) for atom in atoms))
        centroids.append(np.mean([atom.coord for atom in atoms], axis=0))
        for atom in atoms:
            atom_sasa = float(getattr(atom, "sasa", 0.0))
            if str(atom.element or "").strip().upper() == "S":
                sulfur_sasa += atom_sasa
            if str(atom.name).strip().upper() in BACKBONE_ATOMS:
                backbone_sasa += atom_sasa

    sasa = np.asarray(residue_sasa, dtype=np.float64)
    coords = np.asarray(centroids, dtype=np.float64)
    total_sasa = max(float(sasa.sum()), 1e-8)
    letters = np.asarray(list(sequence))
    hydrophobic = np.asarray([residue in HYDROPHOBIC_RESIDUES for residue in sequence])
    positive = np.asarray([residue in POSITIVE_RESIDUES for residue in sequence])
    negative = np.asarray([residue in NEGATIVE_RESIDUES for residue in sequence])
    charged = positive | negative
    max_sasa = np.asarray([MAX_RESIDUE_SASA[residue] for residue in sequence])
    exposed = sasa / max_sasa >= float(relative_exposure_threshold)
    hydrophobicity = np.asarray([HYDROPHOBICITY[residue] for residue in sequence]) / 4.5
    charges = np.asarray([CHARGE.get(residue, 0.0) for residue in sequence])
    positive_sasa = float(sasa[positive].sum())
    negative_sasa = float(sasa[negative].sum())
    charged_sasa = positive_sasa + negative_sasa
    charge_imbalance = (
        abs(positive_sasa - negative_sasa) / charged_sasa
        if charged_sasa > 0.0
        else 0.0
    )

    values = np.asarray(
        [
            total_sasa / len(sequence),
            float(sasa[hydrophobic].sum() / total_sasa),
            float(sasa[charged].sum() / total_sasa),
            charge_imbalance,
            sulfur_sasa / total_sasa,
            backbone_sasa / total_sasa,
            _largest_weighted_patch(
                exposed & hydrophobic, coords, sasa, total_sasa
            ),
            _largest_weighted_patch(exposed & positive, coords, sasa, total_sasa),
            _largest_weighted_patch(exposed & negative, coords, sasa, total_sasa),
            _weighted_spatial_moment(coords, sasa, hydrophobicity),
            _weighted_spatial_moment(coords, sasa, charges),
        ],
        dtype=np.float64,
    )
    if letters.shape[0] != len(sequence):
        raise AssertionError("Unexpected residue array shape.")
    return values, hydrogens_removed


def surface_descriptor_vector(
    pdb_path: str | Path,
    expected_sequence: str,
    *,
    sphere_points: int = 100,
    probe_radius: float = 1.4,
    relative_exposure_threshold: float = 0.25,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return the frozen 12-feature heavy-atom surface descriptor vector."""

    structure = PDBParser(QUIET=True).get_structure(Path(pdb_path).stem, pdb_path)
    per_model = []
    hydrogens_removed = 0
    for model in structure.get_models():
        values, removed = _model_surface_descriptors(
            model,
            expected_sequence,
            sphere_points=sphere_points,
            probe_radius=probe_radius,
            relative_exposure_threshold=relative_exposure_threshold,
        )
        per_model.append(values)
        hydrogens_removed += removed
    if not per_model:
        raise ValueError(f"PDB contains no models: {pdb_path}")
    stacked = np.stack(per_model)
    total_sasa_values = stacked[:, 0]
    coefficient_variation = float(
        total_sasa_values.std() / max(float(total_sasa_values.mean()), 1e-8)
    )
    result = np.concatenate([stacked.mean(axis=0), [coefficient_variation]])
    result = np.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)
    if result.shape != (len(SURFACE_DESCRIPTOR_NAMES),):
        raise AssertionError("Surface descriptor contract dimension changed.")
    return result, {
        "models": len(per_model),
        "hydrogens_removed": int(hydrogens_removed),
    }


def _resolve_pdb(row: dict[str, str], pdb_root: Path) -> Path:
    source_text = str(row.get("source_pdb") or "").strip()
    if source_text and Path(source_text).exists():
        return Path(source_text)
    fallback = pdb_root / str(row["sample_id"]) / f"{row['sequence']}.pdb"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing surface PDB for {row['sample_id']}: {fallback}")


def build_surface_rows(
    fold_manifest: str | Path,
    manifests: list[str | Path],
    pdb_roots: list[str | Path],
    *,
    sphere_points: int,
    probe_radius: float,
    relative_exposure_threshold: float,
) -> list[dict[str, Any]]:
    if len(manifests) != len(pdb_roots):
        raise ValueError("Each surface manifest requires one PDB root.")
    folds = {row["sample_id"]: row for row in _read_csv(fold_manifest)}
    entries: dict[str, dict[str, Any]] = {}
    for manifest, root in zip(manifests, pdb_roots):
        root_path = Path(root)
        for row in _read_csv(manifest):
            sample_id = str(row.get("original_id") or row["sample_id"])
            if sample_id in entries:
                continue
            fold = folds.get(sample_id)
            if fold is None:
                raise ValueError(f"Surface sample is absent from canonical folds: {sample_id}")
            sequence = str(row["sequence"])
            label = int(row["label"])
            if sequence != str(fold["sequence"]) or label != int(fold["label"]):
                raise ValueError(f"Surface manifest/fold mismatch for {sample_id}.")
            if not 11 <= len(sequence) <= 15:
                raise ValueError(f"Non-short row entered Phase 28: {sample_id}")
            entries[sample_id] = {
                "source_index": int(fold["source_index"]),
                "sample_id": sample_id,
                "fold": int(fold["fold"]),
                "label": label,
                "sequence": sequence,
                "length": len(sequence),
                "length_stratum": "11-15",
                "generator_source": "PepFlow",
                "pdb_path": str(_resolve_pdb(row, root_path)),
            }
    output = []
    for entry in sorted(entries.values(), key=lambda item: int(item["source_index"])):
        descriptors, metadata = surface_descriptor_vector(
            entry["pdb_path"],
            str(entry["sequence"]),
            sphere_points=sphere_points,
            probe_radius=probe_radius,
            relative_exposure_threshold=relative_exposure_threshold,
        )
        output.append(
            {
                **entry,
                **metadata,
                **{
                    name: float(value)
                    for name, value in zip(SURFACE_DESCRIPTOR_NAMES, descriptors)
                },
            }
        )
    return output


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


def evaluate_surface_candidate(
    reference_rows: list[dict[str, str]],
    surface_rows: list[dict[str, Any]],
    *,
    threshold: float,
    residual_weight: float,
    regularization_c: float,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference = {int(row["source_index"]): row for row in reference_rows}
    surfaces = {int(row["source_index"]): row for row in surface_rows}
    if set(surfaces) - set(reference):
        raise ValueError("At least one surface row is absent from reference OOF.")
    candidate_probability = {
        index: float(row["toxicity_probability"]) for index, row in reference.items()
    }
    surface_probability: dict[int, float] = {}
    residual_delta: dict[int, float] = {}
    for held_fold in range(5):
        train_rows = [row for row in surface_rows if int(row["fold"]) != held_fold]
        held_rows = [row for row in surface_rows if int(row["fold"]) == held_fold]
        train_x = np.asarray(
            [
                [float(row[name]) for name in SURFACE_DESCRIPTOR_NAMES]
                for row in train_rows
            ]
        )
        held_x = np.asarray(
            [
                [float(row[name]) for name in SURFACE_DESCRIPTOR_NAMES]
                for row in held_rows
            ]
        )
        train_y = np.asarray([int(row["label"]) for row in train_rows])
        scaler = StandardScaler().fit(train_x)
        train_x = np.clip(scaler.transform(train_x), -5.0, 5.0)
        held_x = np.clip(scaler.transform(held_x), -5.0, 5.0)
        model = LogisticRegression(
            C=regularization_c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_seed,
        ).fit(train_x, train_y)
        probability = model.predict_proba(held_x)[:, 1]
        delta = residual_weight * np.clip(_logit(probability), -1.0, 1.0)
        base_probability = np.asarray(
            [
                float(reference[int(row["source_index"])]["toxicity_probability"])
                for row in held_rows
            ]
        )
        changed = _sigmoid(_logit(base_probability) + delta)
        for row, phys, shift, value in zip(held_rows, probability, delta, changed):
            index = int(row["source_index"])
            surface_probability[index] = float(phys)
            residual_delta[index] = float(shift)
            candidate_probability[index] = float(value)

    output_rows = []
    for index in sorted(reference):
        base = reference[index]
        is_target = index in surfaces
        probability = candidate_probability[index]
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
                "surface_specialist_available": int(is_target),
                "reference_probability": float(base["toxicity_probability"]),
                "surface_probability": surface_probability.get(index, ""),
                "residual_logit_delta": residual_delta.get(index, 0.0),
                "candidate_probability": probability,
                "reference_prediction": int(
                    float(base["toxicity_probability"]) >= threshold
                ),
                "candidate_prediction": int(probability >= threshold),
            }
        )

    overall = _cohort_metrics(output_rows, threshold)
    short_rows = [row for row in output_rows if row["surface_specialist_available"]]
    short = _cohort_metrics(short_rows, threshold)
    folds = {
        str(fold): _cohort_metrics(
            [row for row in output_rows if int(row["fold"]) == fold], threshold
        )
        for fold in range(5)
    }
    corrected = sum(
        row["reference_prediction"] != row["label"]
        and row["candidate_prediction"] == row["label"]
        for row in output_rows
    )
    regressed = sum(
        row["reference_prediction"] == row["label"]
        and row["candidate_prediction"] != row["label"]
        for row in output_rows
    )
    fold_non_decrease = sum(
        value["candidate"]["mcc"] >= value["reference"]["mcc"]
        for value in folds.values()
    )
    gate = {
        "overall_mcc_strictly_higher": (
            overall["candidate"]["mcc"] > overall["reference"]["mcc"]
        ),
        "overall_auprc_at_least_reference": (
            overall["candidate"]["auprc"] >= overall["reference"]["auprc"]
        ),
        "short_mcc_strictly_higher": (
            short["candidate"]["mcc"] > short["reference"]["mcc"]
        ),
        "short_auprc_at_least_reference": (
            short["candidate"]["auprc"] >= short["reference"]["auprc"]
        ),
        "at_least_three_fold_mcc_non_decreases": fold_non_decrease >= 3,
        "brier_within_tolerance": (
            overall["candidate"]["brier"] <= overall["reference"]["brier"] + 0.001
        ),
        "ece_10_within_tolerance": (
            overall["candidate"]["ece_10"] <= overall["reference"]["ece_10"] + 0.002
        ),
        "net_decision_corrections_strictly_positive": corrected > regressed,
    }
    gate["all_conditions_passed"] = all(gate.values())
    report = {
        "status": "complete",
        "external_test_labels_read": False,
        "descriptor_count": len(SURFACE_DESCRIPTOR_NAMES),
        "target_rows": len(surface_rows),
        "fallback_rows": len(output_rows) - len(surface_rows),
        "overall": overall,
        "short_pepflow": short,
        "folds": folds,
        "fold_mcc_non_decrease_count": int(fold_non_decrease),
        "decision_transitions": {
            "decision_disagreements": sum(
                row["reference_prediction"] != row["candidate_prediction"]
                for row in output_rows
            ),
            "corrected_reference_errors": int(corrected),
            "regressed_reference_correct": int(regressed),
            "net_corrections": int(corrected - regressed),
        },
        "gate": gate,
        "decision": (
            "promote_short_surface_3d_specialist"
            if gate["all_conditions_passed"]
            else "retain_phase25_3d_handoff"
        ),
    }
    return report, output_rows


def run_protocol(protocol_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    protocol_file = Path(protocol_path)
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 28 protocol must be frozen before evaluation.")
    inputs = protocol["inputs"]
    sasa = protocol["sasa_contract"]
    candidate = protocol["outer_oof_candidate"]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    surface_rows = build_surface_rows(
        inputs["fold_manifest"],
        inputs["manifests"],
        inputs["pdb_roots"],
        sphere_points=int(sasa["sphere_points"]),
        probe_radius=float(sasa["probe_radius_angstrom"]),
        relative_exposure_threshold=float(sasa["relative_exposure_threshold"]),
    )
    if len(surface_rows) != int(inputs["expected_unique_rows"]):
        raise ValueError("Surface coverage differs from frozen protocol.")
    if {int(row["models"]) for row in surface_rows} != {
        int(inputs["expected_models_per_row"])
    }:
        raise ValueError("Surface model count differs from frozen protocol.")
    descriptor_path = output / "surface_descriptors.csv"
    _write_csv(surface_rows, descriptor_path)
    report, prediction_rows = evaluate_surface_candidate(
        _read_csv(protocol["reference"]["artifact"]),
        surface_rows,
        threshold=float(candidate["threshold"]),
        residual_weight=float(candidate["residual_weight"]),
        regularization_c=float(candidate["C"]),
        random_seed=int(candidate["random_seed"]),
    )
    prediction_path = output / "oof_predictions.csv"
    _write_csv(prediction_rows, prediction_path)
    report["provenance"] = {
        "protocol": str(protocol_file),
        "protocol_sha256": _sha256(protocol_file),
        "reference_oof": protocol["reference"]["artifact"],
        "reference_oof_sha256": _sha256(protocol["reference"]["artifact"]),
        "surface_descriptors_sha256": _sha256(descriptor_path),
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
        "--protocol", default="runs/surface_3d_short_specialist/protocol.json"
    )
    parser.add_argument(
        "--output-root", default="runs/surface_3d_short_specialist"
    )
    args = parser.parse_args()
    report = run_protocol(args.protocol, args.output_root)
    print(json.dumps({"decision": report["decision"], "gate": report["gate"]}))


if __name__ == "__main__":
    main()
