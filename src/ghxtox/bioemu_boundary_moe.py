"""Cross-fitted BioEmu boundary mixture-of-experts adapter.

This exploratory screen trains a pooled-ProtT5 logistic specialist only on the
frozen BioEmu decision-boundary cohort.  Three-way 1D/3D consensus rows use a
protected boundary-1D residual.  Every other boundary row uses a fold-pure,
quality-conditioned gate that estimates whether applying the independent
BioEmu evidence is beneficial.  The routes are mutually exclusive and their
absolute residual remains bounded by 0.15.

The rule was formed after OOF error analysis.  It is the promoted research
default requested by the project owner after passing the predefined retention
gate, but it is not an independent generalisation estimate and not yet a
production-inference checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ghxtox._legacy.plm_textcnn import _probability_metrics
from ghxtox.utils import save_json


EPSILON = 1e-7


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _probability(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"Invalid probability values in column {column!r}.")
    return values


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def _align_inputs(
    one_d_rows: list[dict[str, str]],
    default_rows: list[dict[str, str]],
    phase33_rows: list[dict[str, str]],
) -> dict[str, np.ndarray | list[str]]:
    if not len(one_d_rows) == len(default_rows) == len(phase33_rows):
        raise ValueError("1D, default and BioEmu-BR3D OOF files have different row counts.")
    result: dict[str, list[Any]] = {
        "source_indices": [],
        "sample_ids": [],
        "sequences": [],
        "labels": [],
        "folds": [],
        "boundary": [],
    }
    for position, (one_d, default, phase33) in enumerate(
        zip(one_d_rows, default_rows, phase33_rows, strict=True)
    ):
        indices = [int(row.get("source_index", position)) for row in (one_d, default, phase33)]
        if indices != [position, position, position]:
            raise ValueError(f"source_index mismatch at row {position}: {indices}")
        sequences = [row["sequence"] for row in (one_d, default, phase33)]
        labels = [int(row["label"]) for row in (one_d, default, phase33)]
        if len(set(sequences)) != 1 or len(set(labels)) != 1:
            raise ValueError(f"Prediction alignment mismatch at source_index {position}.")
        fold = int(phase33["fold"])
        if default.get("fold") not in (None, "") and int(default["fold"]) != fold:
            raise ValueError(f"Fold mismatch at source_index {position}.")
        if fold not in range(5):
            raise ValueError(f"Invalid fold {fold} at source_index {position}.")
        result["source_indices"].append(position)
        result["sample_ids"].append(
            str(phase33.get("sample_id", one_d.get("sample_id", position)))
        )
        result["sequences"].append(sequences[0])
        result["labels"].append(labels[0])
        result["folds"].append(fold)
        result["boundary"].append(bool(int(phase33["phase33_boundary_member"])))
    return {
        "source_indices": np.asarray(result["source_indices"], dtype=np.int64),
        "sample_ids": result["sample_ids"],
        "sequences": result["sequences"],
        "labels": np.asarray(result["labels"], dtype=np.int64),
        "folds": np.asarray(result["folds"], dtype=np.int64),
        "boundary": np.asarray(result["boundary"], dtype=bool),
    }


def _decision_transitions(
    labels: np.ndarray,
    reference_probability: np.ndarray,
    candidate_probability: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    reference = reference_probability >= threshold
    candidate = candidate_probability >= threshold
    changed = reference != candidate
    corrected = changed & (candidate == labels)
    regressed = changed & (reference == labels)
    return {
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "net": int(corrected.sum() - regressed.sum()),
    }


def apply_consensus_residual(
    default_probability: np.ndarray,
    specialist_probability: np.ndarray,
    active_mask: np.ndarray,
    *,
    residual_bound: float = 0.15,
) -> np.ndarray:
    """Apply a signed, bounded specialist residual to active rows only."""

    default = np.asarray(default_probability, dtype=np.float64)
    specialist = np.asarray(specialist_probability, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if not default.shape == specialist.shape == active.shape:
        raise ValueError("Consensus residual inputs must have identical shapes.")
    if not 0.0 <= float(residual_bound) <= 1.0:
        raise ValueError("residual_bound must lie in [0, 1].")
    residual = float(residual_bound) * np.tanh(_logit(specialist))
    probability = _sigmoid(_logit(default) + active.astype(np.float64) * residual)
    probability[~active] = default[~active]
    return probability


def apply_boundary_mixture(
    default_probability: np.ndarray,
    specialist_probability: np.ndarray,
    bioemu_probability: np.ndarray,
    benefit_gate: np.ndarray,
    boundary_mask: np.ndarray,
    consensus_mask: np.ndarray,
    *,
    residual_bound: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mutually exclusive consensus or soft BioEmu boundary evidence."""

    arrays = [
        np.asarray(default_probability, dtype=np.float64),
        np.asarray(specialist_probability, dtype=np.float64),
        np.asarray(bioemu_probability, dtype=np.float64),
        np.asarray(benefit_gate, dtype=np.float64),
    ]
    boundary = np.asarray(boundary_mask, dtype=bool)
    consensus = np.asarray(consensus_mask, dtype=bool)
    if not all(array.shape == arrays[0].shape for array in arrays[1:]):
        raise ValueError("Boundary mixture probability inputs must have identical shapes.")
    if boundary.shape != arrays[0].shape or consensus.shape != arrays[0].shape:
        raise ValueError("Boundary mixture masks must match the probability inputs.")
    if np.any(consensus & ~boundary):
        raise ValueError("Consensus rows must be a subset of boundary rows.")
    if np.any((arrays[3] < 0.0) | (arrays[3] > 1.0)):
        raise ValueError("Benefit-gate probabilities must lie in [0, 1].")
    if not 0.0 <= float(residual_bound) <= 1.0:
        raise ValueError("residual_bound must lie in [0, 1].")

    soft_evidence = arrays[3] * np.tanh(_logit(arrays[2]))
    consensus_evidence = np.tanh(_logit(arrays[1]))
    evidence = np.where(consensus, consensus_evidence, soft_evidence)
    evidence[~boundary] = 0.0
    probability = _sigmoid(_logit(arrays[0]) + float(residual_bound) * evidence)
    probability[~boundary] = arrays[0][~boundary]
    return probability, evidence


def _load_features(path: str | Path, expected_sequences: list[str]) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    features = payload.get("features")
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("ProtT5 payload must contain a rank-2 'features' tensor.")
    sequences = list(payload.get("sequences", []))
    if sequences and sequences != expected_sequences:
        raise ValueError("ProtT5 feature sequences do not align with OOF predictions.")
    if len(features) != len(expected_sequences):
        raise ValueError("ProtT5 features do not cover every OOF row.")
    values = features.float().numpy()
    if not np.isfinite(values).all():
        raise ValueError("ProtT5 features contain non-finite values.")
    return values


def _cross_fitted_specialist(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    boundary: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = np.full(len(labels), 0.5, dtype=np.float64)
    reports: list[dict[str, Any]] = []
    for fold in range(5):
        train = (folds != fold) & boundary
        held = (folds == fold) & boundary
        if not np.any(train) or len(np.unique(labels[train])) != 2:
            raise ValueError(f"Fold {fold} boundary training rows do not contain both labels.")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train])
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        )
        model.fit(train_features, labels[train])
        if np.any(held):
            probability[held] = model.predict_proba(scaler.transform(features[held]))[:, 1]
        reports.append(
            {
                "fold": fold,
                "boundary_train_rows": int(train.sum()),
                "boundary_held_rows": int(held.sum()),
            }
        )
    return probability, reports


def _boundary_probability(
    rows: list[dict[str, str]], column: str, boundary: np.ndarray
) -> np.ndarray:
    values = np.full(len(rows), 0.5, dtype=np.float64)
    for index, row in enumerate(rows):
        if boundary[index]:
            values[index] = float(row[column])
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"Invalid boundary probability values in column {column!r}.")
    return values


def _gate_features(
    phase33_rows: list[dict[str, str]],
    one_d: np.ndarray,
    default: np.ndarray,
    camc: np.ndarray,
    bioemu: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    z_one_d = _logit(one_d)
    z_default = _logit(default)
    z_camc = _logit(camc)
    z_bioemu = _logit(bioemu)
    numeric = np.asarray(
        [
            [
                float(row["mean_plddt"]),
                float(row["min_plddt"]),
                float(row["low_plddt_fraction"]),
                float(row["global_gate"]),
                np.log1p(float(row["length"])),
            ]
            for row in phase33_rows
        ],
        dtype=np.float64,
    )
    strata = np.zeros((len(phase33_rows), 3), dtype=np.float64)
    stratum_index = {"11-15": 0, "16-30": 1, "31-50": 2}
    for index, row in enumerate(phase33_rows):
        stratum = row["length_stratum"]
        if stratum not in stratum_index:
            raise ValueError(f"Unknown length stratum {stratum!r}.")
        strata[index, stratum_index[stratum]] = 1.0
    values = np.column_stack(
        [
            z_one_d,
            z_default,
            z_camc,
            z_bioemu,
            np.abs(z_one_d),
            np.abs(z_default - _logit(np.asarray([threshold]))[0]),
            np.abs(z_camc - _logit(np.asarray([0.677819]))[0]),
            np.abs(z_bioemu),
            z_one_d * z_bioemu,
            z_one_d - z_bioemu,
            numeric,
            strata,
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("BioEmu benefit-gate features contain non-finite values.")
    return values


def _cross_fitted_benefit_gate(
    features: np.ndarray,
    benefit_target: np.ndarray,
    folds: np.ndarray,
    boundary: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probability = np.zeros(len(benefit_target), dtype=np.float64)
    reports: list[dict[str, Any]] = []
    for fold in range(5):
        train = (folds != fold) & boundary
        held = (folds == fold) & boundary
        if not np.any(train) or len(np.unique(benefit_target[train])) != 2:
            raise ValueError(f"Fold {fold} benefit-gate training rows lack both targets.")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train])
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        )
        model.fit(train_features, benefit_target[train])
        if np.any(held):
            probability[held] = model.predict_proba(scaler.transform(features[held]))[:, 1]
        reports.append(
            {
                "fold": fold,
                "benefit_train_rows": int(train.sum()),
                "benefit_held_rows": int(held.sum()),
                "benefit_target_rate_train": float(benefit_target[train].mean()),
                "benefit_gate_mean_held": float(probability[held].mean()),
            }
        )
    return probability, reports


def run_screen(
    *,
    one_d_path: str | Path,
    default_path: str | Path,
    phase33_path: str | Path,
    prott5_features_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.6,
    one_d_threshold: float = 0.45,
    residual_bound: float = 0.15,
    c_value: float = 0.1,
    seed: int = 20260802,
) -> dict[str, Any]:
    one_d_rows = _read_csv(one_d_path)
    default_rows = _read_csv(default_path)
    phase33_rows = _read_csv(phase33_path)
    aligned = _align_inputs(one_d_rows, default_rows, phase33_rows)
    labels = np.asarray(aligned["labels"])
    folds = np.asarray(aligned["folds"])
    boundary = np.asarray(aligned["boundary"])
    sequences = list(aligned["sequences"])
    features = _load_features(prott5_features_path, sequences)

    one_d = _probability(one_d_rows, "toxicity_probability")
    default = _probability(default_rows, "toxicity_probability")
    phase33 = _probability(phase33_rows, "phase33_candidate_probability")
    camc = _probability(phase33_rows, "phase33_reference_probability")
    bioemu = _boundary_probability(
        phase33_rows, "phase33_standalone_3d_probability", boundary
    )
    specialist, fold_reports = _cross_fitted_specialist(
        features, labels, folds, boundary, c_value=c_value, seed=seed
    )

    gate_features = _gate_features(
        phase33_rows, one_d, default, camc, bioemu, threshold=threshold
    )
    default_logit = _logit(default)
    full_bioemu_logit = default_logit + float(residual_bound) * np.tanh(_logit(bioemu))
    default_loss = np.logaddexp(0.0, default_logit) - labels * default_logit
    full_bioemu_loss = np.logaddexp(0.0, full_bioemu_logit) - labels * full_bioemu_logit
    benefit_target = (full_bioemu_loss < default_loss).astype(np.int64)
    benefit_gate, gate_reports = _cross_fitted_benefit_gate(
        gate_features,
        benefit_target,
        folds,
        boundary,
        c_value=c_value,
        seed=seed,
    )

    default_decision = default >= float(threshold)
    one_d_decision = one_d >= float(one_d_threshold)
    specialist_decision = specialist >= 0.5
    phase33_direction = phase33 >= 0.5
    consensus = (
        boundary
        & (one_d_decision != default_decision)
        & (specialist_decision == one_d_decision)
        & (phase33_direction == one_d_decision)
    )
    candidate, evidence = apply_boundary_mixture(
        default,
        specialist,
        bioemu,
        benefit_gate,
        boundary,
        consensus,
        residual_bound=residual_bound,
    )

    reference_metrics = _probability_metrics(labels, default, threshold)
    candidate_metrics = _probability_metrics(labels, candidate, threshold)
    transitions = _decision_transitions(labels, default, candidate, threshold)
    fold_non_decreases = 0
    for report, gate_report in zip(fold_reports, gate_reports, strict=True):
        fold = int(report["fold"])
        held = folds == fold
        reference_fold = _probability_metrics(labels[held], default[held], threshold)
        candidate_fold = _probability_metrics(labels[held], candidate[held], threshold)
        report.update(gate_report)
        report["consensus_rows"] = int(consensus[held].sum())
        report["soft_gate_rows"] = int((boundary & ~consensus & held).sum())
        report["reference"] = reference_fold
        report["candidate"] = candidate_fold
        fold_non_decreases += int(candidate_fold["mcc"] >= reference_fold["mcc"])

    gates = {
        "mcc_strictly_higher": candidate_metrics["mcc"] > reference_metrics["mcc"],
        "balanced_accuracy_not_lower": candidate_metrics["balanced_accuracy"]
        >= reference_metrics["balanced_accuracy"],
        "f1_not_lower": candidate_metrics["f1"] >= reference_metrics["f1"],
        "auprc_not_lower": candidate_metrics["auprc"] >= reference_metrics["auprc"],
        "brier_not_higher": candidate_metrics["brier"] <= reference_metrics["brier"],
        "ece_10_increase_at_most_0_0001": candidate_metrics["ece_10"]
        <= reference_metrics["ece_10"] + 0.0001,
        "at_least_three_fold_mcc_non_decreases": fold_non_decreases >= 3,
        "positive_net_decision_corrections": transitions["net"] > 0,
    }
    gates["all_conditions_passed"] = all(gates.values())
    decision = (
        "promote_as_project_research_default"
        if gates["all_conditions_passed"]
        else "retain_previous_default"
    )
    summary = {
        "protocol": {
            "public_model_name": "BioEmu Boundary Mixture-of-Experts adapter (BioEmu-BMoE)",
            "legacy_internal_stage_id": "Phase33",
            "provenance": "exploratory_outer_fold_pure_specialist_on_existing_oof_streams",
            "rule_frozen_before_this_report": False,
            "project_owner_promotion_authorized": True,
            "strict_nested_claim_authorized": False,
            "production_unseen_inference_ready": False,
            "test1_or_test2_labels_read": False,
            "specialist": "StandardScaler + balanced logistic regression on pooled ProtT5 boundary rows",
            "benefit_gate": "StandardScaler + balanced logistic regression on fixed scalar/quality features",
            "benefit_target": "whether the full bounded BioEmu residual lowers per-row BCE",
            "c_value": float(c_value),
            "seed": int(seed),
            "routing_rule": "consensus rows use boundary-1D evidence; all other BioEmu boundary rows use gated BioEmu evidence",
            "formula": "z_candidate=z_default+0.15*(consensus?tanh(z_boundary_1d):g_benefit*tanh(z_bioemu))",
            "residual_bound": float(residual_bound),
            "decision_threshold": float(threshold),
        },
        "coverage": {
            "rows": int(len(labels)),
            "phase33_boundary_rows": int(boundary.sum()),
            "active_consensus_rows": int(consensus.sum()),
            "soft_bioemu_gate_rows": int((boundary & ~consensus).sum()),
            "benefit_target_rate": float(benefit_target[boundary].mean()),
            "benefit_gate_mean": float(benefit_gate[boundary].mean()),
        },
        "reference": reference_metrics,
        "candidate": candidate_metrics,
        "delta": {
            key: float(candidate_metrics[key] - reference_metrics[key])
            for key in (
                "balanced_accuracy",
                "f1",
                "mcc",
                "auroc",
                "auprc",
                "brier",
                "ece_10",
            )
        },
        "decision_transitions": transitions,
        "fold_mcc_non_decrease_count": fold_non_decreases,
        "folds": fold_reports,
        "retention_gate": gates,
        "decision": decision,
        "interpretation": (
            "The adapter passes the retention gate and is promoted as the project research "
            "default by owner decision. A pre-frozen replication or new external set is "
            "still required for an independent generalisation claim."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_index",
            "sample_id",
            "fold",
            "label",
            "sequence",
            "one_d_probability",
            "boundary_one_d_probability",
            "phase33_3d_probability",
            "bioemu_standalone_probability",
            "bioemu_benefit_gate",
            "current_default_probability",
            "phase33_boundary_member",
            "consensus_active",
            "boundary_route",
            "bounded_evidence",
            "toxicity_probability",
            "prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(labels)):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": aligned["sample_ids"][index],
                    "fold": int(folds[index]),
                    "label": int(labels[index]),
                    "sequence": sequences[index],
                    "one_d_probability": f"{one_d[index]:.10g}",
                    "boundary_one_d_probability": f"{specialist[index]:.10g}",
                    "phase33_3d_probability": f"{phase33[index]:.10g}",
                    "bioemu_standalone_probability": f"{bioemu[index]:.10g}",
                    "bioemu_benefit_gate": f"{benefit_gate[index]:.10g}",
                    "current_default_probability": f"{default[index]:.10g}",
                    "phase33_boundary_member": int(boundary[index]),
                    "consensus_active": int(consensus[index]),
                    "boundary_route": (
                        "consensus"
                        if consensus[index]
                        else "soft_bioemu"
                        if boundary[index]
                        else "fallback"
                    ),
                    "bounded_evidence": f"{evidence[index]:.10g}",
                    "toxicity_probability": f"{candidate[index]:.10g}",
                    "prediction": int(candidate[index] >= threshold),
                }
            )
    save_json(summary, output / "summary.json")
    save_json(
        {
            "status": "complete",
            "decision": decision,
            "promotion_authorized": bool(gates["all_conditions_passed"]),
            "promotion_scope": "project_research_default",
            "pre_frozen_replication_required": True,
            "production_unseen_inference_ready": False,
            "retention_gate": gates,
            "metrics": candidate_metrics,
        },
        output / "decision.json",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d", required=True)
    parser.add_argument("--current-default", required=True)
    parser.add_argument(
        "--bioemu-boundary-3d",
        "--phase33",
        dest="phase33",
        required=True,
        help="BioEmu Boundary-Residual 3D OOF table; --phase33 is a compatibility alias.",
    )
    parser.add_argument("--prott5-features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_screen(
        one_d_path=args.one_d,
        default_path=args.current_default,
        phase33_path=args.phase33,
        prott5_features_path=args.prott5_features,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    print(json.dumps(result["candidate"], indent=2))
    print(json.dumps(result["retention_gate"], indent=2))


if __name__ == "__main__":
    main()
