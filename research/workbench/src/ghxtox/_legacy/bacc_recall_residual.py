"""BACC-prioritised, disagreement-only ProtT5 recall residual screen.

The adapter learns whether a deployable three-seed ProtT5 teacher should replace
the class decision of the frozen 1D/3D chemical-interaction model.  Training and
architecture selection use group-aware OOF predictions only.  Test1/Test2 are
evaluated once after the architecture and gate threshold have been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ghxtox.bioemu_boundary_moe import _logit, _read_csv
from ghxtox._legacy.plm_textcnn import _probability_metrics
from ghxtox.utils import save_json


@dataclass(frozen=True)
class ResidualConfig:
    name: str
    hidden_layers: tuple[int, ...]
    epochs: int
    alpha: float = 0.01


CONFIGS = (
    ResidualConfig("shallow_e80", (16,), 80),
    ResidualConfig("shallow_e240", (16,), 240),
    ResidualConfig("medium_e80", (32, 16), 80),
    ResidualConfig("medium_e240", (32, 16), 240),
    ResidualConfig("deep_e240", (64, 32, 16), 240),
)

COMPLEXITY_BACC_MARGIN = 0.001


def _probability(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError(f"Invalid probability column {column!r}.")
    return values


def _sequence_features(sequence: str) -> list[float]:
    length = max(len(sequence), 1)
    groups = (
        "KRH",
        "DE",
        "AILMFWVY",
        "FWY",
        "CGP",
        "C",
        "GP",
        "STNQ",
    )
    fractions = [sum(sequence.count(letter) for letter in group) / length for group in groups]
    charge_proxy = (
        sum(sequence.count(letter) for letter in "KRH")
        - sum(sequence.count(letter) for letter in "DE")
    ) / length
    return [np.log1p(length), charge_proxy, *fractions]


def _align(
    teacher_rows: list[dict[str, str]], base_rows: list[dict[str, str]]
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    if len(teacher_rows) != len(base_rows):
        raise ValueError("Teacher and base prediction tables have different row counts.")
    labels = []
    folds = []
    sequences = []
    source_indices = []
    for index, (teacher, base) in enumerate(zip(teacher_rows, base_rows, strict=True)):
        if teacher["sequence"] != base["sequence"]:
            raise ValueError(f"Sequence mismatch at row {index}.")
        if int(teacher["label"]) != int(base["label"]):
            raise ValueError(f"Label mismatch at row {index}.")
        labels.append(int(teacher["label"]))
        sequences.append(teacher["sequence"])
        source_indices.append(int(teacher.get("source_index", index)))
        if base.get("fold") not in (None, ""):
            folds.append(int(base["fold"]))
        elif teacher.get("fold") not in (None, ""):
            folds.append(int(teacher["fold"]))
        else:
            folds.append(-1)
    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(folds, dtype=np.int64),
        sequences,
        np.asarray(source_indices, dtype=np.int64),
    )


def _features(
    teacher_rows: list[dict[str, str]],
    base_rows: list[dict[str, str]],
    sequences: list[str],
    *,
    teacher_threshold: float,
    base_threshold: float,
) -> np.ndarray:
    teacher = _probability(teacher_rows, "toxicity_probability")
    base = _probability(base_rows, "toxicity_probability")
    members = np.column_stack(
        [
            _probability(teacher_rows, f"member{index}_probability")
            for index in range(3)
        ]
    )
    baseline = np.asarray(
        [float(row.get("baseline_probability", row["toxicity_probability"])) for row in base_rows],
        dtype=np.float64,
    )
    interaction = np.asarray(
        [float(row.get("interaction_delta", 0.0)) for row in base_rows],
        dtype=np.float64,
    )
    sequence = np.asarray([_sequence_features(value) for value in sequences], dtype=np.float64)
    teacher_logit = _logit(teacher)
    base_logit = _logit(base)
    values = np.column_stack(
        [
            teacher,
            base,
            baseline,
            interaction,
            members,
            members.std(axis=1),
            members.min(axis=1),
            members.max(axis=1),
            (members >= teacher_threshold).sum(axis=1) / 3.0,
            teacher_logit,
            base_logit,
            teacher_logit - base_logit,
            np.abs(teacher - teacher_threshold),
            np.abs(base - base_threshold),
            sequence,
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("Residual features contain non-finite values.")
    return values


def apply_minimal_decision_residual(
    base_probability: np.ndarray,
    teacher_probability: np.ndarray,
    use_teacher_decision: np.ndarray,
    *,
    threshold: float,
    teacher_threshold: float | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Move selected disagreement rows only just across the frozen threshold."""

    base = np.asarray(base_probability, dtype=np.float64)
    teacher = np.asarray(teacher_probability, dtype=np.float64)
    use_teacher = np.asarray(use_teacher_decision, dtype=bool)
    if not base.shape == teacher.shape == use_teacher.shape:
        raise ValueError("Residual inputs must have identical shapes.")
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must lie strictly in (0, 1).")
    teacher_cutoff = float(threshold if teacher_threshold is None else teacher_threshold)
    if not 0.0 < teacher_cutoff < 1.0:
        raise ValueError("teacher_threshold must lie strictly in (0, 1).")
    disagreement = (base >= threshold) != (teacher >= teacher_cutoff)
    active = use_teacher & disagreement
    result = base.copy()
    teacher_positive = teacher >= teacher_cutoff
    result[active & teacher_positive] = float(threshold) + float(epsilon)
    result[active & ~teacher_positive] = float(threshold) - float(epsilon)
    return result


def _balanced_indices(target: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(target == value) for value in (0, 1)]
    if any(len(indices) == 0 for indices in classes):
        raise ValueError("Gate training target must contain both classes.")
    size = max(len(indices) for indices in classes)
    balanced = np.concatenate(
        [rng.choice(indices, size=size, replace=len(indices) < size) for indices in classes]
    )
    rng.shuffle(balanced)
    return balanced


def _fit_gate(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    config: ResidualConfig,
    *,
    seed: int,
) -> tuple[StandardScaler, MLPClassifier]:
    train_features = features[train_mask]
    train_target = target[train_mask]
    balanced = _balanced_indices(train_target, seed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_features[balanced])
    model = MLPClassifier(
        hidden_layer_sizes=config.hidden_layers,
        activation="relu",
        solver="adam",
        alpha=float(config.alpha),
        batch_size=min(32, len(balanced)),
        learning_rate_init=1e-3,
        max_iter=int(config.epochs),
        shuffle=True,
        random_state=int(seed),
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(scaled, train_target[balanced])
    return scaler, model


def _nondegradation_gate(reference: dict[str, float], candidate: dict[str, float]) -> bool:
    higher = ("accuracy", "f1", "mcc", "auroc", "auprc", "specificity")
    lower = ("brier", "ece_10")
    return all(candidate[key] >= reference[key] - 1e-12 for key in higher) and all(
        candidate[key] <= reference[key] + 1e-12 for key in lower
    )


def _decision_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    decision: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Combine an explicit decision head with the unchanged calibrated probability."""

    metrics = _probability_metrics(labels, probability, threshold)
    prediction = np.asarray(decision, dtype=bool)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    metrics.update(
        {
            "accuracy": float(accuracy_score(labels, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "precision": float(precision_score(labels, prediction, zero_division=0)),
            "sensitivity": float(recall_score(labels, prediction, zero_division=0)),
            "sn": float(recall_score(labels, prediction, zero_division=0)),
            "recall": float(recall_score(labels, prediction, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "sp": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(labels, prediction, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, prediction)),
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }
    )
    return metrics


def _apply_decision_residual(
    base: np.ndarray,
    teacher: np.ndarray,
    active: np.ndarray,
    *,
    threshold: float,
    teacher_threshold: float,
) -> np.ndarray:
    decision = np.asarray(base >= threshold, dtype=bool)
    disagreement = decision != (teacher >= teacher_threshold)
    selected = np.asarray(active, dtype=bool) & disagreement
    decision[selected] = teacher[selected] >= teacher_threshold
    return decision


def _select_threshold(
    labels: np.ndarray,
    base: np.ndarray,
    teacher: np.ndarray,
    gate_probability: np.ndarray,
    validation_mask: np.ndarray,
    *,
    threshold: float,
    teacher_threshold: float,
) -> tuple[float, dict[str, Any]]:
    base_decision = base >= threshold
    reference = _decision_metrics(
        labels[validation_mask],
        base[validation_mask],
        base_decision[validation_mask],
        threshold,
    )
    best: tuple[tuple[float, ...], float, dict[str, float]] | None = None
    disagreement = base_decision != (teacher >= teacher_threshold)
    candidates = np.unique(gate_probability[validation_mask & disagreement])
    for gate_threshold in candidates:
        active = gate_probability >= gate_threshold
        candidate_decision = _apply_decision_residual(
            base,
            teacher,
            active,
            threshold=threshold,
            teacher_threshold=teacher_threshold,
        )
        metrics = _decision_metrics(
            labels[validation_mask],
            base[validation_mask],
            candidate_decision[validation_mask],
            threshold,
        )
        if metrics["balanced_accuracy"] <= reference["balanced_accuracy"] + 1e-12:
            continue
        if not _nondegradation_gate(reference, metrics):
            continue
        score = (
            metrics["balanced_accuracy"],
            metrics["mcc"],
            metrics["f1"],
            metrics["auprc"],
            -metrics["brier"],
        )
        if best is None or score > best[0]:
            best = (score, float(gate_threshold), metrics)
    if best is None:
        return 1.1, {
            "accepted": False,
            "gate_threshold": 1.1,
            "reference": reference,
            "candidate": reference,
        }
    return best[1], {
        "accepted": True,
        "gate_threshold": best[1],
        "reference": reference,
        "candidate": best[2],
    }


def _crossfit_config(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    base: np.ndarray,
    teacher: np.ndarray,
    disagreement: np.ndarray,
    config: ResidualConfig,
    *,
    threshold: float,
    teacher_threshold: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = (teacher >= teacher_threshold) == labels
    gate_probability = np.zeros(len(labels), dtype=np.float64)
    fold_reports = []
    for held_fold in range(5):
        validation_fold = (held_fold + 1) % 5
        train = disagreement & (folds != held_fold) & (folds != validation_fold)
        validation = folds == validation_fold
        held = folds == held_fold
        scaler, model = _fit_gate(
            features, target.astype(np.int64), train, config, seed=seed + held_fold
        )
        validation_disagreement = validation & disagreement
        held_disagreement = held & disagreement
        if np.any(held_disagreement):
            held_gate = model.predict_proba(scaler.transform(features[held_disagreement]))[:, 1]
            gate_probability[held_disagreement] = held_gate
        fold_reports.append(
            {
                "held_fold": held_fold,
                "validation_fold": validation_fold,
                "train_disagreement_rows": int(train.sum()),
                "validation_disagreement_rows": int(validation_disagreement.sum()),
                "held_disagreement_rows": int(held_disagreement.sum()),
            }
        )
    gate_threshold, selection = _select_threshold(
        labels,
        base,
        teacher,
        gate_probability,
        np.ones(len(labels), dtype=bool),
        threshold=threshold,
        teacher_threshold=teacher_threshold,
    )
    active = disagreement & (gate_probability >= gate_threshold)
    candidate_decision = _apply_decision_residual(
        base,
        teacher,
        active,
        threshold=threshold,
        teacher_threshold=teacher_threshold,
    )
    reference_decision = base >= threshold
    reference_metrics = _decision_metrics(labels, base, reference_decision, threshold)
    metrics = _decision_metrics(labels, base, candidate_decision, threshold)
    return candidate_decision, gate_probability, {
        "config": {
            "name": config.name,
            "hidden_layers": list(config.hidden_layers),
            "epochs": config.epochs,
            "alpha": config.alpha,
        },
        "reference": reference_metrics,
        "metrics": metrics,
        "delta": {
            key: float(metrics[key] - reference_metrics[key])
            for key in (
                "accuracy",
                "balanced_accuracy",
                "sensitivity",
                "specificity",
                "f1",
                "mcc",
                "auroc",
                "auprc",
                "brier",
                "ece_10",
            )
        },
        "active_rows": int(active.sum()),
        "gate_threshold": gate_threshold,
        "threshold_selection": selection,
        "transitions": {
            "changed": int(np.sum(candidate_decision != reference_decision)),
            "corrected": int(
                np.sum((candidate_decision != reference_decision) & (candidate_decision == labels))
            ),
            "regressed": int(
                np.sum((candidate_decision != reference_decision) & (candidate_decision != labels))
            ),
            "net": int(
                np.sum((candidate_decision != reference_decision) & (candidate_decision == labels))
                - np.sum((candidate_decision != reference_decision) & (candidate_decision != labels))
            ),
        },
        "other_metrics_nondecreasing": _nondegradation_gate(reference_metrics, metrics),
        "folds": fold_reports,
    }


def _fit_full_and_predict(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_disagreement: np.ndarray,
    eval_features: np.ndarray,
    eval_disagreement: np.ndarray,
    config: ResidualConfig,
    gate_threshold: float,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaler, model = _fit_gate(
        train_features,
        train_target,
        train_disagreement,
        config,
        seed=seed,
    )
    gate = np.zeros(len(eval_features), dtype=np.float64)
    active = np.zeros(len(eval_features), dtype=bool)
    if np.any(eval_disagreement):
        gate[eval_disagreement] = model.predict_proba(
            scaler.transform(eval_features[eval_disagreement])
        )[:, 1]
        active[eval_disagreement] = gate[eval_disagreement] >= gate_threshold
    return gate, active


def _serialize_gate(
    path: Path,
    scaler: StandardScaler,
    model: MLPClassifier,
    config: ResidualConfig,
    *,
    gate_threshold: float,
    teacher_threshold: float,
    base_threshold: float,
) -> None:
    """Write the small gate in a dependency-light, inference-ready format."""

    save_json(
        {
            "artifact_type": "ghxtox_bacc_decision_residual",
            "version": 1,
            "feature_schema": {
                "name": "bacc_recall_residual_v1",
                "dimension": 26,
                "sequence_features": [
                    "log1p_length",
                    "charge_proxy",
                    "fraction_KRH",
                    "fraction_DE",
                    "fraction_AILMFWVY",
                    "fraction_FWY",
                    "fraction_CGP",
                    "fraction_C",
                    "fraction_GP",
                    "fraction_STNQ",
                ],
            },
            "model": {
                "hidden_layers": list(config.hidden_layers),
                "epochs": int(config.epochs),
                "alpha": float(config.alpha),
                "activation": "relu",
                "coefs": [array.tolist() for array in model.coefs_],
                "intercepts": [array.tolist() for array in model.intercepts_],
            },
            "scaler": {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            "teacher_threshold": float(teacher_threshold),
            "base_threshold": float(base_threshold),
            "gate_threshold": float(gate_threshold),
            "decision_only": True,
            "selection_source": "training-only group-aware OOF",
        },
        path,
    )


def _write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    base: np.ndarray,
    teacher: np.ndarray,
    gate: np.ndarray,
    gate_active: np.ndarray,
    decision: np.ndarray,
    *,
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_index",
            "sample_id",
            "sequence",
            "label",
            "base_probability",
            "prott5_teacher_probability",
            "teacher_gate_probability",
            "residual_active",
            "toxicity_probability",
            "prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": row.get("sample_id", index),
                    "sequence": row["sequence"],
                    "label": int(labels[index]),
                    "base_probability": f"{base[index]:.10g}",
                    "prott5_teacher_probability": f"{teacher[index]:.10g}",
                    "teacher_gate_probability": f"{gate[index]:.10g}",
                    "residual_active": int(gate_active[index]),
                    "toxicity_probability": f"{base[index]:.10g}",
                    "prediction": int(decision[index]),
                }
            )


def run_screen(
    *,
    teacher_oof_path: str | Path,
    base_oof_path: str | Path,
    teacher_test1_path: str | Path,
    base_test1_path: str | Path,
    teacher_test2_path: str | Path,
    base_test2_path: str | Path,
    output_dir: str | Path,
    research_default_oof_path: str | Path | None = None,
    threshold: float = 0.6,
    teacher_threshold: float = 0.45,
    seed: int = 20260802,
) -> dict[str, Any]:
    teacher_oof_rows = _read_csv(teacher_oof_path)
    base_oof_rows = _read_csv(base_oof_path)
    labels, folds, sequences, _ = _align(teacher_oof_rows, base_oof_rows)
    if set(folds) != set(range(5)):
        raise ValueError("OOF input must contain canonical folds 0-4.")
    teacher_oof = _probability(teacher_oof_rows, "toxicity_probability")
    base_oof = _probability(base_oof_rows, "toxicity_probability")
    disagreement = (teacher_oof >= teacher_threshold) != (base_oof >= threshold)
    features = _features(
        teacher_oof_rows,
        base_oof_rows,
        sequences,
        teacher_threshold=teacher_threshold,
        base_threshold=threshold,
    )
    screens = []
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for config in CONFIGS:
        candidate, gate, report = _crossfit_config(
            features,
            labels,
            folds,
            base_oof,
            teacher_oof,
            disagreement,
            config,
            threshold=threshold,
            teacher_threshold=teacher_threshold,
            seed=seed,
        )
        screens.append(report)
        candidates[config.name] = (candidate, gate)
    eligible = [
        report
        for report in screens
        if report["metrics"]["balanced_accuracy"]
        > report["reference"]["balanced_accuracy"]
        and report["other_metrics_nondecreasing"]
    ]
    if eligible:
        best_bacc = max(report["metrics"]["balanced_accuracy"] for report in eligible)
        near_best = [
            report
            for report in eligible
            if report["metrics"]["balanced_accuracy"]
            >= best_bacc - COMPLEXITY_BACC_MARGIN
        ]
        selected = min(
            near_best,
            key=lambda report: (
                len(report["config"]["hidden_layers"]),
                sum(report["config"]["hidden_layers"]),
                report["config"]["epochs"],
            ),
        )
        selection_reason = (
            "simplest nondegrading model within 0.001 BACC of the best OOF screen"
        )
    else:
        selected = max(
            screens,
            key=lambda report: (
                report["metrics"]["balanced_accuracy"],
                report["metrics"]["mcc"],
            ),
        )
        selection_reason = "no OOF candidate passed the nondegradation gate"
    selected_config = next(
        config for config in CONFIGS if config.name == selected["config"]["name"]
    )
    oof_decision, oof_gate = candidates[selected_config.name]
    full_gate_threshold = float(selected["gate_threshold"])
    oof_active = disagreement & (oof_gate >= full_gate_threshold)
    train_target = (((teacher_oof >= teacher_threshold) == labels)).astype(np.int64)
    full_scaler, full_model = _fit_gate(
        features,
        train_target,
        disagreement,
        selected_config,
        seed=seed,
    )

    evaluations = {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _serialize_gate(
        output / "deployable_gate.json",
        full_scaler,
        full_model,
        selected_config,
        gate_threshold=full_gate_threshold,
        teacher_threshold=teacher_threshold,
        base_threshold=threshold,
    )
    _write_predictions(
        output / "oof_predictions.csv",
        base_oof_rows,
        labels,
        base_oof,
        teacher_oof,
        oof_gate,
        oof_active,
        oof_decision,
        threshold=threshold,
    )
    research_default = None
    if research_default_oof_path is not None:
        research_rows = _read_csv(research_default_oof_path)
        research_labels, research_folds, research_sequences, _ = _align(
            teacher_oof_rows, research_rows
        )
        if not np.array_equal(research_labels, labels) or not np.array_equal(
            research_folds, folds
        ):
            raise ValueError("Research-default OOF rows do not match the gate-training OOF.")
        if research_sequences != sequences:
            raise ValueError("Research-default OOF sequences do not match gate-training OOF.")
        research_probability = _probability(research_rows, "toxicity_probability")
        research_reference_decision = research_probability >= threshold
        research_disagreement = research_reference_decision != (
            teacher_oof >= teacher_threshold
        )
        research_active = research_disagreement & (oof_gate >= full_gate_threshold)
        research_decision = _apply_decision_residual(
            research_probability,
            teacher_oof,
            research_active,
            threshold=threshold,
            teacher_threshold=teacher_threshold,
        )
        research_reference = _decision_metrics(
            labels, research_probability, research_reference_decision, threshold
        )
        research_metrics = _decision_metrics(
            labels, research_probability, research_decision, threshold
        )
        research_changed = research_decision != research_reference_decision
        research_default = {
            "reference": research_reference,
            "metrics": research_metrics,
            "delta": {
                key: float(research_metrics[key] - research_reference[key])
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "sensitivity",
                    "specificity",
                    "f1",
                    "mcc",
                    "auroc",
                    "auprc",
                    "brier",
                    "ece_10",
                )
            },
            "active_rows": int(research_active.sum()),
            "other_metrics_nondecreasing": _nondegradation_gate(
                research_reference, research_metrics
            ),
            "transitions": {
                "changed": int(research_changed.sum()),
                "corrected": int(
                    np.sum(research_changed & (research_decision == labels))
                ),
                "regressed": int(
                    np.sum(research_changed & (research_decision != labels))
                ),
                "net": int(
                    np.sum(research_changed & (research_decision == labels))
                    - np.sum(research_changed & (research_decision != labels))
                ),
            },
        }
        _write_predictions(
            output / "research_default_oof_predictions.csv",
            research_rows,
            labels,
            research_probability,
            teacher_oof,
            oof_gate,
            research_active,
            research_decision,
            threshold=threshold,
        )
    for name, teacher_path, base_path in (
        ("test1", teacher_test1_path, base_test1_path),
        ("test2", teacher_test2_path, base_test2_path),
    ):
        teacher_rows = _read_csv(teacher_path)
        base_rows = _read_csv(base_path)
        eval_labels, _, eval_sequences, _ = _align(teacher_rows, base_rows)
        teacher_probability = _probability(teacher_rows, "toxicity_probability")
        base_probability = _probability(base_rows, "toxicity_probability")
        eval_disagreement = (teacher_probability >= teacher_threshold) != (
            base_probability >= threshold
        )
        eval_features = _features(
            teacher_rows,
            base_rows,
            eval_sequences,
            teacher_threshold=teacher_threshold,
            base_threshold=threshold,
        )
        eval_gate = np.zeros(len(eval_features), dtype=np.float64)
        eval_active = np.zeros(len(eval_features), dtype=bool)
        if np.any(eval_disagreement):
            eval_gate[eval_disagreement] = full_model.predict_proba(
                full_scaler.transform(eval_features[eval_disagreement])
            )[:, 1]
            eval_active[eval_disagreement] = (
                eval_gate[eval_disagreement] >= full_gate_threshold
            )
        eval_decision = _apply_decision_residual(
            base_probability,
            teacher_probability,
            eval_active,
            threshold=threshold,
            teacher_threshold=teacher_threshold,
        )
        reference_decision = base_probability >= threshold
        reference = _decision_metrics(
            eval_labels, base_probability, reference_decision, threshold
        )
        metrics = _decision_metrics(eval_labels, base_probability, eval_decision, threshold)
        changed = eval_decision != reference_decision
        evaluations[name] = {
            "rows": int(len(eval_labels)),
            "disagreement_rows": int(eval_disagreement.sum()),
            "active_rows": int(eval_active.sum()),
            "reference": reference,
            "metrics": metrics,
            "delta": {
                key: float(metrics[key] - reference[key])
                for key in (
                    "accuracy",
                    "balanced_accuracy",
                    "sensitivity",
                    "specificity",
                    "f1",
                    "mcc",
                    "auroc",
                    "auprc",
                    "brier",
                    "ece_10",
                )
            },
            "other_metrics_nondecreasing": _nondegradation_gate(reference, metrics),
            "transitions": {
                "changed": int(changed.sum()),
                "corrected": int(np.sum(changed & (eval_decision == eval_labels))),
                "regressed": int(np.sum(changed & (eval_decision != eval_labels))),
                "net": int(
                    np.sum(changed & (eval_decision == eval_labels))
                    - np.sum(changed & (eval_decision != eval_labels))
                ),
            },
        }
        _write_predictions(
            output / f"{name}_predictions.csv",
            base_rows,
            eval_labels,
            base_probability,
            teacher_probability,
            eval_gate,
            eval_active,
            eval_decision,
            threshold=threshold,
        )

    oof_report = research_default or selected
    oof_promoted = (
        oof_report["metrics"]["balanced_accuracy"]
        > oof_report["reference"]["balanced_accuracy"]
        and oof_report["other_metrics_nondecreasing"]
    )
    test2_promoted = (
        evaluations["test2"]["metrics"]["balanced_accuracy"]
        > evaluations["test2"]["reference"]["balanced_accuracy"]
        and evaluations["test2"]["other_metrics_nondecreasing"]
    )
    test1_safe = evaluations["test1"]["other_metrics_nondecreasing"]
    promote = bool(oof_promoted and test2_promoted and test1_safe)
    summary = {
        "protocol": {
            "name": "BACC-prioritised disagreement-only ProtT5 recall residual",
            "selection_data": "group-aware training OOF only",
            "test1_test2_role": "single frozen historical evaluation; never used for layer/epoch/threshold selection",
            "teacher_threshold": teacher_threshold,
            "base_threshold": threshold,
            "residual": "decision-only correction on selected disagreement rows; calibrated probability remains unchanged",
            "other_metric_constraint": "ACC/F1/MCC/AUROC/AUPRC/SP nondecrease and Brier/ECE nonincrease",
            "complexity_rule": "choose the simplest eligible architecture within 0.001 BACC of the best OOF screen",
            "seed": seed,
        },
        "coverage": {
            "oof_rows": int(len(labels)),
            "oof_disagreement_rows": int(disagreement.sum()),
        },
        "screens": screens,
        "selected": selected,
        "selection_reason": selection_reason,
        "research_default_oof": research_default,
        "full_gate_threshold": full_gate_threshold,
        "evaluations": evaluations,
        "decision": "promote_recall_residual" if promote else "retain_existing_default",
        "promotion_authorized": promote,
    }
    save_json(summary, output / "summary.json")
    save_json(
        {
            "status": "complete",
            "decision": summary["decision"],
            "promotion_authorized": promote,
            "selected_config": selected["config"],
            "full_gate_threshold": full_gate_threshold,
            "research_default_oof": research_default,
            "test2": evaluations["test2"],
        },
        output / "decision.json",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-oof", required=True)
    parser.add_argument("--base-oof", required=True)
    parser.add_argument("--teacher-test1", required=True)
    parser.add_argument("--base-test1", required=True)
    parser.add_argument("--teacher-test2", required=True)
    parser.add_argument("--base-test2", required=True)
    parser.add_argument("--research-default-oof")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_screen(
        teacher_oof_path=args.teacher_oof,
        base_oof_path=args.base_oof,
        teacher_test1_path=args.teacher_test1,
        base_test1_path=args.base_test1,
        teacher_test2_path=args.teacher_test2,
        base_test2_path=args.base_test2,
        research_default_oof_path=args.research_default_oof,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )
    print(json.dumps(summary["selected"], indent=2))
    print(json.dumps(summary["evaluations"], indent=2))
    print(summary["decision"])


if __name__ == "__main__":
    main()
