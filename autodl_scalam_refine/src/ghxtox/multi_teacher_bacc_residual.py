"""Screen a constrained ProtT5+ESM2 recall gate over the public default.

The gate is trained and selected on group-aware OOF predictions. It may only
promote a current negative decision when ProtT5 or ESM2 supplies positive
evidence. Probabilities remain unchanged, so ranking and calibration metrics
are preserved by construction.
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ghxtox.bacc_recall_residual import (
    _decision_metrics,
    _logit,
    _probability,
    _read_csv,
    _sequence_features,
)
from ghxtox.utils import save_json


@dataclass(frozen=True)
class GateConfig:
    name: str
    hidden_layers: tuple[int, ...]
    epochs: int
    alpha: float = 0.01


CONFIGS = (
    GateConfig("linear_e160", (), 160),
    GateConfig("shallow_e80", (16,), 80),
    GateConfig("shallow_e240", (16,), 240),
    GateConfig("medium_e160", (32, 16), 160),
)

PROTT5_THRESHOLD = 0.45
ESM2_THRESHOLD = 0.35
BASE_THRESHOLD = 0.60
COMPLEXITY_BACC_MARGIN = 0.001
STABILITY_TOLERANCES = {
    "accuracy": 0.001,
    "f1": 0.001,
    "mcc": 0.001,
    "specificity": 0.002,
}


def _rows_by_source_index(rows: list[dict[str, str]], name: str) -> dict[int, dict[str, str]]:
    aligned: dict[int, dict[str, str]] = {}
    for position, row in enumerate(rows):
        source_index = int(row.get("source_index", position))
        if source_index in aligned:
            raise ValueError(f"Duplicate source_index {source_index} in {name}.")
        aligned[source_index] = row
    return aligned


def _align_tables(
    context_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
    prott5_rows: list[dict[str, str]],
    esm2_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    tables = {
        "current": _rows_by_source_index(current_rows, "current"),
        "prott5": _rows_by_source_index(prott5_rows, "prott5"),
        "esm2": _rows_by_source_index(esm2_rows, "esm2"),
    }
    context = _rows_by_source_index(context_rows, "context")
    expected = set(context)
    for name, table in tables.items():
        if set(table) != expected:
            missing = sorted(expected - set(table))[:5]
            extra = sorted(set(table) - expected)[:5]
            raise ValueError(f"{name} source_index mismatch: missing={missing}, extra={extra}.")

    ordered_context = [context[index] for index in sorted(expected)]
    ordered = [[table[index] for index in sorted(expected)] for table in tables.values()]
    for position, rows in enumerate(zip(ordered_context, *ordered, strict=True)):
        anchor = rows[0]
        for name, row in zip(("current", "prott5", "esm2"), rows[1:], strict=True):
            if row["sequence"] != anchor["sequence"]:
                raise ValueError(f"Sequence mismatch for {name} at aligned row {position}.")
            if int(row["label"]) != int(anchor["label"]):
                raise ValueError(f"Label mismatch for {name} at aligned row {position}.")
    return ordered_context, ordered[0], ordered[1], ordered[2]


def _column(rows: list[dict[str, str]], name: str, fallback: str | None = None) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name)
        if value in (None, "") and fallback is not None:
            value = row[fallback]
        if value in (None, ""):
            raise ValueError(f"Missing required column {name!r}.")
        values.append(float(value))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Column {name!r} contains non-finite values.")
    return result


def _candidate_mask(
    current_decision: np.ndarray,
    prott5: np.ndarray,
    esm2: np.ndarray,
    *,
    prott5_threshold: float = PROTT5_THRESHOLD,
    esm2_threshold: float = ESM2_THRESHOLD,
) -> np.ndarray:
    current = np.asarray(current_decision, dtype=bool)
    teacher_positive = (prott5 >= prott5_threshold) | (esm2 >= esm2_threshold)
    return ~current & teacher_positive


def _features(
    current_rows: list[dict[str, str]],
    prott5_rows: list[dict[str, str]],
    esm2_rows: list[dict[str, str]],
) -> np.ndarray:
    current = _probability(current_rows, "toxicity_probability")
    prott5 = _probability(prott5_rows, "toxicity_probability")
    esm2 = _probability(esm2_rows, "toxicity_probability")
    members = np.column_stack(
        [_probability(prott5_rows, f"member{index}_probability") for index in range(3)]
    )
    baseline = _column(current_rows, "baseline_probability", "toxicity_probability")
    interaction = _column(current_rows, "interaction_delta")
    sequences = [row["sequence"] for row in current_rows]
    sequence = np.asarray([_sequence_features(value) for value in sequences], dtype=np.float64)
    prott5_positive = prott5 >= PROTT5_THRESHOLD
    esm2_positive = esm2 >= ESM2_THRESHOLD
    current_logit = _logit(current)
    prott5_logit = _logit(prott5)
    esm2_logit = _logit(esm2)
    values = np.column_stack(
        [
            current,
            baseline,
            interaction,
            prott5,
            esm2,
            members,
            members.std(axis=1),
            members.min(axis=1),
            members.max(axis=1),
            (members >= PROTT5_THRESHOLD).mean(axis=1),
            current_logit,
            prott5_logit,
            esm2_logit,
            prott5_logit - current_logit,
            esm2_logit - current_logit,
            prott5_logit - esm2_logit,
            np.abs(current - BASE_THRESHOLD),
            np.abs(prott5 - PROTT5_THRESHOLD),
            np.abs(esm2 - ESM2_THRESHOLD),
            prott5_positive.astype(np.float64),
            esm2_positive.astype(np.float64),
            (prott5_positive & esm2_positive).astype(np.float64),
            (prott5_positive == esm2_positive).astype(np.float64),
            sequence,
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("Multi-teacher residual features contain non-finite values.")
    return values


def _balanced_indices(target: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(target == value) for value in (0, 1)]
    if any(len(indices) == 0 for indices in classes):
        raise ValueError("Gate training target must contain both classes.")
    size = max(len(indices) for indices in classes)
    result = np.concatenate(
        [rng.choice(indices, size=size, replace=len(indices) < size) for indices in classes]
    )
    rng.shuffle(result)
    return result


def _fit_gate(
    features: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    config: GateConfig,
    seed: int,
) -> tuple[StandardScaler, MLPClassifier]:
    target = labels[train_mask]
    balanced = _balanced_indices(target, seed)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[train_mask][balanced])
    model = MLPClassifier(
        hidden_layer_sizes=config.hidden_layers,
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        batch_size=min(32, len(balanced)),
        learning_rate_init=1e-3,
        max_iter=config.epochs,
        shuffle=True,
        random_state=seed,
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(scaled, target[balanced])
    return scaler, model


def _apply_positive_residual(reference: np.ndarray, active: np.ndarray) -> np.ndarray:
    decision = np.asarray(reference, dtype=bool).copy()
    decision[np.asarray(active, dtype=bool)] = True
    return decision


def _passes_stability(reference: dict[str, float], candidate: dict[str, float]) -> bool:
    if candidate["balanced_accuracy"] <= reference["balanced_accuracy"] + 1e-12:
        return False
    for name, tolerance in STABILITY_TOLERANCES.items():
        if candidate[name] < reference[name] - tolerance - 1e-12:
            return False
    for name in ("auroc", "auprc"):
        if abs(candidate[name] - reference[name]) > 1e-12:
            return False
    for name in ("brier", "ece_10"):
        if abs(candidate[name] - reference[name]) > 1e-12:
            return False
    return True


def _metrics_delta(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    names = (
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
    return {name: float(candidate[name] - reference[name]) for name in names}


def _transition_report(reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    changed = np.asarray(reference, dtype=bool) != np.asarray(candidate, dtype=bool)
    corrected = changed & (candidate == labels)
    regressed = changed & (candidate != labels)
    return {
        "changed": int(changed.sum()),
        "corrected": int(corrected.sum()),
        "regressed": int(regressed.sum()),
        "net": int(corrected.sum() - regressed.sum()),
    }


def _crossfit_gate(
    features: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    candidate_mask: np.ndarray,
    config: GateConfig,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    gate = np.zeros(len(labels), dtype=np.float64)
    reports = []
    for held_fold in range(5):
        validation_fold = (held_fold + 1) % 5
        train = candidate_mask & (folds != held_fold) & (folds != validation_fold)
        held = candidate_mask & (folds == held_fold)
        scaler, model = _fit_gate(features, labels, train, config, seed + held_fold)
        if np.any(held):
            gate[held] = model.predict_proba(scaler.transform(features[held]))[:, 1]
        reports.append(
            {
                "held_fold": held_fold,
                "validation_fold": validation_fold,
                "train_candidate_rows": int(train.sum()),
                "held_candidate_rows": int(held.sum()),
            }
        )
    return gate, reports


def _select_gate_threshold(
    labels: np.ndarray,
    probability: np.ndarray,
    reference_decision: np.ndarray,
    candidate_mask: np.ndarray,
    gate: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, float]]:
    reference = _decision_metrics(labels, probability, reference_decision, BASE_THRESHOLD)
    best: tuple[tuple[float, ...], float, dict[str, float]] | None = None
    for threshold in np.unique(gate[candidate_mask]):
        active = candidate_mask & (gate >= threshold)
        decision = _apply_positive_residual(reference_decision, active)
        metrics = _decision_metrics(labels, probability, decision, BASE_THRESHOLD)
        if not _passes_stability(reference, metrics):
            continue
        score = (metrics["balanced_accuracy"], metrics["mcc"], metrics["f1"], -active.sum())
        if best is None or score > best[0]:
            best = (score, float(threshold), metrics)
    if best is None:
        return 1.1, reference, reference
    return best[1], reference, best[2]


def _write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    probability: np.ndarray,
    reference: np.ndarray,
    prott5: np.ndarray,
    esm2: np.ndarray,
    gate: np.ndarray,
    active: np.ndarray,
    decision: np.ndarray,
) -> None:
    fieldnames = [
        "source_index",
        "sample_id",
        "sequence",
        "label",
        "toxicity_probability",
        "reference_prediction",
        "prott5_teacher_probability",
        "esm2_teacher_probability",
        "multi_teacher_gate_probability",
        "multi_teacher_residual_active",
        "prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": row.get("source_index", index),
                    "sample_id": row.get("sample_id", index),
                    "sequence": row["sequence"],
                    "label": int(row["label"]),
                    "toxicity_probability": f"{probability[index]:.10g}",
                    "reference_prediction": int(reference[index]),
                    "prott5_teacher_probability": f"{prott5[index]:.10g}",
                    "esm2_teacher_probability": f"{esm2[index]:.10g}",
                    "multi_teacher_gate_probability": f"{gate[index]:.10g}",
                    "multi_teacher_residual_active": int(active[index]),
                    "prediction": int(decision[index]),
                }
            )


def _serialize_gate(
    path: Path,
    scaler: StandardScaler,
    model: MLPClassifier,
    config: GateConfig,
    gate_threshold: float,
    feature_dimension: int,
) -> None:
    save_json(
        {
            "artifact_type": "ghxtox_multi_teacher_bacc_residual",
            "version": 1,
            "feature_schema": {
                "name": "prott5_esm2_recall_gate_v1",
                "dimension": feature_dimension,
            },
            "teachers": {
                "prott5_threshold": PROTT5_THRESHOLD,
                "esm2_threshold": ESM2_THRESHOLD,
            },
            "base_threshold": BASE_THRESHOLD,
            "gate_threshold": gate_threshold,
            "decision_only": True,
            "positive_promotions_only": True,
            "model": {
                "hidden_layers": list(config.hidden_layers),
                "epochs": config.epochs,
                "alpha": config.alpha,
                "coefs": [value.tolist() for value in model.coefs_],
                "intercepts": [value.tolist() for value in model.intercepts_],
            },
            "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
            "selection_source": "training-only group-aware OOF",
        },
        path,
    )


def _load_aligned(
    context_path: str | Path,
    current_path: str | Path,
    prott5_path: str | Path,
    esm2_path: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    return _align_tables(
        _read_csv(context_path),
        _read_csv(current_path),
        _read_csv(prott5_path),
        _read_csv(esm2_path),
    )


def run_screen(
    *,
    context_oof_path: str | Path,
    current_oof_path: str | Path,
    prott5_oof_path: str | Path,
    esm2_oof_path: str | Path,
    current_test1_path: str | Path,
    prott5_test1_path: str | Path,
    esm2_test1_path: str | Path,
    current_test2_path: str | Path,
    prott5_test2_path: str | Path,
    esm2_test2_path: str | Path,
    output_dir: str | Path,
    seed: int = 20260803,
) -> dict[str, Any]:
    context, current_rows, prott5_rows, esm2_rows = _load_aligned(
        context_oof_path, current_oof_path, prott5_oof_path, esm2_oof_path
    )
    for context_row, current_row in zip(context, current_rows, strict=True):
        current_row.setdefault("baseline_probability", context_row["baseline_probability"])
        current_row.setdefault("interaction_delta", context_row["interaction_delta"])
    labels = np.asarray([int(row["label"]) for row in context], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in context], dtype=np.int64)
    if set(folds) != set(range(5)):
        raise ValueError("OOF context must contain canonical folds 0-4.")
    probability = _probability(current_rows, "toxicity_probability")
    reference_decision = np.asarray([int(row["prediction"]) for row in current_rows], dtype=bool)
    prott5 = _probability(prott5_rows, "toxicity_probability")
    esm2 = _probability(esm2_rows, "toxicity_probability")
    features = _features(current_rows, prott5_rows, esm2_rows)
    candidates = _candidate_mask(reference_decision, prott5, esm2)

    screens = []
    gates: dict[str, np.ndarray] = {}
    for config in CONFIGS:
        gate, fold_reports = _crossfit_gate(features, labels, folds, candidates, config, seed)
        gate_threshold, reference, metrics = _select_gate_threshold(
            labels, probability, reference_decision, candidates, gate
        )
        active = candidates & (gate >= gate_threshold)
        decision = _apply_positive_residual(reference_decision, active)
        report = {
            "config": {
                "name": config.name,
                "hidden_layers": list(config.hidden_layers),
                "epochs": config.epochs,
                "alpha": config.alpha,
            },
            "reference": reference,
            "metrics": metrics,
            "delta": _metrics_delta(reference, metrics),
            "gate_threshold": gate_threshold,
            "active_rows": int(active.sum()),
            "transitions": _transition_report(reference_decision, decision, labels),
            "passed_stability_gate": _passes_stability(reference, metrics),
            "folds": fold_reports,
        }
        screens.append(report)
        gates[config.name] = gate

    eligible = [report for report in screens if report["passed_stability_gate"]]
    if eligible:
        best_bacc = max(report["metrics"]["balanced_accuracy"] for report in eligible)
        near_best = [
            report
            for report in eligible
            if report["metrics"]["balanced_accuracy"] >= best_bacc - COMPLEXITY_BACC_MARGIN
        ]
        selected = min(
            near_best,
            key=lambda report: (
                len(report["config"]["hidden_layers"]),
                sum(report["config"]["hidden_layers"]),
                report["config"]["epochs"],
            ),
        )
        selection_reason = "simplest stable model within 0.001 BACC of the best OOF screen"
    else:
        selected = max(
            screens,
            key=lambda report: (report["metrics"]["balanced_accuracy"], report["metrics"]["mcc"]),
        )
        selection_reason = "no OOF candidate passed the predeclared stability gate"

    config = next(value for value in CONFIGS if value.name == selected["config"]["name"])
    gate = gates[config.name]
    gate_threshold = float(selected["gate_threshold"])
    active = candidates & (gate >= gate_threshold)
    oof_decision = _apply_positive_residual(reference_decision, active)
    full_scaler, full_model = _fit_gate(features, labels, candidates, config, seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _serialize_gate(
        output / "research_gate.json",
        full_scaler,
        full_model,
        config,
        gate_threshold,
        features.shape[1],
    )
    _write_predictions(
        output / "oof_predictions.csv",
        current_rows,
        probability,
        reference_decision,
        prott5,
        esm2,
        gate,
        active,
        oof_decision,
    )

    evaluations: dict[str, Any] = {}
    eval_inputs = {
        "test1": (current_test1_path, prott5_test1_path, esm2_test1_path),
        "test2": (current_test2_path, prott5_test2_path, esm2_test2_path),
    }
    for name, (current_path, prott5_path, esm2_path) in eval_inputs.items():
        eval_current_raw = _read_csv(current_path)
        eval_current, eval_prott5, eval_esm2 = _align_tables(
            eval_current_raw, eval_current_raw, _read_csv(prott5_path), _read_csv(esm2_path)
        )[1:]
        eval_labels = np.asarray([int(row["label"]) for row in eval_current], dtype=np.int64)
        eval_probability = _probability(eval_current, "toxicity_probability")
        eval_reference = np.asarray([int(row["prediction"]) for row in eval_current], dtype=bool)
        eval_prott5_probability = _probability(eval_prott5, "toxicity_probability")
        eval_esm2_probability = _probability(eval_esm2, "toxicity_probability")
        eval_features = _features(eval_current, eval_prott5, eval_esm2)
        eval_candidates = _candidate_mask(
            eval_reference, eval_prott5_probability, eval_esm2_probability
        )
        eval_gate = np.zeros(len(eval_labels), dtype=np.float64)
        if np.any(eval_candidates):
            eval_gate[eval_candidates] = full_model.predict_proba(
                full_scaler.transform(eval_features[eval_candidates])
            )[:, 1]
        eval_active = eval_candidates & (eval_gate >= gate_threshold)
        eval_decision = _apply_positive_residual(eval_reference, eval_active)
        reference_metrics = _decision_metrics(
            eval_labels, eval_probability, eval_reference, BASE_THRESHOLD
        )
        metrics = _decision_metrics(eval_labels, eval_probability, eval_decision, BASE_THRESHOLD)
        evaluations[name] = {
            "rows": len(eval_labels),
            "candidate_rows": int(eval_candidates.sum()),
            "active_rows": int(eval_active.sum()),
            "reference": reference_metrics,
            "metrics": metrics,
            "delta": _metrics_delta(reference_metrics, metrics),
            "transitions": _transition_report(eval_reference, eval_decision, eval_labels),
            "passed_stability_gate": _passes_stability(reference_metrics, metrics),
        }
        _write_predictions(
            output / f"{name}_predictions.csv",
            eval_current,
            eval_probability,
            eval_reference,
            eval_prott5_probability,
            eval_esm2_probability,
            eval_gate,
            eval_active,
            eval_decision,
        )

    oof_passed = bool(selected["passed_stability_gate"])
    historical_safe = all(
        report["metrics"]["balanced_accuracy"] >= report["reference"]["balanced_accuracy"]
        and report["delta"]["accuracy"] >= -STABILITY_TOLERANCES["accuracy"]
        and report["delta"]["mcc"] >= -STABILITY_TOLERANCES["mcc"]
        and report["delta"]["specificity"] >= -STABILITY_TOLERANCES["specificity"]
        for report in evaluations.values()
    )
    historical_bacc_gain = any(
        report["metrics"]["balanced_accuracy"]
        > report["reference"]["balanced_accuracy"] + 1e-12
        for report in evaluations.values()
    )
    promotion_authorized = bool(oof_passed and historical_safe and historical_bacc_gain)
    deployable_gate = output / "deployable_gate.json"
    if promotion_authorized:
        (output / "research_gate.json").replace(deployable_gate)
    else:
        deployable_gate.unlink(missing_ok=True)
    summary = {
        "protocol": {
            "name": "constrained ProtT5+ESM2 positive-recall residual",
            "selection_data": "group-aware training OOF only",
            "historical_test_role": "frozen migration audit; not used for architecture or threshold selection",
            "base_model": "current GHXTox public default including the ProtT5 decision residual",
            "inputs": {
                "context_oof": str(context_oof_path),
                "current_oof": str(current_oof_path),
                "prott5_oof": str(prott5_oof_path),
                "esm2_oof": str(esm2_oof_path),
                "current_test1": str(current_test1_path),
                "prott5_test1": str(prott5_test1_path),
                "esm2_test1": str(esm2_test1_path),
                "current_test2": str(current_test2_path),
                "prott5_test2": str(prott5_test2_path),
                "esm2_test2": str(esm2_test2_path),
            },
            "prott5_threshold": PROTT5_THRESHOLD,
            "esm2_threshold": ESM2_THRESHOLD,
            "base_threshold": BASE_THRESHOLD,
            "stability_tolerances": STABILITY_TOLERANCES,
            "probability_preserved": True,
            "seed": seed,
        },
        "coverage": {"oof_rows": len(labels), "oof_candidate_rows": int(candidates.sum())},
        "screens": screens,
        "selected": selected,
        "selection_reason": selection_reason,
        "evaluations": evaluations,
        "decision": "promote_multi_teacher_gate" if promotion_authorized else "retain_existing_default",
        "promotion_authorized": promotion_authorized,
        "historical_migration": {
            "nondegrading": historical_safe,
            "at_least_one_bacc_gain": historical_bacc_gain,
        },
    }
    save_json(summary, output / "summary.json")
    save_json(
        {
            "status": "complete",
            "decision": summary["decision"],
            "promotion_authorized": promotion_authorized,
            "selected_config": selected["config"],
            "gate_threshold": selected["gate_threshold"],
            "oof": selected,
            "evaluations": evaluations,
        },
        output / "decision.json",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-oof", required=True)
    parser.add_argument("--current-oof", required=True)
    parser.add_argument("--prott5-oof", required=True)
    parser.add_argument("--esm2-oof", required=True)
    parser.add_argument("--current-test1", required=True)
    parser.add_argument("--prott5-test1", required=True)
    parser.add_argument("--esm2-test1", required=True)
    parser.add_argument("--current-test2", required=True)
    parser.add_argument("--prott5-test2", required=True)
    parser.add_argument("--esm2-test2", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_screen(
        context_oof_path=args.context_oof,
        current_oof_path=args.current_oof,
        prott5_oof_path=args.prott5_oof,
        esm2_oof_path=args.esm2_oof,
        current_test1_path=args.current_test1,
        prott5_test1_path=args.prott5_test1,
        esm2_test1_path=args.esm2_test1,
        current_test2_path=args.current_test2,
        prott5_test2_path=args.prott5_test2,
        esm2_test2_path=args.esm2_test2,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary["selected"], indent=2))
    print(json.dumps(summary["evaluations"], indent=2))
    print(summary["decision"])


if __name__ == "__main__":
    main()
