"""Frozen train-only bounded fusion probe for the 200-row BioEmu pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ghxtox.oof_source_diagnostics import (
    CONFORMER_FEATURES,
    _read_csv,
    _source_entries,
)
from ghxtox.utils import save_json


DEFAULT_SEED = 20260731
DEFAULT_ONE_D_THRESHOLD = 0.45
DEFAULT_RESIDUAL_WEIGHT = 0.25
DEFAULT_RESIDUAL_CLIP = 1.0


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    positive = labels == 1
    negative = ~positive
    tp = float(np.sum(predictions & positive))
    tn = float(np.sum(~predictions & negative))
    fp = float(np.sum(predictions & negative))
    fn = float(np.sum(~predictions & positive))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(selected):
            ece += float(
                np.mean(selected)
                * abs(
                    float(np.mean(probabilities[selected]))
                    - float(np.mean(labels[selected]))
                )
            )
    if len(np.unique(labels)) > 1:
        auroc = float(roc_auc_score(labels, probabilities))
        auprc = float(average_precision_score(labels, probabilities))
    else:
        auroc = float("nan")
        auprc = float("nan")
    return {
        "samples": int(len(labels)),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": (tp * tn - fp * fn) / denominator,
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece_10": ece,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _comparison(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    baseline_predictions = baseline_probabilities >= threshold
    candidate_predictions = candidate_probabilities >= threshold
    baseline_correct = baseline_predictions == labels
    candidate_correct = candidate_predictions == labels
    corrected = ~baseline_correct & candidate_correct
    newly_incorrect = baseline_correct & ~candidate_correct
    return {
        "samples": int(len(labels)),
        "mean_probability_delta": float(
            np.mean(candidate_probabilities - baseline_probabilities)
        ),
        "mean_absolute_probability_delta": float(
            np.mean(np.abs(candidate_probabilities - baseline_probabilities))
        ),
        "decision_disagreement_count": int(
            np.sum(baseline_predictions != candidate_predictions)
        ),
        "decision_disagreement_rate": float(
            np.mean(baseline_predictions != candidate_predictions)
        ),
        "corrected_by_candidate": int(np.sum(corrected)),
        "newly_incorrect": int(np.sum(newly_incorrect)),
        "net_corrected": int(np.sum(corrected) - np.sum(newly_incorrect)),
    }


def _evaluate_subset(
    selected: np.ndarray,
    labels: np.ndarray,
    one_d_probabilities: np.ndarray,
    structure_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return {
        "one_d": _metrics(labels[selected], one_d_probabilities[selected], threshold),
        "structure_only": _metrics(
            labels[selected], structure_probabilities[selected], 0.5
        ),
        "bounded_candidate": _metrics(
            labels[selected], candidate_probabilities[selected], threshold
        ),
        "bounded_candidate_vs_one_d": _comparison(
            labels[selected],
            one_d_probabilities[selected],
            candidate_probabilities[selected],
            threshold,
        ),
    }


def evaluate_bioemu_bounded_fusion(
    frozen_manifest: str | Path,
    bioemu_manifest: str | Path,
    bioemu_cache_root: str | Path,
    one_d_oof: str | Path,
    feasibility_json: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    one_d_threshold: float = DEFAULT_ONE_D_THRESHOLD,
    residual_weight: float = DEFAULT_RESIDUAL_WEIGHT,
    residual_clip: float = DEFAULT_RESIDUAL_CLIP,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run the single preregistered bounded candidate on canonical OOF folds."""

    frozen_rows = _read_csv(frozen_manifest)
    bioemu_rows = _read_csv(bioemu_manifest)
    bioemu_by_index = {int(row["source_index"]): row for row in bioemu_rows}
    bioemu_features = _source_entries(
        [bioemu_manifest], [bioemu_cache_root], ["BioEmu-v1.2"]
    )
    one_d = {int(row["source_index"]): row for row in _read_csv(one_d_oof)}
    feasibility = json.loads(Path(feasibility_json).read_text(encoding="utf-8"))
    strict_usable = {
        str(row["sample_id"]): bool(row["usable"]) for row in feasibility["samples"]
    }

    rows: list[dict[str, Any]] = []
    feature_values = []
    labels = []
    folds = []
    one_d_probabilities = []
    quality_gates = []
    for frozen in frozen_rows:
        source_index = int(frozen["source_index"])
        sample_id = str(frozen["sample_id"])
        bioemu_row = bioemu_by_index.get(source_index)
        one_d_row = one_d.get(source_index)
        features = bioemu_features.get(sample_id)
        if bioemu_row is None or one_d_row is None or features is None:
            raise ValueError(f"Missing aligned input for source_index {source_index}.")
        label = int(frozen["label"])
        if int(bioemu_row["label"]) != label or int(one_d_row["label"]) != label:
            raise ValueError(f"Label mismatch for source_index {source_index}.")
        if str(bioemu_row["sequence"]) != str(frozen["sequence"]):
            raise ValueError(f"Sequence mismatch for source_index {source_index}.")
        requested = int(frozen["requested_conformers"])
        valid = int(bioemu_row["conformer_count"])
        quality_gate = float(np.clip(valid / requested, 0.0, 1.0))
        vector = [float(features[name]) for name in CONFORMER_FEATURES]
        feature_values.append(vector)
        labels.append(label)
        folds.append(int(frozen["fold"]))
        one_d_probabilities.append(float(one_d_row["toxicity_probability"]))
        quality_gates.append(quality_gate)
        rows.append(
            {
                "source_index": source_index,
                "sample_id": sample_id,
                "fold": int(frozen["fold"]),
                "label": label,
                "length": int(frozen["length"]),
                "length_stratum": str(frozen["length_bin"]),
                "valid_conformers": valid,
                "requested_conformers": requested,
                "quality_gate": quality_gate,
                "strict_usable": strict_usable.get(sample_id, False),
                **{name: value for name, value in zip(CONFORMER_FEATURES, vector)},
            }
        )

    features_array = np.asarray(feature_values, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    folds_array = np.asarray(folds, dtype=np.int64)
    one_d_array = np.asarray(one_d_probabilities, dtype=np.float64)
    quality_array = np.asarray(quality_gates, dtype=np.float64)
    structure_probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    fold_models = []
    for fold in sorted(np.unique(folds_array).tolist()):
        train = folds_array != fold
        test = folds_array == fold
        if len(np.unique(labels_array[train])) != 2:
            raise ValueError(f"Fold {fold} training partition lacks both labels.")
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features_array[train])
        test_features = scaler.transform(features_array[test])
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            penalty="l2",
            max_iter=1000,
            random_state=seed,
        )
        classifier.fit(train_features, labels_array[train])
        structure_probabilities[test] = classifier.predict_proba(test_features)[:, 1]
        fold_models.append(
            {
                "fold": int(fold),
                "train_samples": int(np.sum(train)),
                "test_samples": int(np.sum(test)),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coefficient": classifier.coef_[0].tolist(),
                "intercept": float(classifier.intercept_[0]),
            }
        )
    if not np.all(np.isfinite(structure_probabilities)):
        raise RuntimeError("Cross-fitted structure probabilities are incomplete.")

    structure_residual = np.clip(
        _logit(structure_probabilities), -residual_clip, residual_clip
    )
    candidate_logits = (
        _logit(one_d_array) + residual_weight * quality_array * structure_residual
    )
    candidate_probabilities = _sigmoid(candidate_logits)
    for index, row in enumerate(rows):
        row["one_d_probability"] = float(one_d_array[index])
        row["structure_probability"] = float(structure_probabilities[index])
        row["bounded_residual"] = float(
            residual_weight * quality_array[index] * structure_residual[index]
        )
        row["candidate_probability"] = float(candidate_probabilities[index])
        row["one_d_prediction"] = int(one_d_array[index] >= one_d_threshold)
        row["candidate_prediction"] = int(
            candidate_probabilities[index] >= one_d_threshold
        )

    all_rows = np.ones(len(rows), dtype=bool)
    by_length = {
        length_stratum: _evaluate_subset(
            np.asarray(
                [row["length_stratum"] == length_stratum for row in rows],
                dtype=bool,
            ),
            labels_array,
            one_d_array,
            structure_probabilities,
            candidate_probabilities,
            one_d_threshold,
        )
        for length_stratum in ("11-15", "16-30", "31-50")
    }
    by_fold = {
        str(fold): _evaluate_subset(
            folds_array == fold,
            labels_array,
            one_d_array,
            structure_probabilities,
            candidate_probabilities,
            one_d_threshold,
        )
        for fold in sorted(np.unique(folds_array).tolist())
    }
    primary = _evaluate_subset(
        all_rows,
        labels_array,
        one_d_array,
        structure_probabilities,
        candidate_probabilities,
        one_d_threshold,
    )
    usable_mask = np.asarray([bool(row["strict_usable"]) for row in rows])
    sensitivity = _evaluate_subset(
        usable_mask,
        labels_array,
        one_d_array,
        structure_probabilities,
        candidate_probabilities,
        one_d_threshold,
    )

    baseline = primary["one_d"]
    candidate = primary["bounded_candidate"]
    comparison = primary["bounded_candidate_vs_one_d"]
    nondecreasing_folds = sum(
        fold_report["bounded_candidate"]["mcc"] >= fold_report["one_d"]["mcc"] - 1e-12
        for fold_report in by_fold.values()
    )
    gate = {
        "mcc_strictly_higher": candidate["mcc"] > baseline["mcc"],
        "auprc_drop_at_most_0_002": candidate["auprc"] >= baseline["auprc"] - 0.002,
        "brier_increase_at_most_0_002": candidate["brier"] <= baseline["brier"] + 0.002,
        "ece_10_increase_at_most_0_005": candidate["ece_10"]
        <= baseline["ece_10"] + 0.005,
        "net_corrected_positive": comparison["net_corrected"] > 0,
        "at_least_three_fold_mcc_nondecreasing": nondecreasing_folds >= 3,
    }
    gate["all_conditions_passed"] = all(gate.values())
    decision_name = (
        "retain_bounded_bioemu_candidate"
        if gate["all_conditions_passed"]
        else "stop_bounded_bioemu_candidate"
    )
    report = {
        "protocol": {
            "scope": "frozen 200-row training-only BioEmu bounded fusion probe",
            "external_test_labels_read": False,
            "features": list(CONFORMER_FEATURES),
            "canonical_folds": sorted(np.unique(folds_array).tolist()),
            "structure_model": {
                "standardization": "training partition only",
                "model": "L2 LogisticRegression",
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "liblinear",
                "seed": seed,
            },
            "fusion_formula": (
                "logit(p_final)=logit(p_1d)+0.25*(valid_conformers/32)"
                "*clip(logit(p_3d),-1,1)"
            ),
            "one_d_threshold": one_d_threshold,
            "residual_weight": residual_weight,
            "residual_clip": residual_clip,
            "selection_warning": "Single preregistered candidate; no parameter search.",
        },
        "cohort": {
            "samples": len(rows),
            "positive": int(labels_array.sum()),
            "negative": int(len(rows) - labels_array.sum()),
            "strict_usable": int(np.sum(usable_mask)),
        },
        "primary_all_200": primary,
        "by_length": by_length,
        "by_fold": by_fold,
        "strict_usable_198_sensitivity": sensitivity,
        "fold_models": fold_models,
        "retention_gate": {
            **gate,
            "fold_mcc_nondecreasing_count": int(nondecreasing_folds),
        },
        "decision": decision_name,
    }
    decision = {
        "status": "complete",
        "decision": decision_name,
        "bounded_candidate_retained": bool(gate["all_conditions_passed"]),
        "full_generation_authorized": False,
        "default_model_change_authorized": False,
        "external_test_labels_read": False,
        "retention_gate": report["retention_gate"],
        "primary_metrics": {
            "one_d": baseline,
            "bounded_candidate": candidate,
            "comparison": comparison,
        },
        "next_authorized_stage": (
            "independent train-only validation before any generation expansion"
            if gate["all_conditions_passed"]
            else "keep the frozen 1D anchor and stop this bounded BioEmu candidate"
        ),
    }
    return report, rows, decision


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
    parser.add_argument("--one-d-oof", required=True)
    parser.add_argument("--feasibility-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-decision", required=True)
    args = parser.parse_args()
    report, rows, decision = evaluate_bioemu_bounded_fusion(
        args.frozen_manifest,
        args.bioemu_manifest,
        args.bioemu_cache_root,
        args.one_d_oof,
        args.feasibility_json,
    )
    save_json(report, args.output_json)
    save_json(decision, args.output_decision)
    _write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
