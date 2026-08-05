"""Cross-fitted shared-slope, length-intercept calibration for retained 3D OOF.

This is an exploratory screening layer, not a directly promotable handoff. The
already-generated Phase 25 OOF covariates are not strict nested upstream
predictions for a second-level learner. A passing screen can only authorize a
future strict nested retraining of the complete Phase 25 path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

from ghxtox.bootstrap_ci import _metrics
from ghxtox.oof_source_diagnostics import _read_csv
from ghxtox.utils import save_json


DEFAULT_PROTOCOL = "runs/length_calibration_3d/protocol.json"


@dataclass(frozen=True)
class LengthCalibrator:
    alpha: float
    global_intercept: float
    short_intercept: float
    long_intercept: float

    @property
    def slope(self) -> float:
        return float(np.exp(self.alpha))

    def transform(self, logits: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        short = ((lengths >= 11) & (lengths <= 15)).astype(np.float64)
        long = ((lengths >= 31) & (lengths <= 50)).astype(np.float64)
        calibrated_logit = (
            self.slope * logits
            + self.global_intercept
            + self.short_intercept * short
            + self.long_intercept * long
        )
        return expit(calibrated_logit).astype(np.float64, copy=False)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 31 protocol must be frozen before execution.")
    if bool(protocol.get("external_test_labels_read", True)):
        raise ValueError("Phase 31 must keep Test1/Test2 labels sealed.")
    if (
        protocol.get("evaluation_tier")
        != "cross_fitted_screen_not_strict_nested_upstream"
    ):
        raise ValueError("Phase 31 must declare its non-nested upstream limitation.")
    if bool(protocol.get("direct_handoff_promotion_allowed", True)):
        raise ValueError("This screen may not directly promote the 3D handoff.")
    for input_path, expected in protocol["inputs"]["sha256"].items():
        actual = _sha256(input_path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {input_path}: {actual} != {expected}."
            )
    return protocol


def _safe_logit(probabilities: np.ndarray, clip: float) -> np.ndarray:
    bounded = np.clip(probabilities.astype(np.float64), clip, 1.0 - clip)
    return np.log(bounded) - np.log1p(-bounded)


def _length_stratum(length: int) -> str:
    if 11 <= length <= 15:
        return "11-15"
    if 16 <= length <= 30:
        return "16-30"
    if 31 <= length <= 50:
        return "31-50"
    raise ValueError(f"Unsupported peptide length {length}; expected 11-50.")


def _objective_and_gradient(
    parameters: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    slope_log_l2: float,
    length_offset_l2: float,
) -> tuple[float, np.ndarray]:
    alpha, global_intercept, short_intercept, long_intercept = parameters
    slope = float(np.exp(alpha))
    short = ((lengths >= 11) & (lengths <= 15)).astype(np.float64)
    long = ((lengths >= 31) & (lengths <= 50)).astype(np.float64)
    calibrated_logit = (
        slope * logits
        + global_intercept
        + short_intercept * short
        + long_intercept * long
    )
    residual = expit(calibrated_logit) - labels
    objective = float(
        np.mean(np.logaddexp(0.0, calibrated_logit) - labels * calibrated_logit)
        + slope_log_l2 * alpha**2
        + length_offset_l2 * (short_intercept**2 + long_intercept**2)
    )
    gradient = np.asarray(
        [
            np.mean(residual * slope * logits) + 2.0 * slope_log_l2 * alpha,
            np.mean(residual),
            np.mean(residual * short) + 2.0 * length_offset_l2 * short_intercept,
            np.mean(residual * long) + 2.0 * length_offset_l2 * long_intercept,
        ],
        dtype=np.float64,
    )
    return objective, gradient


def fit_length_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    config: dict[str, Any],
) -> tuple[LengthCalibrator, dict[str, Any]]:
    bounds = config["alpha_bounds"]
    slope_log_l2 = float(config["slope_log_l2"])
    length_offset_l2 = float(config["length_offset_l2"])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_and_gradient(
            parameters,
            logits,
            labels,
            lengths,
            slope_log_l2,
            length_offset_l2,
        )

    result = minimize(
        objective,
        np.zeros(4, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (float(bounds[0]), float(bounds[1])),
            (None, None),
            (None, None),
            (None, None),
        ],
        options={
            "maxiter": int(config["maximum_iterations"]),
            "gtol": float(config["gradient_tolerance"]),
            "ftol": float(config["function_tolerance"]),
        },
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    calibrator = LengthCalibrator(
        alpha=float(parameters[0]),
        global_intercept=float(parameters[1]),
        short_intercept=float(parameters[2]),
        long_intercept=float(parameters[3]),
    )
    fit_report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "objective": float(result.fun),
        "gradient_max_abs": float(np.max(np.abs(result.jac))),
        "alpha": calibrator.alpha,
        "slope": calibrator.slope,
        "global_intercept": calibrator.global_intercept,
        "short_intercept": calibrator.short_intercept,
        "long_intercept": calibrator.long_intercept,
    }
    return calibrator, fit_report


def cross_fit_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    lengths: np.ndarray,
    folds: np.ndarray,
    model_config: dict[str, Any],
    optimization_config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    clip = float(model_config["probability_clip"])
    logits = _safe_logit(probabilities, clip)
    calibrated = np.full(probabilities.shape, np.nan, dtype=np.float64)
    reports = []
    fit_config = {**model_config, **optimization_config}
    for held_fold in sorted(np.unique(folds).tolist()):
        train = folds != held_fold
        held = folds == held_fold
        calibrator, fit_report = fit_length_calibrator(
            logits[train], labels[train], lengths[train], fit_config
        )
        calibrated[held] = calibrator.transform(logits[held], lengths[held])
        reports.append(
            {
                "held_fold": int(held_fold),
                "training_rows": int(train.sum()),
                "held_rows": int(held.sum()),
                **fit_report,
            }
        )
    if not np.isfinite(calibrated).all():
        raise RuntimeError("Cross-fitted calibration did not cover every row.")
    return calibrated, reports


def _metrics_with_confusion(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    metrics: dict[str, Any] = _metrics(labels, probabilities, threshold)
    predicted = probabilities >= threshold
    negatives = labels == 0
    positives = labels == 1
    metrics["confusion_matrix"] = [
        [
            int(np.sum(~predicted & negatives)),
            int(np.sum(predicted & negatives)),
        ],
        [
            int(np.sum(~predicted & positives)),
            int(np.sum(predicted & positives)),
        ],
    ]
    return metrics


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_aligned_rows(protocol: dict[str, Any]) -> list[dict[str, str]]:
    inputs = protocol["inputs"]
    rows = sorted(
        _read_csv(inputs["reference_oof"]), key=lambda row: int(row["source_index"])
    )
    folds = {
        int(row["source_index"]): row for row in _read_csv(inputs["fold_manifest"])
    }
    if len(rows) != len({int(row["source_index"]) for row in rows}):
        raise ValueError("Phase 25 OOF contains duplicate source indices.")
    if set(folds) != {int(row["source_index"]) for row in rows}:
        raise ValueError("Phase 25 OOF and canonical fold manifest do not align.")
    for row in rows:
        source_index = int(row["source_index"])
        fold_row = folds[source_index]
        for field in ("fold", "label", "sequence"):
            if str(row[field]) != str(fold_row[field]):
                raise ValueError(f"Canonical {field} mismatch at {source_index}.")
        length = int(row["length"])
        if length != len(row["sequence"]):
            raise ValueError(f"Sequence length mismatch at {source_index}.")
        if row["length_stratum"] != _length_stratum(length):
            raise ValueError(f"Length stratum mismatch at {source_index}.")
    expected = protocol["cohort"]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    observed = {
        "rows": len(rows),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "folds": len({int(row["fold"]) for row in rows}),
    }
    for key, value in observed.items():
        if int(expected[key]) != int(value):
            raise ValueError(f"Frozen cohort mismatch for {key}: {value}.")
    for name, count in expected["length_strata"].items():
        if int(
            np.sum([_length_stratum(int(value)) == name for value in lengths])
        ) != int(count):
            raise ValueError(f"Frozen length cohort mismatch for {name}.")
    return rows


def _validate_reference(
    protocol: dict[str, Any],
    labels: np.ndarray,
    lengths: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    evaluation = protocol["evaluation"]
    threshold = float(evaluation["reference_threshold"])
    actual = _metrics(labels, probabilities, threshold)
    for name, expected in evaluation["reference"].items():
        if abs(float(actual[name]) - float(expected)) > 1e-9:
            raise ValueError(f"Frozen reference {name} mismatch: {actual[name]}.")
    for stratum in ("11-15", "16-30", "31-50"):
        selected = np.asarray(
            [_length_stratum(int(length)) == stratum for length in lengths]
        )
        current = _metrics(labels[selected], probabilities[selected], threshold)
        if (
            abs(current["mcc"] - float(evaluation["length_reference_mcc"][stratum]))
            > 1e-9
        ):
            raise ValueError(f"Frozen {stratum} reference MCC mismatch.")
        if (
            abs(current["auprc"] - float(evaluation["length_reference_auprc"][stratum]))
            > 1e-9
        ):
            raise ValueError(f"Frozen {stratum} reference AUPRC mismatch.")


def run_phase31(
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    output_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _load_protocol(protocol_path)
    rows = _load_aligned_rows(protocol)
    output = Path(output_root or Path(protocol_path).parent)
    output.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    lengths = np.asarray([int(row["length"]) for row in rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    reference_probability = np.asarray(
        [float(row["toxicity_probability"]) for row in rows], dtype=np.float64
    )
    _validate_reference(protocol, labels, lengths, reference_probability)

    candidate_probability, coefficients = cross_fit_probabilities(
        reference_probability,
        labels,
        lengths,
        folds,
        protocol["model"],
        protocol["optimization"],
    )
    evaluation = protocol["evaluation"]
    reference_threshold = float(evaluation["reference_threshold"])
    candidate_threshold = float(evaluation["calibrated_threshold"])
    reference_metrics = _metrics_with_confusion(
        labels, reference_probability, reference_threshold
    )
    candidate_metrics = _metrics_with_confusion(
        labels, candidate_probability, candidate_threshold
    )

    by_fold: dict[str, Any] = {}
    fold_non_decreases = 0
    for fold in sorted(np.unique(folds).tolist()):
        selected = folds == fold
        reference_fold = _metrics_with_confusion(
            labels[selected], reference_probability[selected], reference_threshold
        )
        candidate_fold = _metrics_with_confusion(
            labels[selected], candidate_probability[selected], candidate_threshold
        )
        if candidate_fold["mcc"] >= reference_fold["mcc"]:
            fold_non_decreases += 1
        by_fold[str(fold)] = {
            "reference": reference_fold,
            "candidate": candidate_fold,
        }

    by_length: dict[str, Any] = {}
    for stratum in ("11-15", "16-30", "31-50"):
        selected = np.asarray(
            [_length_stratum(int(length)) == stratum for length in lengths]
        )
        by_length[stratum] = {
            "rows": int(selected.sum()),
            "reference": _metrics_with_confusion(
                labels[selected], reference_probability[selected], reference_threshold
            ),
            "candidate": _metrics_with_confusion(
                labels[selected], candidate_probability[selected], candidate_threshold
            ),
        }

    reference_prediction = reference_probability >= reference_threshold
    candidate_prediction = candidate_probability >= candidate_threshold
    reference_correct = reference_prediction == labels
    candidate_correct = candidate_prediction == labels
    corrected_mask = ~reference_correct & candidate_correct
    regressed_mask = reference_correct & ~candidate_correct
    changed_mask = reference_prediction != candidate_prediction
    transitions = {
        "changed": int(changed_mask.sum()),
        "corrected": int(corrected_mask.sum()),
        "regressed": int(regressed_mask.sum()),
        "net": int(corrected_mask.sum() - regressed_mask.sum()),
        "corrected_source_indices": [
            int(rows[index]["source_index"])
            for index in np.flatnonzero(corrected_mask).tolist()
        ],
        "regressed_source_indices": [
            int(rows[index]["source_index"])
            for index in np.flatnonzero(regressed_mask).tolist()
        ],
    }
    correlations = {
        "pearson": float(
            np.corrcoef(reference_probability, candidate_probability)[0, 1]
        ),
        "spearman": float(
            spearmanr(reference_probability, candidate_probability).statistic
        ),
    }

    gates = protocol["advance_gate"]
    other_mcc_ok = all(
        by_length[stratum]["candidate"]["mcc"]
        >= by_length[stratum]["reference"]["mcc"]
        - float(gates["maximum_other_length_mcc_regression"])
        for stratum in ("16-30", "31-50")
    )
    length_auprc_ok = all(
        values["candidate"]["auprc"]
        >= values["reference"]["auprc"]
        - float(gates["maximum_length_auprc_regression"])
        for values in by_length.values()
    )
    gate = {
        "global_mcc_strictly_higher": candidate_metrics["mcc"]
        > float(gates["global_mcc_strictly_above"]),
        "global_auprc_within_tolerance": candidate_metrics["auprc"]
        >= float(gates["global_auprc_at_least"]),
        "brier_not_worse": candidate_metrics["brier"] <= float(gates["maximum_brier"]),
        "ece_10_within_tolerance": candidate_metrics["ece_10"]
        <= float(gates["maximum_ece_10"]),
        "at_least_three_fold_mcc_non_decreases": fold_non_decreases
        >= int(gates["minimum_fold_mcc_non_decreases"]),
        "short_mcc_not_lower": by_length["11-15"]["candidate"]["mcc"]
        >= float(gates["short_mcc_at_least"]),
        "other_length_mcc_within_tolerance": other_mcc_ok,
        "all_length_auprc_within_tolerance": length_auprc_ok,
        "positive_net_decision_corrections": transitions["net"]
        >= int(gates["minimum_net_decision_corrections"]),
        "probability_pearson_at_least_minimum": correlations["pearson"]
        >= float(gates["minimum_probability_pearson"]),
        "optimizer_converged_every_fold": all(
            bool(item["success"]) for item in coefficients
        ),
        "positive_shared_slope_every_fold": all(
            float(item["slope"]) > 0.0 for item in coefficients
        ),
    }
    gate["all_conditions_passed"] = all(gate.values())

    prediction_rows = []
    handoff_rows = []
    for index, row in enumerate(rows):
        coefficient = coefficients[int(row["fold"])]
        common = {
            "source_index": int(row["source_index"]),
            "sample_id": row["sample_id"],
            "fold": int(row["fold"]),
            "label": int(row["label"]),
            "sequence": row["sequence"],
            "length": int(row["length"]),
            "length_stratum": row["length_stratum"],
            "phase25_probability": f"{reference_probability[index]:.12g}",
            "length_calibrated_probability": f"{candidate_probability[index]:.12g}",
        }
        handoff_rows.append(common)
        prediction_rows.append(
            {
                **common,
                "phase25_threshold": reference_threshold,
                "calibrated_threshold": candidate_threshold,
                "phase25_prediction": int(reference_prediction[index]),
                "length_calibrated_prediction": int(candidate_prediction[index]),
                "decision_changed": int(changed_mask[index]),
                "phase25_correct": int(reference_correct[index]),
                "length_calibrated_correct": int(candidate_correct[index]),
                "calibration_slope": f"{float(coefficient['slope']):.12g}",
                "global_intercept": f"{float(coefficient['global_intercept']):.12g}",
                "short_intercept": f"{float(coefficient['short_intercept']):.12g}",
                "long_intercept": f"{float(coefficient['long_intercept']):.12g}",
            }
        )
    _write_csv(prediction_rows, output / "oof_predictions.csv")
    _write_csv(handoff_rows, output / "fusion_handoff.csv")
    _write_csv(coefficients, output / "fold_coefficients.csv")

    report = {
        "status": "complete",
        "phase": 31,
        "evaluation_tier": protocol["evaluation_tier"],
        "strict_nested_upstream": False,
        "direct_handoff_promotion_allowed": False,
        "external_test_labels_read": False,
        "protocol": protocol,
        "rows": len(rows),
        "overall": {
            "reference": reference_metrics,
            "candidate": candidate_metrics,
        },
        "folds": by_fold,
        "length_strata": by_length,
        "fold_coefficients": coefficients,
        "fold_mcc_non_decrease_count": int(fold_non_decreases),
        "probability_correlation": correlations,
        "decision_transitions": transitions,
        "advance_gate": gate,
        "output_contract": {
            "raw_probability_column": "phase25_probability",
            "candidate_probability_column": "length_calibrated_probability",
            "official_phase25_handoff_overwritten": False,
        },
    }
    decision_name = (
        "advance_to_strict_nested_phase25_retraining"
        if gate["all_conditions_passed"]
        else "retain_phase25_and_stop_length_calibration"
    )
    decision = {
        "status": "complete",
        "decision": decision_name,
        "screen_passed": bool(gate["all_conditions_passed"]),
        "strict_nested_retraining_authorized": bool(gate["all_conditions_passed"]),
        "promotion_authorized": False,
        "handoff_regeneration_authorized": False,
        "official_phase25_handoff_modified": False,
        "external_test_labels_read": False,
        "overall": report["overall"],
        "decision_transitions": transitions,
        "advance_gate": gate,
        "provenance_limitation": protocol["provenance_limitation"],
    }
    save_json(report, output / "summary.json")
    save_json(decision, output / "decision.json")
    return report, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    _, decision = run_phase31(args.protocol, args.output_root)
    print(json.dumps({"decision": decision["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
