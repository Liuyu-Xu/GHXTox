"""Calibrate saved prediction CSV probabilities with a validation split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ghxtox.metrics import binary_metrics
from ghxtox.utils import save_json


def _logit(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities.float().clamp(1e-6, 1.0 - 1e-6))


def _read_prediction_csv(path: str | Path) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no prediction rows.")
    missing = {"label", "toxicity_probability"} - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32)
    probabilities = torch.tensor([float(row["toxicity_probability"]) for row in rows], dtype=torch.float32)
    return rows, labels, probabilities


def _best_threshold(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> tuple[float, dict[str, float]]:
    candidates = torch.linspace(0.01, 0.99, 99).tolist()
    candidates.extend(float(value) for value in torch.sigmoid(logits).unique().tolist())
    candidates = sorted(set(round(float(value), 6) for value in candidates if 0.0 < value < 1.0))
    best_threshold = 0.5
    best_metrics = binary_metrics(logits, labels, threshold=best_threshold)
    best_score = float(best_metrics[metric])
    for threshold in candidates:
        metrics = binary_metrics(logits, labels, threshold=threshold)
        score = float(metrics[metric])
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def fit_temperature(probabilities: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> dict[str, float | str]:
    logits = _logit(probabilities)
    labels = labels.float()
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max(int(steps), 1), line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "temperature",
        "temperature": float(log_temperature.detach().exp().clamp(0.05, 20.0)),
    }


def fit_platt(probabilities: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> dict[str, float | str]:
    logits = _logit(probabilities)
    labels = labels.float()
    scale = torch.nn.Parameter(torch.ones(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.1, max_iter=max(int(steps), 1), line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scale * logits + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "platt",
        "scale": float(scale.detach()),
        "bias": float(bias.detach()),
    }


def apply_calibration(probabilities: torch.Tensor, calibration: dict[str, Any]) -> torch.Tensor:
    method = str(calibration["method"])
    logits = _logit(probabilities)
    if method == "temperature":
        temperature = max(float(calibration["temperature"]), 1e-6)
        return torch.sigmoid(logits / temperature)
    if method == "platt":
        return torch.sigmoid(float(calibration["scale"]) * logits + float(calibration["bias"]))
    if method in {"none", "identity"}:
        return probabilities.float().clamp(0.0, 1.0)
    raise ValueError(f"Unsupported calibration method: {method}")


def _write_calibrated_csv(
    rows: list[dict[str, str]],
    calibrated_probabilities: torch.Tensor,
    output_path: str | Path,
    threshold: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for column in ("raw_toxicity_probability", "toxicity_probability", "prediction", "decision_threshold"):
        if column not in fieldnames:
            fieldnames.append(column)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, calibrated_probabilities.tolist(), strict=True):
            output_row = dict(row)
            output_row["raw_toxicity_probability"] = row["toxicity_probability"]
            output_row["toxicity_probability"] = f"{float(probability):.9g}"
            output_row["prediction"] = int(float(probability) >= threshold)
            output_row["decision_threshold"] = f"{threshold:.9g}"
            writer.writerow(output_row)


def calibrate_predictions(args: argparse.Namespace) -> dict[str, Any]:
    val_rows, val_labels, val_probs = _read_prediction_csv(args.validation)
    if str(args.method) == "temperature":
        calibration = fit_temperature(val_probs, val_labels, steps=args.steps)
    elif str(args.method) == "platt":
        calibration = fit_platt(val_probs, val_labels, steps=args.steps)
    elif str(args.method) in {"none", "identity"}:
        calibration = {"method": "identity"}
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    calibrated_val = apply_calibration(val_probs, calibration)
    val_logits_before = _logit(val_probs)
    val_logits_after = _logit(calibrated_val)
    threshold = float(args.threshold)
    threshold_search = None
    if args.optimize_threshold_metric:
        threshold, threshold_metrics = _best_threshold(val_logits_after, val_labels, args.optimize_threshold_metric)
        threshold_search = {
            "metric": args.optimize_threshold_metric,
            "threshold": float(threshold),
            "metrics": threshold_metrics,
            "selection_data": "validation prediction CSV",
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "calibration": calibration,
        "validation": str(args.validation),
        "validation_num_samples": len(val_rows),
        "threshold": float(threshold),
        "validation_before": binary_metrics(val_logits_before, val_labels, threshold=threshold),
        "validation_after": binary_metrics(val_logits_after, val_labels, threshold=threshold),
        "applied": [],
    }
    if threshold_search is not None:
        summary["threshold_search"] = threshold_search

    for path_text in args.apply:
        rows, labels, probabilities = _read_prediction_csv(path_text)
        calibrated = apply_calibration(probabilities, calibration)
        output_path = output_dir / f"{Path(path_text).stem}_calibrated.csv"
        _write_calibrated_csv(rows, calibrated, output_path, threshold)
        entry = {
            "input": str(path_text),
            "output": str(output_path),
            "num_samples": len(rows),
            "before": binary_metrics(_logit(probabilities), labels, threshold=threshold),
            "after": binary_metrics(_logit(calibrated), labels, threshold=threshold),
        }
        summary["applied"].append(entry)

    save_json(summary, args.output_json)
    print(f"Calibration summary saved to {args.output_json}")
    print(f"method: {calibration['method']}")
    print(f"threshold: {threshold:.6f}")
    print(f"validation_brier: {summary['validation_before']['brier']:.6f} -> {summary['validation_after']['brier']:.6f}")
    print(f"validation_ece_10: {summary['validation_before']['ece_10']:.6f} -> {summary['validation_after']['ece_10']:.6f}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, help="Validation prediction CSV used to fit calibration.")
    parser.add_argument("--apply", nargs="+", required=True, help="Prediction CSV files to calibrate.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--method", choices=["temperature", "platt", "identity"], default="temperature")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--optimize-threshold-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "mcc"],
        default=None,
        help="Select decision threshold on calibrated validation probabilities.",
    )
    return parser


def main() -> None:
    calibrate_predictions(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
