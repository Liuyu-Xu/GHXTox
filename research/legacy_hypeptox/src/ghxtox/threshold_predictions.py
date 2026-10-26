"""Select a decision threshold on validation predictions and apply it to CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from ghxtox.metrics import binary_metrics
from ghxtox.utils import save_json


LOWER_IS_BETTER = {"brier", "ece_10"}


def _logit(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities.float().clamp(1e-6, 1.0 - 1e-6))


def _read_prediction_csv(path: str | Path) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"label", "toxicity_probability"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32)
    probabilities = torch.tensor([float(row["toxicity_probability"]) for row in rows], dtype=torch.float32)
    return rows, labels, probabilities


def _score(metrics: dict[str, float], objective: str, min_sn: float | None, min_sp: float | None) -> float:
    value = float(metrics[objective])
    score = -value if objective in LOWER_IS_BETTER else value
    penalty = 0.0
    if min_sn is not None and float(metrics["sn"]) < min_sn:
        penalty += min_sn - float(metrics["sn"])
    if min_sp is not None and float(metrics["sp"]) < min_sp:
        penalty += min_sp - float(metrics["sp"])
    return score - 10.0 * penalty


def find_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    min_sn: float | None = None,
    min_sp: float | None = None,
) -> dict[str, Any]:
    logits = _logit(probabilities)
    candidates = torch.linspace(0.01, 0.99, 99).tolist()
    candidates.extend(float(value) for value in probabilities.unique().tolist())
    candidates = sorted(set(round(float(value), 6) for value in candidates if 0.0 < value < 1.0))
    best_threshold = 0.5
    best_metrics = binary_metrics(logits, labels, threshold=best_threshold)
    best_score = _score(best_metrics, objective, min_sn=min_sn, min_sp=min_sp)
    trace = []
    for threshold in candidates:
        metrics = binary_metrics(logits, labels, threshold=threshold)
        score = _score(metrics, objective, min_sn=min_sn, min_sp=min_sp)
        trace.append(
            {
                "threshold": float(threshold),
                "score": float(score),
                "objective_value": float(metrics[objective]),
                "sn": float(metrics["sn"]),
                "sp": float(metrics["sp"]),
                "mcc": float(metrics["mcc"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return {
        "threshold": float(best_threshold),
        "objective": objective,
        "score": float(best_score),
        "metrics": best_metrics,
        "constraints": {"min_sn": min_sn, "min_sp": min_sp},
        "top_candidates": sorted(trace, key=lambda row: row["score"], reverse=True)[:20],
    }


def _write_predictions(
    rows: list[dict[str, str]],
    probabilities: torch.Tensor,
    output_path: str | Path,
    threshold: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for column in ("prediction", "decision_threshold"):
        if column not in fieldnames:
            fieldnames.append(column)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, probabilities.tolist(), strict=True):
            output = dict(row)
            output["prediction"] = int(float(probability) >= threshold)
            output["decision_threshold"] = f"{threshold:.9g}"
            writer.writerow(output)


def optimize_and_apply(args: argparse.Namespace) -> dict[str, Any]:
    _, val_labels, val_probabilities = _read_prediction_csv(args.validation)
    selection = find_threshold(
        val_probabilities,
        val_labels,
        args.objective,
        min_sn=args.min_sn,
        min_sp=args.min_sp,
    )
    threshold = float(selection["threshold"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "validation": str(args.validation),
        "selection": selection,
        "applied": [],
    }
    for path_text in args.apply:
        rows, labels, probabilities = _read_prediction_csv(path_text)
        output_path = output_dir / f"{Path(path_text).stem}_thr{threshold:.6f}.csv"
        _write_predictions(rows, probabilities, output_path, threshold)
        metrics = binary_metrics(_logit(probabilities), labels, threshold=threshold)
        metrics["threshold"] = threshold
        summary["applied"].append(
            {
                "input": str(path_text),
                "output": str(output_path),
                "num_samples": len(rows),
                "metrics": metrics,
            }
        )
    save_json(summary, args.output_json)
    print(f"Threshold summary saved to {args.output_json}")
    print(f"threshold: {threshold:.6f}")
    print(f"validation_{args.objective}: {selection['metrics'][args.objective]:.6f}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", required=True, help="Validation prediction CSV used for threshold selection.")
    parser.add_argument("--apply", nargs="+", required=True, help="Prediction CSVs to rewrite with selected threshold.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece_10"],
        default="mcc",
    )
    parser.add_argument("--min-sn", type=float, default=None)
    parser.add_argument("--min-sp", type=float, default=None)
    return parser


def main() -> None:
    optimize_and_apply(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
