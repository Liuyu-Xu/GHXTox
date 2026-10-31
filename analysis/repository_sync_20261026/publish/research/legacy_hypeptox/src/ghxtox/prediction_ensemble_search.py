"""Search prediction-level ensembles from saved validation/test CSVs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ghxtox.metrics import binary_metrics
from ghxtox.threshold_predictions import LOWER_IS_BETTER, find_threshold
from ghxtox.utils import save_json


@dataclass(frozen=True)
class PredictionCandidate:
    name: str
    run_dir: Path
    val_path: Path
    test_path: Path
    val_probs: torch.Tensor
    test_probs: torch.Tensor


def _read_prediction_csv(path: str | Path) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor, list[str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"label", "toxicity_probability"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float32)
    probabilities = torch.tensor([float(row["toxicity_probability"]) for row in rows], dtype=torch.float32)
    if "sample_id" in rows[0]:
        ids = [str(row["sample_id"]) for row in rows]
    elif "sequence" in rows[0]:
        ids = [str(row["sequence"]) for row in rows]
    else:
        ids = [str(index) for index in range(len(rows))]
    return rows, labels, probabilities, ids


def _logit(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities.float().clamp(1e-6, 1.0 - 1e-6))


def _metric_score(metrics: dict[str, float], objective: str, min_sn: float | None, min_sp: float | None) -> float:
    value = float(metrics[objective])
    score = -value if objective in LOWER_IS_BETTER else value
    penalty = 0.0
    if min_sn is not None and float(metrics["sn"]) < min_sn:
        penalty += float(min_sn) - float(metrics["sn"])
    if min_sp is not None and float(metrics["sp"]) < min_sp:
        penalty += float(min_sp) - float(metrics["sp"])
    return float(score - 10.0 * penalty)


def _load_manifest_run_dirs(manifest_path: str | Path) -> list[Path]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Path(row["run_dir"]) for row in manifest.get("variants", [])]


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def collect_candidates(args: argparse.Namespace) -> tuple[list[PredictionCandidate], dict[str, Any]]:
    run_dirs = _load_manifest_run_dirs(args.manifest)
    run_dirs.extend(Path(path) for path in args.run_dirs)
    run_dirs = _unique_paths(run_dirs)
    candidates: list[PredictionCandidate] = []
    skipped: list[dict[str, str]] = []
    reference_val_labels = None
    reference_test_labels = None
    reference_val_ids = None
    reference_test_ids = None

    for run_dir in run_dirs:
        val_path = run_dir / args.val_predictions_name
        test_path = run_dir / args.test_predictions_name
        if not val_path.exists() or not test_path.exists():
            skipped.append({"run_dir": run_dir.as_posix(), "reason": "missing val/test prediction CSV"})
            continue
        _, val_labels, val_probs, val_ids = _read_prediction_csv(val_path)
        _, test_labels, test_probs, test_ids = _read_prediction_csv(test_path)
        if reference_val_labels is None:
            reference_val_labels = val_labels
            reference_test_labels = test_labels
            reference_val_ids = val_ids
            reference_test_ids = test_ids
        else:
            assert reference_test_labels is not None
            assert reference_val_ids is not None and reference_test_ids is not None
            if not torch.equal(reference_val_labels, val_labels):
                raise ValueError(f"Validation label order mismatch: {val_path}")
            if not torch.equal(reference_test_labels, test_labels):
                raise ValueError(f"Test label order mismatch: {test_path}")
            if reference_val_ids != val_ids:
                raise ValueError(f"Validation sample order mismatch: {val_path}")
            if reference_test_ids != test_ids:
                raise ValueError(f"Test sample order mismatch: {test_path}")
        candidates.append(
            PredictionCandidate(
                name=run_dir.name,
                run_dir=run_dir,
                val_path=val_path,
                test_path=test_path,
                val_probs=val_probs,
                test_probs=test_probs,
            )
        )
    if not candidates:
        raise ValueError("No complete prediction candidates were found.")
    metadata = {
        "num_candidates": len(candidates),
        "candidate_names": [candidate.name for candidate in candidates],
        "skipped": skipped,
    }
    return candidates, metadata


def _combine(probability_stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.float()
    weights = weights / weights.sum().clamp_min(1e-12)
    return (probability_stack * weights.view(-1, 1)).sum(dim=0)


def _evaluate_probabilities(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    objective: str,
    min_sn: float | None,
    min_sp: float | None,
) -> tuple[float, dict[str, float], float]:
    selection = find_threshold(probabilities, labels, objective=objective, min_sn=min_sn, min_sp=min_sp)
    threshold = float(selection["threshold"])
    metrics = selection["metrics"]
    score = _metric_score(metrics, objective=objective, min_sn=min_sn, min_sp=min_sp)
    return threshold, metrics, score


def _row_from_metrics(
    names: list[str],
    weights: torch.Tensor,
    threshold: float,
    metrics: dict[str, float],
    score: float,
    stage: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "models": names,
        "num_models": len(names),
        "weights": [float(item) for item in weights.tolist()],
        "threshold": float(threshold),
        "score": float(score),
        "accuracy": float(metrics["accuracy"]),
        "sn": float(metrics["sn"]),
        "sp": float(metrics["sp"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "auroc": float(metrics["auroc"]),
        "auprc": float(metrics["auprc"]),
        "brier": float(metrics["brier"]),
        "ece_10": float(metrics["ece_10"]),
    }


def _uniform_search(
    candidates: list[PredictionCandidate],
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices = range(len(candidates))
    max_k = min(int(args.max_k), len(candidates))
    for k in range(max(int(args.min_k), 1), max_k + 1):
        for combo in itertools.combinations(indices, k):
            stack = torch.stack([candidates[index].val_probs for index in combo])
            weights = torch.full((k,), 1.0 / k, dtype=torch.float32)
            probs = _combine(stack, weights)
            threshold, metrics, score = _evaluate_probabilities(
                probs,
                labels,
                objective=args.objective,
                min_sn=args.min_sn,
                min_sp=args.min_sp,
            )
            names = [candidates[index].name for index in combo]
            rows.append(_row_from_metrics(names, weights, threshold, metrics, score, "uniform"))
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _sample_dirichlet_weights(num_models: int, trials: int, seed: int, alpha: float) -> list[torch.Tensor]:
    weights: list[torch.Tensor] = [torch.full((num_models,), 1.0 / num_models, dtype=torch.float32)]
    weights.extend(torch.eye(num_models, dtype=torch.float32))
    if trials <= 0:
        return weights
    concentration = torch.full((num_models,), max(float(alpha), 1e-3), dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        distribution = torch.distributions.Gamma(concentration, torch.ones_like(concentration))
        for _ in range(int(trials)):
            sample = distribution.sample()
            weights.append(sample / sample.sum().clamp_min(1e-12))
    return weights


def _weighted_refine(
    uniform_rows: list[dict[str, Any]],
    candidates: list[PredictionCandidate],
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    name_to_candidate = {candidate.name: candidate for candidate in candidates}
    refined: list[dict[str, Any]] = []
    for combo_rank, row in enumerate(uniform_rows[: int(args.refine_top)]):
        names = [str(name) for name in row["models"]]
        combo_candidates = [name_to_candidate[name] for name in names]
        stack = torch.stack([candidate.val_probs for candidate in combo_candidates])
        for weights in _sample_dirichlet_weights(
            len(combo_candidates),
            trials=int(args.weight_trials),
            seed=int(args.weight_seed) + combo_rank,
            alpha=float(args.weight_alpha),
        ):
            probs = _combine(stack, weights)
            threshold, metrics, score = _evaluate_probabilities(
                probs,
                labels,
                objective=args.objective,
                min_sn=args.min_sn,
                min_sp=args.min_sp,
            )
            refined.append(_row_from_metrics(names, weights, threshold, metrics, score, "weighted"))
    refined.sort(key=lambda item: item["score"], reverse=True)
    return refined


def _write_prediction_csv(
    template_rows: list[dict[str, str]],
    probabilities: torch.Tensor,
    output_path: str | Path,
    threshold: float,
    model_names: list[str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(template_rows[0])
    for column in ("toxicity_probability", "prediction", "decision_threshold", "ensemble_members"):
        if column not in fieldnames:
            fieldnames.append(column)
    members = ";".join(model_names)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(template_rows, probabilities.tolist(), strict=True):
            output = dict(row)
            output["toxicity_probability"] = f"{float(probability):.9g}"
            output["prediction"] = int(float(probability) >= threshold)
            output["decision_threshold"] = f"{threshold:.9g}"
            output["ensemble_members"] = members
            writer.writerow(output)


def _write_ranking_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "stage",
        "num_models",
        "models",
        "weights",
        "threshold",
        "score",
        "accuracy",
        "sn",
        "sp",
        "f1",
        "mcc",
        "auroc",
        "auprc",
        "brier",
        "ece_10",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            output = dict(row)
            output["rank"] = rank
            output["models"] = ";".join(str(name) for name in row["models"])
            output["weights"] = ";".join(f"{float(weight):.9g}" for weight in row["weights"])
            writer.writerow({key: output.get(key) for key in fieldnames})


def search_prediction_ensembles(args: argparse.Namespace) -> dict[str, Any]:
    candidates, metadata = collect_candidates(args)
    val_rows, val_labels, _, _ = _read_prediction_csv(candidates[0].val_path)
    test_rows, test_labels, _, _ = _read_prediction_csv(candidates[0].test_path)
    uniform_rows = _uniform_search(candidates, val_labels, args)
    refined_rows = _weighted_refine(uniform_rows, candidates, val_labels, args)
    all_rows = sorted(uniform_rows[: int(args.keep_uniform)] + refined_rows, key=lambda row: row["score"], reverse=True)
    best = all_rows[0]
    name_to_candidate = {candidate.name: candidate for candidate in candidates}
    best_candidates = [name_to_candidate[str(name)] for name in best["models"]]
    weights = torch.tensor([float(item) for item in best["weights"]], dtype=torch.float32)
    val_probs = _combine(torch.stack([candidate.val_probs for candidate in best_candidates]), weights)
    test_probs = _combine(torch.stack([candidate.test_probs for candidate in best_candidates]), weights)
    threshold = float(best["threshold"])
    val_metrics = binary_metrics(_logit(val_probs), val_labels, threshold=threshold)
    test_metrics = binary_metrics(_logit(test_probs), test_labels, threshold=threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_ranking_csv(all_rows[: int(args.output_top)], output_dir / "ensemble_search_ranking.csv")
    _write_prediction_csv(val_rows, val_probs, output_dir / "best_val_predictions.csv", threshold, best["models"])
    _write_prediction_csv(test_rows, test_probs, output_dir / "best_test_predictions.csv", threshold, best["models"])
    summary = {
        "objective": args.objective,
        "constraints": {"min_sn": args.min_sn, "min_sp": args.min_sp},
        "search": {
            "min_k": int(args.min_k),
            "max_k": int(args.max_k),
            "uniform_candidates": len(uniform_rows),
            "refine_top": int(args.refine_top),
            "weight_trials": int(args.weight_trials),
            "weight_seed": int(args.weight_seed),
            "weight_alpha": float(args.weight_alpha),
        },
        "metadata": metadata,
        "best": {
            "models": best["models"],
            "weights": best["weights"],
            "threshold": threshold,
            "validation_score": float(best["score"]),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        "top_rows": all_rows[: int(args.output_top)],
        "outputs": {
            "ranking_csv": "ensemble_search_ranking.csv",
            "val_predictions": "best_val_predictions.csv",
            "test_predictions": "best_test_predictions.csv",
        },
    }
    save_json(summary, output_dir / "ensemble_search_summary.json")
    print(f"Prediction ensemble search saved to {output_dir}")
    print(f"best_models: {';'.join(str(name) for name in best['models'])}")
    print(f"threshold: {threshold:.6f}")
    for key in ("accuracy", "sn", "sp", "f1", "mcc", "auroc", "auprc", "brier", "ece_10"):
        print(f"test_{key}: {test_metrics[key]:.6f}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/hypeptox_tuning/manifest.json")
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--val-predictions-name", default="val_predictions.csv")
    parser.add_argument("--test-predictions-name", default="test_predictions.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--objective",
        choices=["accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece_10"],
        default="mcc",
    )
    parser.add_argument("--min-sn", type=float, default=None)
    parser.add_argument("--min-sp", type=float, default=None)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--refine-top", type=int, default=32)
    parser.add_argument("--weight-trials", type=int, default=256)
    parser.add_argument("--weight-seed", type=int, default=42)
    parser.add_argument("--weight-alpha", type=float, default=1.0)
    parser.add_argument("--keep-uniform", type=int, default=64)
    parser.add_argument("--output-top", type=int, default=100)
    return parser


def main() -> None:
    search_prediction_ensembles(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
