"""Evaluate probability ensembles of trained GHXTox checkpoints."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device, resolve_device, save_json


def _best_threshold(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> tuple[float, dict[str, Any]]:
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


def _metric_score(metrics: dict[str, float], metric: str) -> float:
    score = float(metrics[metric])
    return -score if metric in {"loss", "brier", "ece_10"} else score


def _search_weights(
    probability_stack: torch.Tensor,
    labels: torch.Tensor,
    metric: str,
    threshold: float,
    threshold_metric: str | None,
    trials: int,
    seed: int,
    alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    num_models = probability_stack.shape[0]
    candidates: list[torch.Tensor] = []
    candidates.append(torch.full((num_models,), 1.0 / max(num_models, 1), dtype=torch.float32))
    candidates.extend(torch.eye(num_models, dtype=torch.float32))
    if trials > 0:
        concentration = torch.full((num_models,), max(float(alpha), 1e-3), dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            gamma = torch.distributions.Gamma(concentration, torch.ones_like(concentration))
            for _ in range(int(trials)):
                sample = gamma.sample((1,)).squeeze(0)
                candidates.append(sample / sample.sum().clamp_min(1e-12))

    best_weights = candidates[0]
    best_metrics = {}
    best_score = -float("inf")
    trace = []
    for index, weights in enumerate(candidates):
        probs = (probability_stack * weights.view(-1, 1)).sum(dim=0)
        logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
        candidate_threshold = float(threshold)
        if threshold_metric is not None:
            candidate_threshold, metrics = _best_threshold(logits, labels, threshold_metric)
        else:
            metrics = binary_metrics(logits, labels, threshold=threshold)
        score = _metric_score(metrics, metric)
        row = {
            "candidate_index": index,
            "score": float(score),
            "metric_value": float(metrics[metric]),
            "threshold": float(candidate_threshold),
            "weights": [float(item) for item in weights.tolist()],
        }
        trace.append(row)
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics
    return best_weights, {
        "metric": metric,
        "threshold": float(threshold),
        "threshold_metric": threshold_metric,
        "trials": int(trials),
        "seed": int(seed),
        "alpha": float(alpha),
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "best_weights": [float(item) for item in best_weights.tolist()],
        "top_candidates": sorted(trace, key=lambda row: row["score"], reverse=True)[:10],
        "selection_data": "current labeled input; use this on validation data only",
    }


def _model_inputs_for_config(config: dict[str, Any]) -> tuple[bool, bool]:
    modality = str(config.get("model", {}).get("modality", "fusion")).lower()
    include_structure = modality not in {"sequence_only", "atom_only", "sequence_atom"}
    include_atom = modality in {"atom_only", "sequence_atom", "fusion_atom_residual", "residual_experts"}
    return include_structure, include_atom


def _predict_checkpoint(
    checkpoint_path: str | Path,
    dataset: PeptideTensorDataset,
    processed_path: str | Path,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    records = dataset.records
    required_plm_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(records, required_plm_dim, processed_path)
    include_structure, include_atom = _model_inputs_for_config(config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_peptides,
            include_structure=include_structure,
            include_atom=include_atom,
        ),
    )
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    probs_all = []
    labels_all = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            if labels is None:
                raise RuntimeError("Ensemble evaluation requires labels.")
            output = model(batch)
            probs = torch.sigmoid(output["logits"].detach().cpu())
            labels_cpu = labels.detach().cpu()
            probs_all.append(probs)
            labels_all.append(labels_cpu)
            if not rows:
                for sample_id, sequence, label in zip(
                    batch["sample_id"],
                    batch["sequence"],
                    labels_cpu.tolist(),
                ):
                    rows.append({"sample_id": sample_id, "sequence": sequence, "label": label})
            else:
                for sample_id, sequence, label in zip(
                    batch["sample_id"],
                    batch["sequence"],
                    labels_cpu.tolist(),
                ):
                    rows.append({"sample_id": sample_id, "sequence": sequence, "label": label})
    return torch.cat(probs_all), torch.cat(labels_all), rows, checkpoint


def _write_predictions(
    rows: list[dict[str, Any]],
    probabilities: torch.Tensor,
    output_path: str | Path,
    threshold: float,
    checkpoints: list[str],
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "sequence",
                "label",
                "toxicity_probability",
                "prediction",
                "decision_threshold",
                "ensemble_checkpoints",
            ],
        )
        writer.writeheader()
        checkpoint_text = ";".join(checkpoints)
        for row, prob in zip(rows, probabilities.tolist(), strict=True):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sequence": row["sequence"],
                    "label": int(row["label"]),
                    "toxicity_probability": f"{float(prob):.9g}",
                    "prediction": int(float(prob) >= threshold),
                    "decision_threshold": f"{threshold:.9g}",
                    "ensemble_checkpoints": checkpoint_text,
                }
            )


def evaluate_ensemble(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    checkpoints = [str(path) for path in args.checkpoints]
    if args.processed_per_checkpoint:
        processed_paths = [str(path) for path in args.processed_per_checkpoint]
        if len(processed_paths) != len(checkpoints):
            raise ValueError("--processed-per-checkpoint length must match --checkpoints length.")
    elif args.processed:
        processed_paths = [str(args.processed)] * len(checkpoints)
    else:
        raise ValueError("Provide --processed or --processed-per-checkpoint.")
    if args.weights:
        weights = torch.tensor([float(item) for item in args.weights], dtype=torch.float32)
        if weights.numel() != len(checkpoints):
            raise ValueError("--weights length must match --checkpoints length.")
    else:
        weights = torch.ones(len(checkpoints), dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-12)

    probability_stack = []
    reference_labels = None
    reference_rows: list[dict[str, Any]] | None = None
    reference_ids: list[str] | None = None
    checkpoint_epochs = []
    for checkpoint_path, processed_path in zip(checkpoints, processed_paths, strict=True):
        dataset = PeptideTensorDataset(processed_path, require_labels=True)
        probs, labels, rows, checkpoint = _predict_checkpoint(
            checkpoint_path,
            dataset,
            processed_path,
            device,
            args.batch_size,
        )
        probability_stack.append(probs)
        checkpoint_epochs.append(float(checkpoint.get("epoch", -1)))
        sample_ids = [str(row["sample_id"]) for row in rows]
        if reference_labels is None:
            reference_labels = labels
            reference_rows = rows
            reference_ids = sample_ids
        elif not torch.equal(reference_labels, labels):
            raise ValueError(f"Label order mismatch for checkpoint {checkpoint_path}.")
        elif reference_ids != sample_ids:
            raise ValueError(f"Sample order mismatch for checkpoint {checkpoint_path}.")
    assert reference_labels is not None and reference_rows is not None
    probability_tensor = torch.stack(probability_stack)
    weight_search = None
    if args.optimize_weights_metric:
        weights, weight_search = _search_weights(
            probability_tensor,
            reference_labels,
            args.optimize_weights_metric,
            threshold=float(args.threshold),
            threshold_metric=args.optimize_threshold_metric,
            trials=int(args.weight_search_trials),
            seed=int(args.weight_search_seed),
            alpha=float(args.weight_search_alpha),
        )
    weighted_probabilities = probability_tensor * weights.view(-1, 1)
    ensemble_probs = weighted_probabilities.sum(dim=0)
    logits = torch.logit(ensemble_probs.clamp(1e-6, 1.0 - 1e-6))
    threshold = float(args.threshold)
    threshold_search = None
    if args.optimize_threshold_metric:
        threshold, optimized_metrics = _best_threshold(logits, reference_labels, args.optimize_threshold_metric)
        threshold_search = {
            "metric": args.optimize_threshold_metric,
            "threshold": float(threshold),
            "metrics": optimized_metrics,
            "selection_data": "current labeled --processed input; use this on validation data only",
        }
    metrics = binary_metrics(logits, reference_labels, threshold=threshold)
    metrics.update(
        {
            "threshold": float(threshold),
            "num_samples": float(len(reference_labels)),
            "ensemble_size": float(len(checkpoints)),
            "checkpoints": checkpoints,
            "processed_files": processed_paths,
            "weights": [float(item) for item in weights.tolist()],
            "checkpoint_epochs": checkpoint_epochs,
        }
    )
    if weight_search is not None:
        metrics["weight_search"] = weight_search
    if threshold_search is not None:
        metrics["threshold_search"] = threshold_search
    save_json(metrics, args.output)
    if args.predictions:
        _write_predictions(reference_rows, ensemble_probs, args.predictions, threshold, checkpoints)
    print(f"Ensemble metrics saved to {args.output}")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "auroc",
        "auprc",
        "brier",
        "ece_10",
    ]:
        print(f"{key}: {metrics[key]:.6f}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", default=None, help="Labeled processed .pt file shared by all checkpoints.")
    parser.add_argument(
        "--processed-per-checkpoint",
        nargs="*",
        default=None,
        help="Optional one processed .pt file per checkpoint, for heterogeneous PLM dimensions.",
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--optimize-weights-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "mcc", "auroc", "auprc", "brier", "ece_10"],
        default=None,
        help="Search ensemble weights on the current labeled input. Use on validation data only.",
    )
    parser.add_argument("--weight-search-trials", type=int, default=256)
    parser.add_argument("--weight-search-seed", type=int, default=42)
    parser.add_argument("--weight-search-alpha", type=float, default=1.0)
    parser.add_argument(
        "--optimize-threshold-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "mcc"],
        default=None,
        help="Select threshold on the current labeled input. Use on validation data, then pass that threshold to test.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    evaluate_ensemble(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
