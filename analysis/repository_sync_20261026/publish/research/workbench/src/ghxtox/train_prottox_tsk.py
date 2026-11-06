"""Train the lightweight ProtTox-TSK MLP or KAN head from frozen features."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, ProtToxTSKConfig, ProtToxTSKModel, checkpoint_payload, file_sha256, write_manifest


def _load_cache(path: str | Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CACHE_FORMAT_VERSION or payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Input is not a compatible ProtTox-TSK frozen-feature cache")
    records = payload.get("records", [])
    if not records or any(record.get("label") is None for record in records):
        raise ValueError("ProtTox-TSK training cache must contain labels for every record")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    physic = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    if tri.shape[1] != 3 * int(payload["encoder"]["embedding_dim"]):
        raise ValueError("TriStat cache dimension does not match encoder metadata")
    return payload, torch.cat((tri, physic), dim=1), labels, torch.arange(len(records))


def _folds(cache: dict[str, Any], labels: torch.Tensor, manifest_path: str | None, folds: int, seed: int) -> np.ndarray:
    records = cache["records"]
    if manifest_path:
        with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = {str(row["sample_id"]): int(row["fold"]) for row in csv.DictReader(handle)}
        missing = [record["sample_id"] for record in records if record["sample_id"] not in rows]
        if missing:
            raise ValueError(f"Fold manifest does not cover cached sample IDs, e.g. {missing[:3]}")
        result = np.asarray([rows[record["sample_id"]] for record in records], dtype=np.int64)
        if len(set(result.tolist())) != folds:
            raise ValueError(f"Fold manifest must contain exactly {folds} folds for this run")
        return result
    # This fallback is for a 128-record smoke test only; reported experiments must pass an audited manifest.
    # Keep labels represented in every feasible fold so basic smoke metrics do
    # not degenerate merely because the source order is label-clustered.
    result = np.empty(len(records), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for label in sorted(set(int(value) for value in labels.tolist())):
        indices = np.flatnonzero(np.asarray(labels.tolist(), dtype=np.int64) == label)
        rng.shuffle(indices)
        result[indices] = np.arange(len(indices), dtype=np.int64) % folds
    return result


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predicted = probabilities >= threshold
    truth = labels.astype(bool)
    tp, tn = int(np.sum(predicted & truth)), int(np.sum(~predicted & ~truth))
    fp, fn = int(np.sum(predicted & ~truth)), int(np.sum(~predicted & truth))
    denominator = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    result = {"accuracy": (tp + tn) / max(len(labels), 1), "balanced_accuracy": (sensitivity + specificity) / 2.0, "precision": precision, "sensitivity": sensitivity, "specificity": specificity, "f1": 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-12), "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0, "brier": float(np.mean((probabilities - labels) ** 2)), "threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        result["auroc"] = float(roc_auc_score(labels, probabilities))
        result["auprc"] = float(average_precision_score(labels, probabilities))
    except (ImportError, ValueError):
        result["auroc"], result["auprc"] = float("nan"), float("nan")
    return result


def _fit(features: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, *, config: ProtToxTSKConfig, epochs: int, learning_rate: float, seed: int, device: torch.device) -> tuple[ProtToxTSKModel, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = ProtToxTSKModel(config).to(device)
    train_x, train_y = features[train_indices], labels[train_indices]
    mean, std = train_x.mean(dim=0), train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x, y = ((train_x - mean) / std).to(device), train_y.to(device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model.forward_features(x), y)
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def train(cache_path: str | Path, output_dir: str | Path, *, head_kind: str, fold_manifest: str | None, folds: int, epochs: int, learning_rate: float, hidden_dim: int, grid_size: int, dropout: float, seed: int, device_name: str) -> dict[str, Any]:
    cache, features, labels, _ = _load_cache(cache_path)
    fold_ids = _folds(cache, labels, fold_manifest, folds, seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    config = ProtToxTSKConfig(embedding_dim=int(cache["encoder"]["embedding_dim"]), head_kind=head_kind, hidden_dim=hidden_dim, grid_size=grid_size, dropout=dropout)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(folds):
        train_indices = torch.as_tensor(np.flatnonzero(fold_ids != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(fold_ids == fold), dtype=torch.long)
        model, mean, std = _fit(features, labels, train_indices, config=config, epochs=epochs, learning_rate=learning_rate, seed=seed + fold, device=device)
        with torch.inference_mode():
            probabilities = np.asarray(torch.sigmoid(model.forward_features(((features[test_indices] - mean) / std).to(device))).cpu().tolist(), dtype=np.float64)
        oof[test_indices.tolist()] = probabilities
        payload = checkpoint_payload(model, feature_mean=mean, feature_std=std, metadata={"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold": fold, "fold_manifest": fold_manifest, "epochs": epochs, "seed": seed + fold, "trained_records": int(len(train_indices)), "held_out_records": int(len(test_indices))})
        torch.save(payload, output / f"fold_{fold}.pt")
        fold_rows.append({"fold": fold, "n_train": int(len(train_indices)), "n_test": int(len(test_indices)), **_metrics(np.asarray(labels[test_indices].tolist(), dtype=np.float64), probabilities)})
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF training did not emit exactly one probability per cached record")
    final_model, final_mean, final_std = _fit(features, labels, torch.arange(len(labels)), config=config, epochs=epochs, learning_rate=learning_rate, seed=seed + 1000, device=device)
    torch.save(checkpoint_payload(final_model, feature_mean=final_mean, feature_std=final_std, metadata={"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold_manifest": fold_manifest, "epochs": epochs, "run_seed": seed, "refit_seed": seed + 1000, "trained_records": int(len(labels)), "selection_scope": "full-data refit after OOF; not an OOF metric source"}), output / "final_model.pt")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence_sha256", "label", "fold", "probability"])
        writer.writeheader()
        for record, fold, probability in zip(cache["records"], fold_ids.tolist(), oof.tolist()):
            writer.writerow({"sample_id": record["sample_id"], "sequence_sha256": record["sequence_sha256"], "label": record["label"], "fold": fold, "probability": f"{probability:.9g}"})
    summary = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "head_kind": head_kind, "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path), "records": len(labels)}, "fold_manifest": fold_manifest, "folds": fold_rows, "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof), "epochs": epochs, "learning_rate": learning_rate, "seed": seed, "parameter_count": sum(parameter.numel() for parameter in final_model.parameters()), "elapsed_seconds": time.perf_counter() - started, "device": str(device), "final_checkpoint": str(output / "final_model.pt"), "warning": "KAN is a candidate only. Compare this OOF summary against a same-protocol MLP run before promotion."}
    write_manifest(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head", choices=["mlp", "kan"], default="mlp")
    parser.add_argument("--fold-manifest", default=None, help="Audited CSV with sample_id and fold; required for reported results.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.output_dir, head_kind=args.head, fold_manifest=args.fold_manifest, folds=args.folds, epochs=args.epochs, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim or (192 if args.head == "mlp" else 128), grid_size=args.grid_size, dropout=args.dropout if args.dropout is not None else (0.10 if args.head == "mlp" else 0.05), seed=args.seed, device_name=args.device)
    print(json.dumps(result["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
