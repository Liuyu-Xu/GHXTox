"""Five-fold trainer for shared and Cys-regime-conditioned residual controls."""

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

from ghxtox.models.cys_conditional import CysConditionalConfig, CysConditionalResidualModel, cys_regime
from ghxtox.prottox_cys_conditional import MODEL_NAME, MODEL_VERSION, checkpoint_payload
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_prottox_tsk import _folds, _load_cache, _metrics


def _regimes(cache: dict[str, Any]) -> torch.Tensor:
    counts = torch.tensor([float(torch.as_tensor(record["physchem"])[6]) for record in cache["records"]])
    return cys_regime(counts)


def _fit(features: torch.Tensor, regimes: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, *, config: CysConditionalConfig, epochs: int, learning_rate: float, seed: int, device: torch.device) -> tuple[CysConditionalResidualModel, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CysConditionalResidualModel(config).to(device)
    train_features, train_labels = features[indices], labels[indices]
    mean, std = train_features.mean(dim=0), train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    inputs, target, condition = ((train_features - mean) / std).to(device), train_labels.to(device), regimes[indices].to(device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(inputs, condition), target)
        loss.backward()
        optimizer.step()
    return model.eval(), mean, std


def train(cache_path: str | Path, output_dir: str | Path, *, variant: str, fold_manifest: str, folds: int, epochs: int, learning_rate: float, hidden_dim: int, residual_hidden_dim: int, dropout: float, seed: int, device_name: str) -> dict[str, Any]:
    cache, features, labels, _ = _load_cache(cache_path)
    if not fold_manifest:
        raise ValueError("Reported Cys-conditional runs require the audited five-fold manifest")
    fold_ids, regimes = _folds(cache, labels, fold_manifest, folds, seed), _regimes(cache)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    config = CysConditionalConfig(base_input_dim=int(features.shape[1]), variant=variant, hidden_dim=hidden_dim, residual_hidden_dim=residual_hidden_dim, dropout=dropout)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started, oof, fold_rows = time.perf_counter(), np.full(len(labels), np.nan, dtype=np.float64), []
    for fold in range(folds):
        train_indices, held_indices = torch.as_tensor(np.flatnonzero(fold_ids != fold), dtype=torch.long), torch.as_tensor(np.flatnonzero(fold_ids == fold), dtype=torch.long)
        model, mean, std = _fit(features, regimes, labels, train_indices, config=config, epochs=epochs, learning_rate=learning_rate, seed=seed + fold, device=device)
        with torch.inference_mode():
            probability = torch.sigmoid(model(((features[held_indices] - mean) / std).to(device), regimes[held_indices].to(device))).cpu().tolist()
        values = np.asarray(probability, dtype=np.float64)
        oof[held_indices.tolist()] = values
        torch.save(checkpoint_payload(model, base_mean=mean, base_std=std, metadata={"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold_manifest": fold_manifest, "fold": fold, "epochs": epochs, "seed": seed + fold, "variant": variant, "trained_records": int(len(train_indices)), "held_out_records": int(len(held_indices))}), output / f"fold_{fold}.pt")
        fold_rows.append({"fold": fold, "n_train": int(len(train_indices)), "n_test": int(len(held_indices)), **_metrics(np.asarray(labels[held_indices].tolist(), dtype=np.float64), values)})
    if not np.isfinite(oof).all():
        raise RuntimeError("Cys-conditional OOF training did not emit exactly one probability per cached record")
    final_model, mean, std = _fit(features, regimes, labels, torch.arange(len(labels)), config=config, epochs=epochs, learning_rate=learning_rate, seed=seed + 1000, device=device)
    torch.save(checkpoint_payload(final_model, base_mean=mean, base_std=std, metadata={"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold_manifest": fold_manifest, "epochs": epochs, "run_seed": seed, "refit_seed": seed + 1000, "variant": variant, "trained_records": int(len(labels)), "selection_scope": "full-data refit after OOF; not an OOF metric source"}), output / "final_model.pt")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence_sha256", "label", "fold", "probability", "cys_regime"])
        writer.writeheader()
        for record, fold, probability, regime in zip(cache["records"], fold_ids.tolist(), oof.tolist(), regimes.tolist()):
            writer.writerow({"sample_id": record["sample_id"], "sequence_sha256": record["sequence_sha256"], "label": record["label"], "fold": fold, "probability": f"{probability:.9g}", "cys_regime": int(regime)})
    summary = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "base_model_version": cache["model_version"], "variant": variant, "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path), "records": len(labels)}, "fold_manifest": fold_manifest, "folds": fold_rows, "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof), "regime_counts": {"cys_0_1": int((regimes == 0).sum()), "cys_2_3": int((regimes == 1).sum()), "cys_4_plus": int((regimes == 2).sum())}, "epochs": epochs, "learning_rate": learning_rate, "seed": seed, "parameter_count": sum(parameter.numel() for parameter in final_model.parameters() if parameter.requires_grad), "elapsed_seconds": time.perf_counter() - started, "device": str(device), "final_checkpoint": str(output / "final_model.pt"), "selection_scope": "training OOF only; Test1/Test2 excluded from variant selection"}
    write_manifest(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["shared_residual", "cys_conditional_residual"], required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--residual-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = train(args.cache, args.output_dir, variant=args.variant, fold_manifest=args.fold_manifest, folds=args.folds, epochs=args.epochs, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim, residual_hidden_dim=args.residual_hidden_dim, dropout=args.dropout, seed=args.seed, device_name=args.device)
    print(json.dumps(summary["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
