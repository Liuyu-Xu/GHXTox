"""Train spectral-only, concatenation, or gated-residual ProtTox candidates."""

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

from ghxtox.models import SpectralRefineConfig, SpectralRefineModel, spectral_physchem_features
from ghxtox.models.spectral_physchem import SPECTRAL_FEATURE_VERSION
from ghxtox.prottox_spectral_refine import MODEL_NAME, MODEL_VERSION, checkpoint_payload, file_sha256, write_manifest
from ghxtox.train_prottox_tsk import _folds, _load_cache, _metrics


def _fit(base_features: torch.Tensor, spectral_features: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, *, config: SpectralRefineConfig, epochs: int, learning_rate: float, seed: int, device: torch.device) -> tuple[SpectralRefineModel, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SpectralRefineModel(config).to(device)
    train_base, train_spectral, train_labels = base_features[train_indices], spectral_features[train_indices], labels[train_indices]
    base_mean, base_std = train_base.mean(dim=0), train_base.std(dim=0, unbiased=False).clamp_min(1e-6)
    spectral_mean, spectral_std = train_spectral.mean(dim=0), train_spectral.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    normalized_base = ((train_base - base_mean) / base_std).to(device)
    normalized_spectral = ((train_spectral - spectral_mean) / spectral_std).to(device)
    target = train_labels.to(device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(normalized_base, normalized_spectral), target)
        loss.backward()
        optimizer.step()
    return model.eval(), base_mean, base_std, spectral_mean, spectral_std


def train(cache_path: str | Path, output_dir: str | Path, *, variant: str, fold_manifest: str, folds: int, epochs: int, learning_rate: float, hidden_dim: int, spectral_hidden_dim: int, dropout: float, seed: int, device_name: str) -> dict[str, Any]:
    cache, base_features, labels, _ = _load_cache(cache_path)
    if not fold_manifest:
        raise ValueError("Reported spectral-refinement runs require the audited five-fold manifest")
    fold_ids = _folds(cache, labels, fold_manifest, folds, seed)
    spectral_features = spectral_physchem_features([str(record["sequence"]) for record in cache["records"]])
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    config = SpectralRefineConfig(base_input_dim=int(base_features.shape[1]), spectral_dim=int(spectral_features.shape[1]), variant=variant, hidden_dim=hidden_dim, spectral_hidden_dim=spectral_hidden_dim, dropout=dropout)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(folds):
        train_indices = torch.as_tensor(np.flatnonzero(fold_ids != fold), dtype=torch.long)
        held_indices = torch.as_tensor(np.flatnonzero(fold_ids == fold), dtype=torch.long)
        model, base_mean, base_std, spectral_mean, spectral_std = _fit(base_features, spectral_features, labels, train_indices, config=config, epochs=epochs, learning_rate=learning_rate, seed=seed + fold, device=device)
        with torch.inference_mode():
            probability = torch.sigmoid(model(((base_features[held_indices] - base_mean) / base_std).to(device), ((spectral_features[held_indices] - spectral_mean) / spectral_std).to(device))).cpu().tolist()
        probability_array = np.asarray(probability, dtype=np.float64)
        oof[held_indices.tolist()] = probability_array
        torch.save(checkpoint_payload(model, base_mean=base_mean, base_std=base_std, spectral_mean=spectral_mean, spectral_std=spectral_std, metadata={"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold_manifest": fold_manifest, "fold": fold, "epochs": epochs, "seed": seed + fold, "trained_records": int(len(train_indices)), "held_out_records": int(len(held_indices)), "variant": variant}), output / f"fold_{fold}.pt")
        fold_rows.append({"fold": fold, "n_train": int(len(train_indices)), "n_test": int(len(held_indices)), **_metrics(np.asarray(labels[held_indices].tolist(), dtype=np.float64), probability_array)})
    if not np.isfinite(oof).all():
        raise RuntimeError("Spectral-refinement OOF training did not emit exactly one probability per cached record")
    final_model, base_mean, base_std, spectral_mean, spectral_std = _fit(base_features, spectral_features, labels, torch.arange(len(labels)), config=config, epochs=epochs, learning_rate=learning_rate, seed=seed + 1000, device=device)
    torch.save(checkpoint_payload(final_model, base_mean=base_mean, base_std=base_std, spectral_mean=spectral_mean, spectral_std=spectral_std, metadata={"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold_manifest": fold_manifest, "epochs": epochs, "run_seed": seed, "refit_seed": seed + 1000, "trained_records": int(len(labels)), "variant": variant, "selection_scope": "full-data refit after OOF; not an OOF metric source"}), output / "final_model.pt")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence_sha256", "label", "fold", "probability"])
        writer.writeheader()
        for record, fold, probability in zip(cache["records"], fold_ids.tolist(), oof.tolist()):
            writer.writerow({"sample_id": record["sample_id"], "sequence_sha256": record["sequence_sha256"], "label": record["label"], "fold": fold, "probability": f"{probability:.9g}"})
    summary = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "base_model_version": cache["model_version"], "spectral_feature_version": SPECTRAL_FEATURE_VERSION, "variant": variant, "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path), "records": len(labels)}, "fold_manifest": fold_manifest, "folds": fold_rows, "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof), "epochs": epochs, "learning_rate": learning_rate, "seed": seed, "parameter_count": sum(parameter.numel() for parameter in final_model.parameters() if parameter.requires_grad), "elapsed_seconds": time.perf_counter() - started, "device": str(device), "final_checkpoint": str(output / "final_model.pt"), "selection_scope": "training OOF only; Test1/Test2 excluded from variant selection"}
    write_manifest(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["spectral_only", "concat", "gated_residual"], required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--spectral-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = train(args.cache, args.output_dir, variant=args.variant, fold_manifest=args.fold_manifest, folds=args.folds, epochs=args.epochs, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim, spectral_hidden_dim=args.spectral_hidden_dim, dropout=args.dropout, seed=args.seed, device_name=args.device)
    print(json.dumps(summary["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
