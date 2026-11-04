"""Five-fold OOF trainer for the frozen-residue LocalMotif control and candidate."""

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

from ghxtox.models.local_motif import LOCAL_MOTIF_VARIANTS, LocalMotifConfig, LocalMotifResidualModel
from ghxtox.prottox_local_motif import MODEL_NAME, MODEL_VERSION, checkpoint_payload
from ghxtox.prottox_local_motif_cache import CACHE_VERSION, MODEL_NAME as CACHE_MODEL_NAME
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_prottox_tsk import _folds, _metrics


def _load_cache(path: str | Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CACHE_VERSION or payload.get("model_name") != CACHE_MODEL_NAME or payload.get("model_version") != CACHE_VERSION:
        raise ValueError("Input is not a compatible ProtTox-LocalMotif frozen residue cache")
    records = payload.get("records", [])
    if not records or any(record.get("label") is None or "residue_embeddings" not in record for record in records):
        raise ValueError("LocalMotif cache requires labeled records with frozen residue embeddings")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    physic = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    residues = [torch.as_tensor(record["residue_embeddings"], dtype=torch.float16).contiguous() for record in records]
    embedding_dim = int(payload["encoder"]["embedding_dim"])
    if tri.shape[1] != 3 * embedding_dim or any(value.ndim != 2 or value.shape[0] != len(record["sequence"]) or value.shape[1] != embedding_dim for record, value in zip(records, residues, strict=True)):
        raise ValueError("LocalMotif cache has inconsistent base or residue embedding shapes")
    return payload, torch.cat((tri, physic), dim=1), torch.tensor([int(record["label"]) for record in records], dtype=torch.float32), residues


def _batch(indices: torch.Tensor, features: torch.Tensor, residues: list[torch.Tensor], mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [residues[int(index)] for index in indices.tolist()]
    maximum = max(value.shape[0] for value in selected)
    embedding_dim = selected[0].shape[1]
    embeddings = torch.zeros((len(selected), maximum, embedding_dim), dtype=torch.float32)
    mask = torch.zeros((len(selected), maximum), dtype=torch.bool)
    for row, value in enumerate(selected):
        length = value.shape[0]
        embeddings[row, :length] = value.float()
        mask[row, :length] = True
    return ((features[indices] - mean) / std).to(device), embeddings.to(device), mask.to(device)


def _fit(features: torch.Tensor, residues: list[torch.Tensor], labels: torch.Tensor, indices: torch.Tensor, *, config: LocalMotifConfig, epochs: int, learning_rate: float, batch_size: int, seed: int, device: torch.device) -> tuple[LocalMotifResidualModel, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = LocalMotifResidualModel(config).to(device)
    train_features = features[indices]
    mean, std = train_features.mean(dim=0), train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(shuffled), batch_size):
            current = shuffled[start : start + batch_size]
            base, embedding, mask = _batch(current, features, residues, mean, std, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(base, embedding, mask), labels[current].to(device))
            loss.backward()
            optimizer.step()
    return model.eval(), mean, std


def _predict(model: LocalMotifResidualModel, features: torch.Tensor, residues: list[torch.Tensor], indices: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, batch_size: int, device: torch.device) -> np.ndarray:
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            current = indices[start : start + batch_size]
            base, embedding, mask = _batch(current, features, residues, mean, std, device)
            probabilities.extend(torch.sigmoid(model(base, embedding, mask)).cpu().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def train(cache_path: str | Path, output_dir: str | Path, *, variant: str, fold_manifest: str, folds: int, epochs: int, learning_rate: float, hidden_dim: int, local_dim: int, convolution_dim: int, residual_hidden_dim: int, dropout: float, batch_size: int, seed: int, device_name: str) -> dict[str, Any]:
    cache, features, labels, residues = _load_cache(cache_path)
    if not fold_manifest:
        raise ValueError("Reported LocalMotif runs require the audited five-fold manifest")
    fold_ids = _folds(cache, labels, fold_manifest, folds, seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    config = LocalMotifConfig(base_input_dim=int(features.shape[1]), residue_embedding_dim=int(cache["encoder"]["embedding_dim"]), variant=variant, hidden_dim=hidden_dim, local_dim=local_dim, convolution_dim=convolution_dim, residual_hidden_dim=residual_hidden_dim, dropout=dropout)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started, oof, fold_rows = time.perf_counter(), np.full(len(labels), np.nan, dtype=np.float64), []
    for fold in range(folds):
        train_indices = torch.as_tensor(np.flatnonzero(fold_ids != fold), dtype=torch.long)
        held_indices = torch.as_tensor(np.flatnonzero(fold_ids == fold), dtype=torch.long)
        model, mean, std = _fit(features, residues, labels, train_indices, config=config, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed + fold, device=device)
        probabilities = _predict(model, features, residues, held_indices, mean, std, batch_size=batch_size, device=device)
        oof[held_indices.tolist()] = probabilities
        metadata = {"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "base_cache": dict(cache["base_cache"]), "fold_manifest": fold_manifest, "fold": fold, "epochs": epochs, "seed": seed + fold, "variant": variant, "batch_size": batch_size, "trained_records": int(len(train_indices)), "held_out_records": int(len(held_indices))}
        torch.save(checkpoint_payload(model, base_mean=mean, base_std=std, metadata=metadata), output / f"fold_{fold}.pt")
        fold_rows.append({"fold": fold, "n_train": int(len(train_indices)), "n_test": int(len(held_indices)), **_metrics(np.asarray(labels[held_indices].tolist(), dtype=np.float64), probabilities)})
    if not np.isfinite(oof).all():
        raise RuntimeError("LocalMotif OOF training did not emit exactly one probability per cached record")
    final_model, mean, std = _fit(features, residues, labels, torch.arange(len(labels)), config=config, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed + 1000, device=device)
    final_metadata = {"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "base_cache": dict(cache["base_cache"]), "fold_manifest": fold_manifest, "epochs": epochs, "run_seed": seed, "refit_seed": seed + 1000, "variant": variant, "batch_size": batch_size, "trained_records": int(len(labels)), "selection_scope": "full-data refit after OOF; not an OOF metric source"}
    torch.save(checkpoint_payload(final_model, base_mean=mean, base_std=std, metadata=final_metadata), output / "final_model.pt")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence_sha256", "label", "fold", "probability"])
        writer.writeheader()
        for record, fold, probability in zip(cache["records"], fold_ids.tolist(), oof.tolist(), strict=True):
            writer.writerow({"sample_id": record["sample_id"], "sequence_sha256": record["sequence_sha256"], "label": record["label"], "fold": fold, "probability": f"{probability:.9g}"})
    summary = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "variant": variant, "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path), "records": len(labels)}, "base_cache": dict(cache["base_cache"]), "fold_manifest": fold_manifest, "folds": fold_rows, "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof), "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size, "seed": seed, "parameter_count": sum(parameter.numel() for parameter in final_model.parameters() if parameter.requires_grad), "local_branch_parameter_count": final_model.local_branch_parameter_count(), "elapsed_seconds": time.perf_counter() - started, "device": str(device), "final_checkpoint": str(output / "final_model.pt"), "selection_scope": "paired training OOF only; Test1/Test2 excluded from variant selection"}
    write_manifest(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=LOCAL_MOTIF_VARIANTS, required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--local-dim", type=int, default=96)
    parser.add_argument("--convolution-dim", type=int, default=32)
    parser.add_argument("--residual-hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = train(args.cache, args.output_dir, variant=args.variant, fold_manifest=args.fold_manifest, folds=args.folds, epochs=args.epochs, learning_rate=args.learning_rate, hidden_dim=args.hidden_dim, local_dim=args.local_dim, convolution_dim=args.convolution_dim, residual_hidden_dim=args.residual_hidden_dim, dropout=args.dropout, batch_size=args.batch_size, seed=args.seed, device_name=args.device)
    print(json.dumps(summary["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
