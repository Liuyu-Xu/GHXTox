"""Train the isolated two-stage ProtTox structure-quality-proxy student."""

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

from ghxtox.models.structure_quality_proxy import StructureQualityProxyConfig, StructureQualityProxyStudent
from ghxtox.prottox_struct_guided import (
    AUXILIARY_TARGET_NAMES,
    CACHE_FORMAT_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    checkpoint_payload,
)
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_prottox_tsk import _metrics


def _load_cache(path: str | Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CACHE_FORMAT_VERSION or payload.get("model_name") != MODEL_NAME or payload.get("model_version") != MODEL_VERSION:
        raise ValueError("Input is not a compatible ProtTox-StructGuide cache")
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("Structure-guided cache has no records")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    physic = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    auxiliary = torch.stack([torch.as_tensor(record["structure_quality_targets"], dtype=torch.float32) for record in records])
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    expected = 3 * int(payload["encoder"]["embedding_dim"])
    if tri.shape[1] != expected or auxiliary.shape[1] != len(AUXILIARY_TARGET_NAMES):
        raise ValueError("Cache dimensions do not match its recorded feature/auxiliary contract")
    if sorted(set(folds.tolist())) != [0, 1, 2, 3, 4]:
        raise ValueError("Structure-guided cache must preserve the audited five folds")
    return payload, torch.cat((tri, physic), dim=1), labels, auxiliary, folds


def _fit(
    features: torch.Tensor,
    labels: torch.Tensor,
    auxiliary: torch.Tensor,
    indices: torch.Tensor,
    *,
    config: StructureQualityProxyConfig,
    variant: str,
    auxiliary_weight: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[StructureQualityProxyStudent, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = StructureQualityProxyStudent(config).to(device)
    train_x, train_y, train_aux = features[indices], labels[indices], auxiliary[indices]
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    auxiliary_mean = train_aux.mean(dim=0)
    auxiliary_std = train_aux.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    normalized_x = (train_x - feature_mean) / feature_std
    normalized_aux = (train_aux - auxiliary_mean) / auxiliary_std
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(indices), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = normalized_x[rows].to(device)
            y = train_y[rows].to(device)
            target_aux = normalized_aux[rows].to(device)
            logits, predicted_aux = model(x)
            toxicity_loss = F.binary_cross_entropy_with_logits(logits, y)
            auxiliary_loss = F.huber_loss(predicted_aux, target_aux) if variant == "structure_quality_guided" else torch.zeros((), device=device)
            loss = toxicity_loss + float(auxiliary_weight) * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), feature_mean, feature_std, auxiliary_mean, auxiliary_std


def train(
    cache_path: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    shared_dim: int,
    proxy_hidden_dim: int,
    decision_hidden_dim: int,
    dropout: float,
    auxiliary_weight: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if variant not in {"toxicity_only_control", "structure_quality_guided"}:
        raise ValueError("variant must be toxicity_only_control or structure_quality_guided")
    cache, features, labels, auxiliary, fold_ids = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    config = StructureQualityProxyConfig(input_dim=features.shape[1], shared_dim=shared_dim, proxy_hidden_dim=proxy_hidden_dim, decision_hidden_dim=decision_hidden_dim, dropout=dropout)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    oof = np.full(len(labels), np.nan, dtype=np.float64)
    proxy_oof = np.full((len(labels), len(AUXILIARY_TARGET_NAMES)), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold in range(5):
        train_indices = torch.as_tensor(np.flatnonzero(fold_ids != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(fold_ids == fold), dtype=torch.long)
        model, feature_mean, feature_std, auxiliary_mean, auxiliary_std = _fit(features, labels, auxiliary, train_indices, config=config, variant=variant, auxiliary_weight=auxiliary_weight, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed + fold, device=device)
        with torch.inference_mode():
            logits, proxy = model(((features[test_indices] - feature_mean) / feature_std).to(device))
            probabilities = np.asarray(torch.sigmoid(logits).cpu().tolist(), dtype=np.float64)
            predicted_proxy = (proxy.cpu() * auxiliary_std + auxiliary_mean).tolist()
        oof[test_indices.tolist()] = probabilities
        proxy_oof[test_indices.tolist()] = np.asarray(predicted_proxy, dtype=np.float64)
        metadata = {"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "fold": fold, "epochs": epochs, "seed": seed + fold, "variant": variant, "auxiliary_weight": auxiliary_weight, "trained_records": int(len(train_indices)), "held_out_records": int(len(test_indices)), "selection_scope": "training OOF only; Test1/Test2 excluded"}
        torch.save(checkpoint_payload(model, feature_mean=feature_mean, feature_std=feature_std, auxiliary_mean=auxiliary_mean, auxiliary_std=auxiliary_std, metadata=metadata), output / f"fold_{fold}.pt")
        fold_rows.append({"fold": fold, "n_train": int(len(train_indices)), "n_test": int(len(test_indices)), **_metrics(np.asarray(labels[test_indices].tolist(), dtype=np.float64), probabilities)})
    if not np.isfinite(oof).all() or not np.isfinite(proxy_oof).all():
        raise RuntimeError("OOF training did not produce exactly one finite toxicity/proxy output per record")
    full_indices = torch.arange(len(labels))
    final_model, feature_mean, feature_std, auxiliary_mean, auxiliary_std = _fit(features, labels, auxiliary, full_indices, config=config, variant=variant, auxiliary_weight=auxiliary_weight, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, seed=seed + 1000, device=device)
    final_metadata = {"cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path), "epochs": epochs, "run_seed": seed, "refit_seed": seed + 1000, "variant": variant, "auxiliary_weight": auxiliary_weight, "trained_records": int(len(labels)), "selection_scope": "full-data refit after OOF; not an OOF metric source"}
    torch.save(checkpoint_payload(final_model, feature_mean=feature_mean, feature_std=feature_std, auxiliary_mean=auxiliary_mean, auxiliary_std=auxiliary_std, metadata=final_metadata), output / "final_model.pt")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability", *AUXILIARY_TARGET_NAMES])
        writer.writeheader()
        for record, probability, proxy in zip(cache["records"], oof.tolist(), proxy_oof.tolist(), strict=True):
            writer.writerow({"sample_id": record["sample_id"], "historical_sample_id": record["historical_sample_id"], "sequence_sha256": record["sequence_sha256"], "label": record["label"], "fold": record["fold"], "probability": f"{probability:.9g}", **{name: f"{value:.9g}" for name, value in zip(AUXILIARY_TARGET_NAMES, proxy, strict=True)}})
    summary = {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "variant": variant, "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path), "records": len(labels)}, "base_cache": dict(cache["base_cache"]), "structure_quality_source": dict(cache["structure_quality_source"]), "fold_manifest": dict(cache["fold_manifest"]), "auxiliary_targets": list(AUXILIARY_TARGET_NAMES), "auxiliary_loss": {"kind": "huber", "weight": auxiliary_weight if variant == "structure_quality_guided" else 0.0}, "folds": fold_rows, "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof), "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size, "seed": seed, "parameter_count": sum(parameter.numel() for parameter in final_model.parameters()), "elapsed_seconds": time.perf_counter() - started, "device": str(device), "final_checkpoint": str(output / "final_model.pt"), "selection_scope": "training OOF only; Test1/Test2 excluded from model selection"}
    write_manifest(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["toxicity_only_control", "structure_quality_guided"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shared-dim", type=int, default=192)
    parser.add_argument("--proxy-hidden-dim", type=int, default=64)
    parser.add_argument("--decision-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--auxiliary-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.output_dir, variant=args.variant, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, shared_dim=args.shared_dim, proxy_hidden_dim=args.proxy_hidden_dim, decision_hidden_dim=args.decision_hidden_dim, dropout=args.dropout, auxiliary_weight=args.auxiliary_weight, seed=args.seed, device_name=args.device)
    print(json.dumps(result["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
