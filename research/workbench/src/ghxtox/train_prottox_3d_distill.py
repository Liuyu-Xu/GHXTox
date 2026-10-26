"""Train outer-safe 3D-expert-distilled lightweight ProtTox students.

Variants:
- mlp_teacher_kd: retained-style TriStat-MLP trained with label loss and
  temperature-scaled distillation from teacher_outer{k} targets.
- proxy_teacher_kd: StructureQualityProxyStudent (two-stage proxy design)
  trained with label loss, proxy Huber loss and the same teacher targets.

Only training-OOF metrics are produced; no Test1/Test2 access and no final
deployment refit is created in this validation phase.
"""
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
from ghxtox.models.tristat_head import TriStatMLPHead
from ghxtox.prottox_struct_guided import AUXILIARY_TARGET_NAMES
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_prottox_tsk import _metrics


MODEL_NAME = "ProtTox-3D-Distill"
MODEL_VERSION = "prottox-3d-distill-v1"


def _load_cache(cache_path: str | Path):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "ProtTox-StructGuide" or payload.get("model_version") != "prottox-struct-guide-v1":
        raise ValueError("Input cache is not the audited ProtTox-StructGuide feature cache")
    records = list(payload.get("records", []))
    if not records:
        raise ValueError("Cache has no records")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    physic = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    auxiliary = torch.stack([torch.as_tensor(record["structure_quality_targets"], dtype=torch.float32) for record in records])
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    if tri.shape[1] != 3072 or physic.shape[1] != 9 or auxiliary.shape[1] != 3:
        raise ValueError("Cache dimensions do not match the ProtTox feature contract")
    if sorted(set(folds.tolist())) != [0, 1, 2, 3, 4]:
        raise ValueError("Cache must preserve the audited five folds")
    return payload, torch.cat((tri, physic), dim=1), labels, auxiliary, folds, records


def _load_teacher_targets(target_dir: str | Path, outer_fold: int) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    path = Path(target_dir) / f"teacher_outer{outer_fold}_targets.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["outer_fold"]) != outer_fold:
                raise ValueError("Teacher target file contains a mismatched outer_fold")
            rows.append(row)
    mapping = {
        str(row["sample_id"]): {
            "teacher_probability": float(row["teacher_probability"]),
            "teacher_logit": float(row["teacher_logit"]),
            "teacher_sample_role": str(row["teacher_sample_role"]),
        }
        for row in rows
    }
    if len(mapping) != len(rows):
        raise ValueError("Teacher target file has duplicate sample IDs")
    return mapping, rows


def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        torch.sigmoid(teacher_logits / temperature),
    )


def _fit_proxy(
    features,
    labels,
    auxiliary,
    teacher_logits,
    train_indices,
    *,
    config,
    auxiliary_weight,
    kd_weight,
    temperature,
    epochs,
    learning_rate,
    batch_size,
    seed,
    device,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = StructureQualityProxyStudent(config).to(device)
    train_x = features[train_indices]
    train_y = labels[train_indices]
    train_aux = auxiliary[train_indices]
    train_teacher = teacher_logits[train_indices]
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
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = normalized_x[rows].to(device)
            y = train_y[rows].to(device)
            target_aux = normalized_aux[rows].to(device)
            teacher = train_teacher[rows].to(device)
            logits, predicted_aux = model(x)
            toxicity_loss = F.binary_cross_entropy_with_logits(logits, y)
            auxiliary_loss = F.huber_loss(predicted_aux, target_aux)
            distill_loss = _kd_loss(logits, teacher, temperature)
            loss = toxicity_loss + auxiliary_weight * auxiliary_loss + kd_weight * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), feature_mean, feature_std, auxiliary_mean, auxiliary_std


def _fit_mlp(
    features,
    labels,
    teacher_logits,
    train_indices,
    *,
    hidden_dim,
    dropout,
    kd_weight,
    temperature,
    epochs,
    learning_rate,
    batch_size,
    seed,
    device,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TriStatMLPHead(int(features.shape[1]), hidden_dim=hidden_dim, dropout=dropout).to(device)
    train_x = features[train_indices]
    train_y = labels[train_indices]
    train_teacher = teacher_logits[train_indices]
    feature_mean = train_x.mean(dim=0)
    feature_std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    normalized_x = (train_x - feature_mean) / feature_std
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = normalized_x[rows].to(device)
            y = train_y[rows].to(device)
            teacher = train_teacher[rows].to(device)
            logits = model(x)
            toxicity_loss = F.binary_cross_entropy_with_logits(logits, y)
            distill_loss = _kd_loss(logits, teacher, temperature)
            loss = toxicity_loss + kd_weight * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), feature_mean, feature_std


def train(
    cache_path,
    target_dir,
    output_dir,
    *,
    variant,
    seed,
    device_name,
    epochs,
    learning_rate,
    batch_size,
    shared_dim,
    proxy_hidden_dim,
    decision_hidden_dim,
    dropout,
    auxiliary_weight,
    kd_temperature,
    kd_weight,
):
    if variant not in {"mlp_teacher_kd", "proxy_teacher_kd"}:
        raise ValueError("variant must be mlp_teacher_kd or proxy_teacher_kd")
    payload, features, labels, auxiliary, fold_ids, records = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    target_manifest = json.loads((Path(target_dir) / "target_manifest.json").read_text(encoding="utf-8"))
    teacher_logit_full = np.full(len(records), np.nan, dtype=np.float64)
    teacher_role_full = np.full(len(records), "", dtype=object)
    for fold in range(5):
        mapping, rows = _load_teacher_targets(target_dir, fold)
        for row in rows:
            index = int(row["source_index"])
            if fold_ids[index] == fold:
                raise ValueError(f"Teacher outer-{fold} targets contain held-out sample {index}")
        # A record's own outer table excludes it; the per-fold training loop below
        # loads the fold-specific mapping again from CSV for explicitness.

    oof = np.full(len(records), np.nan, dtype=np.float64)
    fold_rows = []
    config_proxy = StructureQualityProxyConfig(
        input_dim=int(features.shape[1]),
        shared_dim=shared_dim,
        proxy_hidden_dim=proxy_hidden_dim,
        decision_hidden_dim=decision_hidden_dim,
        dropout=dropout,
    )
    for fold in range(5):
        mapping, rows = _load_teacher_targets(target_dir, fold)
        expected_train = int(np.count_nonzero(fold_ids != fold))
        if len(rows) != expected_train:
            raise ValueError(f"Teacher outer-{fold} target count {len(rows)} != student train count {expected_train}")
        teacher_logits = torch.full((len(records),), float("nan"), dtype=torch.float32)
        for record_index, record in enumerate(records):
            if int(record["fold"]) == fold:
                continue
            target = mapping.get(str(record["historical_sample_id"]))
            if target is None:
                raise ValueError(f"No teacher target for {record['historical_sample_id']} in outer fold {fold}")
            teacher_logits[record_index] = float(target["teacher_logit"])
        train_indices = torch.as_tensor(np.flatnonzero(fold_ids != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(fold_ids == fold), dtype=torch.long)
        if variant == "proxy_teacher_kd":
            model, feature_mean, feature_std, auxiliary_mean, auxiliary_std = _fit_proxy(
                features, labels, auxiliary, teacher_logits, train_indices,
                config=config_proxy, auxiliary_weight=auxiliary_weight, kd_weight=kd_weight,
                temperature=kd_temperature, epochs=epochs, learning_rate=learning_rate,
                batch_size=batch_size, seed=seed + fold, device=device,
            )
            with torch.inference_mode():
                logits, _proxy = model(((features[test_indices] - feature_mean) / feature_std).to(device))
            checkpoint_payload = {
                "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "variant": variant,
                "config": dict(config_proxy.__dict__), "model_state": model.state_dict(),
                "feature_mean": feature_mean.detach().cpu(), "feature_std": feature_std.detach().cpu(),
                "auxiliary_mean": auxiliary_mean.detach().cpu(), "auxiliary_std": auxiliary_std.detach().cpu(),
            }
        else:
            model, feature_mean, feature_std = _fit_mlp(
                features, labels, teacher_logits, train_indices,
                hidden_dim=shared_dim, dropout=dropout, kd_weight=kd_weight, temperature=kd_temperature,
                epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
                seed=seed + fold, device=device,
            )
            with torch.inference_mode():
                logits = model(((features[test_indices] - feature_mean) / feature_std).to(device))
            checkpoint_payload = {
                "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "variant": variant,
                "config": {"input_dim": int(features.shape[1]), "hidden_dim": shared_dim, "dropout": dropout},
                "model_state": model.state_dict(),
                "feature_mean": feature_mean.detach().cpu(), "feature_std": feature_std.detach().cpu(),
            }
        probabilities = np.asarray(torch.sigmoid(logits).cpu().tolist(), dtype=np.float64)
        oof[test_indices.tolist()] = probabilities
        metadata = {
            "cache_path": str(cache_path), "cache_sha256": file_sha256(cache_path),
            "teacher_targets": str(target_dir), "teacher_target_manifest_sha256": file_sha256(Path(target_dir) / "target_manifest.json"),
            "fold": fold, "seed": seed + fold, "variant": variant, "epochs": epochs,
            "kd_temperature": kd_temperature, "kd_weight": kd_weight,
            "auxiliary_weight": auxiliary_weight, "trained_records": int(len(train_indices)),
            "held_out_records": int(len(test_indices)), "selection_scope": "training OOF only; Test1/Test2 excluded",
        }
        checkpoint_payload["metadata"] = metadata
        torch.save(checkpoint_payload, output / f"fold_{fold}.pt")
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                **_metrics(np.asarray(labels[test_indices].tolist(), dtype=np.float64), probabilities),
            }
        )
        print(f"variant={variant} seed={seed} fold={fold} done", flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError("OOF training did not produce one finite probability per record")
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability"],
        )
        writer.writeheader()
        for record, probability in zip(records, oof.tolist(), strict=True):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "historical_sample_id": record["historical_sample_id"],
                    "sequence_sha256": record["sequence_sha256"],
                    "label": record["label"],
                    "fold": record["fold"],
                    "probability": f"{probability:.9g}",
                }
            )
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "variant": variant,
        "seed": seed,
        "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path), "records": len(records)},
        "teacher_targets": {
            "dir": str(target_dir),
            "manifest": target_manifest,
            "kd_temperature": kd_temperature,
            "kd_weight": kd_weight,
        },
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "folds": fold_rows,
        "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof),
        "selection_scope": "training OOF only; Test1/Test2 excluded from model selection",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    write_manifest(output / "manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=["mlp_teacher_kd", "proxy_teacher_kd"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shared-dim", type=int, default=192)
    parser.add_argument("--proxy-hidden-dim", type=int, default=64)
    parser.add_argument("--decision-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--auxiliary-weight", type=float, default=0.10)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    args = parser.parse_args()
    result = train(
        args.cache, args.target_dir, args.output_dir, variant=args.variant, seed=args.seed,
        device_name=args.device, epochs=args.epochs, learning_rate=args.learning_rate,
        batch_size=args.batch_size, shared_dim=args.shared_dim, proxy_hidden_dim=args.proxy_hidden_dim,
        decision_hidden_dim=args.decision_hidden_dim, dropout=args.dropout,
        auxiliary_weight=args.auxiliary_weight, kd_temperature=args.kd_temperature, kd_weight=args.kd_weight,
    )
    print(json.dumps(result["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
