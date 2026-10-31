"""Train explicit module-ablation controls for the frozen MPCE-3DKD model.

Supported variants:
- no_cross: multi-view and prototypes remain, low-rank cross interaction is
  removed;
- no_proto: multi-view, low-rank cross and teacher KD remain, prototype
  branch/auxiliary loss is removed.

All ablations use the same outer-safe teacher targets, normalization,
five audited folds, seeds and fixed hyperparameters as the full candidate.
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
from torch import nn
from torch.nn import functional as F

from ghxtox.models.mpce_3dkd_ablations import MPCE3KDAblation
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_mpce_net import _prototype_loss
from ghxtox.train_prottox_tsk import _metrics


MODEL_NAME = "MPCE-3DKD-Ablation"
MODEL_VERSION = "v1"


def _load_cache(cache_path: str | Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, list[dict[str, Any]]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    if not records:
        raise ValueError("Cache has no records")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    if tri.shape[1] != 3072 or phys.shape[1] != 9:
        raise ValueError("Ablation requires the audited 3072+9 ProtT5 feature contract")
    return payload, tri, phys, labels, folds, records


def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        torch.sigmoid(teacher_logits / temperature),
    )


def _load_teacher_targets(target_dir: str | Path, outer_fold: int) -> dict[str, dict[str, float]]:
    path = Path(target_dir) / f"teacher_outer{outer_fold}_targets.csv"
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["outer_fold"]) != outer_fold:
                raise ValueError("Teacher file contains a mismatched outer_fold")
            rows.append(row)
    mapping = {
        str(row["sample_id"]): {
            "teacher_probability": float(row["teacher_probability"]),
            "teacher_logit": float(row["teacher_logit"]),
        }
        for row in rows
    }
    if len(mapping) != len(rows):
        raise ValueError("Teacher target file has duplicate sample IDs")
    return mapping


def _loss_for_batch(
    model: nn.Module,
    variant: str,
    tri_x: torch.Tensor,
    phys_x: torch.Tensor,
    y: torch.Tensor,
    teacher: torch.Tensor,
    *,
    prototype_weight: float,
    kd_weight: float,
    temperature: float,
) -> torch.Tensor:
    toxicity_loss = None
    prototype_loss = torch.zeros((), device=y.device)
    if variant == "no_proto":
        z_final = model(tri_x, phys_x)
        toxicity_loss = F.binary_cross_entropy_with_logits(z_final, y)
    else:
        z_final, _z_global, _z_proto, cosine = model(tri_x, phys_x)
        toxicity_loss = F.binary_cross_entropy_with_logits(z_final, y)
        prototype_loss = _prototype_loss(cosine, y)
    distill_loss = _kd_loss(z_final, teacher, temperature)
    return toxicity_loss + prototype_weight * prototype_loss + kd_weight * distill_loss


def _fit(
    variant: str,
    tri: torch.Tensor,
    phys: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    prototype_weight: float,
    kd_weight: float,
    temperature: float,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MPCE3KDAblation(variant=variant).to(device)
    tri_train = tri[train_indices]
    phys_train = phys[train_indices]
    tri_mean = tri_train.mean(dim=0)
    tri_std = tri_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    phys_mean = phys_train.mean(dim=0)
    phys_std = phys_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_tri = (tri - tri_mean) / tri_std
    normalized_phys = (phys - phys_mean) / phys_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            selected = train_indices[rows]
            tri_x = normalized_tri[selected].to(device)
            phys_x = normalized_phys[selected].to(device)
            y = labels[selected].to(device)
            teacher = teacher_logits[selected].to(device)
            loss = _loss_for_batch(
                model,
                variant,
                tri_x,
                phys_x,
                y,
                teacher,
                prototype_weight=prototype_weight,
                kd_weight=kd_weight,
                temperature=temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), tri_mean, tri_std, phys_mean, phys_std


def train(
    cache_path: str | Path,
    target_dir: str | Path,
    output_dir: str | Path,
    *,
    variant: str,
    seed: int,
    device_name: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    prototype_weight: float,
    kd_weight: float,
    temperature: float,
) -> dict[str, Any]:
    payload, tri, phys, labels, folds, records = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    target_manifest = json.loads((Path(target_dir) / "target_manifest.json").read_text(encoding="utf-8"))
    for fold in range(5):
        mapping = _load_teacher_targets(target_dir, fold)
        expected = int(np.count_nonzero(folds != fold))
        if len(mapping) != expected:
            raise ValueError(f"Teacher outer-{fold} coverage {len(mapping)} != train count {expected}")

    oof = np.full(len(records), np.nan, dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(5):
        mapping = _load_teacher_targets(target_dir, fold)
        teacher_logits = torch.full((len(records),), float("nan"), dtype=torch.float32)
        for record_index, record in enumerate(records):
            if int(record["fold"]) == fold:
                continue
            target = mapping.get(str(record["historical_sample_id"]))
            if target is None:
                raise ValueError(f"No teacher target for {record['historical_sample_id']} outer-{fold}")
            teacher_logits[record_index] = float(target["teacher_logit"])
        train_indices = torch.as_tensor(np.flatnonzero(folds != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(folds == fold), dtype=torch.long)
        model, tri_mean, tri_std, phys_mean, phys_std = _fit(
            variant,
            tri,
            phys,
            labels,
            teacher_logits,
            train_indices,
            seed=seed + fold,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            prototype_weight=prototype_weight,
            kd_weight=kd_weight,
            temperature=temperature,
        )
        normalized_tri = (tri[test_indices] - tri_mean) / tri_std
        normalized_phys = (phys[test_indices] - phys_mean) / phys_std
        with torch.inference_mode():
            if variant == "no_proto":
                z_final = model(normalized_tri.to(device), normalized_phys.to(device))
            else:
                z_final, _z_global, _z_proto, _cosine = model(normalized_tri.to(device), normalized_phys.to(device))
            probabilities = np.asarray(torch.sigmoid(z_final).cpu().tolist(), dtype=np.float64)
        oof[test_indices.tolist()] = probabilities
        metadata = {
            "variant": variant,
            "cache_sha256": file_sha256(cache_path),
            "teacher_target_manifest_sha256": file_sha256(Path(target_dir) / "target_manifest.json"),
            "fold": fold,
            "run_seed": seed,
            "per_fold_seed": seed + fold,
            "epochs": epochs,
            "kd_temperature": temperature,
            "kd_weight": kd_weight,
            "prototype_weight": prototype_weight,
            "selection_scope": "training OOF only; Test1/Test2 excluded",
        }
        torch.save(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "variant": variant,
                "model_state": model.state_dict(),
                "tri_mean": tri_mean.detach().cpu(),
                "tri_std": tri_std.detach().cpu(),
                "phys_mean": phys_mean.detach().cpu(),
                "phys_std": phys_std.detach().cpu(),
                "metadata": metadata,
            },
            output / f"fold_{fold}.pt",
        )
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                **_metrics(np.asarray(labels[test_indices].tolist(), dtype=np.float64), probabilities),
            }
        )
        print(f"ablation={variant} seed={seed} fold={fold} done", flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError("Ablation OOF did not produce one finite probability per record")
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
            "manifest_sha256": file_sha256(Path(target_dir) / "target_manifest.json"),
            "kd_temperature": temperature,
            "kd_weight": kd_weight,
            "prototype_weight": prototype_weight,
        },
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "folds": fold_rows,
        "oof_metrics": _metrics(np.asarray(labels.tolist(), dtype=np.float64), oof),
        "selection_scope": "training OOF only; Test1/Test2 excluded",
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
    parser.add_argument("--variant", choices=["no_cross", "no_proto"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prototype-weight", type=float, default=0.5)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    args = parser.parse_args()
    result = train(
        args.cache,
        args.target_dir,
        args.output_dir,
        variant=args.variant,
        seed=args.seed,
        device_name=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        prototype_weight=args.prototype_weight,
        kd_weight=args.kd_weight,
        temperature=args.kd_temperature,
    )
    print(json.dumps(result["oof_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
