"""Full-data refit of the frozen MPCE-3DKD architecture for deployment audit.

Every training sample receives exactly one teacher soft target from the
strict outer-fold teacher that never saw the sample's outer fold
(crossfit_teacher_targets_6387.csv). Two pre-registered full-data seeds are
refit; deployment aggregates the two probabilities by arithmetic mean.

This is an audit/full-data artifact only: it does not change the OOF model
selection, ProtTox-3DKD v1, the website or Test1/Test2.
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

from ghxtox.prottox_3d_distill_targets import sha256_file
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_mpce_3dkd import _kd_loss, _prototype_loss
from ghxtox.train_mpce_net import MPCEB


MODEL_NAME = "MPCE-3DKD-Product"
MODEL_VERSION = "mpce-3dkd-product-v1"


def _load_cache(cache_path: str | Path) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    if not records:
        raise ValueError("Cache has no records")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    if tri.shape[1] != 3072 or phys.shape[1] != 9:
        raise ValueError("Full refit requires the audited 3072+9 feature contract")
    return records, tri, phys, labels


def _fit_full(
    tri: torch.Tensor,
    phys: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    prototype_weight: float,
    kd_weight: float,
    temperature: float,
) -> tuple[MPCEB, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MPCEB().to(device)
    tri_mean = tri.mean(dim=0)
    tri_std = tri.std(dim=0, unbiased=False).clamp_min(1e-6)
    phys_mean = phys.mean(dim=0)
    phys_std = phys.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_tri = (tri - tri_mean) / tri_std
    normalized_phys = (phys - phys_mean) / phys_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(labels), generator=generator)
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            tri_x = normalized_tri[rows].to(device)
            phys_x = normalized_phys[rows].to(device)
            y = labels[rows].to(device)
            teacher = teacher_logits[rows].to(device)
            z_final, _z_global, _z_proto, cosine = model(tri_x, phys_x)
            toxicity_loss = F.binary_cross_entropy_with_logits(z_final, y)
            prototype_loss = _prototype_loss(cosine, y)
            distill_loss = _kd_loss(z_final, teacher, temperature)
            loss = (
                toxicity_loss
                + prototype_weight * prototype_loss
                + kd_weight * distill_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(rows)
            count += len(rows)
        final_loss = loss_sum / count
    return model.eval(), tri_mean, tri_std, phys_mean, phys_std, final_loss


def refit(cache_path: str | Path, crossfit_csv: str | Path, output_dir: str | Path, seeds: list[int], device_name: str = "cuda") -> dict[str, Any]:
    records, tri, phys, labels = _load_cache(cache_path)
    with Path(crossfit_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        crossfit_rows = list(csv.DictReader(handle))
    if len(crossfit_rows) != len(records):
        raise ValueError("Crossfit table row count does not match cache records")
    records_by_historical = {str(record["historical_sample_id"]): record for record in records}
    if len(records_by_historical) != len(records):
        raise ValueError("Duplicate historical_sample_id in cache")
    crossfit_by_sample = {}
    for row in crossfit_rows:
        record = records_by_historical.get(str(row["sample_id"]))
        if record is None:
            raise ValueError(f"No student record for crossfit sample {row['sample_id']}")
        crossfit_by_sample[str(record["sample_id"])] = row
    if len(crossfit_by_sample) != len(records):
        raise ValueError("Crossfit table does not cover every cache record")
    teacher_logits = torch.full((len(records),), float("nan"), dtype=torch.float32)
    for record_index, record in enumerate(records):
        row = crossfit_by_sample[str(record["sample_id"])]
        if int(row["outer_fold"]) != int(record["fold"]):
            raise ValueError(f"Crossfit teacher fold mismatch for {record['sample_id']}")
        if str(row["teacher_sample_role"]) != "outer_holdout":
            raise ValueError("Crossfit teacher role must be outer_holdout")
        if str(row["sequence"]) != str(record["sequence"]):
            raise ValueError(f"Crossfit sequence mismatch for {record['sample_id']}")
        teacher_logits[record_index] = float(row["teacher_logit"])
    if not torch.isfinite(teacher_logits).all():
        raise ValueError("Crossfit teacher logits must all be finite")

    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoints = []
    for seed in seeds:
        model, tri_mean, tri_std, phys_mean, phys_std, final_loss = _fit_full(
            tri,
            phys,
            labels,
            teacher_logits,
            seed=seed,
            device=device,
            epochs=20,
            learning_rate=1e-3,
            batch_size=256,
            prototype_weight=0.5,
            kd_weight=1.0,
            temperature=2.0,
        )
        checkpoint_path = output / f"full_model_seed{seed}.pt"
        checkpoint_payload = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "seed": seed,
            "model_state": model.state_dict(),
            "tri_mean": tri_mean.detach().cpu().float(),
            "tri_std": tri_std.detach().cpu().float(),
            "phys_mean": phys_mean.detach().cpu().float(),
            "phys_std": phys_std.detach().cpu().float(),
            "metadata": {
                "seed": seed,
                "cache_path": str(cache_path),
                "cache_sha256": file_sha256(cache_path),
                "crossfit_targets": str(crossfit_csv),
                "crossfit_targets_sha256": sha256_file(crossfit_csv),
                "epochs": 20,
                "learning_rate": 1e-3,
                "batch_size": 256,
                "prototype_weight": 0.5,
                "kd_temperature": 2.0,
                "kd_weight": 1.0,
                "selection_scope": "full training-data refit; Test1/Test2 excluded",
                "final_train_loss": final_loss,
            },
        }
        torch.save(checkpoint_payload, checkpoint_path)
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
                "final_train_loss": final_loss,
            }
        )
        print(f"full refit seed={seed} final_train_loss={final_loss:.6g} done", flush=True)
    manifest = {
        "artifact_type": "mpce_3dkd_product_full_refit",
        "version": 1,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "architecture": "Frozen ProtT5 TriStat views + PhysChem -> MPCE multi-view/cross/prototype -> probability; 3D teacher is training-only",
        "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        "crossfit_targets": {"path": str(crossfit_csv), "sha256": sha256_file(crossfit_csv), "rows": len(crossfit_rows)},
        "checkpoints": checkpoints,
        "ensemble": {"kind": "mean_probability", "members": seeds, "threshold": 0.5},
        "hyperparameters": {
            "epochs": 20,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "prototype_weight": 0.5,
            "kd_temperature": 2.0,
            "kd_weight": 1.0,
        },
        "selection_scope": "full training-data refit after frozen OOF architecture decision; Test1/Test2 excluded",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    write_manifest(output / "ensemble_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--crossfit-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if sorted(args.seed) != [20260905, 20260906]:
        raise ValueError("Product refit seeds are pre-registered as 20260905 and 20260906")
    result = refit(args.cache, args.crossfit_csv, args.output_dir, sorted(args.seed), device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
