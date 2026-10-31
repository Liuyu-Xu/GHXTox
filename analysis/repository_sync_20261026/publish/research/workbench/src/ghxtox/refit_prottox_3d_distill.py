"""Full-data refit of the product mlp_teacher_kd student.

Every training sample receives exactly one teacher soft target from the
strict-outer teacher that never saw the sample's outer fold
(crossfit_teacher_targets_6387.csv). Two pre-registered full-data seeds are
trained; the deployment ensemble averages their probabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ghxtox.models.tristat_head import TriStatMLPHead
from ghxtox.prottox_3d_distill_targets import sha256_file
from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_prottox_3d_distill import _load_cache


MODEL_NAME = "ProtTox-3D-Distill-Product"
MODEL_VERSION = "prottox-3d-distill-product-v1"


def _fit_full_mlp(
    features,
    labels,
    teacher_logits,
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
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_x = (features - feature_mean) / feature_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    last_loss = float("nan")
    for _ in range(epochs):
        order = torch.randperm(len(labels), generator=generator)
        loss_sum = 0.0
        count = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = normalized_x[rows].to(device)
            y = labels[rows].to(device)
            teacher = teacher_logits[rows].to(device)
            logits = model(x)
            toxicity_loss = F.binary_cross_entropy_with_logits(logits, y)
            teacher_target = torch.sigmoid(teacher / temperature)
            distill_loss = F.binary_cross_entropy_with_logits(logits / temperature, teacher_target)
            loss = toxicity_loss + kd_weight * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * len(rows)
            count += len(rows)
        last_loss = loss_sum / count
    return model.eval(), feature_mean, feature_std, last_loss


def refit(cache_path, crossfit_csv, output_dir, seeds, device_name="cuda"):
    payload, features, labels, _auxiliary, _fold_ids, records = _load_cache(cache_path)
    crossfit_rows = []
    with Path(crossfit_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        crossfit_rows = list(reader)
    if len(crossfit_rows) != len(records):
        raise ValueError(f"Cross-fitted table rows {len(crossfit_rows)} != cache records {len(records)}")
    records_by_historical = {str(record["historical_sample_id"]): record for record in records}
    if len(records_by_historical) != len(records):
        raise ValueError("Student cache contains duplicate historical_sample_id values")
    crossfit_by_student = {}
    for row in crossfit_rows:
        record = records_by_historical.get(str(row["sample_id"]))
        if record is None:
            raise ValueError(f"No student record for crossfit sample {row['sample_id']}")
        crossfit_by_student[str(record["sample_id"])] = row
    if len(crossfit_by_student) != len(records):
        raise ValueError("Cross-fitted table does not cover every student record")
    teacher_logits = torch.full((len(records),), float("nan"), dtype=torch.float32)
    for record_index, record in enumerate(records):
        row = crossfit_by_student[str(record["sample_id"])]
        if int(row["outer_fold"]) != int(record["fold"]):
            raise ValueError(f"Crossfit teacher fold mismatch for {record['sample_id']}")
        if row["teacher_sample_role"] != "outer_holdout":
            raise ValueError("Cross-fitted target must carry role outer_holdout")
        if str(record["sequence"]) != str(row["sequence"]):
            raise ValueError(f"Sequence mismatch for {record['sample_id']}")
        teacher_logits[record_index] = float(row["teacher_logit"])
    if not torch.isfinite(teacher_logits).all():
        raise ValueError("Cross-fitted teacher logits must all be finite")
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoints = []
    for seed in seeds:
        model, feature_mean, feature_std, final_loss = _fit_full_mlp(
            features,
            labels,
            teacher_logits,
            hidden_dim=192,
            dropout=0.10,
            kd_weight=1.0,
            temperature=2.0,
            epochs=20,
            learning_rate=1e-3,
            batch_size=256,
            seed=seed,
            device=device,
        )
        checkpoint_payload = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "variant": "mlp_teacher_kd_full_refit",
            "config": {"input_dim": int(features.shape[1]), "hidden_dim": 192, "dropout": 0.10},
            "model_state": model.state_dict(),
            "feature_mean": feature_mean.detach().cpu().float(),
            "feature_std": feature_std.detach().cpu().float(),
            "metadata": {
                "seed": seed,
                "cache_path": str(cache_path),
                "cache_sha256": file_sha256(cache_path),
                "crossfit_targets": str(crossfit_csv),
                "crossfit_targets_sha256": sha256_file(crossfit_csv),
                "epochs": 20,
                "learning_rate": 1e-3,
                "batch_size": 256,
                "kd_temperature": 2.0,
                "kd_weight": 1.0,
                "selection_scope": "full training-data refit; Test1/Test2 excluded",
                "final_train_loss": final_loss,
            },
        }
        checkpoint_path = output / f"full_model_seed{seed}.pt"
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
        "artifact_type": "prottox_3d_distill_product_full_refit",
        "version": 1,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "architecture": "Frozen ProtT5 TriStat + PhysChem -> TriStat-MLP -> probability (KD only during training)",
        "cache": {"path": str(cache_path), "sha256": file_sha256(cache_path)},
        "crossfit_targets": {"path": str(crossfit_csv), "sha256": sha256_file(crossfit_csv), "rows": len(crossfit_rows)},
        "checkpoints": checkpoints,
        "ensemble": {"kind": "mean_probability", "members": [int(seed) for seed in seeds], "threshold": 0.5},
        "hyperparameters": {"epochs": 20, "learning_rate": 1e-3, "batch_size": 256, "kd_temperature": 2.0, "kd_weight": 1.0, "hidden_dim": 192, "dropout": 0.10},
        "selection_scope": "full training-data refit after frozen OOF product decision; Test1/Test2 excluded",
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    (output / "ensemble_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
