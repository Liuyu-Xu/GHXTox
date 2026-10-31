"""Train ProtTox-3DKD-V2A: teacher-guided residual over frozen v1 fold base."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ghxtox.prottox_3dkd_v2a_targets import _load_v1_fold_checkpoint
from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_prottox_tsk import _metrics


MODEL_NAME = "ProtTox-3DKD-V2A"
MODEL_VERSION = "v2a-1"
SEEDS = (20260905, 20260906)


class TeacherGuidedResidual(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.10) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features: torch.Tensor, base_logit: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(features)).squeeze(-1)
        delta = self.residual(features).squeeze(-1)
        return base_logit + gate * delta


def _load_cache(cache_path):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    features = torch.cat(
        (
            torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records]),
            torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records]),
        ),
        dim=1,
    )
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    return payload, features, labels, folds, records


def _load_targets(dir_path):
    mapping = {}
    for fold in range(5):
        path = Path(dir_path) / f"v2a_residual_targets_outer{fold}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                mapping[(fold, str(row["historical_sample_id"]))] = row
    return mapping


def _fit_candidate(features, labels, teacher_logits, base_logits, train_indices, seed, epochs, learning_rate, batch_size, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TeacherGuidedResidual(int(features.shape[1])).to(device)
    train_x = features[train_indices]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = normalized[train_indices[rows]].to(device)
            y = labels[train_indices[rows]].to(device)
            teacher = teacher_logits[train_indices[rows]].to(device)
            base = base_logits[train_indices[rows]].to(device)
            final_logits = model(x, base)
            label_loss = F.binary_cross_entropy_with_logits(final_logits, y)
            kd_loss = F.binary_cross_entropy_with_logits(final_logits / 2.0, torch.sigmoid(teacher / 2.0))
            residual_loss = F.huber_loss(final_logits - base, teacher - base)
            loss = label_loss + kd_loss + residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), mean, std


def train(cache_path, target_dir, fold_template, output_dir, seed, device_name="cuda"):
    payload, features, labels, folds, records = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(target_dir)
    started = time.perf_counter()
    oof = np.full(len(records), np.nan)
    fold_rows = []
    for fold in range(5):
        fold_models = []
        for model_seed in SEEDS:
            path = Path(str(fold_template).format(seed=model_seed, k=fold))
            model, mean, std, _metadata, _payload = _load_v1_fold_checkpoint(path)
            fold_models.append((model, mean, std))
        with torch.inference_mode():
            probabilities = []
            for model, mean, std in fold_models:
                probabilities.append(torch.sigmoid(model((features - mean) / std.clamp_min(1e-6))))
            base_logits_all = torch.logit((probabilities[0] + probabilities[1]) / 2.0).clamp(-50.0, 50.0)
        teacher_logits = torch.full((len(records),), float("nan"))
        residual_rows = 0
        for index, record in enumerate(records):
            if int(record["fold"]) == fold:
                continue
            row = targets.get((fold, str(record["historical_sample_id"])))
            if row is None:
                raise ValueError(f"Missing V2-A residual target for {record['sample_id']}")
            teacher_logits[index] = float(row["z_teacher_logit"])
            residual_rows += 1
        train_indices = torch.as_tensor(np.flatnonzero(folds != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(folds == fold), dtype=torch.long)
        model, mean, std = _fit_candidate(
            features,
            labels,
            teacher_logits,
            base_logits_all,
            train_indices,
            seed=seed + fold,
            epochs=20,
            learning_rate=1e-3,
            batch_size=256,
            device=device,
        )
        normalized = (features - mean) / std
        with torch.inference_mode():
            final_logits = model(normalized[test_indices].to(device), base_logits_all[test_indices].to(device))
            probabilities = torch.sigmoid(final_logits).cpu().numpy()
        oof[test_indices.numpy()] = probabilities
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "residual_rows": residual_rows,
                **_metrics(labels[test_indices].numpy(), probabilities),
            }
        )
        checkpoint = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "seed": seed,
            "fold": fold,
            "model_state": model.state_dict(),
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
            "metadata": {"frozen_v1_fold_base": "mlp_teacher_kd_seed{20260905,20260906}_v1", "selection_scope": "training OOF only; Test1/Test2 excluded"},
        }
        torch.save(checkpoint, output / f"fold_{fold}.pt")
        print(f"seed={seed} fold={fold} done", flush=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability"])
        for record, probability in zip(records, oof.tolist(), strict=True):
            writer.writerow([record["sample_id"], record["historical_sample_id"], record["sequence_sha256"], record["label"], record["fold"], f"{probability:.9g}"])
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "seed": seed,
        "cache_sha256": file_sha256(cache_path),
        "folds": fold_rows,
        "oof_metrics": _metrics(labels.numpy(), oof),
        "selection_scope": "training OOF only; Test1/Test2 excluded",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--fold-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.target_dir, args.fold_template, args.output_dir, args.seed, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
