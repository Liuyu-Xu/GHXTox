"""Train 3DKD-HyperNet v1 arms: static+KD and dynamic+KD."""
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

from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_prottox_tsk import _metrics


MODEL_NAME = "3DKD-HyperNet"
MODEL_VERSION = "v1"
RANK = 16
HIDDEN = 192


class HyperNetworkPredictorKD(nn.Module):
    def __init__(self, dynamic: bool) -> None:
        super().__init__()
        self.dynamic = dynamic
        self.W0 = nn.Linear(3072, HIDDEN)
        self.U = nn.Parameter(torch.zeros(RANK, HIDDEN))
        self.V = nn.Parameter(torch.zeros(RANK, 3072))
        if dynamic:
            self.context = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, RANK))
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)
            self.fixed_g = None
        else:
            self.fixed_g = nn.Parameter(torch.zeros(RANK))
        self.body = nn.Sequential(nn.LayerNorm(HIDDEN), nn.GELU(), nn.Dropout(0.10), nn.Linear(HIDDEN, 1))

    def forward(self, tri, phys):
        base = self.W0(tri)
        projected = tri @ self.V.t()
        g = self.context(phys) if self.dynamic else self.fixed_g.unsqueeze(0).expand(tri.shape[0], -1)
        hidden = base + (projected * g) @ self.U
        return self.body(hidden).squeeze(-1)


def _load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    return payload, tri, phys, labels, folds, records


def _teacher_logits(teacher_dir, fold, size, folds):
    path = Path(teacher_dir) / f"teacher_outer{fold}_targets.csv"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
    if len(rows) != int(np.sum(folds != fold)):
        raise ValueError("teacher row mismatch")
    values = np.full(size, np.nan)
    for row in rows:
        values[int(row["source_index"])] = float(row["teacher_logit"])
    return torch.as_tensor(values, dtype=torch.float32)


def _fit(tri, phys, labels, teacher, train_indices, dynamic, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = HyperNetworkPredictorKD(dynamic=dynamic).to(device)
    tri_mean = tri[train_indices].mean(dim=0)
    tri_std = tri[train_indices].std(dim=0, unbiased=False).clamp_min(1e-6)
    phys_mean = phys[train_indices].mean(dim=0)
    phys_std = phys[train_indices].std(dim=0, unbiased=False).clamp_min(1e-6)
    tri_norm = (tri - tri_mean) / tri_std
    phys_norm = (phys - phys_mean) / phys_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(20):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), 256):
            rows = order[start : start + 256]
            x_tri = tri_norm[train_indices[rows]].to(device)
            x_phys = phys_norm[train_indices[rows]].to(device)
            y = labels[train_indices[rows]].to(device)
            t = teacher[train_indices[rows]].to(device)
            logits = model(x_tri, x_phys)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            kd = F.binary_cross_entropy_with_logits(logits / 2.0, torch.sigmoid(t / 2.0))
            loss = loss + kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), tri_mean, tri_std, phys_mean, phys_std


def train(cache_path, teacher_dir, output_dir, dynamic, seed, device_name="cuda"):
    payload, tri, phys, labels, folds, records = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    oof = np.full(len(labels), np.nan)
    fold_rows = []
    for fold in range(5):
        teacher = _teacher_logits(teacher_dir, fold, len(labels), folds)
        train_indices = torch.as_tensor(np.flatnonzero(folds != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(folds == fold), dtype=torch.long)
        model, tri_mean, tri_std, phys_mean, phys_std = _fit(
            tri, phys, labels, teacher, train_indices, dynamic, seed=seed + fold, device=device
        )
        tri_norm = (tri - tri_mean) / tri_std
        phys_norm = (phys - phys_mean) / phys_std
        with torch.inference_mode():
            logits = model(tri_norm[test_indices].to(device), phys_norm[test_indices].to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        oof[test_indices.numpy()] = probabilities
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                **_metrics(labels[test_indices].numpy(), probabilities),
            }
        )
        torch.save(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "dynamic": dynamic,
                "seed": seed,
                "fold": fold,
                "model_state": model.state_dict(),
                "tri_mean": tri_mean.cpu(),
                "tri_std": tri_std.cpu(),
                "phys_mean": phys_mean.cpu(),
                "phys_std": phys_std.cpu(),
            },
            output / f"fold_{fold}.pt",
        )
        print(f"3DKD-HyperNet dynamic={dynamic} seed={seed} fold={fold} done", flush=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability"])
        for record, probability in zip(records, oof.tolist(), strict=True):
            writer.writerow([record["sample_id"], record["historical_sample_id"], record["sequence_sha256"], record["label"], record["fold"], f"{probability:.9g}"])
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dynamic": dynamic,
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
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dynamic", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.teacher_dir, args.output_dir, bool(args.dynamic), args.seed, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
