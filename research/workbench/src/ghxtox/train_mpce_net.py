"""Train MPCE-Net B: multi-view cross interactions + multi-prototype branch."""
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


MODEL_NAME = "MPCE-Net"
MODEL_VERSION = "b-v1"
VIEW_DIM = 64
LATENT_DIM = 96
PROTOTYPES_PER_CLASS = 4


class MPCEB(nn.Module):
    def __init__(self, embedding_dim: int = 1024, phys_dim: int = 9) -> None:
        super().__init__()
        self.proj_mean = nn.Sequential(nn.Linear(embedding_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_max = nn.Sequential(nn.Linear(embedding_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_std = nn.Sequential(nn.Linear(embedding_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_phys = nn.Sequential(nn.Linear(phys_dim, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        global_input = VIEW_DIM * 8
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 1),
        )
        self.latent = nn.Sequential(nn.Linear(VIEW_DIM * 4, LATENT_DIM), nn.LayerNorm(LATENT_DIM), nn.GELU())
        self.prototypes = nn.Parameter(
            torch.randn(2 * PROTOTYPES_PER_CLASS, LATENT_DIM) * 0.02
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def _views(self, tri, phys):
        mean_view = tri[:, :1024]
        max_view = tri[:, 1024:2048]
        std_view = tri[:, 2048:]
        h_mean = self.proj_mean(mean_view)
        h_max = self.proj_max(max_view)
        h_std = self.proj_std(std_view)
        h_phys = self.proj_phys(phys)
        return h_mean, h_max, h_std, h_phys

    def forward(self, tri, phys):
        h_mean, h_max, h_std, h_phys = self._views(tri, phys)
        mm = h_mean * h_max
        ms = h_mean * h_std
        mx = h_max * h_std
        sp = h_mean * h_phys
        interactions = torch.cat([mm, ms, mx, sp], dim=-1)
        vector = torch.cat([h_mean, h_max, h_std, h_phys, interactions], dim=-1)
        z_global = self.global_mlp(vector).squeeze(-1)
        latent = self.latent(interactions)
        latent = F.normalize(latent, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        cosine = latent @ prototypes.t()  # [B, 2K]
        toxic_scores = cosine[:, :PROTOTYPES_PER_CLASS]
        nontoxic_scores = cosine[:, PROTOTYPES_PER_CLASS:]
        affinity_toxic = torch.logsumexp(toxic_scores, dim=-1)
        affinity_nontoxic = torch.logsumexp(nontoxic_scores, dim=-1)
        z_proto = affinity_toxic - affinity_nontoxic
        z_final = z_global + torch.sigmoid(self.alpha) * z_proto
        return z_final, z_global, z_proto, cosine


def _load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    return payload, tri, phys, labels, folds, records


def _prototype_loss(cosine, labels):
    logits = cosine[:, :PROTOTYPES_PER_CLASS].logsumexp(dim=1)
    logits_non = cosine[:, PROTOTYPES_PER_CLASS:].logsumexp(dim=1)
    class_logits = torch.stack([logits_non, logits], dim=1)
    return F.cross_entropy(class_logits, labels.long())


def _fit(tri, phys, labels, train_indices, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MPCEB().to(device)
    tri_train = tri[train_indices]
    phys_train = phys[train_indices]
    tri_mean = tri_train.mean(dim=0)
    tri_std = tri_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    phys_mean = phys_train.mean(dim=0)
    phys_std = phys_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_tri = (tri - tri_mean) / tri_std
    normalized_phys = (phys - phys_mean) / phys_std
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(20):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), 256):
            rows = order[start : start + 256]
            tri_x = normalized_tri[train_indices[rows]].to(device)
            phys_x = normalized_phys[train_indices[rows]].to(device)
            y = labels[train_indices[rows]].to(device)
            z_final, _z_global, _z_proto, cosine = model(tri_x, phys_x)
            loss = F.binary_cross_entropy_with_logits(z_final, y) + 0.5 * _prototype_loss(cosine, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), tri_mean, tri_std, phys_mean, phys_std


def train(cache_path, output_dir, seed, device_name="cuda"):
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
        train_indices = torch.as_tensor(np.flatnonzero(folds != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(folds == fold), dtype=torch.long)
        model, tri_mean, tri_std, phys_mean, phys_std = _fit(tri, phys, labels, train_indices, seed=seed + fold, device=device)
        normalized_tri = (tri - tri_mean) / tri_std
        normalized_phys = (phys - phys_mean) / phys_std
        with torch.inference_mode():
            z_final, *_rest = model(normalized_tri[test_indices].to(device), normalized_phys[test_indices].to(device))
            probabilities = torch.sigmoid(z_final).cpu().numpy()
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
        print(f"MPCE-B seed={seed} fold={fold} done", flush=True)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.output_dir, args.seed, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
