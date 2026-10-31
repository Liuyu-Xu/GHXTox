"""Train MPCE-B2 dual-branch arms: global_only, fixed fusion, gated fusion."""
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


MODEL_NAME = "MPCE-Net-B2"
MODEL_VERSION = "b2-v1"
VIEW_DIM = 64
LATENT_DIM = 96
PROTOTYPES_PER_CLASS = 4


class MPCEB2(nn.Module):
    def __init__(self, fusion: str) -> None:
        super().__init__()
        self.fusion = fusion
        self.proj_mean = nn.Sequential(nn.Linear(1024, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_max = nn.Sequential(nn.Linear(1024, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_std = nn.Sequential(nn.Linear(1024, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_phys = nn.Sequential(nn.Linear(9, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.global_mlp = nn.Sequential(nn.Linear(VIEW_DIM * 8, 192), nn.GELU(), nn.Dropout(0.10), nn.Linear(192, 1))
        self.latent = nn.Sequential(nn.Linear(VIEW_DIM * 4, LATENT_DIM), nn.LayerNorm(LATENT_DIM), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(2 * PROTOTYPES_PER_CLASS, LATENT_DIM) * 0.02)
        if fusion == "gated":
            self.gate_mlp = nn.Sequential(nn.Linear(VIEW_DIM * 8, 16), nn.GELU(), nn.Linear(16, 1))

    def _interaction_vector(self, tri, phys):
        mean_view = tri[:, :1024]
        max_view = tri[:, 1024:2048]
        std_view = tri[:, 2048:]
        h_mean = self.proj_mean(mean_view)
        h_max = self.proj_max(max_view)
        h_std = self.proj_std(std_view)
        h_phys = self.proj_phys(phys)
        interactions = torch.cat([h_mean * h_max, h_mean * h_std, h_max * h_std, h_mean * h_phys], dim=-1)
        vector = torch.cat([h_mean, h_max, h_std, h_phys, interactions], dim=-1)
        return vector, interactions

    def forward(self, tri, phys):
        vector, interactions = self._interaction_vector(tri, phys)
        z_global = self.global_mlp(vector).squeeze(-1)
        latent = F.normalize(self.latent(interactions), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        cosine = latent @ prototypes.t()
        toxic = cosine[:, :PROTOTYPES_PER_CLASS].logsumexp(dim=-1)
        nontoxic = cosine[:, PROTOTYPES_PER_CLASS:].logsumexp(dim=-1)
        z_proto = toxic - nontoxic
        if self.fusion == "global_only":
            z_final = z_global
        elif self.fusion == "fixed":
            z_final = 0.5 * z_global + 0.5 * z_proto
        else:
            gate = torch.sigmoid(self.gate_mlp(vector)).squeeze(-1)
            z_final = z_global + gate * (z_proto - z_global)
        return z_final, z_global, z_proto, cosine


def _prototype_loss(cosine, labels):
    toxic = cosine[:, :PROTOTYPES_PER_CLASS].logsumexp(dim=1)
    nontoxic = cosine[:, PROTOTYPES_PER_CLASS:].logsumexp(dim=1)
    return F.cross_entropy(torch.stack([nontoxic, toxic], dim=1), labels.long())


def _fit(tri, phys, labels, train_indices, fusion, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MPCEB2(fusion).to(device)
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
            z_final, *_rest, cosine = model(x_tri, x_phys)
            loss = F.binary_cross_entropy_with_logits(z_final, y) + 0.5 * _prototype_loss(cosine, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), tri_mean, tri_std, phys_mean, phys_std


def train(cache_path, output_dir, fusion, seed, device_name="cuda"):
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
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
        model, tri_mean, tri_std, phys_mean, phys_std = _fit(tri, phys, labels, train_indices, fusion, seed=seed + fold, device=device)
        tri_norm = (tri - tri_mean) / tri_std
        phys_norm = (phys - phys_mean) / phys_std
        with torch.inference_mode():
            z_final, *_rest = model(tri_norm[test_indices].to(device), phys_norm[test_indices].to(device))
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
                "fusion": fusion,
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
        print(f"MPCE-B2 {fusion} seed={seed} fold={fold} done", flush=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability"])
        for record, probability in zip(records, oof.tolist(), strict=True):
            writer.writerow([record["sample_id"], record["historical_sample_id"], record["sequence_sha256"], record["label"], record["fold"], f"{probability:.9g}"])
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "fusion": fusion,
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
    parser.add_argument("--fusion", choices=["global_only", "fixed", "gated"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.output_dir, args.fusion, args.seed, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
