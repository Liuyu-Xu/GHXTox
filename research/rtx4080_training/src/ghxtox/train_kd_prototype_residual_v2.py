"""MPCE prototype residual amplitude validation v2 (inner-selected beta)."""
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


MODEL_NAME = "KD-Prototype-Residual-V2"
MODEL_VERSION = "kpr-v2"
VIEW_DIM = 64
LATENT_DIM = 96
PROTOTYPES_PER_CLASS = 4
BETA_CANDIDATES = (0.25, 0.5, 0.75)


class FlatMLPResidual(nn.Module):
    def __init__(self, input_dim: int = 3081, hidden: int = 72) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features):
        return torch.tanh(self.net(features).squeeze(-1))


class PrototypeResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_mean = nn.Sequential(nn.Linear(1024, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_max = nn.Sequential(nn.Linear(1024, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_std = nn.Sequential(nn.Linear(1024, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.proj_phys = nn.Sequential(nn.Linear(9, VIEW_DIM), nn.LayerNorm(VIEW_DIM), nn.GELU())
        self.latent = nn.Sequential(nn.Linear(VIEW_DIM * 4, LATENT_DIM), nn.LayerNorm(LATENT_DIM), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(2 * PROTOTYPES_PER_CLASS, LATENT_DIM) * 0.02)

    def forward(self, tri, phys):
        h_mean = self.proj_mean(tri[:, :1024])
        h_max = self.proj_max(tri[:, 1024:2048])
        h_std = self.proj_std(tri[:, 2048:])
        h_phys = self.proj_phys(phys)
        interactions = torch.cat([h_mean * h_max, h_mean * h_std, h_max * h_std, h_mean * h_phys], dim=-1)
        latent = F.normalize(self.latent(interactions), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        cosine = latent @ prototypes.t()
        toxic = cosine[:, :PROTOTYPES_PER_CLASS].logsumexp(dim=-1)
        nontoxic = cosine[:, PROTOTYPES_PER_CLASS:].logsumexp(dim=-1)
        return torch.tanh(toxic - nontoxic), cosine


def _load_cache(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.float32)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    return payload, tri, phys, labels, folds, records


def _v1_global_logits(tri, phys, fold_template, fold, device):
    probabilities = []
    for seed in (20260905, 20260906):
        path = Path(str(fold_template).format(seed=seed, k=fold))
        model, mean, std, _metadata, _payload = _load_v1_fold_checkpoint(path)
        model.to(device)
        normalized = (torch.cat((tri, phys), dim=1) - mean) / std.clamp_min(1e-6)
        with torch.inference_mode():
            probabilities.append(torch.sigmoid(model(normalized.to(device))).cpu())
    p_global = (probabilities[0] + probabilities[1]) / 2.0
    return torch.logit(p_global.clamp(1e-7, 1 - 1e-7))


def _inner_split(train_indices, labels, fold, seed):
    rng = np.random.RandomState(20260905 + fold)
    train_arr = np.asarray(train_indices)
    y = labels[train_arr].numpy()
    selected = []
    for label_value in (0, 1):
        positions = np.flatnonzero(y == label_value)
        count = max(1, int(round(0.2 * len(positions))))
        chosen = rng.choice(positions, size=count, replace=False)
        selected.extend(train_arr[chosen].tolist())
    val = np.asarray(sorted(selected), dtype=np.int64)
    train = np.asarray(sorted(set(train_arr.tolist()) - set(val.tolist())), dtype=np.int64)
    return torch.as_tensor(train), torch.as_tensor(val)


def _fit_residual(tri, phys, labels, z_global, train_indices, arm, beta, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tri_mean = tri[train_indices].mean(dim=0)
    tri_std = tri[train_indices].std(dim=0, unbiased=False).clamp_min(1e-6)
    phys_mean = phys[train_indices].mean(dim=0)
    phys_std = phys[train_indices].std(dim=0, unbiased=False).clamp_min(1e-6)
    tri_norm = (tri - tri_mean) / tri_std
    phys_norm = (phys - phys_mean) / phys_std
    residual = (FlatMLPResidual() if arm == "B" else PrototypeResidual()).to(device)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    residual.train()
    for _ in range(20):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), 256):
            rows = order[start : start + 256]
            x_tri = tri_norm[train_indices[rows]].to(device)
            x_phys = phys_norm[train_indices[rows]].to(device)
            y = labels[train_indices[rows]].to(device)
            z_row = z_global[train_indices[rows]].to(device)
            if arm == "B":
                delta = residual(torch.cat((x_tri, x_phys), dim=-1))
                z_final = z_row + beta * delta
                loss = F.binary_cross_entropy_with_logits(z_final, y)
            else:
                delta, cosine = residual(x_tri, x_phys)
                z_final = z_row + beta * delta
                toxic = cosine[:, :PROTOTYPES_PER_CLASS].logsumexp(dim=1)
                nontoxic = cosine[:, PROTOTYPES_PER_CLASS:].logsumexp(dim=1)
                loss = F.binary_cross_entropy_with_logits(z_final, y) + 0.5 * F.cross_entropy(
                    torch.stack([nontoxic, toxic], dim=1), y.long()
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return residual.eval(), tri_mean, tri_std, phys_mean, phys_std


def _predict(residual, tri, phys, z_global, indices, arm, beta, tri_mean, tri_std, phys_mean, phys_std, device):
    tri_norm = (tri - tri_mean) / tri_std
    phys_norm = (phys - phys_mean) / phys_std
    x_tri = tri_norm[indices].to(device)
    x_phys = phys_norm[indices].to(device)
    with torch.inference_mode():
        if arm == "B":
            delta = residual(torch.cat((x_tri, x_phys), dim=-1))
        else:
            delta, _cosine = residual(x_tri, x_phys)
        z_final = z_global[indices].to(device) + beta * delta
        probabilities = torch.sigmoid(z_final).cpu().numpy()
        delta = delta.cpu().numpy()
    return probabilities, delta


def train(cache_path, fold_template, output_dir, arm, seed, device_name="cuda"):
    payload, tri, phys, labels, folds, records = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    oof = np.full(len(labels), np.nan)
    audit_rows = []
    fold_rows = []
    beta_selection = []
    for fold in range(5):
        z_global = _v1_global_logits(tri, phys, fold_template, fold, device)
        train_indices = torch.as_tensor(np.flatnonzero(folds != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(folds == fold), dtype=torch.long)
        inner_train, inner_val = _inner_split(train_indices, labels, fold, seed)
        baseline_prob = torch.sigmoid(z_global[inner_val]).numpy()
        baseline_metrics = _metrics(labels[inner_val].numpy(), baseline_prob)
        candidates = []
        for beta in BETA_CANDIDATES:
            residual, tri_mean, tri_std, phys_mean, phys_std = _fit_residual(
                tri, phys, labels, z_global, inner_train, arm, beta, seed=seed + fold, device=device
            )
            probabilities, _delta = _predict(
                residual, tri, phys, z_global, inner_val, arm, beta,
                tri_mean, tri_std, phys_mean, phys_std, device,
            )
            metrics = _metrics(labels[inner_val].numpy(), probabilities)
            candidates.append({"beta": beta, **{k: metrics[k] for k in ("mcc", "auprc", "brier")}})
        eligible = [
            c for c in candidates
            if c["auprc"] >= baseline_metrics["auprc"] - 0.002
            and c["brier"] <= baseline_metrics["brier"] + 0.002
        ]
        selected = max(eligible, key=lambda c: (c["mcc"], -c["beta"])) if eligible else min(candidates, key=lambda c: c["beta"])
        beta = selected["beta"]
        beta_selection.append(
            {
                "fold": fold,
                "baseline_inner": {k: baseline_metrics[k] for k in ("mcc", "auprc", "brier")},
                "candidates": candidates,
                "eligible": [c["beta"] for c in eligible],
                "selected_beta": beta,
            }
        )
        residual, tri_mean, tri_std, phys_mean, phys_std = _fit_residual(
            tri, phys, labels, z_global, train_indices, arm, beta, seed=seed + fold, device=device
        )
        probabilities, delta = _predict(
            residual, tri, phys, z_global, test_indices, arm, beta,
            tri_mean, tri_std, phys_mean, phys_std, device,
        )
        oof[test_indices.numpy()] = probabilities
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "selected_beta": beta,
                **_metrics(labels[test_indices].numpy(), probabilities),
            }
        )
        for position, record_index in enumerate(test_indices.tolist()):
            audit_rows.append(
                {
                    "sample_id": records[record_index]["sample_id"],
                    "historical_sample_id": records[record_index]["historical_sample_id"],
                    "label": int(labels[record_index]),
                    "fold": fold,
                    "v1_probability": float(torch.sigmoid(z_global[record_index])),
                    "final_probability": float(probabilities[position]),
                    "residual_tanh": float(delta[position]),
                    "beta": beta,
                }
            )
        print(f"{arm} seed={seed} fold={fold} beta={beta} done", flush=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability"])
        for record, probability in zip(records, oof.tolist(), strict=True):
            writer.writerow([record["sample_id"], record["historical_sample_id"], record["sequence_sha256"], record["label"], record["fold"], f"{probability:.9g}"])
    with (output / "audit_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "historical_sample_id", "label", "fold", "v1_probability", "final_probability", "residual_tanh", "beta"])
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "arm": arm,
        "seed": seed,
        "cache_sha256": file_sha256(cache_path),
        "beta_candidates": list(BETA_CANDIDATES),
        "beta_selection": beta_selection,
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
    parser.add_argument("--fold-template", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", choices=["B", "C"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.fold_template, args.output_dir, args.arm, args.seed, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
