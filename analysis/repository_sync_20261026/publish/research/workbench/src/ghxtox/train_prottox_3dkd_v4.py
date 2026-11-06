"""Train V4 uncertainty-gated privileged refinement arms B/C/D."""
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


MODEL_NAME = "ProtTox-3DKD-V4"
MODEL_VERSION = "v4-1"


class UncertaintyGatedPrivilegedStudent(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.10) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.base_head = nn.Linear(hidden_dim, 1)
        self.priv_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.priv_head.weight)
        nn.init.zeros_(self.priv_head.bias)

    def forward(self, features: torch.Tensor, gate_enabled: bool):
        hidden = self.shared(features)
        z_base = self.base_head(hidden).squeeze(-1)
        z_priv = self.priv_head(hidden).squeeze(-1)
        margin = torch.abs(torch.sigmoid(z_base) - 0.5)
        gate = (1.0 - 2.0 * margin).clamp(0.0, 1.0).detach() if gate_enabled else torch.full_like(margin, 0.5)
        z_final = (1.0 - gate) * z_base + gate * z_priv
        return z_final, z_base, z_priv, gate


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


def _load_teacher(features, fold_ids, teacher_dir, fold):
    path = Path(teacher_dir) / f"teacher_outer{fold}_targets.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(np.sum(fold_ids != fold)):
        raise ValueError("Teacher target count mismatch")
    values = np.full(len(features), np.nan)
    for row in rows:
        index = int(row["source_index"])
        values[index] = float(row["teacher_logit"])
    if not np.isfinite(values[fold_ids != fold]).all():
        raise ValueError("Teacher logits incomplete")
    return torch.as_tensor(values, dtype=torch.float32)


def _fit(features, labels, teacher_logits, train_indices, arm, seed, epochs, lr, batch_size, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = UncertaintyGatedPrivilegedStudent(int(features.shape[1])).to(device)
    train_x = features[train_indices]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    gate_enabled = arm in {"C", "D"}
    use_kd = arm in {"B", "D"}
    for _ in range(epochs):
        order = torch.randperm(len(train_indices), generator=generator)
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            x = normalized[train_indices[rows]].to(device)
            y = labels[train_indices[rows]].to(device)
            z_final, z_base, z_priv, gate = model(x, gate_enabled)
            loss = F.binary_cross_entropy_with_logits(z_final, y) + F.binary_cross_entropy_with_logits(z_priv, y)
            if use_kd:
                teacher = teacher_logits[train_indices[rows]].to(device)
                kd = F.binary_cross_entropy_with_logits(z_priv / 2.0, torch.sigmoid(teacher / 2.0), reduction="none")
                weight = gate if arm == "D" else torch.ones_like(gate)
                loss = loss + torch.mean(kd * weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), mean, std


def train(cache_path, teacher_dir, output_dir, arm, seed, device_name="cuda"):
    payload, features, labels, folds, records = _load_cache(cache_path)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    oof = np.full(len(labels), np.nan)
    fold_rows = []
    for fold in range(5):
        teacher_logits = _load_teacher(features, folds, teacher_dir, fold)
        train_indices = torch.as_tensor(np.flatnonzero(folds != fold), dtype=torch.long)
        test_indices = torch.as_tensor(np.flatnonzero(folds == fold), dtype=torch.long)
        model, mean, std = _fit(
            features, labels, teacher_logits, train_indices, arm,
            seed=seed + fold, epochs=20, lr=1e-3, batch_size=256, device=device,
        )
        normalized = (features - mean) / std
        with torch.inference_mode():
            z_final, _base, _priv, _gate = model(normalized[test_indices].to(device), gate_enabled=arm in {"C", "D"})
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
        checkpoint = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "arm": arm,
            "seed": seed,
            "fold": fold,
            "model_state": model.state_dict(),
            "feature_mean": mean.detach().cpu(),
            "feature_std": std.detach().cpu(),
        }
        torch.save(checkpoint, output / f"fold_{fold}.pt")
        print(f"arm={arm} seed={seed} fold={fold} done", flush=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "historical_sample_id", "sequence_sha256", "label", "fold", "probability"])
        for record, probability in zip(records, oof.tolist(), strict=True):
            writer.writerow([record["sample_id"], record["historical_sample_id"], record["sequence_sha256"], record["label"], record["fold"], f"{probability:.9g}"])
    summary = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "arm": arm,
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
    parser.add_argument("--arm", choices=["B", "C", "D"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = train(args.cache, args.teacher_dir, args.output_dir, args.arm, args.seed, device_name=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
