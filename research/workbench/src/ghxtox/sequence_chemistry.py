"""SCALAM: sequence chemistry-aware length-adaptive multiscale 1D module.

Four variants (selected by config "variant"):
  v1 (recommended): A2 grouped chemistry + B1 cross-channel attention + B3 gate + residual
  v2 (simple)     : A1 direct chemistry projection + B3 gate
  v3 (dual-stream): A2 grouped chemistry + B5 separate multiscale pooling then concat
  v4 (site-style) : A3 group-id embedding chemistry + B1 + B3

Chemistry channel is fully sequence-derived (functional-group ids + 42D
physicochemical descriptors), independent of 3D, preserving the no-structure
fallback design. Output is a per-peptide toxicity logit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ghxtox.folds import load_fold_indices
from ghxtox.utils import save_json, resolve_device


# ---------------------------------------------------------------------------
# 42D residue-feature column groups (see ghxtox.features.residue_feature_matrix)
# ---------------------------------------------------------------------------
CHEM_GROUPS: dict[str, list[int]] = {
    "charge": [1, 5, 6, 9, 31, 32],
    "hydro_aromatic": [0, 3, 10, 12, 15, 30],
    "sulfur": [4, 11, 20, 21, 34],
    "polar_functional": [2, 22, 23, 24, 25, 26, 27, 28, 29, 33, 35, 36, 37, 38],
    "structure_position": [7, 8, 13, 14, 16, 17, 18, 19, 39, 40, 41],
}
GROUP_NAMES = list(CHEM_GROUPS)

# length-aware kernel sets
LENGTH_KERNELS = {
    "11-15": (3, 5),
    "16-30": (3, 5, 7),
    "31-50": (3, 5, 7, 9),
}


def _kernels_for_length(length: int) -> tuple[int, ...]:
    if length <= 15:
        return LENGTH_KERNELS["11-15"]
    if length <= 30:
        return LENGTH_KERNELS["16-30"]
    return LENGTH_KERNELS["31-50"]


class SequenceChemistryModule(nn.Module):
    """SCALAM classifier; see module docstring for variant semantics."""

    def __init__(
        self,
        token_dim: int = 1024,
        sem_hidden: int = 128,
        chem_hidden: int = 32,
        group_embed_dim: int = 16,
        num_filters: int = 64,
        dropout: float = 0.2,
        variant: str = "v1",
        num_group_classes: int = 16,
    ) -> None:
        super().__init__()
        if variant not in ("v1", "v2", "v3", "v4"):
            raise ValueError(f"Unknown variant {variant!r}.")
        self.variant = variant

        self.sem_proj = nn.Sequential(
            nn.Linear(token_dim, sem_hidden), nn.LayerNorm(sem_hidden), nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

        # ---- chemistry encoder ----
        if variant in ("v1", "v3"):
            # A2 grouped chemistry
            self.chem_groups = nn.ModuleDict(
                {name: nn.Linear(len(idx), chem_hidden) for name, idx in CHEM_GROUPS.items()}
            )
            chem_out = chem_hidden * len(GROUP_NAMES)
        elif variant == "v2":
            # A1 direct projection
            self.chem_proj = nn.Linear(42, chem_hidden)
            chem_out = chem_hidden
        else:  # v4 site-style
            self.group_embed = nn.Embedding(num_group_classes, group_embed_dim)
            self.site_proj = nn.Linear(42, group_embed_dim)
            chem_out = group_embed_dim * 2

        # ---- interaction ----
        interact_in = sem_hidden + chem_out
        if variant in ("v1", "v4"):
            # B1 cross-channel attention (semantic query x chemistry key/value)
            self.attn_q = nn.Linear(sem_hidden, sem_hidden)
            self.attn_k = nn.Linear(chem_out, sem_hidden)
            self.attn_v = nn.Linear(chem_out, sem_hidden)
            self.attn_out = nn.Linear(sem_hidden, sem_hidden)

        # gate (B3) used by v1/v2/v4
        if variant in ("v1", "v2", "v4"):
            self.gate = nn.Sequential(nn.Linear(interact_in, 1), nn.Sigmoid())
            self.alpha = nn.Parameter(torch.tensor(0.1))

        # ---- length-adaptive multiscale on the fused representation ----
        self.bottleneck = nn.Conv1d(interact_in, 64, kernel_size=1)
        self.multiscale = nn.ModuleList(
            [
                nn.Conv1d(64, num_filters, kernel_size=size, padding=0)
                for size in sorted(set(k for k in LENGTH_KERNELS.values() for k in k))
            ]
        )
        self.kernel_to_index = {
            size: index for index, size in enumerate(sorted(set(k for k in LENGTH_KERNELS.values() for k in k)))
        }
        max_windows = len(LENGTH_KERNELS["31-50"])
        self.hidden = nn.Sequential(
            nn.Linear(max_windows * num_filters, 128),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(128, 1)
        self._reset()

    def _reset(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if hasattr(self, "alpha"):
            nn.init.constant_(self.alpha, 0.1)

    def _chemistry(self, group_ids: torch.Tensor, residue_features: torch.Tensor) -> torch.Tensor:
        if self.variant in ("v1", "v3"):
            encoded = [
                self.chem_groups[name](residue_features[:, :, idx]) for name, idx in CHEM_GROUPS.items()
            ]
            return torch.relu(torch.cat(encoded, dim=-1))
        if self.variant == "v2":
            return torch.relu(self.chem_proj(residue_features))
        # v4
        emb = self.group_embed(group_ids)
        phys = torch.relu(self.site_proj(residue_features))
        return torch.cat([emb, phys], dim=-1)

    def forward(
        self,
        token: torch.Tensor,
        group_ids: torch.Tensor,
        residue_features: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = token.shape
        sem = self.sem_proj(token)  # (B, L, H)
        chem = self._chemistry(group_ids, residue_features)  # (B, L, C)

        if self.variant in ("v1", "v4"):
            q = self.attn_q(sem)
            k = self.attn_k(chem)
            v = self.attn_v(chem)
            scores = torch.einsum("blh,bjh->blj", q, k) / (k.shape[-1] ** 0.5)
            mask = torch.arange(seq_len, device=token.device)[None, :] < lengths[:, None]
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
            scores = torch.softmax(scores, dim=-1)
            attended = torch.einsum("blj,bjh->blh", scores, v)
            sem_rep = self.dropout(self.attn_out(attended)) + sem  # residual keeps pure semantics
        elif self.variant == "v2":
            sem_rep = sem
        else:  # v3
            sem_rep = sem
        fused = torch.cat([sem_rep, chem], dim=-1)
        if self.variant in ("v1", "v2", "v4"):
            g = self.gate(fused)
            fused = torch.cat([fused[..., : sem_rep.shape[-1]], fused[..., sem_rep.shape[-1]:] * g], dim=-1)

        # length-adaptive multiscale
        kernels = _kernels_for_length(int(lengths.max().item())) if lengths is not None else (3, 5, 7)
        fused = fused.transpose(1, 2)  # (B, D, L)
        fused = torch.relu(self.bottleneck(fused))
        pooled = []
        for size in kernels:
            conv = self.multiscale[self.kernel_to_index[size]]
            out = torch.relu(conv(fused))
            pooled.append(out.amax(dim=-1))
        while len(pooled) < len(LENGTH_KERNELS["31-50"]):
            pooled.append(torch.zeros_like(pooled[0]))
        representation = self.dropout(torch.cat(pooled, dim=-1))
        return self.classifier(self.hidden(representation)).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class ChemistrySequenceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        r = self.records[index]
        return {
            "token": r["token"],
            "group_ids": r["group_ids"],
            "residue_features": r["residue_features"],
            "label": torch.tensor(int(r["label"]), dtype=torch.float32),
            "length": torch.tensor(int(r["length"]), dtype=torch.long),
        }


def collate_chemistry(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    token = torch.zeros(len(batch), max_len, batch[0]["token"].shape[-1], dtype=batch[0]["token"].dtype)
    group_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    residue = torch.zeros(len(batch), max_len, batch[0]["residue_features"].shape[-1], dtype=torch.float32)
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.stack([item["length"] for item in batch])
    for i, item in enumerate(batch):
        L = int(item["length"])
        token[i, :L] = item["token"]
        group_ids[i, :L] = item["group_ids"]
        residue[i, :L] = item["residue_features"]
    return {
        "token": token,
        "group_ids": group_ids,
        "residue_features": residue,
        "label": labels,
        "length": lengths,
    }


# ---------------------------------------------------------------------------
# Data assembly from existing caches
# ---------------------------------------------------------------------------
def build_records(
    processed_path: str | Path,
    adapter_path: str | Path,
) -> list[dict[str, Any]]:
    processed = torch.load(processed_path, map_location="cpu", weights_only=False)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    records = processed["records"]
    tokens = adapter["token_embeddings"]
    offsets = adapter["offsets"]
    sample_ids = list(adapter["sample_ids"])
    by_id = {str(r["sample_id"]): r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate sample_id in processed records.")
    if len(sample_ids) != len(records):
        raise ValueError("Adapter/train sample count mismatch.")
    out: list[dict[str, Any]] = []
    for i, sample_id in enumerate(sample_ids):
        if sample_id not in by_id:
            raise ValueError(f"Adapter sample {sample_id} absent from processed.")
        r = by_id[sample_id]
        start = int(offsets[i])
        end = int(offsets[i + 1])
        length = end - start
        seq_len = int(r["aa_ids"].shape[0])
        if length != seq_len:
            raise ValueError(f"Length mismatch {sample_id}: {length} != {seq_len}")
        out.append(
            {
                "sample_id": sample_id,
                "label": int(r["label"]),
                "length": length,
                "token": tokens[start:end].float(),
                "group_ids": r["group_ids"].long(),
                "residue_features": r["residue_features"].float(),
            }
        )
    return out


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probabilities >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
    return {
        "accuracy": float(np.mean(pred == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "sensitivity": sn,
        "specificity": sp,
        "mcc": float(matthews_corrcoef(labels, pred)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray, tolerance: float = 0.0005) -> float:
    from sklearn.metrics import balanced_accuracy_score
    candidates = np.sort(np.unique(probabilities))
    best_bacc = -1.0
    best_t = 0.5
    for t in candidates:
        bacc = balanced_accuracy_score(labels, (probabilities >= t).astype(int))
        if bacc > best_bacc:
            best_bacc = bacc
            best_t = t
    plateau = [t for t in candidates if balanced_accuracy_score(labels, (probabilities >= t).astype(int)) >= best_bacc - tolerance]
    return float(max(plateau))


def train_fold(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    max_epochs: int | None = None,
) -> tuple[SequenceChemistryModule, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SequenceChemistryModule(
        token_dim=int(config.get("token_dim", 1024)),
        sem_hidden=int(config.get("sem_hidden", 128)),
        chem_hidden=int(config.get("chem_hidden", 32)),
        num_filters=int(config.get("num_filters", 64)),
        dropout=float(config.get("dropout", 0.2)),
        variant=str(config.get("variant", "v1")),
        num_group_classes=int(config.get("num_group_classes", 16)),
    ).to(device)
    train_ds = ChemistrySequenceDataset(train_records)
    val_ds = ChemistrySequenceDataset(val_records)
    loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 64)), shuffle=True, collate_fn=collate_chemistry, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
    n_pos = sum(1 for r in train_records if r["label"] == 1)
    n_neg = len(train_records) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = int(config.get("epochs", 30)) if max_epochs is None else max_epochs
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(batch["token"], batch["group_ids"], batch["residue_features"], batch["length"])
            loss = loss_fn(logits, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        train_prob = predict(model, train_ds, device)
        val_prob = predict(model, val_ds, device)
    return model, train_prob, val_prob


def predict(model: nn.Module, dataset: Dataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_chemistry)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(batch["token"], batch["group_ids"], batch["residue_features"], batch["length"])
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def run_oof(
    records: list[dict[str, Any]],
    fold_manifest: str | Path,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    labels = np.asarray([r["label"] for r in records])
    n = len(records)
    oof = np.zeros(n, dtype=np.float64)
    folds = [0] * n
    for fold in range(5):
        train_idx, val_idx = load_fold_indices(fold_manifest, fold, n)
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        model, _, val_prob = train_fold(train_records, val_records, config, seed + fold, device, max_epochs)
        oof[val_idx] = val_prob
        for i in val_idx:
            folds[i] = fold
    threshold = _best_threshold(labels, oof)
    metrics = _probability_metrics(labels, oof, threshold)
    return {
        "seed": seed,
        "variant": str(config.get("variant")),
        "threshold": threshold,
        "oof_metrics": metrics,
        "oof": oof.tolist(),
        "labels": labels.tolist(),
        "folds": folds,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCALAM sequence-chemistry module")
    sub = parser.add_subparsers(dest="command", required=True)
    p_cv = sub.add_parser("cv")
    p_cv.add_argument("--processed", required=True)
    p_cv.add_argument("--adapter", required=True)
    p_cv.add_argument("--fold-manifest", required=True)
    p_cv.add_argument("--config", required=True)
    p_cv.add_argument("--seed", type=int, default=42)
    p_cv.add_argument("--output-dir", required=True)
    p_cv.add_argument("--device", default="auto")
    p_cv.add_argument("--max-epochs", type=int, default=None)
    p_cv.add_argument("--variant", default=None, help="Override config variant (v1/v2/v3/v4).")
    p_full = sub.add_parser("full")
    p_full.add_argument("--processed", required=True)
    p_full.add_argument("--adapter", required=True)
    p_full.add_argument("--config", required=True)
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--variant", default=None)
    p_full.add_argument("--output", required=True)
    p_full.add_argument("--device", default="auto")
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--processed", required=True)
    p_pred.add_argument("--adapter", required=True)
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--output-csv", required=True)
    p_pred.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    if args.command in ("cv", "full"):
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if getattr(args, "variant", None):
            config["variant"] = args.variant
    if args.command == "cv":
        records = build_records(args.processed, args.adapter)
        result = run_oof(records, args.fold_manifest, config, args.seed, device, args.max_epochs)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(result, out_dir / f"oof_seed{args.seed}.json")
        print(json.dumps({"variant": result["variant"], "threshold": result["threshold"], "oof_metrics": result["oof_metrics"]}, indent=2))
    elif args.command == "full":
        records = build_records(args.processed, args.adapter)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model = SequenceChemistryModule(
            token_dim=int(config.get("token_dim", 1024)),
            sem_hidden=int(config.get("sem_hidden", 128)),
            chem_hidden=int(config.get("chem_hidden", 32)),
            num_filters=int(config.get("num_filters", 64)),
            dropout=float(config.get("dropout", 0.2)),
            variant=str(config.get("variant")),
            num_group_classes=int(config.get("num_group_classes", 16)),
        ).to(device)
        dataset = ChemistrySequenceDataset(records)
        loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=True, collate_fn=collate_chemistry)
        n_pos = sum(1 for r in records if r["label"] == 1); n_neg = len(records) - n_pos
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1), device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4)))
        for _ in range(int(config.get("epochs", 30))):
            model.train()
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                loss = loss_fn(model(batch["token"], batch["group_ids"], batch["residue_features"], batch["length"]), batch["label"])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": config}, args.output)
        print("saved full checkpoint:", args.output)
    elif args.command == "predict":
        records = build_records(args.processed, args.adapter)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = SequenceChemistryModule(
            token_dim=int(cfg.get("token_dim", 1024)), sem_hidden=int(cfg.get("sem_hidden", 128)),
            chem_hidden=int(cfg.get("chem_hidden", 32)), num_filters=int(cfg.get("num_filters", 64)),
            dropout=float(cfg.get("dropout", 0.2)), variant=str(cfg.get("variant")),
            num_group_classes=int(cfg.get("num_group_classes", 16)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        probs = predict(model, ChemistrySequenceDataset(records), device)
        import csv as _csv
        with open(args.output_csv, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
            writer.writeheader()
            for r, p in zip(records, probs):
                writer.writerow({"sample_id": r["sample_id"], "sequence": r.get("sequence", ""), "label": r["label"], "toxicity_probability": f"{float(p):.9g}"})
        print("saved predictions:", args.output_csv)
    else:
        raise SystemExit("Unknown command.")


if __name__ == "__main__":
    main()