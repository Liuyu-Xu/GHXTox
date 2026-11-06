"""Combination-A 1D expert: LoRA task-adaptive ProtT5 + Bidirectional Mamba (SSM) classifier.

This module replaces two of the four default 1D pipeline modules:
  1. 蛋白质大模型 module  : frozen ProtT5 mean-pooled embeddings
       -> ProtT5 encoder + LoRA task-adaptive fine-tuning (PLMTox-style)
  2. 分类网络 module       : multi-scale TextCNN on a pooled vector
       -> bidirectional Mamba / state-space sequence classifier (BiMA-DTI-style)
The 数据增强 and 多人投票 (multi-seed ensemble) modules remain external.

Commands mirror ghxtox.sequence_chemistry:
  cv      : 5-fold group-aware OOF, threshold selected on OOF BACC
  full    : train on the whole training set and save a checkpoint
  predict : predict a test set and write a probability CSV

Design notes
------------
* ProtT5 is tokenized residue-aligned (" ".join(sequence), add_special_tokens=True),
  matching the existing data/processed/prott5_tokens cache protocol.  The EOS /
  padding positions are excluded from the Mamba scan and from pooling.
* LoRA is implemented without a peft dependency: every nn.Linear inside the T5
  encoder is wrapped by LoRALinear (frozen base + low-rank update).  The wrapper
  is applied in-place, so the HuggingFace forward path is unchanged.
* The Mamba S6 selective scan is implemented in pure PyTorch (sequential loop over
  the sequence).  Peptide lengths are <= 50, so the loop is cheap and avoids the
  platform-specific mamba-ssm CUDA dependency (it cannot be installed on Windows).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from pathlib import Path
from typing import Any

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PUBLIC_1D_EXPERIMENT = "lora_mamba_1d_combinationA_autodl"
PUBLIC_1D_SEEDS = (42, 123, 2025)
PUBLIC_1D_ARCHITECTURE_KEYS = (
    "plm",
    "token_dim",
    "d_model",
    "d_state",
    "d_conv",
    "expand",
    "n_layers",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "dropout",
    "max_length",
)


# ---------------------------------------------------------------------------
# Local helpers (self-contained; no dependency on the rest of the ghxtox package)
# ---------------------------------------------------------------------------
def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        return torch.device("cuda")
    return torch.device(requested)


try:  # torch >= 2.3
    from torch.amp import GradScaler as _GradScaler

    def _make_scaler(enabled: bool):
        return _GradScaler("cuda", enabled=enabled)

    def _amp_autocast(enabled: bool):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

except (ImportError, AttributeError):  # older torch (<= 2.2); GradScaler takes no device arg
    from torch.cuda.amp import GradScaler as _GradScaler

    def _make_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)

    def _amp_autocast(enabled: bool):
        return torch.cuda.amp.autocast(enabled=enabled)


def _save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_fold_indices(
    manifest: str | Path, validation_fold: int, expected_size: int
) -> tuple[list[int], list[int]]:
    """Load train/validation indices from a fixed fold manifest (source_index -> fold)."""
    assignments: dict[int, int] = {}
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assignments[int(row["source_index"])] = int(row["fold"])
    if set(assignments) != set(range(expected_size)):
        raise ValueError(
            f"Fold manifest contains {len(assignments)} indices, expected exactly {expected_size}."
        )
    validation = [index for index, fold in assignments.items() if fold == validation_fold]
    training = [index for index, fold in assignments.items() if fold != validation_fold]
    if not training or not validation:
        raise ValueError(f"Fold {validation_fold} produced an empty train or validation subset.")
    return training, validation


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Linear layer with a frozen base and a trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear only wraps nn.Linear.")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / max(int(rank), 1)
        in_features, out_features = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(int(rank), int(in_features)))
        self.lora_B = nn.Parameter(torch.zeros(int(out_features), int(rank)))
        self.lora_dropout = nn.Dropout(float(dropout))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base_weight_grad_off = True

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_out = self.base(hidden)
        low_rank = (self.lora_dropout(hidden) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * low_rank


def wrap_encoder_lora(encoder: nn.Module, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1) -> int:
    """Wrap every nn.Linear in the encoder in-place with LoRALinear.  Returns the number wrapped."""
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    if rank == 0:
        return 0
    if rank < 0:
        raise ValueError("lora_rank must be nonnegative; zero explicitly disables LoRA.")
    wrapped = 0
    for name, module in list(encoder.named_modules()):
        if isinstance(module, nn.Linear):
            parent = encoder
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacement = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, parts[-1], replacement)
            wrapped += 1
    return wrapped


def remove_inactive_lora(encoder: nn.Module) -> int:
    """Remove zero-update adapters atomically; reject active or invalid adapters."""
    adapters = [(name, module) for name, module in encoder.named_modules()
                if isinstance(module, LoRALinear)]
    active = [name for name, module in adapters if torch.count_nonzero(module.lora_B.detach()).item() != 0]
    if active:
        raise ValueError("Cannot remove active LoRA adapters: " + ", ".join(active[:3]))
    for name, module in adapters:
        parent = encoder
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.base)
    return len(adapters)


# ---------------------------------------------------------------------------
# Mamba (S6) building blocks, pure PyTorch
# ---------------------------------------------------------------------------
def selective_scan_loop(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Sequential selective scan.

    u     : (B, L, D)
    delta : (B, L, D)
    A     : (D, N)
    B, C  : (B, L, N)
    returns (B, L, D)
    """
    batch, length, dim = u.shape
    state_dim = A.shape[1]
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))          # (B, L, D, N)
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)                                  # (B, L, D, N)
    state = torch.zeros(batch, dim, state_dim, device=u.device, dtype=u.dtype)
    outputs: list[torch.Tensor] = []
    for step in range(length):
        state = dA[:, step] * state + dB[:, step] * u[:, step].unsqueeze(-1)
        outputs.append((state * C[:, step].unsqueeze(1)).sum(dim=-1))          # (B, D)
    return torch.stack(outputs, dim=1)


class MambaScan(nn.Module):
    """One-directional S6 scan (input stays in the given order)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=int(d_conv),
            padding=int(d_conv) - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        nn.init.uniform_(self.A_log, a=-5.0, b=-1.0)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """hidden: (B, L, D); valid: (B, L) float mask of valid residue positions."""
        batch, length, _ = hidden.shape
        xz = self.in_proj(hidden)
        x, z = xz.chunk(2, dim=-1)                                             # (B, L, D), (B, L, D)
        x = x.transpose(1, 2)                                                  # (B, D, L)
        x = self.conv1d(x)[..., :length].transpose(1, 2)                       # (B, L, D)
        x = nn.functional.silu(x)
        z = nn.functional.silu(z)
        dt_b_c = self.x_proj(x)
        dt, B_scan, C_scan = dt_b_c.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = nn.functional.softplus(self.dt_proj(dt)).clamp(max=20.0)
        # mask padded positions so they do not inject information into the state
        mask = valid.unsqueeze(-1)                                             # (B, L, 1)
        x_masked = x * mask
        dt_masked = dt * mask
        # run the scan in fp32 to avoid exp() overflow in fp16, then cast back
        A_scan = -torch.exp(self.A_log.float())                                  # (D, N), negative eigenvalues
        y = selective_scan_loop(x_masked.float(), dt_masked.float(), A_scan,
                                B_scan.float(), C_scan.float())                # (B, L, D)
        y = y.to(x.dtype)
        merged = y * z + self.D.unsqueeze(0).unsqueeze(0) * x_masked
        return self.out_proj(merged)


class MambaLayer(nn.Module):
    """Bidirectional Mamba layer: forward scan + backward scan + residual."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.forward_scan = MambaScan(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_scan = MambaScan(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.merge = nn.Sequential(
            nn.Linear(2 * int(d_model), int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residual = hidden
        normalized = self.norm(hidden)
        forward_out = self.forward_scan(normalized, valid)
        reversed_hidden = normalized.flip(1)
        reversed_valid = valid.flip(1)
        backward_out = self.backward_scan(reversed_hidden, reversed_valid).flip(1)
        merged = self.merge(torch.cat([forward_out, backward_out], dim=-1))
        return residual + self.dropout(merged)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class LoraMamba1D(nn.Module):
    """ProtT5(+LoRA) -> projection -> bidirectional Mamba -> masked mean pool -> logit."""

    def __init__(
        self,
        plm: nn.Module,
        token_dim: int = 1024,
        d_model: int = 192,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        lora_rank: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.plm = plm
        self.lora_wrapped = wrap_encoder_lora(plm.encoder, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        self.proj = nn.Sequential(
            nn.Linear(int(token_dim), int(d_model)),
            nn.LayerNorm(int(d_model)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.mamba_layers = nn.ModuleList(
            [MambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout) for _ in range(n_layers)]
        )
        self.pool_norm = nn.LayerNorm(int(d_model))
        self.classifier = nn.Linear(int(d_model), 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
        valid = (positions < lengths[:, None]).float()                          # (B, L)
        x = self.proj(hidden)
        for layer in self.mamba_layers:
            x = layer(x, valid)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).float()
        pooled = self.pool_norm(pooled)
        return self.classifier(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class SequenceRecord:
    __slots__ = ("sample_id", "sequence", "label", "length")


def load_train_records(fold_manifest: str | Path) -> list[SequenceRecord]:
    """Records from the CD-HIT 0.80 group-aware fold manifest (sample_id, sequence, label, fold)."""
    records: list[SequenceRecord] = []
    with Path(fold_manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = SequenceRecord()
            record.sample_id = row["sample_id"]
            record.sequence = row["sequence"]
            record.label = int(row["label"])
            record.length = len(row["sequence"])
            records.append(record)
    return records


def load_test_records(adapter_path: str | Path) -> list[SequenceRecord]:
    """Records from a *_prott5_structure_adapter.pt test payload (sample_ids, sequences, labels)."""
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    records: list[SequenceRecord] = []
    for sample_id, sequence, label in zip(payload["sample_ids"], payload["sequences"], payload["labels"]):
        record = SequenceRecord()
        record.sample_id = str(sample_id)
        record.sequence = str(sequence)
        record.label = int(label)
        record.length = len(str(sequence))
        records.append(record)
    return records


AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _point_mutate(sequence: str, rng, max_mutations: int) -> str:
    if not sequence:
        return sequence
    positions = rng.choice(len(sequence), size=int(rng.randint(1, max_mutations + 1)), replace=False)
    chars = list(sequence)
    for pos in positions:
        options = [c for c in AA_ALPHABET if c != chars[pos]]
        chars[pos] = options[int(rng.randint(0, len(options) - 1))]
    return "".join(chars)


def _identity_at_least(first: str, second: str, threshold: float) -> bool:
    la, lb = len(first), len(second)
    if min(la, lb) / max(la, lb) < threshold:
        return False
    if la == lb:
        same = sum(1 for a, b in zip(first, second) if a == b)
        return same / la >= threshold
    prev = list(range(lb + 1))
    for i, ca in enumerate(first, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(second, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return (min(la, lb) - prev[lb]) / min(la, lb) >= threshold


def mutation_augment_records(
    train_records: list[SequenceRecord],
    filter_records: list[SequenceRecord],
    config: dict[str, Any],
    rng,
) -> list[SequenceRecord]:
    """Point-mutate positive training sequences; drop any augmented sequence whose
    identity to a *filter* sequence (the fold validation set) reaches the threshold.

    This is the sequence-space analogue of the leakage-controlled Borderline-SMOTE
    protocol: synthetic samples are generated only from the training fold and are
    never allowed to become near-duplicates of validation sequences.  The parent
    sequence itself is intentionally NOT filtered against, since a 1-2 point
    mutation is by construction a near-copy of its parent.
    """
    cfg = config.get("mutation_aug") or {}
    if not cfg.get("enabled", False):
        return train_records
    rate = float(cfg.get("rate", 0.2))
    max_mutations = int(cfg.get("max_mutations", 2))
    threshold = float(cfg.get("identity_threshold", 0.8))

    filter_seqs = [record.sequence for record in filter_records]
    exact = set(filter_seqs)
    kmer_size = 3
    kmer_index: dict[str, list[int]] = {}
    for idx, seq in enumerate(filter_seqs):
        for k in {seq[i:i + kmer_size] for i in range(max(len(seq) - kmer_size + 1, 0))}:
            kmer_index.setdefault(k, []).append(idx)

    added: list[SequenceRecord] = []
    counter = 0
    for record in train_records:
        if record.label != 1 or not record.sequence:
            continue
        if rng.random() >= rate:
            continue
        mutated = _point_mutate(record.sequence, rng, max_mutations)
        if mutated in exact:
            continue
        candidates: set[int] = set()
        for i in range(len(mutated) - kmer_size + 1):
            candidates.update(kmer_index.get(mutated[i:i + kmer_size], ()))
        if any(_identity_at_least(mutated, filter_seqs[idx], threshold) for idx in candidates):
            continue
        new_record = SequenceRecord()
        new_record.sample_id = f"{record.sample_id}__mut{counter}"
        new_record.sequence = mutated
        new_record.label = 1
        new_record.length = len(mutated)
        added.append(new_record)
        counter += 1
    return train_records + added


class LoraMambaDataset(Dataset):
    def __init__(self, records: list[SequenceRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "sequence": record.sequence,
            "label": torch.tensor(float(record.label), dtype=torch.float32),
            "length": torch.tensor(int(record.length), dtype=torch.long),
        }


class LoraMambaCollator:
    def __init__(self, tokenizer: Any, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences = [item["sequence"] for item in batch]
        encoded = self.tokenizer(
            [" ".join(sequence) for sequence in sequences],
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": torch.stack([item["label"] for item in batch]),
            "length": torch.stack([item["length"] for item in batch]),
        }


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------
def build_model(plm: nn.Module, config: dict[str, Any], device: torch.device) -> LoraMamba1D:
    model = LoraMamba1D(
        plm,
        token_dim=int(config.get("token_dim", 1024)),
        d_model=int(config.get("d_model", 192)),
        d_state=int(config.get("d_state", 16)),
        d_conv=int(config.get("d_conv", 4)),
        expand=int(config.get("expand", 2)),
        n_layers=int(config.get("n_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        lora_rank=int(config.get("lora_rank", 0)),
        lora_alpha=float(config.get("lora_alpha", 16.0)),
        lora_dropout=float(config.get("lora_dropout", 0.1)),
    ).to(device)
    if config.get("grad_checkpoint", False):
        # The frozen embedding output has no grad. Reentrant checkpointing would
        # then discard the encoder graph, silently preventing LoRA updates.
        if "gradient_checkpointing_kwargs" in inspect.signature(plm.gradient_checkpointing_enable).parameters:
            plm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        else:
            # Older supported Transformers versions lack the kwargs API.
            plm.gradient_checkpointing_enable()
            plm.enable_input_require_grads()
    return model


def make_training_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Keep head and adapter learning rates independently configurable."""
    base_lr = float(config.get("learning_rate", 3e-4))
    groups = []
    for is_lora, lr in [(False, base_lr), (True, float(config.get("lora_learning_rate", base_lr)))]:
        parameters = [p for n, p in model.named_parameters()
                      if p.requires_grad and ((".lora_A" in n or ".lora_B" in n) == is_lora)]
        if parameters:
            groups.append({"params": parameters, "lr": lr, "name": "lora" if is_lora else "head"})
    return torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", 0.01)))


def check_lora_gradients(model: nn.Module) -> dict[str, int]:
    """Fail loudly when adapters exist but backward did not reach them."""
    adapters = [(n, p) for n, p in model.named_parameters()
                if p.requires_grad and (".lora_A" in n or ".lora_B" in n)]
    missing = [n for n, p in adapters if p.grad is None]
    if missing:
        raise RuntimeError("LoRA gradient disconnected: " + ", ".join(missing[:3]))
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("Frozen parameters unexpectedly received gradients.")
    return {"adapter_tensors": len(adapters), "grad_present": len(adapters) - len(missing)}


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score
    pred = (probabilities >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
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
    for threshold in candidates:
        bacc = balanced_accuracy_score(labels, (probabilities >= threshold).astype(int))
        if bacc > best_bacc:
            best_bacc = bacc
            best_t = threshold
    plateau = [
        threshold for threshold in candidates
        if balanced_accuracy_score(labels, (probabilities >= threshold).astype(int)) >= best_bacc - tolerance
    ]
    return float(max(plateau))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: _GradScaler | None,
) -> None:
    model.train()
    adapter_before = {n: p.detach().clone() for n, p in model.named_parameters()
                      if p.requires_grad and ".lora_B" in n}
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with _amp_autocast(use_amp):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["length"])
            loss = loss_fn(logits, batch["label"])
        if scaler is not None:
            scaler.scale(loss).backward()
            check_lora_gradients(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            check_lora_gradients(model)
            optimizer.step()
    if adapter_before:
        current = dict(model.named_parameters())
        if not any(not torch.equal(old, current[n].detach()) for n, old in adapter_before.items()):
            raise RuntimeError("LoRA B did not update during the epoch; inspect learning rate, AMP and gradients.")


@torch.no_grad()
def predict_probs(
    model: nn.Module,
    dataset: Dataset,
    collator: LoraMambaCollator,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    probabilities: list[np.ndarray] = []
    for batch in loader:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        with _amp_autocast(use_amp):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["length"])
        probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probabilities)


def _trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict restricted to trainable parameters (LoRA + head), excluding the frozen PLM."""
    keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name in keys}


def validate_public_checkpoint_payload(
    payload: dict[str, Any],
    reference_config: dict[str, Any],
    *,
    expected_seed: int | None = None,
    model: nn.Module | None = None,
    checkpoint_path: str | Path = "<checkpoint>",
) -> None:
    """Reject old, incomplete, or architecture-mismatched 1D checkpoints."""

    if not isinstance(payload, dict) or "model_state" not in payload or "config" not in payload:
        raise ValueError(f"{checkpoint_path}: not a LoRA-Mamba checkpoint payload.")
    checkpoint_config = payload["config"]
    experiment = checkpoint_config.get("experiment", {}).get("name")
    if experiment != PUBLIC_1D_EXPERIMENT:
        raise ValueError(
            f"{checkpoint_path}: experiment {experiment!r} is not the public 1D expert "
            f"{PUBLIC_1D_EXPERIMENT!r}."
        )
    for key in PUBLIC_1D_ARCHITECTURE_KEYS:
        if checkpoint_config.get(key) != reference_config.get(key):
            raise ValueError(
                f"{checkpoint_path}: checkpoint/config mismatch for {key}: "
                f"{checkpoint_config.get(key)!r} != {reference_config.get(key)!r}."
            )
    if checkpoint_config != reference_config:
        raise ValueError(
            f"{checkpoint_path}: checkpoint training configuration differs from the "
            "declared public 1D configuration."
        )
    seed = payload.get("seed")
    if expected_seed is not None and int(seed) != int(expected_seed):
        raise ValueError(
            f"{checkpoint_path}: checkpoint seed {seed!r} != expected seed {expected_seed}."
        )
    state = payload["model_state"]
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{checkpoint_path}: model_state is empty or invalid.")
    keys = set(state)
    if not any(".lora_A" in key for key in keys) or not any(".lora_B" in key for key in keys):
        raise ValueError(f"{checkpoint_path}: LoRA parameters are absent.")
    if not any("forward_scan" in key for key in keys) or not any("backward_scan" in key for key in keys):
        raise ValueError(f"{checkpoint_path}: bidirectional Mamba parameters are absent.")
    if model is not None:
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(expected - keys)
        extra = sorted(keys - expected)
        if missing_trainable or extra:
            raise ValueError(
                f"{checkpoint_path}: trainable-state contract mismatch; "
                f"missing={missing_trainable[:5]}, extra={extra[:5]}."
            )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def validate_no_lora_checkpoint_payload(
    payload: dict[str, Any], reference_config: dict[str, Any], *,
    model: nn.Module | None = None, checkpoint_path: str | Path = "<checkpoint>",
) -> None:
    """Validate new no-LoRA training artifacts without weakening legacy checks."""
    if not isinstance(payload, dict) or payload.get("config") != reference_config:
        raise ValueError(f"{checkpoint_path}: checkpoint/config mismatch.")
    if int(reference_config.get("lora_rank", 0)) != 0:
        raise ValueError(f"{checkpoint_path}: expected no-LoRA configuration.")
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state or any(".lora_" in name for name in state):
        raise ValueError(f"{checkpoint_path}: invalid no-LoRA checkpoint state.")
    if not any("forward_scan" in k for k in state) or not any("backward_scan" in k for k in state):
        raise ValueError(f"{checkpoint_path}: bidirectional Mamba parameters are absent.")
    if model is not None:
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError(f"{checkpoint_path}: trainable-state contract mismatch.")


def run_cv(
    records: list[SequenceRecord],
    fold_manifest: str | Path,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    tokenizer: Any,
    max_epochs: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    labels = np.asarray([record.label for record in records])
    sample_ids = [record.sample_id for record in records]
    count = len(records)
    oof = np.zeros(count, dtype=np.float64)
    fold_assignments = [0] * count
    timing: dict[str, float] = {}

    for fold in range(5):
        train_idx, val_idx = _load_fold_indices(fold_manifest, fold, count)
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        fold_rng = np.random.RandomState(seed * 1000 + fold)
        train_records = mutation_augment_records(train_records, val_records, config, fold_rng)
        n_pos = sum(1 for record in train_records if record.label == 1)
        n_neg = len(train_records) - n_pos
        plm = _load_plm(config, device)
        model = build_model(plm, config, device)
        pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = make_training_optimizer(model, config)
        collator = LoraMambaCollator(tokenizer, max_length=int(config.get("max_length", 128)))
        train_loader = DataLoader(
            LoraMambaDataset(train_records), batch_size=int(config.get("batch_size", 16)),
            shuffle=True, collate_fn=collator, drop_last=False,
        )
        use_amp = bool(config.get("fp16", True)) and device.type == "cuda"
        scaler = _make_scaler(use_amp)
        epochs = int(config.get("epochs", 10)) if max_epochs is None else max_epochs
        fold_start = time.time()
        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, scaler)
        timing[f"fold{fold}_train_seconds"] = float(time.time() - fold_start)
        val_prob = predict_probs(model, LoraMambaDataset(val_records), collator, device, int(config.get("batch_size", 16)), use_amp)
        oof[val_idx] = val_prob
        for index in val_idx:
            fold_assignments[index] = fold
        if output_dir is not None:
            checkpoint_path = Path(output_dir) / f"fold{fold}_seed{seed}.pt"
            torch.save({
                "artifact_type": ("ghxtox_no_lora_bimamba_1d_checkpoint" if int(config.get("lora_rank", 0)) == 0 else "ghxtox_lora_bimamba_1d_checkpoint"),
                "schema_version": 1,
                "model_state": _trainable_state_dict(model),
                "config": config,
                "fold": fold,
                "seed": seed,
            }, checkpoint_path)
        del plm, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    threshold = _best_threshold(labels, oof)
    metrics = _probability_metrics(labels, oof, threshold)
    return {
        "seed": seed,
        "config_name": str(config.get("experiment", {}).get("name", "lora_mamba_1d")),
        "threshold": threshold,
        "oof_metrics": metrics,
        "oof": oof.tolist(),
        "labels": labels.tolist(),
        "sample_ids": sample_ids,
        "folds": fold_assignments,
        "timing_seconds": timing,
    }


def _load_plm(config: dict[str, Any], device: torch.device) -> nn.Module:
    from transformers import T5EncoderModel
    model_path = str(config.get("plm", "Rostlab/prot_t5_xl_half_uniref50-enc"))
    plm = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    plm.eval()
    return plm.to(device)


def train_full(
    records: list[SequenceRecord],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    tokenizer: Any,
    output: str | Path,
    max_epochs: int | None = None,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_rng = np.random.RandomState(seed * 1000)
    records = mutation_augment_records(records, [], config, full_rng)
    plm = _load_plm(config, device)
    model = build_model(plm, config, device)
    n_pos = sum(1 for record in records if record.label == 1)
    n_neg = len(records) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1), device=device))
    optimizer = make_training_optimizer(model, config)
    collator = LoraMambaCollator(tokenizer, max_length=int(config.get("max_length", 128)))
    loader = DataLoader(
        LoraMambaDataset(records), batch_size=int(config.get("batch_size", 16)),
        shuffle=True, collate_fn=collator, drop_last=False,
    )
    use_amp = bool(config.get("fp16", True)) and device.type == "cuda"
    scaler = _make_scaler(use_amp)
    epochs = int(config.get("epochs", 10)) if max_epochs is None else max_epochs
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, loss_fn, device, use_amp, scaler)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "artifact_type": ("ghxtox_no_lora_bimamba_1d_checkpoint" if int(config.get("lora_rank", 0)) == 0 else "ghxtox_lora_bimamba_1d_checkpoint"),
        "schema_version": 1,
        "model_state": _trainable_state_dict(model),
        "config": config,
        "seed": seed,
    }, output)
    print("saved full checkpoint:", output)


def predict_csv(
    records: list[SequenceRecord],
    checkpoint: str | Path,
    config: dict[str, Any],
    device: torch.device,
    tokenizer: Any,
    output_csv: str | Path,
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validator = validate_no_lora_checkpoint_payload if int(config.get("lora_rank", 0)) == 0 else validate_public_checkpoint_payload
    validator(payload, config, checkpoint_path=checkpoint)
    plm = _load_plm(config, device)
    model = build_model(plm, config, device)
    validator(payload, config, model=model, checkpoint_path=checkpoint)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    expected_missing = set(model.state_dict()) - set(payload["model_state"])
    if unexpected or set(missing) != expected_missing:
        raise ValueError(
            f"{checkpoint}: unsafe partial load; missing={missing[:5]}, unexpected={unexpected[:5]}."
        )
    remove_inactive_lora(model.plm.encoder)
    collator = LoraMambaCollator(tokenizer, max_length=int(config.get("max_length", 128)))
    probabilities = predict_probs(model, LoraMambaDataset(records), collator, device, int(config.get("batch_size", 16)), bool(config.get("fp16", True)))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
        writer.writeheader()
        for record, probability in zip(records, probabilities):
            writer.writerow({
                "sample_id": record.sample_id,
                "sequence": record.sequence,
                "label": record.label,
                "toxicity_probability": f"{float(probability):.9g}",
            })
    print("saved predictions:", output_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen ProtT5 + BiMamba 1D expert; legacy LoRA checkpoint compatibility")
    sub = parser.add_subparsers(dest="command", required=True)
    p_cv = sub.add_parser("cv")
    p_cv.add_argument("--fold-manifest", required=True)
    p_cv.add_argument("--config", default="configs/no_lora_mamba_1d_autodl.json")
    p_cv.add_argument("--seed", type=int, default=42)
    p_cv.add_argument("--output-dir", required=True)
    p_cv.add_argument("--device", default="auto")
    p_cv.add_argument("--max-epochs", type=int, default=None)
    p_cv.add_argument("--max-samples", type=int, default=None, help="Pilot: only keep the first N records.")
    p_cv.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    p_full = sub.add_parser("full")
    p_full.add_argument("--fold-manifest", required=True)
    p_full.add_argument("--config", default="configs/no_lora_mamba_1d_autodl.json")
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--output", required=True)
    p_full.add_argument("--device", default="auto")
    p_full.add_argument("--max-epochs", type=int, default=None)
    p_full.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--adapter", required=True)
    p_pred.add_argument("--checkpoint", required=True)
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("--output-csv", required=True)
    p_pred.add_argument("--device", default="auto")
    p_pred.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = _resolve_device(args.device)
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if getattr(args, "batch_size", None):
        config["batch_size"] = args.batch_size
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(str(config.get("plm", "Rostlab/prot_t5_xl_half_uniref50-enc")), do_lower_case=False)

    if args.command == "cv":
        records = load_train_records(args.fold_manifest)
        if getattr(args, "max_samples", None):
            records = records[: args.max_samples]
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_cv(records, args.fold_manifest, config, args.seed, device, tokenizer, args.max_epochs, args.output_dir)
        _save_json(result, out_dir / f"oof_seed{args.seed}.json")
        print(json.dumps({"threshold": result["threshold"], "oof_metrics": result["oof_metrics"]}, indent=2))
    elif args.command == "full":
        records = load_train_records(args.fold_manifest)
        train_full(records, config, args.seed, device, tokenizer, args.output, args.max_epochs)
    elif args.command == "predict":
        records = load_test_records(args.adapter)
        predict_csv(records, args.checkpoint, config, device, tokenizer, args.output_csv)
    else:
        raise SystemExit("Unknown command.")


if __name__ == "__main__":
    main()
