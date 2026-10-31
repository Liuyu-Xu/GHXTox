"""Cross-fitted 1D-query/3D-evidence attention screen.

The adapter treats the promoted BioEmu-BMoE probability as an immutable
baseline.  It is active only on the frozen BioEmu boundary cohort.  A compact
1D query token attends to three structure-evidence tokens (CAMC-3D, standalone
BioEmu and BioEmu-BR3D), then emits a bounded logit residual.  Every outer fold
uses a disjoint validation fold for early stopping; a fold falls back to the
baseline when validation BCE does not improve.

This remains an exploratory second-level OOF architecture screen.  It does not
create the full-train assets required for previously unseen samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ghxtox.bioemu_boundary_moe import (
    _align_inputs,
    _decision_transitions,
    _logit,
    _probability,
    _read_csv,
    _sigmoid,
)
from ghxtox._legacy.plm_textcnn import _probability_metrics
from ghxtox.utils import save_json


@dataclass(frozen=True)
class AttentionConfig:
    name: str
    hidden_dim: int
    num_heads: int
    residual_bound: float
    delta_l1: float


CONFIGS = (
    AttentionConfig("compact_h1", 8, 1, 0.05, 0.02),
    AttentionConfig("compact_h2", 16, 2, 0.10, 0.03),
    AttentionConfig("conservative_h2", 16, 2, 0.05, 0.05),
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _standardize(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return tuple((value - mean) / scale for value in (train, *others))


def apply_attention_residual(
    baseline_probability: np.ndarray,
    raw_delta: np.ndarray,
    active_mask: np.ndarray,
    *,
    residual_bound: float,
    decision_threshold: float | None = None,
) -> np.ndarray:
    """Apply a bounded logit correction while preserving inactive rows exactly."""

    baseline = np.asarray(baseline_probability, dtype=np.float64)
    delta = np.asarray(raw_delta, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if not baseline.shape == delta.shape == active.shape:
        raise ValueError("Attention residual inputs must have identical shapes.")
    if not 0.0 <= float(residual_bound) <= 1.0:
        raise ValueError("residual_bound must lie in [0, 1].")
    baseline_logit = _logit(baseline)
    candidate_logit = (
        baseline_logit
        + active.astype(np.float64)
        * float(residual_bound)
        * np.tanh(delta)
    )
    if decision_threshold is not None:
        threshold = float(decision_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("decision_threshold must lie strictly in (0, 1).")
        threshold_logit = float(_logit(np.asarray([threshold]))[0])
        positive = baseline >= threshold
        candidate_logit[active & positive] = np.maximum(
            candidate_logit[active & positive], threshold_logit + 1e-7
        )
        candidate_logit[active & ~positive] = np.minimum(
            candidate_logit[active & ~positive], threshold_logit - 1e-7
        )
    probability = _sigmoid(candidate_logit)
    probability[~active] = baseline[~active]
    return probability


class CrossDimensionAttention(nn.Module):
    """Small 1D-query to three-token 3D cross-attention residual."""

    def __init__(
        self,
        query_dim: int,
        structure_dim: int,
        hidden_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden_dim), nn.GELU()
        )
        self.structure_projection = nn.Sequential(
            nn.LayerNorm(structure_dim),
            nn.Linear(structure_dim, hidden_dim),
            nn.GELU(),
        )
        self.structure_type = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        nn.init.normal_(self.structure_type, std=0.02)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, query: torch.Tensor, structure: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_hidden = self.query_projection(query).unsqueeze(1)
        structure_hidden = self.structure_projection(structure) + self.structure_type
        context, weights = self.attention(
            query_hidden,
            structure_hidden,
            structure_hidden,
            need_weights=True,
            average_attn_weights=True,
        )
        query_vector = query_hidden[:, 0]
        context_vector = context[:, 0]
        interaction = torch.cat(
            [
                query_vector,
                context_vector,
                query_vector * context_vector,
                torch.abs(query_vector - context_vector),
            ],
            dim=-1,
        )
        return self.output(interaction).squeeze(-1), weights[:, 0]


def _build_features(
    base_rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]],
    one_d: np.ndarray,
    baseline: np.ndarray,
    camc: np.ndarray,
    bioemu: np.ndarray,
    br3d: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_threshold = float(_logit(np.asarray([threshold]))[0])
    z_one_d = _logit(one_d)
    z_baseline = _logit(baseline)
    z_camc = _logit(camc)
    z_bioemu = _logit(bioemu)
    z_br3d = _logit(br3d)
    length = np.asarray([float(row["length"]) for row in phase_rows])
    mean_plddt = np.asarray([float(row["mean_plddt"]) for row in phase_rows])
    min_plddt = np.asarray([float(row["min_plddt"]) for row in phase_rows])
    low_plddt = np.asarray(
        [float(row["low_plddt_fraction"]) for row in phase_rows]
    )
    global_gate = np.asarray([float(row["global_gate"]) for row in phase_rows])
    benefit_gate = np.asarray(
        [float(row["bioemu_benefit_gate"]) for row in base_rows]
    )
    specialist = np.asarray(
        [float(row["boundary_one_d_probability"]) for row in base_rows]
    )
    consensus = np.asarray([float(row["consensus_active"]) for row in base_rows])
    query = np.column_stack(
        [
            z_one_d,
            _logit(specialist),
            z_baseline,
            np.abs(z_one_d - z_threshold),
            z_one_d - z_baseline,
            benefit_gate,
            np.log1p(length),
            consensus,
        ]
    )
    structure_logits = (z_camc, z_bioemu, z_br3d)
    structure_thresholds = (
        float(_logit(np.asarray([0.677819]))[0]),
        0.0,
        0.0,
    )
    tokens = []
    for logit_value, decision_logit in zip(
        structure_logits, structure_thresholds, strict=True
    ):
        tokens.append(
            np.column_stack(
                [
                    logit_value,
                    np.abs(logit_value - decision_logit),
                    logit_value - z_one_d,
                    mean_plddt,
                    min_plddt,
                    low_plddt,
                    global_gate,
                    np.log1p(length),
                ]
            )
        )
    structure = np.stack(tokens, axis=1)
    if not np.isfinite(query).all() or not np.isfinite(structure).all():
        raise ValueError("Cross-dimension attention features contain non-finite values.")
    return query.astype(np.float32), structure.astype(np.float32)


def _fit_outer_fold(
    query: np.ndarray,
    structure: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    folds: np.ndarray,
    boundary: np.ndarray,
    *,
    held_fold: int,
    config: AttentionConfig,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    validation_fold = (held_fold + 1) % 5
    train = boundary & (folds != held_fold) & (folds != validation_fold)
    validation = boundary & (folds == validation_fold)
    held = boundary & (folds == held_fold)
    if min(int(train.sum()), int(validation.sum()), int(held.sum())) == 0:
        raise ValueError(f"Fold {held_fold} has an empty boundary split.")

    q_train, q_validation, q_held = _standardize(
        query[train], query[validation], query[held]
    )
    s_train_flat = structure[train].reshape(-1, structure.shape[-1])
    s_validation_flat = structure[validation].reshape(-1, structure.shape[-1])
    s_held_flat = structure[held].reshape(-1, structure.shape[-1])
    s_train_flat, s_validation_flat, s_held_flat = _standardize(
        s_train_flat, s_validation_flat, s_held_flat
    )
    s_train = s_train_flat.reshape(structure[train].shape)
    s_validation = s_validation_flat.reshape(structure[validation].shape)
    s_held = s_held_flat.reshape(structure[held].shape)

    _set_seed(seed + held_fold)
    model = CrossDimensionAttention(
        query.shape[-1],
        structure.shape[-1],
        config.hidden_dim,
        config.num_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    tensors = {
        "q_train": torch.from_numpy(q_train),
        "s_train": torch.from_numpy(s_train),
        "y_train": torch.from_numpy(labels[train].astype(np.float32)),
        "z_train": torch.from_numpy(_logit(baseline[train]).astype(np.float32)),
        "q_validation": torch.from_numpy(q_validation),
        "s_validation": torch.from_numpy(s_validation),
        "y_validation": torch.from_numpy(labels[validation].astype(np.float32)),
        "z_validation": torch.from_numpy(
            _logit(baseline[validation]).astype(np.float32)
        ),
    }
    baseline_validation_bce = float(
        criterion(tensors["z_validation"], tensors["y_validation"]).item()
    )
    best_validation_bce = baseline_validation_bce
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        raw_delta, _ = model(tensors["q_train"], tensors["s_train"])
        bounded_delta = float(config.residual_bound) * torch.tanh(raw_delta)
        logits = tensors["z_train"] + bounded_delta
        loss = criterion(logits, tensors["y_train"])
        loss = loss + float(config.delta_l1) * bounded_delta.abs().mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            raw_validation, _ = model(
                tensors["q_validation"], tensors["s_validation"]
            )
            validation_logits = tensors["z_validation"] + float(
                config.residual_bound
            ) * torch.tanh(raw_validation)
            validation_bce = float(
                criterion(validation_logits, tensors["y_validation"]).item()
            )
        if validation_bce < best_validation_bce - 1e-6:
            best_validation_bce = validation_bce
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            break

    raw_held = np.zeros(int(held.sum()), dtype=np.float64)
    attention = np.full((int(held.sum()), 3), 1.0 / 3.0, dtype=np.float64)
    accepted = best_state is not None
    if accepted:
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            raw_tensor, attention_tensor = model(
                torch.from_numpy(q_held), torch.from_numpy(s_held)
            )
        raw_held = raw_tensor.numpy().astype(np.float64)
        attention = attention_tensor.numpy().astype(np.float64)
    report = {
        "held_fold": int(held_fold),
        "validation_fold": int(validation_fold),
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "held_rows": int(held.sum()),
        "best_epoch": int(best_epoch),
        "validation_adapter_accepted": bool(accepted),
        "baseline_validation_bce": baseline_validation_bce,
        "best_validation_bce": best_validation_bce,
    }
    return raw_held, attention, report


def _screen_config(
    query: np.ndarray,
    structure: np.ndarray,
    labels: np.ndarray,
    baseline: np.ndarray,
    folds: np.ndarray,
    boundary: np.ndarray,
    *,
    config: AttentionConfig,
    threshold: float,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw_delta = np.zeros(len(labels), dtype=np.float64)
    attention = np.zeros((len(labels), 3), dtype=np.float64)
    reports = []
    for fold in range(5):
        held_mask = boundary & (folds == fold)
        fold_delta, fold_attention, report = _fit_outer_fold(
            query,
            structure,
            labels,
            baseline,
            folds,
            boundary,
            held_fold=fold,
            config=config,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
        )
        raw_delta[held_mask] = fold_delta
        attention[held_mask] = fold_attention
        reports.append(report)
    unprotected_candidate = apply_attention_residual(
        baseline,
        raw_delta,
        boundary,
        residual_bound=config.residual_bound,
    )
    candidate = apply_attention_residual(
        baseline,
        raw_delta,
        boundary,
        residual_bound=config.residual_bound,
        decision_threshold=threshold,
    )
    metrics = _probability_metrics(labels, candidate, threshold)
    baseline_metrics = _probability_metrics(labels, baseline, threshold)
    fold_non_decreases = 0
    for report in reports:
        held = folds == int(report["held_fold"])
        base_fold = _probability_metrics(labels[held], baseline[held], threshold)
        candidate_fold = _probability_metrics(labels[held], candidate[held], threshold)
        report["baseline"] = base_fold
        report["candidate"] = candidate_fold
        fold_non_decreases += int(candidate_fold["mcc"] >= base_fold["mcc"])
    summary = {
        "config": {
            "name": config.name,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "residual_bound": config.residual_bound,
            "delta_l1": config.delta_l1,
        },
        "metrics": metrics,
        "unprotected_diagnostic": {
            "metrics": _probability_metrics(
                labels, unprotected_candidate, threshold
            ),
            "decision_transitions": _decision_transitions(
                labels, baseline, unprotected_candidate, threshold
            ),
        },
        "delta": {
            key: float(metrics[key] - baseline_metrics[key])
            for key in (
                "balanced_accuracy",
                "f1",
                "mcc",
                "auroc",
                "auprc",
                "brier",
                "ece_10",
            )
        },
        "decision_transitions": _decision_transitions(
            labels, baseline, candidate, threshold
        ),
        "fold_mcc_non_decrease_count": int(fold_non_decreases),
        "accepted_fold_count": int(
            sum(report["validation_adapter_accepted"] for report in reports)
        ),
        "mean_boundary_attention": {
            "camc_3d": float(attention[boundary, 0].mean()),
            "bioemu_standalone_3d": float(attention[boundary, 1].mean()),
            "bioemu_br3d": float(attention[boundary, 2].mean()),
        },
        "folds": reports,
    }
    return candidate, attention, summary


def run_screen(
    *,
    one_d_path: str | Path,
    current_default_path: str | Path,
    phase33_path: str | Path,
    output_dir: str | Path,
    threshold: float = 0.6,
    seed: int = 20260802,
    max_epochs: int = 200,
    patience: int = 25,
) -> dict[str, Any]:
    one_d_rows = _read_csv(one_d_path)
    base_rows = _read_csv(current_default_path)
    phase_rows = _read_csv(phase33_path)
    aligned = _align_inputs(one_d_rows, base_rows, phase_rows)
    labels = np.asarray(aligned["labels"])
    folds = np.asarray(aligned["folds"])
    boundary = np.asarray(aligned["boundary"])
    one_d = _probability(one_d_rows, "toxicity_probability")
    baseline = _probability(base_rows, "toxicity_probability")
    camc = _probability(phase_rows, "phase33_reference_probability")
    br3d = _probability(phase_rows, "phase33_candidate_probability")
    bioemu = np.full(len(labels), 0.5, dtype=np.float64)
    for index, row in enumerate(phase_rows):
        if boundary[index]:
            bioemu[index] = float(row["phase33_standalone_3d_probability"])
    query, structure = _build_features(
        base_rows,
        phase_rows,
        one_d,
        baseline,
        camc,
        bioemu,
        br3d,
        threshold=threshold,
    )
    baseline_metrics = _probability_metrics(labels, baseline, threshold)
    screens = []
    candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for config in CONFIGS:
        candidate, attention, screen = _screen_config(
            query,
            structure,
            labels,
            baseline,
            folds,
            boundary,
            config=config,
            threshold=threshold,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
        )
        screens.append(screen)
        candidates[config.name] = (candidate, attention)

    selected = max(
        screens,
        key=lambda item: (
            item["metrics"]["mcc"],
            item["metrics"]["balanced_accuracy"],
            item["metrics"]["auprc"],
            -item["metrics"]["brier"],
        ),
    )
    selected_probability, selected_attention = candidates[selected["config"]["name"]]
    transitions = selected["decision_transitions"]
    gates = {
        "mcc_not_lower": selected["metrics"]["mcc"]
        >= baseline_metrics["mcc"],
        "balanced_accuracy_not_lower": selected["metrics"]["balanced_accuracy"]
        >= baseline_metrics["balanced_accuracy"],
        "f1_not_lower": selected["metrics"]["f1"] >= baseline_metrics["f1"],
        "auprc_not_lower": selected["metrics"]["auprc"]
        >= baseline_metrics["auprc"],
        "brier_not_higher": selected["metrics"]["brier"]
        <= baseline_metrics["brier"],
        "at_least_three_fold_mcc_non_decreases": selected[
            "fold_mcc_non_decrease_count"
        ]
        >= 3,
        "nonnegative_net_decision_corrections": transitions["net"] >= 0,
        "continuous_probability_metric_improved": (
            selected["metrics"]["auprc"] > baseline_metrics["auprc"]
            or selected["metrics"]["brier"] < baseline_metrics["brier"]
        ),
    }
    gates["all_conditions_passed"] = all(gates.values())
    decision = (
        "promote_cross_dimension_attention_default"
        if gates["all_conditions_passed"]
        else "retain_bioemu_bmoe_default"
    )
    summary = {
        "protocol": {
            "public_model_name": "BioEmu cross-dimension attention adapter (BioEmu-XAttn)",
            "baseline": "BioEmu boundary mixture-of-experts adapter (BioEmu-BMoE)",
            "architecture": "1D query token attends to CAMC-3D, standalone BioEmu and BioEmu-BR3D evidence tokens",
            "scope": "frozen 600-row BioEmu boundary cohort only; exact fallback elsewhere",
            "outer_protocol": "held canonical fold with a disjoint rotating validation fold",
            "validation_fallback": "zero residual unless validation BCE improves",
            "decision_protection": "clip residual at the frozen 0.60-side boundary so the BioEmu-BMoE class decision cannot be reversed",
            "test1_or_test2_labels_read": False,
            "strict_nested_upstream_claim_authorized": False,
            "production_unseen_inference_ready": False,
            "seed": int(seed),
            "threshold": float(threshold),
        },
        "coverage": {
            "rows": int(len(labels)),
            "boundary_rows": int(boundary.sum()),
            "exact_fallback_rows": int((~boundary).sum()),
        },
        "baseline": baseline_metrics,
        "screens": screens,
        "selected": selected,
        "retention_gate": gates,
        "decision": decision,
        "interpretation": (
            "This is a second-level OOF architecture screen. Promotion applies only to the "
            "project research default and does not imply an independent generalisation gain."
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "source_index",
            "sample_id",
            "fold",
            "label",
            "sequence",
            "boundary_member",
            "bioemu_bmoe_probability",
            "camc_attention",
            "bioemu_attention",
            "bioemu_br3d_attention",
            "toxicity_probability",
            "prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(labels)):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": aligned["sample_ids"][index],
                    "fold": int(folds[index]),
                    "label": int(labels[index]),
                    "sequence": aligned["sequences"][index],
                    "boundary_member": int(boundary[index]),
                    "bioemu_bmoe_probability": f"{baseline[index]:.10g}",
                    "camc_attention": f"{selected_attention[index, 0]:.10g}",
                    "bioemu_attention": f"{selected_attention[index, 1]:.10g}",
                    "bioemu_br3d_attention": f"{selected_attention[index, 2]:.10g}",
                    "toxicity_probability": f"{selected_probability[index]:.10g}",
                    "prediction": int(selected_probability[index] >= threshold),
                }
            )
    save_json(summary, output / "summary.json")
    save_json(
        {
            "status": "complete",
            "decision": decision,
            "promotion_authorized": bool(gates["all_conditions_passed"]),
            "production_unseen_inference_ready": False,
            "retention_gate": gates,
            "selected_config": selected["config"],
            "metrics": selected["metrics"],
        },
        output / "decision.json",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d", required=True)
    parser.add_argument("--current-default", required=True)
    parser.add_argument(
        "--bioemu-boundary-3d",
        "--phase33",
        dest="phase33",
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_screen(
        one_d_path=args.one_d,
        current_default_path=args.current_default,
        phase33_path=args.phase33,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print(json.dumps(summary["selected"], indent=2))
    print(json.dumps(summary["retention_gate"], indent=2))


if __name__ == "__main__":
    main()
