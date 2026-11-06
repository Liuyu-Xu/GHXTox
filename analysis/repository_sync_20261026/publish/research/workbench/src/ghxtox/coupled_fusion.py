"""Leakage-controlled dynamic coupling for 1D and 3D toxicity experts.

This module keeps the 1D expert as the safe anchor and learns a small
sample-wise gate that decides how much 3D logit evidence may enter.  Screening
is cross-fitted by fold, so held-out rows are never used to fit either the
coupling parameters or the decision threshold used for their predictions.
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

from ghxtox.metrics import binary_metrics
from ghxtox.utils import save_json, set_seed


EPSILON = 1e-6


def _read_prediction_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    required = {"sequence", "label"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Prediction CSV {path} is missing columns: {sorted(missing)}")
    return rows


def _read_folds(path: str | Path, rows: list[dict[str, str]]) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    by_index = {int(row["source_index"]): row for row in manifest_rows}
    folds: list[int] = []
    for position, row in enumerate(rows):
        source_index = int(row.get("source_index", position))
        if source_index not in by_index:
            raise ValueError(f"Fold manifest is missing source_index {source_index}.")
        fold_row = by_index[source_index]
        if row["sequence"] != fold_row["sequence"] or int(row["label"]) != int(
            fold_row["label"]
        ):
            raise ValueError(f"Fold manifest mismatch at source_index {source_index}.")
        folds.append(int(fold_row["fold"]))
    result = np.asarray(folds, dtype=np.int64)
    if len(np.unique(result)) < 2:
        raise ValueError("Coupled fusion screening requires at least two folds.")
    return result


def _probabilities(rows: list[dict[str, str]], column: str, path: str | Path) -> np.ndarray:
    if column not in rows[0]:
        raise ValueError(f"Prediction CSV {path} has no probability column {column!r}.")
    probabilities = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError(f"Column {column!r} in {path} must contain probabilities.")
    return probabilities


def _logit(probabilities: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(probabilities):
        return torch.logit(probabilities.float().clamp(EPSILON, 1.0 - EPSILON))
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def _optional_float(row: dict[str, str], names: tuple[str, ...], default: float) -> float:
    for name in names:
        value = row.get(name)
        if value not in {None, ""}:
            return float(value)
    return float(default)


def _aligned_inputs(
    one_d_path: str | Path,
    three_d_path: str | Path,
    *,
    fold_manifest: str | Path | None,
    one_d_column: str,
    three_d_column: str,
) -> dict[str, Any]:
    one_d_rows = _read_prediction_csv(one_d_path)
    three_d_rows = _read_prediction_csv(three_d_path)
    if len(one_d_rows) != len(three_d_rows):
        raise ValueError(
            f"Prediction row counts differ: {len(one_d_rows)} versus {len(three_d_rows)}."
        )
    for index, (left, right) in enumerate(zip(one_d_rows, three_d_rows)):
        if left["sequence"] != right["sequence"] or int(left["label"]) != int(right["label"]):
            raise ValueError(f"1D/3D prediction alignment mismatch at row {index}.")
        if "source_index" in left and "source_index" in right:
            if int(left["source_index"]) != int(right["source_index"]):
                raise ValueError(f"1D/3D source_index mismatch at row {index}.")

    labels = np.asarray([int(row["label"]) for row in one_d_rows], dtype=np.int64)
    one_d = _probabilities(one_d_rows, one_d_column, one_d_path)
    three_d = _probabilities(three_d_rows, three_d_column, three_d_path)
    folds = None
    if fold_manifest is not None:
        folds = _read_folds(fold_manifest, one_d_rows)
    elif "fold" in one_d_rows[0]:
        folds = np.asarray([int(row["fold"]) for row in one_d_rows], dtype=np.int64)
    else:
        raise ValueError("Provide --fold-manifest or prediction CSVs with a fold column.")

    z1 = _logit(one_d)
    z3 = _logit(three_d)
    gate = np.asarray(
        [
            _optional_float(
                row,
                ("global_3d_gate", "mean_global_gate", "quality_gate"),
                default=0.0,
            )
            for row in three_d_rows
        ],
        dtype=np.float64,
    )
    fallback = np.asarray(
        [
            _optional_float(row, ("fallback_used", "structure_fallback_used"), default=0.0)
            for row in three_d_rows
        ],
        dtype=np.float64,
    )
    features = np.stack(
        [
            z1,
            z3,
            z3 - z1,
            np.abs(z3 - z1),
            one_d - three_d,
            np.abs(one_d - three_d),
            np.abs(one_d - 0.5),
            np.abs(three_d - 0.5),
            _entropy(one_d),
            _entropy(three_d),
            gate,
            fallback,
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "rows": one_d_rows,
        "labels": labels,
        "folds": folds,
        "one_d": one_d,
        "three_d": three_d,
        "features": features,
        "feature_names": [
            "one_d_logit",
            "three_d_logit",
            "three_d_minus_one_d_logit",
            "abs_logit_disagreement",
            "one_d_minus_three_d_probability",
            "abs_probability_disagreement",
            "one_d_margin",
            "three_d_margin",
            "one_d_entropy",
            "three_d_entropy",
            "structure_quality_gate",
            "fallback_used",
        ],
    }


class DynamicCoupler(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        dropout: float,
        max_three_d_weight: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.max_three_d_weight = float(max_three_d_weight)
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.gate_head = nn.Linear(hidden_dim // 2, 1)
        self.residual_head = nn.Linear(hidden_dim // 2, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        one_d_probability: torch.Tensor,
        three_d_probability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = self.net(features)
        gate = self.max_three_d_weight * torch.sigmoid(self.gate_head(h)).squeeze(-1)
        residual = self.residual_scale * torch.tanh(self.residual_head(h)).squeeze(-1)
        one_d_logit = _logit(one_d_probability)
        three_d_logit = _logit(three_d_probability)
        logits = one_d_logit + gate * (three_d_logit - one_d_logit) + residual
        return {"logits": logits, "gate": gate, "residual": residual}


def _standardize(train: np.ndarray, apply: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    stats = {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}
    return (train - mean) / std, (apply - mean) / std, stats


def _threshold_search(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: str,
    threshold_step: float,
    min_sn: float | None,
    min_sp: float | None,
) -> tuple[float, dict[str, float], bool]:
    labels_t = torch.tensor(labels, dtype=torch.float32)
    logits_t = torch.tensor(_logit(probabilities), dtype=torch.float32)
    best: tuple[bool, tuple[float, ...], float, dict[str, float]] | None = None
    for threshold in np.arange(threshold_step, 1.0, threshold_step):
        metrics = binary_metrics(logits_t, labels_t, threshold=float(threshold))
        feasible = (min_sn is None or metrics["sn"] >= min_sn) and (
            min_sp is None or metrics["sp"] >= min_sp
        )
        score = (
            float(metrics[metric]),
            float(metrics["mcc"]),
            float(metrics["balanced_accuracy"]),
            -abs(float(threshold) - 0.5),
        )
        candidate = (feasible, score, float(threshold), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("Threshold search produced no candidates.")
    return best[2], best[3], best[0]


def _fit_model(
    features: np.ndarray,
    one_d: np.ndarray,
    three_d: np.ndarray,
    labels: np.ndarray,
    *,
    hidden_dim: int,
    dropout: float,
    max_three_d_weight: float,
    residual_scale: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    gate_l1: float,
    residual_l1: float,
    seed: int,
) -> tuple[DynamicCoupler, dict[str, Any]]:
    set_seed(seed)
    device = torch.device("cpu")
    model = DynamicCoupler(
        feature_dim=features.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_three_d_weight=max_three_d_weight,
        residual_scale=residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    x = torch.tensor(features, dtype=torch.float32, device=device)
    p1 = torch.tensor(one_d, dtype=torch.float32, device=device)
    p3 = torch.tensor(three_d, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    history = []
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, p1, p3)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output["logits"], y)
        if gate_l1 > 0:
            loss = loss + float(gate_l1) * output["gate"].mean()
        if residual_l1 > 0:
            loss = loss + float(residual_l1) * output["residual"].abs().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    return model, {"history": history}


def _predict_model(
    model: DynamicCoupler,
    features: np.ndarray,
    one_d: np.ndarray,
    three_d: np.ndarray,
) -> dict[str, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(one_d, dtype=torch.float32),
            torch.tensor(three_d, dtype=torch.float32),
        )
    logits = output["logits"].detach().cpu()
    return {
        "probabilities": torch.sigmoid(logits).numpy(),
        "gates": output["gate"].detach().cpu().numpy(),
        "residuals": output["residual"].detach().cpu().numpy(),
    }


def screen_coupled_fusion(
    *,
    one_d_predictions: str | Path,
    three_d_predictions: str | Path,
    output_dir: str | Path,
    fold_manifest: str | Path | None = None,
    one_d_column: str = "toxicity_probability",
    three_d_column: str = "toxicity_probability",
    hidden_dim: int = 32,
    dropout: float = 0.05,
    max_three_d_weight: float = 0.45,
    residual_scale: float = 0.35,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-3,
    epochs: int = 800,
    gate_l1: float = 0.01,
    residual_l1: float = 0.02,
    threshold_metric: str = "mcc",
    threshold_step: float = 0.005,
    min_sn: float | None = None,
    min_sp: float | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    data = _aligned_inputs(
        one_d_predictions,
        three_d_predictions,
        fold_manifest=fold_manifest,
        one_d_column=one_d_column,
        three_d_column=three_d_column,
    )
    labels = data["labels"]
    folds = data["folds"]
    one_d = data["one_d"]
    three_d = data["three_d"]
    features = data["features"]
    probabilities = np.zeros_like(one_d)
    gates = np.zeros_like(one_d)
    residuals = np.zeros_like(one_d)
    thresholds = np.zeros_like(one_d)
    fold_results = []
    for fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != fold
        test_mask = folds == fold
        train_features, test_features, stats = _standardize(
            features[train_mask], features[test_mask]
        )
        model, fit_summary = _fit_model(
            train_features,
            one_d[train_mask],
            three_d[train_mask],
            labels[train_mask],
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_three_d_weight=max_three_d_weight,
            residual_scale=residual_scale,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            gate_l1=gate_l1,
            residual_l1=residual_l1,
            seed=seed + int(fold),
        )
        train_pred = _predict_model(model, train_features, one_d[train_mask], three_d[train_mask])
        threshold, train_metrics, constraints_ok = _threshold_search(
            labels[train_mask],
            train_pred["probabilities"],
            metric=threshold_metric,
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        test_pred = _predict_model(model, test_features, one_d[test_mask], three_d[test_mask])
        probabilities[test_mask] = test_pred["probabilities"]
        gates[test_mask] = test_pred["gates"]
        residuals[test_mask] = test_pred["residuals"]
        thresholds[test_mask] = threshold
        test_metrics = binary_metrics(
            torch.tensor(_logit(test_pred["probabilities"]), dtype=torch.float32),
            torch.tensor(labels[test_mask], dtype=torch.float32),
            threshold=threshold,
        )
        fold_results.append(
            {
                "fold": int(fold),
                "num_train": int(train_mask.sum()),
                "num_test": int(test_mask.sum()),
                "threshold": float(threshold),
                "constraints_satisfied_on_train": bool(constraints_ok),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_standardization": stats,
                "fit": fit_summary,
            }
        )

    predictions = probabilities >= thresholds
    coupled_metrics = _metrics_from_predictions(labels, probabilities, predictions)
    one_d_threshold, one_d_metrics, _ = _threshold_search(
        labels, one_d, metric=threshold_metric, threshold_step=threshold_step, min_sn=min_sn, min_sp=min_sp
    )
    three_d_threshold, three_d_metrics, _ = _threshold_search(
        labels, three_d, metric=threshold_metric, threshold_step=threshold_step, min_sn=min_sn, min_sp=min_sp
    )

    full_features, _, full_stats = _standardize(features, features)
    full_model, fit_summary = _fit_model(
        full_features,
        one_d,
        three_d,
        labels,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_three_d_weight=max_three_d_weight,
        residual_scale=residual_scale,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        gate_l1=gate_l1,
        residual_l1=residual_l1,
        seed=seed,
    )
    full_pred = _predict_model(full_model, full_features, one_d, three_d)
    frozen_threshold, frozen_metrics, frozen_constraints = _threshold_search(
        labels,
        full_pred["probabilities"],
        metric=threshold_metric,
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        data["rows"],
        one_d,
        three_d,
        probabilities,
        gates,
        residuals,
        thresholds,
        output_root / "crossfit_predictions.csv",
    )
    torch.save(
        {
            "model_state": full_model.state_dict(),
            "model_config": {
                "feature_dim": int(features.shape[1]),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "max_three_d_weight": float(max_three_d_weight),
                "residual_scale": float(residual_scale),
            },
            "standardization": full_stats,
            "feature_names": data["feature_names"],
            "threshold": float(frozen_threshold),
            "threshold_metric": threshold_metric,
        },
        output_root / "frozen_coupler.pt",
    )
    summary = {
        "protocol": {
            "model": "1D-anchored dynamic logit coupling with learned 3D gate and bounded residual",
            "one_d_predictions": str(one_d_predictions),
            "three_d_predictions": str(three_d_predictions),
            "fold_manifest": None if fold_manifest is None else str(fold_manifest),
            "one_d_column": one_d_column,
            "three_d_column": three_d_column,
            "test_labels_used_for_selection": False,
            "threshold_metric": threshold_metric,
            "threshold_step": float(threshold_step),
            "min_sn": min_sn,
            "min_sp": min_sp,
        },
        "model_config": {
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "max_three_d_weight": float(max_three_d_weight),
            "residual_scale": float(residual_scale),
            "gate_l1": float(gate_l1),
            "residual_l1": float(residual_l1),
            "epochs": int(epochs),
        },
        "crossfit_metrics": coupled_metrics,
        "one_d_oof": {"threshold": float(one_d_threshold), "metrics": one_d_metrics},
        "three_d_oof": {"threshold": float(three_d_threshold), "metrics": three_d_metrics},
        "delta_coupled_minus_one_d": {
            key: float(coupled_metrics[key] - one_d_metrics[key])
            for key in ("accuracy", "balanced_accuracy", "mcc", "auroc", "auprc")
        },
        "delta_coupled_minus_three_d": {
            key: float(coupled_metrics[key] - three_d_metrics[key])
            for key in ("accuracy", "balanced_accuracy", "mcc", "auroc", "auprc")
        },
        "gate": {
            "mean": float(gates.mean()),
            "std": float(gates.std()),
            "min": float(gates.min()),
            "max": float(gates.max()),
        },
        "residual": {
            "mean": float(residuals.mean()),
            "mean_absolute": float(np.abs(residuals).mean()),
            "max_absolute": float(np.abs(residuals).max()),
        },
        "fold_results": fold_results,
        "frozen": {
            "checkpoint": str(output_root / "frozen_coupler.pt"),
            "threshold": float(frozen_threshold),
            "metrics_on_all_oof_rows": frozen_metrics,
            "constraints_satisfied": bool(frozen_constraints),
            "fit": fit_summary,
            "warning": "Frozen on all OOF rows only for future untouched application; crossfit metrics control retention.",
        },
    }
    save_json(summary, output_root / "summary.json")
    print(json.dumps(summary["crossfit_metrics"], indent=2))
    return summary


def _metrics_from_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    threshold = 0.5
    logits = torch.tensor(_logit(probabilities), dtype=torch.float32)
    metrics = binary_metrics(
        logits, torch.tensor(labels, dtype=torch.float32), threshold=threshold
    )
    labels_bool = labels.astype(bool)
    pred_bool = predictions.astype(bool)
    tp = float(np.logical_and(pred_bool, labels_bool).sum())
    tn = float(np.logical_and(~pred_bool, ~labels_bool).sum())
    fp = float(np.logical_and(pred_bool, ~labels_bool).sum())
    fn = float(np.logical_and(~pred_bool, labels_bool).sum())
    denominator = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    metrics.update(
        {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "precision": precision,
            "sensitivity": recall,
            "sn": recall,
            "recall": recall,
            "specificity": specificity,
            "sp": specificity,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "mcc": (tp * tn - fp * fn) / denominator,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    )
    return metrics


def _write_predictions(
    rows: list[dict[str, str]],
    one_d: np.ndarray,
    three_d: np.ndarray,
    coupled: np.ndarray,
    gates: np.ndarray,
    residuals: np.ndarray,
    thresholds: np.ndarray,
    output: Path,
) -> None:
    fieldnames = [
        "source_index",
        "sample_id",
        "sequence",
        "label",
        "one_d_probability",
        "three_d_probability",
        "toxicity_probability",
        "three_d_gate",
        "coupling_residual",
        "decision_threshold",
        "prediction",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": row.get("source_index", index),
                    "sample_id": row.get("sample_id", f"sample_{index}"),
                    "sequence": row["sequence"],
                    "label": row["label"],
                    "one_d_probability": f"{float(one_d[index]):.10g}",
                    "three_d_probability": f"{float(three_d[index]):.10g}",
                    "toxicity_probability": f"{float(coupled[index]):.10g}",
                    "three_d_gate": f"{float(gates[index]):.10g}",
                    "coupling_residual": f"{float(residuals[index]):.10g}",
                    "decision_threshold": f"{float(thresholds[index]):.10g}",
                    "prediction": int(coupled[index] >= thresholds[index]),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-d-predictions", required=True)
    parser.add_argument("--three-d-predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold-manifest")
    parser.add_argument("--one-d-column", default="toxicity_probability")
    parser.add_argument("--three-d-column", default="toxicity_probability")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-three-d-weight", type=float, default=0.45)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--gate-l1", type=float, default=0.01)
    parser.add_argument("--residual-l1", type=float, default=0.02)
    parser.add_argument("--threshold-metric", default="mcc")
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--min-sn", type=float)
    parser.add_argument("--min-sp", type=float)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    screen_coupled_fusion(
        one_d_predictions=args.one_d_predictions,
        three_d_predictions=args.three_d_predictions,
        output_dir=args.output_dir,
        fold_manifest=args.fold_manifest,
        one_d_column=args.one_d_column,
        three_d_column=args.three_d_column,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_three_d_weight=args.max_three_d_weight,
        residual_scale=args.residual_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        gate_l1=args.gate_l1,
        residual_l1=args.residual_l1,
        threshold_metric=args.threshold_metric,
        threshold_step=args.threshold_step,
        min_sn=args.min_sn,
        min_sp=args.min_sp,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
