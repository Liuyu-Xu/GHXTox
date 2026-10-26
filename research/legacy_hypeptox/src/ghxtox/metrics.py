"""Small metric helpers with an optional scikit-learn fast path."""

from __future__ import annotations

import math

import torch


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Return standard confidence-bin ECE for binary probabilities."""

    probs = probs.detach().cpu().float().view(-1)
    labels = labels.detach().cpu().float().view(-1)
    if probs.numel() == 0:
        return float("nan")
    n_bins = max(int(n_bins), 1)
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.tensor(0.0)
    for index in range(n_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == n_bins - 1:
            in_bin = (probs >= lower) & (probs <= upper)
        else:
            in_bin = (probs >= lower) & (probs < upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_probs = probs[in_bin]
        bin_labels = labels[in_bin]
        bin_preds = (bin_probs >= 0.5).float()
        accuracy = (bin_preds == bin_labels).float().mean()
        confidence = torch.maximum(bin_probs, 1.0 - bin_probs).mean()
        ece = ece + (count / probs.numel()) * (accuracy - confidence).abs()
    return float(ece)


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    labels = labels.detach().cpu().float()
    probs = torch.sigmoid(logits.detach().cpu().float())
    preds = (probs >= threshold).float()

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    total = max(tp + tn + fp + fn, 1.0)

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    balanced_acc = 0.5 * (recall + specificity)
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    metrics = {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "sensitivity": recall,
        "sn": recall,
        "recall": recall,
        "specificity": specificity,
        "sp": specificity,
        "f1": f1,
        "mcc": ((tp * tn) - (fp * fn)) / mcc_den,
        "brier": float(torch.mean((probs - labels) ** 2)),
        "ece_10": expected_calibration_error(probs, labels, n_bins=10),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = labels.numpy()
        y_score = probs.numpy()
        if len(set(y_true.tolist())) > 1:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    except Exception:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics
