"""Inference for the explicit ProtT5 sequence ablation.

The public default is the 3D GHXTox-3D-Chemical-Adapter-Residual route in
``ghxtox.hybrid_default``. This module intentionally exposes only the
sequence-only reference, which is useful for ablation and explicit fallback.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.plm_textcnn import PooledPLMTextCNN, _predict, _probability_metrics
from ghxtox.utils import load_json, resolve_device, save_json


DEFAULT_CONFIG = "configs/default_prott5.json"


def _load_payload(features_path: str | Path) -> tuple[dict[str, Any], np.ndarray, list[str], list[str], np.ndarray | None]:
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    if "features" not in payload:
        raise ValueError(f"Feature cache has no 'features' field: {features_path}")
    features = payload["features"].float().numpy()
    sequences = [str(value) for value in payload.get("sequences", [])]
    if len(sequences) != len(features):
        raise ValueError("Feature cache sequences and features have different lengths.")
    sample_ids = [str(value) for value in payload.get("sample_ids", range(len(features)))]
    if len(sample_ids) != len(features):
        raise ValueError("Feature cache sample_ids and features have different lengths.")
    labels = payload.get("labels")
    if labels is not None:
        labels = labels.long().numpy()
        if len(labels) != len(features):
            raise ValueError("Feature cache labels and features have different lengths.")
    return payload, features, sample_ids, sequences, labels


def predict_default(
    config_path: str | Path,
    features_path: str | Path,
    output_path: str | Path,
    *,
    metrics_path: str | Path | None,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    config = load_json(config_path)
    payload, features, sample_ids, sequences, labels = _load_payload(features_path)
    device = resolve_device(device_name)
    threshold = float(config["default_threshold"])
    member_probabilities: list[np.ndarray] = []
    checkpoints: list[str] = []
    for checkpoint_dir in config["checkpoint_dirs"]:
        for fold in range(5):
            checkpoint_path = Path(checkpoint_dir) / f"fold{fold}_best_model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing ProtT5 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_config = dict(checkpoint["model"])
            model_config["window_sizes"] = tuple(model_config["window_sizes"])
            model = PooledPLMTextCNN(**model_config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            member_probabilities.append(_predict(model, features, batch_size, device))
            checkpoints.append(str(checkpoint_path))
    if len(member_probabilities) != 15:
        raise ValueError(f"Expected 15 ProtT5 checkpoints, found {len(member_probabilities)}.")
    matrix = np.stack(member_probabilities, axis=0)
    probabilities = matrix.mean(axis=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "sample_id",
        "sequence",
        "label",
        "toxicity_probability",
        "prediction",
        "decision_threshold",
    ] + [f"member{index}_probability" for index in range(matrix.shape[0])]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            row: dict[str, Any] = {
                "source_index": index,
                "sample_id": sample_ids[index],
                "sequence": sequences[index],
                "label": "" if labels is None else int(labels[index]),
                "toxicity_probability": f"{float(probability):.10g}",
                "prediction": int(probability >= threshold),
                "decision_threshold": f"{threshold:.10g}",
            }
            row.update(
                {
                    f"member{member}_probability": f"{float(matrix[member, index]):.10g}"
                    for member in range(matrix.shape[0])
                }
            )
            writer.writerow(row)
    summary: dict[str, Any] = {
        "model": config["public_model_name"],
        "protocol": {
            "config": str(config_path),
            "features": str(features_path),
            "checkpoints": checkpoints,
            "ensemble": config["ensemble"],
            "threshold": threshold,
            "threshold_selected_from_test_labels": False,
            "feature_metadata": payload.get("metadata", {}),
        },
    }
    if labels is not None:
        summary["metrics"] = _probability_metrics(labels, probabilities, threshold=threshold)
    if metrics_path is None:
        metrics_path = output.with_suffix(".metrics.json")
    save_json(summary, metrics_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict with the ProtT5 performance default.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = predict_default(
        args.config,
        args.features,
        args.output,
        metrics_path=args.metrics,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    if "metrics" in summary:
        print(summary["metrics"])


if __name__ == "__main__":
    main()
