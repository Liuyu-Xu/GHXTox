"""Evaluate a frozen ProtTox-TSK checkpoint on a separately cached held-out set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, file_sha256, load_checkpoint, write_manifest
from ghxtox.prottox_tsk_policy import apply_policy, load_policy
from ghxtox.train_prottox_tsk import _metrics


def evaluate_cache(cache_path: str | Path, checkpoint_path: str | Path, output_dir: str | Path, *, split_name: str, expected_records: int, threshold: float = 0.5, policy_path: str | Path | None = None, policy_mode: str = "balanced", device: str = "cuda") -> dict[str, Any]:
    cache_file, checkpoint_file = Path(cache_path), Path(checkpoint_path)
    cache = torch.load(cache_file, map_location="cpu", weights_only=False)
    if cache.get("format_version") != CACHE_FORMAT_VERSION or cache.get("model_name") != MODEL_NAME or cache.get("model_version") != MODEL_VERSION:
        raise ValueError("Held-out cache is not a compatible ProtTox-TSK feature artifact")
    records = cache.get("records", [])
    if len(records) != expected_records:
        raise ValueError(f"{split_name} expected {expected_records} records, found {len(records)}")
    if any(record.get("label") is None for record in records):
        raise ValueError("Held-out evaluation requires labels in the cache")
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model, mean, std, checkpoint_metadata = load_checkpoint(checkpoint_file, target_device)
    if int(cache["encoder"]["embedding_dim"]) != model.config.embedding_dim:
        raise ValueError("Held-out cache embedding dimension does not match checkpoint")
    training_cache_sha = str(checkpoint_metadata.get("cache_sha256", ""))
    if file_sha256(cache_file) == training_cache_sha:
        raise ValueError("Held-out cache is identical to the training cache; refusing leakage-prone evaluation")
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    physic = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    with torch.inference_mode():
        probabilities = torch.sigmoid(model.forward_features((model.combine_features(tri, physic).to(target_device) - mean) / std)).cpu().tolist()
    raw_probabilities_array = np.asarray(probabilities, dtype=np.float64)
    policy = load_policy(policy_path, checkpoint_file, mode=policy_mode) if policy_path else None
    calibrated_probabilities, selected_threshold = apply_policy(torch.as_tensor(raw_probabilities_array), policy, mode=policy_mode) if policy else (torch.as_tensor(raw_probabilities_array), threshold)
    probabilities_array = np.asarray(calibrated_probabilities.tolist(), dtype=np.float64)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence", "sequence_sha256", "label", "raw_probability", "probability", "prediction", "threshold", "policy_mode"])
        writer.writeheader()
        for record, raw_probability, probability in zip(records, raw_probabilities_array.tolist(), probabilities_array.tolist()):
            writer.writerow({"sample_id": record["sample_id"], "sequence": record["sequence"], "sequence_sha256": record["sequence_sha256"], "label": record["label"], "raw_probability": f"{raw_probability:.9g}", "probability": f"{probability:.9g}", "prediction": int(probability >= selected_threshold), "threshold": selected_threshold, "policy_mode": policy_mode if policy else "fixed_threshold"})
    report = {"split": split_name, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "decision_threshold": selected_threshold, "policy_mode": policy_mode if policy else "fixed_threshold", "policy": {"path": str(policy_path), "sha256": file_sha256(policy_path)} if policy_path else None, "metrics": _metrics(labels, probabilities_array, selected_threshold), "records": len(records), "heldout_cache": {"path": str(cache_file), "sha256": file_sha256(cache_file), "source": cache.get("source")}, "checkpoint": {"path": str(checkpoint_file), "sha256": file_sha256(checkpoint_file), "metadata": checkpoint_metadata}, "leakage_checks": {"heldout_cache_not_training_cache": True, "checkpoint_training_cache_sha256": training_cache_sha, "policy_source": "training OOF only" if policy else "fixed caller-provided threshold", "test_set_not_used_for_model_or_threshold_selection": True}}
    write_manifest(output / "metrics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="Pre-frozen decision threshold; never tune this on Test1/Test2.")
    parser.add_argument("--policy", default=None, help="Frozen training-OOF policy artifact.")
    parser.add_argument("--policy-mode", choices=["screening", "balanced", "strict"], default="balanced")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_cache(args.cache, args.checkpoint, args.output_dir, split_name=args.split_name, expected_records=args.expected_records, threshold=args.threshold, policy_path=args.policy, policy_mode=args.policy_mode, device=args.device)
    print(result["metrics"])


if __name__ == "__main__":
    main()
