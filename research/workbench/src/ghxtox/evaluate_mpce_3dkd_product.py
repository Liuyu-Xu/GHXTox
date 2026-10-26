"""Final read-only Test1/Test2 audit for the MPCE-3DKD product ensemble.

Uses the canonical batch-size-1 held-out feature caches and the frozen
single-forward product full refit.  Only metrics are computed after all model
and threshold decisions were frozen; nothing is selected or tuned here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.mpce_3dkd_predict import load_ensemble_manifest, load_product_checkpoint
from ghxtox.prottox_tsk import file_sha256, write_manifest
from ghxtox.train_prottox_tsk import _metrics


def evaluate(cache_path: str | Path, manifest_path: str | Path, output_dir: str | Path, *, split_name: str, expected_records: int, threshold: float = 0.5, device: str = "cuda") -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache.get("records", [])
    if len(records) != expected_records:
        raise ValueError(f"{split_name}: expected {expected_records}, got {len(records)}")
    if any(record.get("label") is None for record in records):
        raise ValueError(f"{split_name}: labels required for final audit")
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = load_ensemble_manifest(manifest_path)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}
    members = {
        seed: load_product_checkpoint(checkpoint_paths[seed], target_device)
        for seed in (20260905, 20260906)
    }
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    member_probabilities = []
    with torch.inference_mode():
        for seed, (model, tri_mean, tri_std, phys_mean, phys_std, metadata) in members.items():
            normalized_tri = (tri.to(target_device) - tri_mean) / tri_std
            normalized_phys = (phys.to(target_device) - phys_mean) / phys_std
            z_final, _z_global, _z_proto, _cosine = model(normalized_tri, normalized_phys)
            member_probabilities.append(torch.sigmoid(z_final).cpu())
    averaged = torch.stack(member_probabilities).mean(dim=0).numpy()
    metrics = _metrics(labels, averaged, threshold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "sequence", "sequence_sha256", "label", "probability", "prediction", "threshold", "seed05_probability", "seed06_probability"],
        )
        writer.writeheader()
        for record, probability, p05, p06 in zip(
            records,
            averaged.tolist(),
            member_probabilities[0].numpy().tolist(),
            member_probabilities[1].numpy().tolist(),
            strict=True,
        ):
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "sequence": record["sequence"],
                    "sequence_sha256": record["sequence_sha256"],
                    "label": record["label"],
                    "probability": f"{probability:.9g}",
                    "prediction": int(probability >= threshold),
                    "threshold": threshold,
                    "seed05_probability": f"{p05:.9g}",
                    "seed06_probability": f"{p06:.9g}",
                }
            )
    report = {
        "split": split_name,
        "model_name": manifest["model_name"],
        "model_version": manifest["model_version"],
        "decision_threshold": threshold,
        "metrics": metrics,
        "records": len(records),
        "heldout_cache": {"path": str(Path(cache_path)), "sha256": file_sha256(cache_path), "records": len(records)},
        "checkpoint_manifest": {"path": str(Path(manifest_path)), "sha256": file_sha256(manifest_path)},
        "selection_scope": "final read-only audit after architecture/checkpoint/threshold were frozen; no tuning on Test1/Test2",
        "notes": ["Test2 is a labeled subset of Test1 and is reported only as a historical audit split"],
    }
    write_manifest(output / "metrics.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate(
        args.cache,
        args.manifest,
        args.output_dir,
        split_name=args.split_name,
        expected_records=args.expected_records,
        threshold=args.threshold,
        device=args.device,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
