"""Read-only historical Test1/Test2 audit for the frozen ProtTox-3DKD v1.0."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ghxtox.prottox_3dkd_predict import predict_sequences
from ghxtox.prottox_tsk import sequence_sha256
from ghxtox.train_prottox_tsk import _metrics


def _ece10(labels: np.ndarray, probabilities: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    indices = np.clip(np.digitize(probabilities, bins) - 1, 0, 9)
    total = 0.0
    for index in range(10):
        mask = indices == index
        count = int(np.sum(mask))
        if count == 0:
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        total += count / len(labels) * abs(accuracy - confidence)
    return total


def audit(test_csv: str, manifest: str, model_path: str | None, training_cache: str, split: str, output: str, device: str = "cuda") -> dict:
    with Path(test_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sequences = [str(row["sequence"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.float64)
    cache = torch.load(training_cache, map_location="cpu", weights_only=False)
    training_hashes = {str(record["sequence_sha256"]) for record in cache["records"]}
    test_hashes = [sequence_sha256(sequence) for sequence in sequences]
    overlap = sorted(set(test_hashes) & training_hashes)
    results = predict_sequences(sequences, manifest, model_path=model_path, device=device)
    probabilities = np.asarray([float(result["probability"]) for result in results], dtype=np.float64)
    if len(probabilities) != len(labels):
        raise ValueError("Prediction count does not match test labels")
    metrics = _metrics(labels, probabilities)
    metrics["ece_10"] = _ece10(labels, probabilities)
    audit_record = {
        "artifact_type": "prottox_3dkd_test_audit",
        "model_name": "ProtTox-3DKD",
        "model_version": "v1.0",
        "split": split,
        "records": len(rows),
        "positive": int(np.sum(labels)),
        "negative": int(len(labels) - np.sum(labels)),
        "threshold": 0.5,
        "training_cache_sha256": None,
        "exact_sequence_overlap_with_training": len(overlap),
        "metrics": metrics,
        "test_csv": str(test_csv),
        "selection_scope": "frozen deployment version; read-only historical audit; no model/threshold/ensemble change",
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit_record, indent=2) + "\n", encoding="utf-8")
    return audit_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--split", required=True, choices=["Test1", "Test2"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = audit(
        args.test_csv,
        args.manifest,
        args.model_path,
        args.training_cache,
        args.split,
        args.output,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
