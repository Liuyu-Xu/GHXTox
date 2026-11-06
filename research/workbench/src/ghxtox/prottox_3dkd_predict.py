"""Frozen deployment inference for ProtTox-3DKD (dual-seed MLP ensemble).

Feature extraction imports exactly the same ProtT5 tokenizer/embedding path,
TriStat pooling and PhysChem functions used by the training cache. Each of the
two frozen checkpoints owns its fitted feature normalization; the final
probability is the arithmetic mean of their sigmoid probabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from ghxtox.models import physchem_features, tristat_pool
from ghxtox.models.tristat_head import TriStatMLPHead
from ghxtox.prottox_tsk import file_sha256
from ghxtox.prottox_tsk_cache import _residue_embeddings


MODEL_NAME = "ProtTox-3DKD"
MODEL_VERSION = "v1.0"
ENSEMBLE_SEEDS = (20260905, 20260906)
DECISION_THRESHOLD = 0.5


def load_product_checkpoint(path: str | Path, device: torch.device) -> tuple[TriStatMLPHead, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "ProtTox-3D-Distill-Product" or payload.get("model_version") != "prottox-3d-distill-product-v1":
        raise ValueError(f"Checkpoint is not a ProtTox-3D-Distill product checkpoint: {path}")
    config = payload["config"]
    model = TriStatMLPHead(int(config["input_dim"]), hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(payload["feature_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    if mean.numel() != config["input_dim"] or std.numel() != config["input_dim"]:
        raise ValueError("Product checkpoint normalization dimension does not match its model config")
    return model, mean, std, dict(payload.get("metadata", {}))


def load_ensemble_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if manifest.get("ensemble", {}).get("kind") != "mean_probability":
        raise ValueError("Ensemble manifest must use mean_probability aggregation")
    members = [int(value) for value in manifest["ensemble"]["members"]]
    if sorted(members) != sorted(ENSEMBLE_SEEDS):
        raise ValueError(f"Ensemble seeds are frozen as {ENSEMBLE_SEEDS}")
    if len(manifest.get("checkpoints", [])) != 2:
        raise ValueError("Product ensemble requires exactly two checkpoints")
    return manifest


def predict_sequences(
    sequences: Sequence[str],
    manifest_path: str | Path,
    *,
    model_path: str | None = None,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = load_ensemble_manifest(manifest_path)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}
    if set(checkpoint_paths) != set(ENSEMBLE_SEEDS):
        raise ValueError("Ensemble manifest must contain exactly seeds 20260905 and 20260906")
    members = []
    for seed in ENSEMBLE_SEEDS:
        members.append((seed, *load_product_checkpoint(checkpoint_paths[seed], target_device)))
    records = [{"sample_id": f"predict_{index + 1}", "sequence": sequence} for index, sequence in enumerate(sequences)]
    embeddings, encoder_identity = _residue_embeddings(
        records,
        model_name="prot_t5_xl_half_uniref50-enc",
        model_path=model_path,
        device_name=str(target_device),
        batch_size=4,
    )
    if any(int(value.shape[-1]) != 1024 for value in embeddings):
        raise ValueError("Frozen ProtT5 embedding dimension must be 1024 for ProtTox-3DKD")
    tri = torch.stack(
        [
            tristat_pool(embedding.unsqueeze(0), torch.ones((1, len(sequence)), dtype=torch.bool)).squeeze(0)
            for sequence, embedding in zip(sequences, embeddings)
        ]
    )
    phys = physchem_features(list(sequences))
    if int(phys.shape[1]) != 9:
        raise ValueError("PhysChem feature dimension must be 9 for ProtTox-3DKD")
    all_probabilities = []
    with torch.inference_mode():
        for seed, model, mean, std, metadata in members:
            features = torch.cat((tri, phys), dim=1).to(target_device)
            normalized = (features - mean) / std
            logits = model(normalized)
            all_probabilities.append(torch.sigmoid(logits).cpu())
    averaged = torch.stack(all_probabilities).mean(dim=0)
    results = []
    for row, sequence in enumerate(sequences):
        probability = float(averaged[row])
        results.append(
            {
                "sequence": sequence,
                "probability": probability,
                "label": "Toxic" if probability >= DECISION_THRESHOLD else "Non-toxic",
                "decision_threshold": DECISION_THRESHOLD,
                "member_probabilities": {
                    str(seed): float(all_probabilities[index][row])
                    for index, (seed, *_rest) in enumerate(members)
                },
                "physchem": {
                    "length": int(phys[row, 0]),
                    "netCharge": float(phys[row, 1]),
                    "basicFraction": float(phys[row, 3]),
                    "hydrophobicFraction": float(phys[row, 4]),
                    "aromaticFraction": float(phys[row, 5]),
                    "cysteineCount": int(phys[row, 6]),
                },
                "provenance": {
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "encoder": encoder_identity,
                    "ensemble_seeds": list(ENSEMBLE_SEEDS),
                    "ensemble_manifest": str(manifest_path),
                    "ensemble_manifest_sha256": file_sha256(manifest_path),
                    "checkpoints": {
                        str(seed): {
                            "path": str(checkpoint_paths[seed]),
                            "sha256": file_sha256(checkpoint_paths[seed]),
                            "metadata": members[index][4],
                        }
                        for index, seed in enumerate(ENSEMBLE_SEEDS)
                    },
                    "kd_temperature": 2.0,
                    "kd_weight": 1.0,
                    "training_scope": "full training-data refit; Test1/Test2 excluded",
                },
            }
        )
    return results


def predict_batch(
    items: Sequence[dict[str, Any]],
    manifest_path: str | Path,
    *,
    model_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Run many sequences through one ProtT5 load with length-sorted batches.

    Sequences are sorted by length, split into chunks of ``batch_size``, and
    processed by the frozen encoder once. Results are restored to input order.
    """
    if not items:
        return []
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = load_ensemble_manifest(manifest_path)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}
    if set(checkpoint_paths) != set(ENSEMBLE_SEEDS):
        raise ValueError("Ensemble manifest must contain exactly seeds 20260905 and 20260906")
    members = []
    for seed in ENSEMBLE_SEEDS:
        members.append((seed, *load_product_checkpoint(checkpoint_paths[seed], target_device)))
    order = sorted(range(len(items)), key=lambda index: len(str(items[index]["sequence"])))
    sorted_records = [
        {"sample_id": str(items[index].get("id") or items[index].get("name") or f"predict_{index + 1}"), "sequence": str(items[index]["sequence"])}
        for index in order
    ]
    embeddings, encoder_identity = _residue_embeddings(
        sorted_records,
        model_name="prot_t5_xl_half_uniref50-enc",
        model_path=model_path,
        device_name=str(target_device),
        batch_size=batch_size,
    )
    if any(int(value.shape[-1]) != 1024 for value in embeddings):
        raise ValueError("Frozen ProtT5 embedding dimension must be 1024 for ProtTox-3DKD")
    sorted_sequences = [str(record["sequence"]) for record in sorted_records]
    tri = torch.stack(
        [
            tristat_pool(embedding.unsqueeze(0), torch.ones((1, len(sequence)), dtype=torch.bool)).squeeze(0)
            for sequence, embedding in zip(sorted_sequences, embeddings)
        ]
    )
    phys = physchem_features(sorted_sequences)
    if int(phys.shape[1]) != 9:
        raise ValueError("PhysChem feature dimension must be 9 for ProtTox-3DKD")
    all_probabilities = []
    with torch.inference_mode():
        for seed, model, mean, std, metadata in members:
            features = torch.cat((tri, phys), dim=1).to(target_device)
            normalized = (features - mean) / std
            logits = model(normalized)
            all_probabilities.append(torch.sigmoid(logits).cpu())
    averaged_sorted = torch.stack(all_probabilities).mean(dim=0)
    reverse_order = [0] * len(order)
    for output_position, original_position in enumerate(order):
        reverse_order[original_position] = output_position
    results = []
    for original_position, output_position in enumerate(reverse_order):
        row = output_position
        probability = float(averaged_sorted[row])
        sequence = str(items[original_position]["sequence"])
        results.append(
            {
                "id": str(items[original_position].get("id") or items[original_position].get("name") or f"predict_{original_position + 1}"),
                "sequence": sequence,
                "probability": probability,
                "label": "Toxic" if probability >= DECISION_THRESHOLD else "Non-toxic",
                "decision_threshold": DECISION_THRESHOLD,
                "member_probabilities": {
                    str(seed): float(all_probabilities[index][row])
                    for index, (seed, *_rest) in enumerate(members)
                },
                "physchem": {
                    "length": int(phys[row, 0]),
                    "netCharge": float(phys[row, 1]),
                    "basicFraction": float(phys[row, 3]),
                    "hydrophobicFraction": float(phys[row, 4]),
                    "aromaticFraction": float(phys[row, 5]),
                    "cysteineCount": int(phys[row, 6]),
                },
                "provenance": {
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "encoder": encoder_identity,
                    "ensemble_seeds": list(ENSEMBLE_SEEDS),
                    "ensemble_manifest": str(manifest_path),
                    "ensemble_manifest_sha256": file_sha256(manifest_path),
                    "checkpoints": {
                        str(seed): {
                            "path": str(checkpoint_paths[seed]),
                            "sha256": file_sha256(checkpoint_paths[seed]),
                            "metadata": members[index][4],
                        }
                        for index, seed in enumerate(ENSEMBLE_SEEDS)
                    },
                    "kd_temperature": 2.0,
                    "kd_weight": 1.0,
                    "training_scope": "full training-data refit; Test1/Test2 excluded",
                },
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--ensemble-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from ghxtox.fasta import read_fasta

    fasta = read_fasta(args.input_fasta)
    results = predict_sequences(
        [record.sequence for record in fasta],
        args.ensemble_manifest,
        model_path=args.model_path,
        device=args.device,
    )
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "sequence", "probability", "prediction", "decision_threshold"],
        )
        writer.writeheader()
        for record, result in zip(fasta, results):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "sequence": result["sequence"],
                    "probability": f"{result['probability']:.9g}",
                    "prediction": result["label"],
                    "decision_threshold": result["decision_threshold"],
                }
            )


if __name__ == "__main__":
    main()
