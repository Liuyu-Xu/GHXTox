"""One-stage final prediction with a trained ProtTox-TSK head."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import torch

from ghxtox.models import physchem_features, tristat_pool
from ghxtox.prottox_tsk import load_checkpoint
from ghxtox.prottox_tsk_cache import _residue_embeddings
from ghxtox.prottox_tsk_policy import apply_policy, load_policy
from ghxtox.fasta import read_fasta


def predict_sequences(sequences: Sequence[str], checkpoint: str | Path, *, model_path: str | None = None, policy_path: str | Path | None = None, policy_mode: str = "balanced", device: str = "cuda") -> list[dict[str, Any]]:
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model, mean, std, metadata = load_checkpoint(checkpoint, target_device)
    records = [{"sample_id": f"predict_{index + 1}", "sequence": sequence} for index, sequence in enumerate(sequences)]
    embeddings, encoder = _residue_embeddings(records, model_name="prot_t5_xl_half_uniref50-enc", model_path=model_path, device_name=str(target_device), batch_size=4)
    if any(embedding.shape[-1] != model.config.embedding_dim for embedding in embeddings):
        raise ValueError("Frozen ProtT5 embedding dimension is incompatible with the selected ProtTox-TSK checkpoint")
    tri = torch.stack([tristat_pool(embedding.unsqueeze(0), torch.ones((1, len(sequence)), dtype=torch.bool)).squeeze(0) for sequence, embedding in zip(sequences, embeddings)])
    phys = physchem_features(list(sequences))
    with torch.inference_mode():
        raw_probabilities = torch.sigmoid(model.forward_features((model.combine_features(tri, phys).to(target_device) - mean) / std)).cpu()
    policy = load_policy(policy_path, checkpoint, mode=policy_mode) if policy_path else None
    probabilities, threshold = apply_policy(raw_probabilities, policy, mode=policy_mode) if policy else (raw_probabilities, 0.5)
    return [{"sequence": sequence, "raw_probability": float(raw_probability), "probability": float(probability), "label": "Toxic" if probability >= threshold else "Non-toxic", "decision_threshold": threshold, "policy_mode": policy_mode if policy else "fixed_0.5", "physchem": {"length": int(len(sequence)), "netCharge": float(phys[row, 1]), "basicFraction": float(phys[row, 3]), "hydrophobicFraction": float(phys[row, 4]), "aromaticFraction": float(phys[row, 5]), "cysteineCount": int(phys[row, 6])}, "provenance": {"checkpoint": str(checkpoint), "head_kind": model.config.head_kind, "encoder": encoder, "checkpoint_metadata": metadata, "policy": str(policy_path) if policy_path else None}} for row, (sequence, raw_probability, probability) in enumerate(zip(sequences, raw_probabilities.tolist(), probabilities.tolist()))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--policy", default=None, help="Frozen training-OOF policy artifact.")
    parser.add_argument("--policy-mode", choices=["screening", "balanced", "strict"], default="balanced")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    fasta = read_fasta(args.input_fasta)
    results = predict_sequences([record.sequence for record in fasta], args.checkpoint, model_path=args.model_path, policy_path=args.policy, policy_mode=args.policy_mode, device=args.device)
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence", "raw_toxicity_probability", "toxicity_probability", "prediction", "decision_threshold", "policy_mode", "net_charge_ph7", "cysteine_count"])
        writer.writeheader()
        for record, result in zip(fasta, results):
            writer.writerow({"sample_id": record.sample_id, "sequence": result["sequence"], "raw_toxicity_probability": f"{result['raw_probability']:.9g}", "toxicity_probability": f"{result['probability']:.9g}", "prediction": result["label"], "decision_threshold": result["decision_threshold"], "policy_mode": result["policy_mode"], "net_charge_ph7": result["physchem"]["netCharge"], "cysteine_count": result["physchem"]["cysteineCount"]})


if __name__ == "__main__":
    main()
