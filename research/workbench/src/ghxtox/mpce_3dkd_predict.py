"""Frozen deployment inference for MPCE-3DKD (dual-seed MPCE ensemble).

Feature extraction imports exactly the same ProtT5 tokenizer/embedding path,
TriStat pooling, PhysChem functions and MPCEB forward used by the training
cache.  Each checkpoint owns fitted feature normalization; the ensemble
probability is the arithmetic mean of the two sigmoid probabilities.

Raw probability is the audited model output.  A display-calibrated
probability (temperature scaling selected once on training OOF) is included
for website display without changing the model or decision threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ghxtox.models import physchem_features, tristat_pool
from ghxtox.plm_embed import _format_prott5_sequence, _load_prott5
from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_mpce_net import MPCEB, PROTOTYPES_PER_CLASS


MODEL_NAME = "MPCE-3DKD"
MODEL_VERSION = "v1.0-product-single-forward-audit"
ENSEMBLE_SEEDS = (20260905, 20260906)
DECISION_THRESHOLD = 0.5
DISPLAY_TEMPERATURE = 2.69177508354187
INTERACTION_BRANCH_NAMES = ("mean_max", "mean_std", "max_std", "sequence_physchem")


def _forward_with_zero_interaction(
    model: MPCEB,
    tri: torch.Tensor,
    phys: torch.Tensor,
    zero_branch: int,
) -> torch.Tensor:
    """Return z_final after zeroing one low-rank interaction group.

    This is a read-only counterfactual used for post-prediction evidence; it
    does not modify model parameters or the deployment prediction math.
    """
    h_mean = model.proj_mean(tri[:, :1024])
    h_max = model.proj_max(tri[:, 1024:2048])
    h_std = model.proj_std(tri[:, 2048:])
    h_phys = model.proj_phys(phys)
    branches = [
        h_mean * h_max,
        h_mean * h_std,
        h_max * h_std,
        h_mean * h_phys,
    ]
    blocks = list(branches)
    blocks[zero_branch] = torch.zeros_like(blocks[zero_branch])
    interactions = torch.cat(blocks, dim=-1)
    vector = torch.cat([h_mean, h_max, h_std, h_phys, interactions], dim=-1)
    z_global = model.global_mlp(vector).squeeze(-1)
    latent = model.latent(interactions)
    latent = F.normalize(latent, dim=-1)
    prototypes = F.normalize(model.prototypes, dim=-1)
    cosine = latent @ prototypes.t()
    toxic_scores = cosine[:, :PROTOTYPES_PER_CLASS]
    nontoxic_scores = cosine[:, PROTOTYPES_PER_CLASS:]
    affinity_toxic = torch.logsumexp(toxic_scores, dim=-1)
    affinity_nontoxic = torch.logsumexp(nontoxic_scores, dim=-1)
    z_proto = affinity_toxic - affinity_nontoxic
    return z_global + torch.sigmoid(model.alpha) * z_proto


def _embed_independently(
    records: Sequence[dict[str, Any]],
    *,
    model_path: str | None,
    device: torch.device,
) -> tuple[list[torch.Tensor], str]:
    """Run ProtT5 once, then embed every sequence in its own forward pass.

    ProtT5 outputs are sensitive to padding/batch composition, so per-record
    forwards are required to reproduce the canonical batch-size-1 product
    cache for arbitrary query sets.
    """
    encoder, tokenizer, expected_dim = _load_prott5(
        "prot_t5_xl_half_uniref50-enc",
        device,
        model_path=model_path,
    )
    encoder.eval()
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for record in records:
            sequence = str(record["sequence"])
            encoded = tokenizer(
                [_format_prott5_sequence(sequence)],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state.detach().cpu().float()
            residue = hidden[0, : len(sequence)].contiguous()
            if residue.shape != (len(sequence), expected_dim):
                raise ValueError(
                    f"Residue alignment failed for {record['sample_id']}: "
                    f"{tuple(residue.shape)}"
                )
            output.append(residue)
    return output, "prot_t5_xl_half_uniref50-enc"


def load_product_checkpoint(path: str | Path, device: torch.device) -> tuple[MPCEB, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "MPCE-3DKD-Product" or payload.get("model_version") != "mpce-3dkd-product-v1":
        raise ValueError(f"Checkpoint is not an MPCE-3DKD product checkpoint: {path}")
    model = MPCEB()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    tri_mean = torch.as_tensor(payload["tri_mean"], dtype=torch.float32, device=device)
    tri_std = torch.as_tensor(payload["tri_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    phys_mean = torch.as_tensor(payload["phys_mean"], dtype=torch.float32, device=device)
    phys_std = torch.as_tensor(payload["phys_std"], dtype=torch.float32, device=device).clamp_min(1e-6)
    return model, tri_mean, tri_std, phys_mean, phys_std, dict(payload.get("metadata", {}))


def load_ensemble_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if manifest.get("ensemble", {}).get("kind") != "mean_probability":
        raise ValueError("Ensemble manifest must use mean_probability aggregation")
    members = sorted(int(item["seed"]) for item in manifest["checkpoints"])
    if members != sorted(ENSEMBLE_SEEDS):
        raise ValueError(f"Ensemble seeds must be {sorted(ENSEMBLE_SEEDS)}")
    return manifest


def predict_sequences(
    sequences: Sequence[str],
    manifest_path: str | Path,
    *,
    model_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    display_temperature: float = DISPLAY_TEMPERATURE,
) -> list[dict[str, Any]]:
    target_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    manifest = load_ensemble_manifest(manifest_path)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}
    profile_map = None
    profile_map_path = Path(manifest_path).parent / "prototype_profile_map.json"
    if profile_map_path.is_file():
        profile_map = json.loads(profile_map_path.read_text(encoding="utf-8"))
    members = []
    for seed in ENSEMBLE_SEEDS:
        members.append((seed, *load_product_checkpoint(checkpoint_paths[seed], target_device)))
    records = [
        {"sample_id": f"predict_{index + 1}", "sequence": str(sequence)}
        for index, sequence in enumerate(sequences)
    ]
    embeddings, encoder_identity = _embed_independently(
        records,
        model_path=model_path,
        device=target_device,
    )
    if any(int(value.shape[-1]) != 1024 for value in embeddings):
        raise ValueError("Frozen ProtT5 embedding dimension must be 1024 for MPCE-3DKD")
    tri = torch.stack(
        [
            tristat_pool(
                embedding.unsqueeze(0),
                torch.ones((1, len(sequence)), dtype=torch.bool),
            ).squeeze(0)
            for sequence, embedding in zip(sequences, embeddings)
        ]
    )
    phys = physchem_features(list(sequences))
    if int(phys.shape[1]) != 9:
        raise ValueError("PhysChem feature dimension must be 9 for MPCE-3DKD")
    member_probabilities = []
    member_cosines = []
    member_z_proto = []
    member_zero_probabilities = []
    with torch.inference_mode():
        for seed, model, tri_mean, tri_std, phys_mean, phys_std, metadata in members:
            normalized_tri = (tri.to(target_device) - tri_mean) / tri_std
            normalized_phys = (phys.to(target_device) - phys_mean) / phys_std
            probabilities = []
            cosines = []
            z_protos = []
            zero_probabilities = {name: [] for name in INTERACTION_BRANCH_NAMES}
            for start in range(0, len(sequences), batch_size):
                stop = min(start + batch_size, len(sequences))
                z_final, _z_global, z_proto, cosine = model(
                    normalized_tri[start:stop],
                    normalized_phys[start:stop],
                )
                probabilities.append(torch.sigmoid(z_final).cpu())
                cosines.append(cosine.cpu())
                z_protos.append(z_proto.cpu())
                for branch_index, branch_name in enumerate(INTERACTION_BRANCH_NAMES):
                    zero_logits = _forward_with_zero_interaction(
                        model,
                        normalized_tri[start:stop],
                        normalized_phys[start:stop],
                        branch_index,
                    )
                    zero_probabilities[branch_name].append(torch.sigmoid(zero_logits).cpu())
            member_probabilities.append(torch.cat(probabilities, dim=0))
            member_cosines.append(torch.cat(cosines, dim=0))
            member_z_proto.append(torch.cat(z_protos, dim=0))
            member_zero_probabilities.append(
                {
                    name: torch.cat(values, dim=0)
                    for name, values in zero_probabilities.items()
                }
            )
    averaged = torch.stack(member_probabilities).mean(dim=0)
    logits = torch.log(averaged.clamp_min(1e-7) / (1 - averaged).clamp_min(1e-7))
    calibrated = torch.sigmoid(logits / display_temperature)
    results = []
    for row, sequence in enumerate(sequences):
        raw_probability = float(averaged[row])
        calibrated_probability = float(calibrated[row])
        member_probability_map = {
            str(seed): float(member_probabilities[index][row])
            for index, (seed, *_rest) in enumerate(members)
        }
        per_seed_evidence = {}
        for index, (seed, *_rest) in enumerate(members):
            cosine = member_cosines[index][row].numpy()
            z_proto_value = float(member_z_proto[index][row])
            toxic_index = int(cosine[:PROTOTYPES_PER_CLASS].argmax())
            nontoxic_index = int(cosine[PROTOTYPES_PER_CLASS:].argmax()) + PROTOTYPES_PER_CLASS
            toxic_similarity = float(cosine[toxic_index])
            nontoxic_similarity = float(cosine[nontoxic_index])

            def pattern_text(prototype_index: int) -> str | None:
                if profile_map is None:
                    return None
                prototype_entry = (
                    profile_map.get("per_seed", {})
                    .get(str(seed), {})
                    .get("prototypes", {})
                    .get(str(prototype_index))
                )
                if not prototype_entry:
                    return None
                return str(prototype_entry.get("model_associated_pattern"))

            def prototype_entry(prototype_index: int) -> dict | None:
                if profile_map is None:
                    return None
                return (
                    profile_map.get("per_seed", {})
                    .get(str(seed), {})
                    .get("prototypes", {})
                    .get(str(prototype_index))
                )

            toxic_entry = prototype_entry(toxic_index) or {}
            nontoxic_entry = prototype_entry(nontoxic_index) or {}
            toxic_stats = toxic_entry.get("similarity_stats", {})
            nontoxic_stats = nontoxic_entry.get("similarity_stats", {})
            per_seed_evidence[str(seed)] = {
                "toxic_pattern_id": f"T{toxic_index + 1}",
                "toxic_similarity": toxic_similarity,
                "toxic_pattern_profile": pattern_text(toxic_index),
                "toxic_pattern_support_count": toxic_entry.get("usage_count"),
                "toxic_pattern_mean_similarity": toxic_stats.get("mean"),
                "toxic_pattern_similarity_q25": toxic_stats.get("q25"),
                "toxic_pattern_similarity_q50": toxic_stats.get("q50"),
                "toxic_pattern_similarity_q75": toxic_stats.get("q75"),
                "non_toxic_pattern_id": f"N{nontoxic_index - PROTOTYPES_PER_CLASS + 1}",
                "non_toxic_similarity": nontoxic_similarity,
                "non_toxic_pattern_profile": pattern_text(nontoxic_index),
                "non_toxic_pattern_support_count": nontoxic_entry.get("usage_count"),
                "non_toxic_pattern_mean_similarity": nontoxic_stats.get("mean"),
                "non_toxic_pattern_similarity_q25": nontoxic_stats.get("q25"),
                "non_toxic_pattern_similarity_q50": nontoxic_stats.get("q50"),
                "non_toxic_pattern_similarity_q75": nontoxic_stats.get("q75"),
                "prototype_affinity_logit": z_proto_value,
            }
        differences = {
            seed: abs(member_probability_map[str(seed)] - raw_probability)
            for seed in ENSEMBLE_SEEDS
        }
        representative_seed = min(
            ENSEMBLE_SEEDS,
            key=lambda seed: (differences[seed], 0 if seed == 20260905 else 1),
        )
        representative_evidence = per_seed_evidence[str(representative_seed)]
        dominant_side = "toxic" if raw_probability >= DECISION_THRESHOLD else "non_toxic"
        if dominant_side == "toxic":
            dominant_id = representative_evidence["toxic_pattern_id"]
            dominant_similarity = representative_evidence["toxic_similarity"]
            dominant_profile = representative_evidence["toxic_pattern_profile"]
        else:
            dominant_id = representative_evidence["non_toxic_pattern_id"]
            dominant_similarity = representative_evidence["non_toxic_similarity"]
            dominant_profile = representative_evidence["non_toxic_pattern_profile"]
        prototype_evidence = {
            "per_seed": per_seed_evidence,
            "representative_seed": str(representative_seed),
            "dominant_side": dominant_side,
            "dominant_pattern_id": dominant_id,
            "dominant_pattern_similarity": dominant_similarity,
            "dominant_model_associated_pattern": dominant_profile,
            "interpretation_note": "model-associated prototype evidence; not an experimental toxicity cause",
        }
        is_toxic_prediction = raw_probability >= DECISION_THRESHOLD
        cross_view_evidence = {
            "branch_removal_delta": {},
            "dominant_interaction": None,
            "dominant_removal_delta": None,
            "support_count": 0,
            "support_fraction": 0.0,
            "interpretation_note": "removal_delta = probability without that interaction group - full probability; model-associated evidence only",
        }
        dominant_abs = -1.0
        support_count = 0
        for branch_name in INTERACTION_BRANCH_NAMES:
            zero_probability = float(
                torch.stack(
                    [member_zero_probabilities[index][branch_name][row] for index in range(len(members))]
                ).mean()
            )
            removal_delta = zero_probability - raw_probability
            cross_view_evidence["branch_removal_delta"][branch_name] = removal_delta
            if abs(removal_delta) > dominant_abs:
                dominant_abs = abs(removal_delta)
                cross_view_evidence["dominant_interaction"] = branch_name
                cross_view_evidence["dominant_removal_delta"] = removal_delta
            supports_prediction = (
                (is_toxic_prediction and removal_delta < 0.0)
                or (not is_toxic_prediction and removal_delta > 0.0)
            )
            if supports_prediction:
                support_count += 1
        cross_view_evidence["support_count"] = support_count
        cross_view_evidence["support_fraction"] = support_count / len(INTERACTION_BRANCH_NAMES)
        results.append(
            {
                "sequence": str(sequence),
                "probability": raw_probability,
                "calibrated_probability": calibrated_probability,
                "label": "Toxic" if raw_probability >= DECISION_THRESHOLD else "Non-toxic",
                "decision_threshold": DECISION_THRESHOLD,
                "member_probabilities": {
                    seed: member_probability_map[seed]
                    for seed in member_probability_map
                },
                "prototype_evidence": prototype_evidence,
                "cross_view_evidence": cross_view_evidence,
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
                            "metadata": members[index][6],
                        }
                        for index, seed in enumerate(ENSEMBLE_SEEDS)
                    },
                    "kd_temperature": 2.0,
                    "kd_weight": 1.0,
                    "display_calibration_temperature": display_temperature,
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
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    from ghxtox.fasta import read_fasta

    fasta = read_fasta(args.input_fasta)
    results = predict_sequences(
        [record.sequence for record in fasta],
        args.ensemble_manifest,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "sequence", "probability", "calibrated_probability", "prediction", "decision_threshold"],
        )
        writer.writeheader()
        for record, result in zip(fasta, results):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "sequence": result["sequence"],
                    "probability": f"{result['probability']:.9g}",
                    "calibrated_probability": f"{result['calibrated_probability']:.9g}",
                    "prediction": result["label"],
                    "decision_threshold": result["decision_threshold"],
                }
            )


if __name__ == "__main__":
    main()
