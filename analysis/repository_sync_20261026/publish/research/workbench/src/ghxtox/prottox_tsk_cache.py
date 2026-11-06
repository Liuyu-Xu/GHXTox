"""Cache frozen ProtT5 TriStat + physicochemical features once per dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from ghxtox.models import physchem_feature_vector, tristat_pool
from ghxtox.plm_embed import PROTT5_MODELS, _format_prott5_sequence, _load_prott5
from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, file_sha256, sequence_sha256, write_manifest
from ghxtox.utils import resolve_device


def _residue_embeddings(records: list[dict[str, Any]], *, model_name: str, model_path: str | None, device_name: str, batch_size: int) -> tuple[list[torch.Tensor], str]:
    """Use embedded source features when supplied, otherwise run frozen ProtT5."""
    existing = [record.get("plm_features") for record in records]
    if all(torch.is_tensor(value) for value in existing):
        tensors = [torch.as_tensor(value, dtype=torch.float32)[: len(record["sequence"])].contiguous() for record, value in zip(records, existing)]
        return tensors, str(records[0].get("plm_model", "precomputed_residue_embeddings"))
    if any(torch.is_tensor(value) for value in existing):
        raise ValueError("Input cache must contain PLM embeddings for every record or for none of them")
    device = resolve_device(device_name)
    encoder, tokenizer, expected_dim = _load_prott5(model_name, device, model_path=model_path)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            encoded = tokenizer([_format_prott5_sequence(str(item["sequence"])) for item in chunk], add_special_tokens=True, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state.detach().cpu().float()
            for item, embedding in zip(chunk, hidden):
                length = len(str(item["sequence"]))
                residue = embedding[:length].contiguous()
                if residue.shape != (length, expected_dim):
                    raise ValueError(f"ProtT5 residue alignment failed for {item['sample_id']}: {tuple(residue.shape)}")
                output.append(residue)
    return output, model_name


def _select_records(records: list[dict[str, Any]], limit: int | None, *, stratify_labels: bool, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    if not stratify_labels:
        return records[:limit]
    buckets: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("label") is None:
            raise ValueError("--stratify-labels requires a label for every selected source record")
        buckets.setdefault(int(record["label"]), []).append(record)
    if len(buckets) < 2:
        raise ValueError("--stratify-labels requires at least two classes")
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = sorted(buckets)
    selected: list[dict[str, Any]] = []
    for position, label in enumerate(labels):
        target = limit // len(labels) + (1 if position < limit % len(labels) else 0)
        if len(buckets[label]) < target:
            raise ValueError(f"Class {label} has only {len(buckets[label])} records; cannot draw {target} stratified smoke records")
        selected.extend(buckets[label][:target])
    rng.shuffle(selected)
    return selected


def cache_features(input_path: str | Path, output_path: str | Path, *, model_name: str, model_path: str | None, device: str, batch_size: int, limit: int | None = None, stratify_labels: bool = False, seed: int = 20260905) -> dict[str, Any]:
    source = Path(input_path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if "records" in payload:
        records = list(payload["records"])
    elif all(key in payload for key in ("sample_ids", "sequences", "labels")):
        sample_ids, sequences, labels = payload["sample_ids"], payload["sequences"], payload["labels"]
        if not (len(sample_ids) == len(sequences) == len(labels)):
            raise ValueError("Sequence payload has inconsistent sample_ids, sequences, and labels lengths")
        records = [{"sample_id": sample_id, "sequence": sequence, "label": label} for sample_id, sequence, label in zip(sample_ids, sequences, labels)]
    else:
        raise ValueError("Source payload must contain records or parallel sample_ids/sequences/labels fields")
    records = _select_records(records, limit, stratify_labels=stratify_labels, seed=seed)
    if not records:
        raise ValueError("No source records available for ProtTox-TSK caching")
    for record in records:
        if not record.get("sample_id") or not record.get("sequence"):
            raise ValueError("Each source record must include sample_id and sequence")
    embeddings, encoder_identity = _residue_embeddings(records, model_name=model_name, model_path=model_path, device_name=device, batch_size=batch_size)
    cached_records: list[dict[str, Any]] = []
    for record, embedding in zip(records, embeddings):
        length = len(str(record["sequence"]))
        tri = tristat_pool(embedding.unsqueeze(0), torch.ones((1, length), dtype=torch.bool)).squeeze(0).cpu().float()
        cached_records.append({
            "sample_id": str(record["sample_id"]), "sequence": str(record["sequence"]), "sequence_sha256": sequence_sha256(str(record["sequence"])),
            "label": None if record.get("label") is None else int(record["label"]), "tristat": tri,
            "physchem": torch.tensor(physchem_feature_vector(str(record["sequence"])), dtype=torch.float32),
        })
    embedding_dim = int(embeddings[0].shape[-1])
    if any(int(value.shape[-1]) != embedding_dim for value in embeddings):
        raise ValueError("Inconsistent frozen embedding dimensions")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"format_version": CACHE_FORMAT_VERSION, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "encoder": {"identifier": encoder_identity, "frozen": True, "embedding_dim": embedding_dim}, "source": {"path": str(source), "sha256": file_sha256(source)}, "records": cached_records}
    torch.save(artifact, output)
    manifest = {"artifact": str(output), "artifact_sha256": file_sha256(output), "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "num_records": len(cached_records), "embedding_dim": embedding_dim, "encoder": artifact["encoder"], "source": artifact["source"], "selection": {"limit": limit, "stratify_labels": stratify_labels, "seed": seed}, "feature_contract": ["tristat_mean", "tristat_max", "tristat_std", "physchem_9"]}
    write_manifest(str(output) + ".manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Processed source .pt with records containing sample_id and sequence.")
    parser.add_argument("--output", required=True, help="Output frozen-feature .pt cache.")
    parser.add_argument("--model", default="prot_t5_xl_half_uniref50-enc", choices=sorted(PROTT5_MODELS))
    parser.add_argument("--model-path", default=None, help="Local ProtT5 directory; defaults to the project model path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test record limit.")
    parser.add_argument("--stratify-labels", action="store_true", help="Select a class-balanced subset when --limit is used; smoke only, not a group split.")
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    result = cache_features(args.input, args.output, model_name=args.model, model_path=args.model_path, device=args.device, batch_size=args.batch_size, limit=args.limit, stratify_labels=args.stratify_labels, seed=args.seed)
    print(f"Cached {result['num_records']} frozen ProtT5 feature records to {result['artifact']}")


if __name__ == "__main__":
    main()
