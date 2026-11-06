"""Build an auditable frozen-residue cache for the LocalMotif candidate only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ghxtox.plm_embed import PROTT5_MODELS
from ghxtox.prottox_tsk import file_sha256, sequence_sha256, write_manifest
from ghxtox.prottox_tsk_cache import _residue_embeddings
from ghxtox.prottox_tsk_error_audit import _cache


CACHE_VERSION = "prottox-local-motif-residue-cache-v1"
MODEL_NAME = "ProtTox-LocalMotif"


def cache_residue_embeddings(base_cache_path: str | Path, output_path: str | Path, *, model_name: str, model_path: str | None, device: str, batch_size: int) -> dict[str, Any]:
    """Encode exactly the audited base-cache sequences and preserve base features verbatim."""
    base_file, output = Path(base_cache_path), Path(output_path)
    base = _cache(base_file)
    base_records = list(base["records"])
    ids = [str(record["sample_id"]) for record in base_records]
    if len(ids) != len(set(ids)):
        raise ValueError("Base feature cache contains duplicate sample IDs")
    expected_encoder = str(base.get("encoder", {}).get("identifier", ""))
    if expected_encoder != model_name:
        raise ValueError(f"Residue cache model {model_name!r} must match base cache encoder {expected_encoder!r}")
    encoder_records = [{"sample_id": str(record["sample_id"]), "sequence": str(record["sequence"]), "label": int(record["label"])} for record in base_records]
    embeddings, encoder_identity = _residue_embeddings(encoder_records, model_name=model_name, model_path=model_path, device_name=device, batch_size=batch_size)
    if encoder_identity != expected_encoder or len(embeddings) != len(base_records):
        raise ValueError("Frozen residue encoder identity or record count is inconsistent with the base cache")
    embedding_dim = int(embeddings[0].shape[-1])
    cached_records: list[dict[str, Any]] = []
    for record, embedding in zip(base_records, embeddings, strict=True):
        sequence = str(record["sequence"])
        if sequence_sha256(sequence) != str(record["sequence_sha256"]) or tuple(embedding.shape) != (len(sequence), embedding_dim):
            raise ValueError(f"Frozen residue embedding alignment failed for {record['sample_id']}")
        cached_records.append({
            "sample_id": str(record["sample_id"]),
            "sequence": sequence,
            "sequence_sha256": str(record["sequence_sha256"]),
            "label": int(record["label"]),
            "tristat": torch.as_tensor(record["tristat"], dtype=torch.float32).contiguous(),
            "physchem": torch.as_tensor(record["physchem"], dtype=torch.float32).contiguous(),
            "residue_embeddings": embedding.to(dtype=torch.float16).contiguous(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CACHE_VERSION,
        "model_name": MODEL_NAME,
        "model_version": CACHE_VERSION,
        "base_cache": {"path": str(base_file), "sha256": file_sha256(base_file), "model_name": base["model_name"], "model_version": base["model_version"]},
        "encoder": {"identifier": encoder_identity, "frozen": True, "embedding_dim": embedding_dim, "storage_dtype": "float16"},
        "source": dict(base.get("source", {})),
        "records": cached_records,
    }
    torch.save(payload, output)
    lengths = [len(record["sequence"]) for record in cached_records]
    manifest = {
        "artifact": str(output),
        "artifact_sha256": file_sha256(output),
        "format_version": CACHE_VERSION,
        "model_name": MODEL_NAME,
        "model_version": CACHE_VERSION,
        "base_cache": payload["base_cache"],
        "encoder": payload["encoder"],
        "source": payload["source"],
        "records": len(cached_records),
        "residues": sum(lengths),
        "max_sequence_length": max(lengths),
        "feature_contract": ["base_tristat_3072", "base_physchem_9", "frozen_residue_embeddings_float16"],
        "scope": "LocalMotif candidate only; training cache; Test1/Test2 excluded",
    }
    write_manifest(str(output) + ".manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="prot_t5_xl_half_uniref50-enc", choices=sorted(PROTT5_MODELS))
    parser.add_argument("--model-path", required=True, help="Explicit local ProtT5 snapshot used by the audited base cache.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    manifest = cache_residue_embeddings(args.base_cache, args.output, model_name=args.model, model_path=args.model_path, device=args.device, batch_size=args.batch_size)
    print(f"Cached {manifest['records']} residue embeddings ({manifest['residues']} residues) to {manifest['artifact']}")


if __name__ == "__main__":
    main()
