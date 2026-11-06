"""Generate audited PepBERT TriStat + PhysChem feature caches.

This script is intentionally additive.  It consumes an official PepBERT
Hugging Face snapshot (model.py/config.py/tokenizer.json + checkpoint),
runs frozen residue-level inference, and stores a ProtTox-TSK-compatible
frozen-feature cache so the existing MLP training script can be reused
unchanged for the representation shootout.

Only the audited training fold manifest is used; Test1/Test2 are never read.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from ghxtox.models import physchem_feature_vector, tristat_pool
from ghxtox.prottox_tsk import (
    CACHE_FORMAT_VERSION,
    MODEL_NAME,
    MODEL_VERSION,
    file_sha256,
    sequence_sha256,
    write_manifest,
)


def _module_from_file(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_against_reference(rows: list[dict[str, Any]], reference_cache: Path) -> None:
    payload = torch.load(reference_cache, map_location="cpu", weights_only=False)
    ref_records = payload["records"]
    if len(ref_records) != len(rows):
        raise ValueError(
            f"Reference cache records {len(ref_records)} != fold manifest rows {len(rows)}"
        )
    for position, (row, reference) in enumerate(zip(rows, ref_records, strict=True)):
        if row["sample_id"] != reference["sample_id"]:
            raise ValueError(
                f"Row {position}: manifest sample_id {row['sample_id']!r} != "
                f"reference cache {reference['sample_id']!r}"
            )
        if row["sequence"] != reference["sequence"]:
            raise ValueError(
                f"Row {position}: manifest sequence differs from reference cache"
            )
        if int(row["label"]) != int(reference["label"]):
            raise ValueError(
                f"Row {position}: manifest label differs from reference cache"
            )


def cache_pepbert_features(
    *,
    repo_dir: Path,
    checkpoint_name: str,
    encoder_id: str,
    embedding_dim: int,
    fold_manifest_path: Path,
    reference_cache_path: Path,
    output_path: Path,
    device_name: str,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing cache: {output_path}")
    if not repo_dir.is_dir():
        raise FileNotFoundError(repo_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(fold_manifest_path)
    if len(rows) != 6387:
        raise ValueError(f"Expected the audited 6387-row fold manifest, got {len(rows)}")
    if sorted({row["fold"] for row in rows}) != ["0", "1", "2", "3", "4"]:
        raise ValueError("Fold manifest must contain exactly five folds")
    _validate_against_reference(rows, reference_cache_path)

    config = _module_from_file(repo_dir / "config.py", f"config_{encoder_id}")
    model_module = _module_from_file(repo_dir / "model.py", f"model_{encoder_id}")
    model_config = config.get_config()
    if int(model_config["seq_len"]) < 52:
        raise ValueError(f"Unexpected PepBERT seq_len {model_config['seq_len']}")
    tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
    expected_vocab = tokenizer.get_vocab_size()
    model = model_module.build_transformer(
        src_vocab_size=expected_vocab,
        src_seq_len=int(model_config["seq_len"]),
        d_model=int(model_config["d_model"]),
    )
    checkpoint_path = repo_dir / checkpoint_name
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise ValueError(f"{checkpoint_path} does not contain model_state_dict")
    model.load_state_dict(state["model_state_dict"])
    if int(model_config["d_model"]) != embedding_dim:
        raise ValueError(
            f"Encoder id {encoder_id} says dim {embedding_dim}, checkpoint says "
            f"{model_config['d_model']}"
        )
    model.eval()

    sos = tokenizer.token_to_id("[SOS]")
    eos = tokenizer.token_to_id("[EOS]")
    pad = tokenizer.token_to_id("[PAD]")
    if None in (sos, eos, pad):
        raise ValueError("PepBERT tokenizer is missing [SOS]/[EOS]/[PAD]")

    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    encoded = []
    for row in rows:
        sequence = str(row["sequence"])
        encoded.append([sos] + tokenizer.encode(sequence).ids + [eos])

    started = time.perf_counter()
    forward_seconds = 0.0
    cached_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            stop = min(start + batch_size, len(rows))
            chunk_ids = encoded[start:stop]
            chunk_rows = rows[start:stop]
            width = max(len(ids) for ids in chunk_ids)
            input_ids = torch.full((len(chunk_ids), width), pad, dtype=torch.int64)
            mask = torch.zeros((len(chunk_ids), width), dtype=torch.bool)
            for index, ids in enumerate(chunk_ids):
                input_ids[index, : len(ids)] = torch.as_tensor(ids, dtype=torch.int64)
                mask[index, : len(ids)] = True
            attention_mask = mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.int64, device=device)
            forward_start = time.perf_counter()
            hidden = model.encode(input_ids.to(device), attention_mask).detach().cpu().float()
            forward_seconds += time.perf_counter() - forward_start
            for row, ids, embedding in zip(chunk_rows, chunk_ids, hidden, strict=True):
                sequence = str(row["sequence"])
                length = len(sequence)
                if len(ids) != length + 2:
                    raise RuntimeError(f"Token alignment failed for {row['sample_id']}")
                residue = embedding[1 : 1 + length].contiguous()
                if tuple(residue.shape) != (length, embedding_dim):
                    raise RuntimeError(
                        f"Residue alignment failed for {row['sample_id']}: "
                        f"{tuple(residue.shape)}"
                    )
                tri = tristat_pool(
                    residue.unsqueeze(0),
                    torch.ones((1, length), dtype=torch.bool),
                ).squeeze(0).float()
                if tri.shape != (3 * embedding_dim,):
                    raise RuntimeError(f"TriStat dimension mismatch for {row['sample_id']}")
                cached_records.append(
                    {
                        "sample_id": str(row["sample_id"]),
                        "sequence": sequence,
                        "sequence_sha256": sequence_sha256(sequence),
                        "label": int(row["label"]),
                        "tristat": tri,
                        "physchem": torch.tensor(
                            physchem_feature_vector(sequence), dtype=torch.float32
                        ),
                    }
                )
    elapsed = time.perf_counter() - started

    if len(cached_records) != len(rows):
        raise RuntimeError("Feature generation did not produce one record per manifest row")
    artifact = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "encoder": {
            "identifier": encoder_id,
            "frozen": True,
            "embedding_dim": embedding_dim,
            "parameters": parameter_count,
            "repo": str(repo_dir),
            "checkpoint_file": checkpoint_name,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "tokenizer_sha256": file_sha256(repo_dir / "tokenizer.json"),
            "model_py_sha256": file_sha256(repo_dir / "model.py"),
            "config_py_sha256": file_sha256(repo_dir / "config.py"),
            "seq_len": int(model_config["seq_len"]),
        },
        "source": {
            "path": str(fold_manifest_path),
            "sha256": file_sha256(fold_manifest_path),
            "scope": "training OOF only; Test1/Test2 excluded",
        },
        "reference_cache": {
            "path": str(reference_cache_path),
            "sha256": file_sha256(reference_cache_path),
        },
        "records": cached_records,
    }
    torch.save(artifact, output_path)

    manifest = {
        "artifact": str(output_path),
        "artifact_sha256": file_sha256(output_path),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "encoder": artifact["encoder"],
        "num_records": len(cached_records),
        "source": artifact["source"],
        "feature_contract": ["tristat_mean", "tristat_max", "tristat_std", "physchem_9"],
        "reference_cache": artifact["reference_cache"],
        "timing": {
            "total_elapsed_seconds": elapsed,
            "forward_only_seconds": forward_seconds,
            "average_seconds_per_record": elapsed / max(len(cached_records), 1),
            "device": str(device),
        },
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if torch.cuda.is_available() and device.type == "cuda"
            else None
        ),
        "checks": {
            "all_features_finite": all(
                torch.isfinite(record["tristat"]).all()
                and torch.isfinite(record["physchem"]).all()
                for record in cached_records
            ),
            "exact_reference_order": True,
            "records": len(cached_records),
        },
    }
    write_manifest(str(output_path) + ".manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, help="Local PepBERT HF snapshot directory")
    parser.add_argument("--checkpoint-file", required=True, help="e.g. tmodel_16.pt")
    parser.add_argument("--encoder-id", required=True, help="e.g. pepbert-small-uniparc")
    parser.add_argument("--embedding-dim", type=int, required=True)
    parser.add_argument(
        "--fold-manifest",
        default=r"D:\dachuang2026\GHXTox\data\folds\prottox_tsk_train_prott5_official_mean_folds_v1.csv",
    )
    parser.add_argument(
        "--reference-cache",
        default=r"D:\dachuang2026\GHXTox\runs\prottox_tsk_v1\features.pt",
        help="Existing audited ProtT5 cache used only to verify exact sample order",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = cache_pepbert_features(
        repo_dir=Path(args.repo_dir),
        checkpoint_name=args.checkpoint_file,
        encoder_id=args.encoder_id,
        embedding_dim=args.embedding_dim,
        fold_manifest_path=Path(args.fold_manifest),
        reference_cache_path=Path(args.reference_cache),
        output_path=Path(args.output),
        device_name=args.device,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
