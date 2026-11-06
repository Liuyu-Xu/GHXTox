"""Generate canonical batch-size-1 ProtT5 feature caches for held-out splits.

Same encoder contract as the MPCE-3DKD product single-forward cache: one
sequence per ProtT5 forward pass so that any Test1/Test2 result can be
reproduced by the standalone runner exactly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ghxtox.models import physchem_feature_vector, tristat_pool
from ghxtox.plm_embed import _format_prott5_sequence, _load_prott5
from ghxtox.prottox_tsk import CACHE_FORMAT_VERSION, MODEL_NAME, MODEL_VERSION, file_sha256, sequence_sha256, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source .pt with sample_ids/sequences/labels")
    parser.add_argument("--prot5-snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    if not all(key in payload for key in ("sample_ids", "sequences", "labels")):
        raise ValueError("Source must contain parallel sample_ids/sequences/labels")
    sample_ids = [str(value) for value in payload["sample_ids"]]
    sequences = [str(value) for value in payload["sequences"]]
    labels = [int(value) for value in payload["labels"]]
    if not (len(sample_ids) == len(sequences) == len(labels)):
        raise ValueError("Source arrays have inconsistent lengths")

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    encoder, tokenizer, embedding_dim = _load_prott5(
        "prot_t5_xl_half_uniref50-enc",
        device,
        model_path=args.prot5_snapshot,
    )
    encoder.eval()
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample_id, sequence, label in zip(sample_ids, sequences, labels, strict=True):
            encoded = tokenizer(
                [_format_prott5_sequence(sequence)],
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = encoder(**encoded).last_hidden_state.detach().cpu().float()
            residue = hidden[0, : len(sequence)].contiguous()
            if residue.shape != (len(sequence), embedding_dim):
                raise ValueError(f"Residue alignment failed for {sample_id}")
            tri = tristat_pool(
                residue.unsqueeze(0),
                torch.ones((1, len(sequence)), dtype=torch.bool),
            ).squeeze(0).float()
            records.append(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "sequence_sha256": sequence_sha256(sequence),
                    "label": label,
                    "tristat": tri,
                    "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32),
                }
            )
    elapsed = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": CACHE_FORMAT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "encoder": {
            "identifier": "prot_t5_xl_half_uniref50-enc",
            "frozen": True,
            "embedding_dim": embedding_dim,
            "snapshot": str(Path(args.prot5_snapshot)),
            "snapshot_sha256": file_sha256(Path(args.prot5_snapshot) / "pytorch_model.bin"),
            "forward_mode": "one_sequence_per_forward",
        },
        "source": {"path": str(Path(args.source)), "sha256": file_sha256(Path(args.source))},
        "records": records,
    }
    torch.save(artifact, output)
    manifest = {
        "artifact": str(output),
        "artifact_sha256": file_sha256(output),
        "records": len(records),
        "encoder": artifact["encoder"],
        "source": artifact["source"],
        "timing_seconds": elapsed,
    }
    write_manifest(str(output) + ".manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
