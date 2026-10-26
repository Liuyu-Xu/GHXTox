"""Generate a canonical ProtT5 feature cache from the frozen deployment snapshot.

The audited legacy MPCE/ProtTox cache cannot be reproduced bit-for-bit from
the currently checked-in HF snapshot (historical snapshot provenance is not
recoverable from this checkout).  For the deployment product only, this script
recomputes all 6387 TriStat+PhysChem records from the explicitly frozen 94a6
snapshot so that the standalone runner and the product full-refit share one
reproducible encoder.

It is additive: no OOF model, v1 checkpoint, legacy cache or Test set is
modified or read for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from ghxtox.models import physchem_feature_vector, tristat_pool
from ghxtox.prottox_tsk import file_sha256, sequence_sha256, write_manifest
from ghxtox.prottox_tsk_cache import _residue_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--prot5-snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = []
    with Path(args.fold_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6387 or sorted({row["fold"] for row in rows}) != ["0", "1", "2", "3", "4"]:
        raise ValueError("Expected the audited 6387-row five-fold manifest")

    records = [
        {"sample_id": str(row["sample_id"]), "sequence": str(row["sequence"])}
        for row in rows
    ]
    embeddings, encoder_identity = _residue_embeddings(
        records,
        model_name="prot_t5_xl_half_uniref50-enc",
        model_path=args.prot5_snapshot,
        device_name=args.device,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cached: list[dict[str, Any]] = []
    for row, embedding in zip(rows, embeddings, strict=True):
        sequence = str(row["sequence"])
        if tuple(embedding.shape) != (len(sequence), 1024):
            raise ValueError(f"Residue alignment failed for {row['sample_id']}")
        tri = tristat_pool(
            embedding.unsqueeze(0),
            torch.ones((1, len(sequence)), dtype=torch.bool),
        ).squeeze(0).float()
        cached.append(
            {
                "sample_id": str(row["sample_id"]),
                "historical_sample_id": str(row["historical_sample_id"]),
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "label": int(row["label"]),
                "fold": int(row["fold"]),
                "tristat": tri,
                "physchem": torch.tensor(physchem_feature_vector(sequence), dtype=torch.float32),
            }
        )
    artifact = {
        "artifact_type": "prot_t5_canonical_product_cache",
        "model_name": "MPCE-3DKD-Product-Canonical",
        "encoder": {
            "identifier": encoder_identity,
            "frozen": True,
            "embedding_dim": 1024,
            "snapshot": str(Path(args.prot5_snapshot)),
            "snapshot_sha256": file_sha256(Path(args.prot5_snapshot) / "pytorch_model.bin"),
        },
        "fold_manifest": str(Path(args.fold_manifest)),
        "fold_manifest_sha256": file_sha256(Path(args.fold_manifest)),
        "records": cached,
    }
    torch.save(artifact, output)
    manifest = {
        "artifact": str(output),
        "artifact_sha256": file_sha256(output),
        "records": len(cached),
        "encoder": artifact["encoder"],
        "fold_manifest_sha256": artifact["fold_manifest_sha256"],
        "timing_seconds": time.perf_counter() - started,
        "purpose": "deployment product full-refit only; frozen OOF artifacts remain untouched",
    }
    write_manifest(str(output) + ".manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
