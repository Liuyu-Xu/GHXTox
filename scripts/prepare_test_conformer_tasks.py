"""Prepare dataset-scoped Test1/Test2 conformer-generation tasks.

The task FASTA identifiers are deliberately prefixed with the dataset name so
that server outputs can never be confused with training caches.  The CSV keeps
the current processed-record ID as ``target_sample_id``; a separate audited
normalization step restores that ID only in ``original_id`` for local feature
attachment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _records(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"No records found in {path}")
    return records


def _write_fasta(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f">{row['sample_id']}|{row['label']}\n{row['sequence']}\n")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _task_rows(dataset: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_scoped: set[str] = set()
    seen_targets: set[str] = set()
    for source_index, record in enumerate(records):
        target = str(record["sample_id"])
        sequence = str(record["sequence"]).strip().upper()
        label = int(record["label"])
        scoped = f"{dataset}__{target}"
        if scoped in seen_scoped or target in seen_targets:
            raise ValueError(f"Duplicate task identifier in {dataset}: {target}")
        seen_scoped.add(scoped)
        seen_targets.add(target)
        rows.append(
            {
                "source_index": source_index,
                "sample_id": scoped,
                "target_sample_id": target,
                "dataset": dataset,
                "label": label,
                "sequence": sequence,
                "length": len(sequence),
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            }
        )
    return rows


def _write_task_family(
    rows: list[dict[str, Any]],
    root: Path,
    stem: str,
    chunk_size: int | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Task {stem} is empty")
    _write_fasta(rows, root / f"{stem}.fasta")
    _write_csv(rows, root / f"{stem}.csv")
    chunks = 0
    if chunk_size is not None:
        chunk_root = root / "chunks" / stem
        for chunk_index, offset in enumerate(range(0, len(rows), chunk_size), start=1):
            chunk = rows[offset : offset + chunk_size]
            _write_fasta(chunk, chunk_root / f"chunk_{chunk_index:04d}.fasta")
            _write_csv(chunk, chunk_root / f"chunk_{chunk_index:04d}.csv")
            chunks += 1
    return {"records": len(rows), "chunks": chunks}


def prepare(
    test1_processed: Path,
    test2_processed: Path,
    output_root: Path,
    chunk_size: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "id_policy": "dataset__processed_sample_id",
        "af2_length_range": [16, 30],
        "pepflow_length_range": [3, 15],
        "routing_policy": "source-matched to the retained training conformer branches",
        "datasets": {},
    }
    for dataset, processed in (("test1", test1_processed), ("test2", test2_processed)):
        rows = _task_rows(dataset, _records(processed))
        pepflow = [row for row in rows if 3 <= int(row["length"]) <= 15]
        # The retained multisource training candidate uses PepFlow for 3--15 aa
        # and AF2/ColabFold only for 16--30 aa.  The 31--50 aa AF2 experiment was
        # a rejected pilot, so expanding AF2 to every 3--50 aa test peptide would
        # both waste server work and create a train/test structure-source shift.
        af2 = [row for row in rows if 16 <= int(row["length"]) <= 30]
        smoke_pepflow = pepflow[:2]
        # Exercise both a short and a longer sequence when possible.
        smoke_af2 = [af2[0], af2[-1]] if len(af2) > 1 else af2
        dataset_summary = {
            "processed": str(processed),
            "total_records": len(rows),
            "pepflow_03_15": _write_task_family(
                pepflow, output_root, f"{dataset}_pepflow_03_15"
            ),
            "af2_16_30": _write_task_family(
                af2, output_root, f"{dataset}_af2_16_30", chunk_size=chunk_size
            ),
            "smoke_pepflow": _write_task_family(
                smoke_pepflow, output_root, f"{dataset}_pepflow_smoke"
            ),
            "smoke_af2": _write_task_family(
                smoke_af2, output_root, f"{dataset}_af2_smoke", chunk_size=chunk_size
            ),
        }
        summary["datasets"][dataset] = dataset_summary
    (output_root / "task_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test1-processed",
        default="data/processed/test1_chemical_sites_final_esm2.pt",
    )
    parser.add_argument(
        "--test2-processed",
        default="data/processed/test2_chemical_sites_final_esm2.pt",
    )
    parser.add_argument("--output-root", default="data/test_conformers/tasks")
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    summary = prepare(
        Path(args.test1_processed),
        Path(args.test2_processed),
        Path(args.output_root),
        int(args.chunk_size),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
