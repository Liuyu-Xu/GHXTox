"""Create an audited fold manifest for a ProtTox-TSK source payload.

The retained ProtT5 training payload and GHXTox group-fold manifest use
different sample IDs.  This tool permits the mapping only after exact,
one-to-one sequence and label checks; it never infers an ID mapping by row
order.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from ghxtox.prottox_tsk import file_sha256, write_manifest


def _source_records(path: str | Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not all(key in payload for key in ("sample_ids", "sequences", "labels")):
        raise ValueError("Source payload must contain sample_ids, sequences, and labels")
    ids, sequences, labels = payload["sample_ids"], payload["sequences"], payload["labels"]
    if not (len(ids) == len(sequences) == len(labels)):
        raise ValueError("Source sample_ids, sequences, and labels lengths differ")
    records = [{"source_index": index, "source_sample_id": str(sample_id), "sequence": str(sequence), "label": int(label)} for index, (sample_id, sequence, label) in enumerate(zip(ids, sequences, labels))]
    if len({record["source_sample_id"] for record in records}) != len(records):
        raise ValueError("Source sample IDs are not unique")
    if len({record["sequence"] for record in records}) != len(records):
        raise ValueError("Source sequences are not unique; exact sequence mapping would be ambiguous")
    return records


def build_manifest(source_path: str | Path, historical_manifest: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = _source_records(source_path)
    with Path(historical_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        historical = list(csv.DictReader(handle))
    required = {"sample_id", "sequence", "label", "group_id", "fold"}
    if not historical or not required <= set(historical[0]):
        raise ValueError(f"Historical fold manifest must contain {sorted(required)}")
    if len(historical) != len(source):
        raise ValueError(f"Record count mismatch: source={len(source)}, manifest={len(historical)}")
    historical_by_sequence = {str(row["sequence"]): row for row in historical}
    if len(historical_by_sequence) != len(historical):
        raise ValueError("Historical manifest sequences are not unique; mapping would be ambiguous")
    output_rows: list[dict[str, Any]] = []
    for record in source:
        row = historical_by_sequence.get(record["sequence"])
        if row is None:
            raise ValueError(f"Source sequence has no historical fold record: {record['source_sample_id']}")
        if int(row["label"]) != record["label"]:
            raise ValueError(f"Label mismatch for source {record['source_sample_id']} and historical {row['sample_id']}")
        output_rows.append({"source_index": record["source_index"], "sample_id": record["source_sample_id"], "sequence": record["sequence"], "label": record["label"], "group_id": row["group_id"], "fold": int(row["fold"]), "historical_sample_id": row["sample_id"]})
    folds = sorted({int(row["fold"]) for row in output_rows})
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in output_rows:
        group_folds[str(row["group_id"])].add(int(row["fold"]))
    split_groups = [group for group, assigned in group_folds.items() if len(assigned) != 1]
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"Expected five folds numbered 0..4, got {folds}")
    if split_groups:
        raise ValueError(f"Group leakage detected, e.g. {split_groups[:3]}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_index", "sample_id", "sequence", "label", "group_id", "fold", "historical_sample_id"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    audit = {"source": {"path": str(source_path), "sha256": file_sha256(source_path), "records": len(source), "label_counts": dict(sorted(Counter(row["label"] for row in source).items()))}, "historical_manifest": {"path": str(historical_manifest), "sha256": file_sha256(historical_manifest), "records": len(historical)}, "output": {"path": str(destination), "sha256": file_sha256(destination), "records": len(output_rows), "fold_counts": dict(sorted(Counter(row["fold"] for row in output_rows).items()))}, "checks": {"sequence_mapping": "exact_one_to_one", "label_mapping": "exact", "group_fold_leakage": False, "folds": folds}}
    write_manifest(str(destination) + ".audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--historical-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = build_manifest(args.source, args.historical_manifest, args.output)
    print(f"Audited {audit['output']['records']} exact source-to-fold mappings: {audit['output']['path']}")


if __name__ == "__main__":
    main()
