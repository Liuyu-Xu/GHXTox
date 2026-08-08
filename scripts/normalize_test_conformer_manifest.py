"""Audit and normalize a dataset-scoped conformer manifest for GHXTox use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize(task_manifest: Path, conformer_manifest: Path, output: Path) -> None:
    tasks = {row["sample_id"]: row for row in _read(task_manifest)}
    rows = _read(conformer_manifest)
    if not rows:
        raise ValueError(f"Empty conformer manifest: {conformer_manifest}")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for row in rows:
        scoped_id = row["sample_id"]
        task = tasks.get(scoped_id)
        if task is None:
            raise ValueError(f"Unexpected scoped ID in conformer manifest: {scoped_id}")
        sequence = row["sequence"].strip().upper()
        expected_hash = task["sequence_sha256"]
        observed_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        if sequence != task["sequence"] or observed_hash != expected_hash:
            raise ValueError(f"Sequence mismatch for {scoped_id}")
        if int(row["label"]) != int(task["label"]):
            raise ValueError(f"Label mismatch for {scoped_id}")
        updated = dict(row)
        updated["original_id"] = task["target_sample_id"]
        updated["dataset"] = task["dataset"]
        updated["scoped_sample_id"] = scoped_id
        updated["target_sample_id"] = task["target_sample_id"]
        updated["sequence_sha256"] = expected_hash
        normalized.append(updated)
        seen.add(scoped_id)
    missing = sorted(set(tasks) - seen)
    if missing:
        raise ValueError(f"Missing conformer outputs for {len(missing)} tasks, e.g. {missing[:3]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    fieldnames = list(normalized[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized)
    os.replace(temporary, output)
    print(
        f"Normalized {len(normalized)} conformers for {len(seen)} samples: {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--conformer-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    normalize(Path(args.task_manifest), Path(args.conformer_manifest), Path(args.output))


if __name__ == "__main__":
    main()
