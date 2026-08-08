"""Strictly audit server-generated Test1/Test2 conformer caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(
    task_manifest: Path,
    cache_root: Path,
    expected_conformers: int,
    strict: bool,
) -> dict[str, object]:
    tasks = {row["sample_id"]: row for row in _read(task_manifest)}
    manifest_path = cache_root / "conformer_manifest.csv"
    rows = _read(manifest_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    failures: list[str] = []
    counts = Counter()
    for scoped_id, task in tasks.items():
        sample_rows = grouped.get(scoped_id, [])
        counts[len(sample_rows)] += 1
        if len(sample_rows) != expected_conformers:
            failures.append(
                f"{scoped_id}: expected {expected_conformers} conformers, got {len(sample_rows)}"
            )
            continue
        expected_sequence = task["sequence"]
        expected_hash = task["sequence_sha256"]
        for row in sample_rows:
            if row.get("dataset") != task["dataset"]:
                failures.append(f"{scoped_id}: dataset tag mismatch")
            if row.get("original_id") != task["target_sample_id"]:
                failures.append(f"{scoped_id}: target ID mapping mismatch")
            sequence = row["sequence"].strip().upper()
            if sequence != expected_sequence:
                failures.append(f"{scoped_id}: manifest sequence mismatch")
            if hashlib.sha256(sequence.encode("ascii")).hexdigest() != expected_hash:
                failures.append(f"{scoped_id}: sequence hash mismatch")
        sample_root = cache_root / scoped_id
        summary_path = sample_root / "ensemble_summary.npz"
        conformer_paths = sorted(sample_root.glob("conformer_*.npz"))
        if len(conformer_paths) != expected_conformers:
            failures.append(f"{scoped_id}: NPZ count={len(conformer_paths)}")
        if not summary_path.exists():
            failures.append(f"{scoped_id}: missing ensemble_summary.npz")
            continue
        with np.load(summary_path) as payload:
            if int(payload["aligned_coords"].shape[0]) != expected_conformers:
                failures.append(f"{scoped_id}: summary conformer count mismatch")
            if int(payload["aligned_coords"].shape[1]) != len(expected_sequence):
                failures.append(f"{scoped_id}: summary sequence length mismatch")
    unexpected = sorted(set(grouped) - set(tasks))
    if unexpected:
        failures.append(f"Unexpected sample IDs: {unexpected[:3]}")
    result = {
        "status": "ok" if not failures else "failed",
        "task_manifest": str(task_manifest),
        "cache_root": str(cache_root),
        "expected_samples": len(tasks),
        "manifest_samples": len(grouped),
        "manifest_conformers": len(rows),
        "conformer_count_distribution": dict(sorted(counts.items())),
        "failures": failures,
    }
    (cache_root / "test_cache_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if strict and failures:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--expected-conformers", type=int, default=8)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    audit(
        Path(args.task_manifest),
        Path(args.cache_root),
        int(args.expected_conformers),
        bool(args.strict),
    )


if __name__ == "__main__":
    main()

