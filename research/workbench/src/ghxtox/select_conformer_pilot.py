"""Select a deterministic fold/label/topology-stratified PepFlow pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import torch

from ghxtox.fasta import read_fasta
from ghxtox.peptide_topology_audit import _risk_class


def _read_folds(path: str | Path) -> dict[int, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {int(row["source_index"]): row for row in csv.DictReader(handle)}


def _read_excluded_indices(path: str | Path | None) -> set[int]:
    if path is None:
        return set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "source_index" not in rows[0]:
        raise ValueError(f"Exclusion manifest lacks source_index: {path}")
    return {int(row["source_index"]) for row in rows}


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def _read_mean_plddt(
    path: str | Path | None,
    sequences: list[str],
) -> list[float | None]:
    if path is None:
        return [None] * len(sequences)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    if len(records) != len(sequences):
        raise ValueError("Processed structure data and FASTA have different lengths.")
    values = []
    for index, (record, sequence) in enumerate(zip(records, sequences)):
        if str(record["sequence"]) != sequence:
            raise ValueError(f"Processed structure sequence mismatch at index {index}.")
        values.append(float(record["plddt"].float().mean()))
    return values


def _length_bin(
    length: int,
    bins: list[tuple[int, int]],
) -> str | None:
    for lower, upper in bins:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    return None


def _round_robin_take(
    candidates: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        strata.setdefault(
            (str(row["confidence_bin"]), str(row["topology_risk"])),
            [],
        ).append(row)
    for values in strata.values():
        values.sort(key=lambda row: _stable_rank(str(row["sample_id"]), seed))
    selected = []
    keys = sorted(strata)
    while len(selected) < count:
        progressed = False
        for key in keys:
            values = strata[key]
            if values and len(selected) < count:
                selected.append(values.pop(0))
                progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} samples from a primary stratum with only "
            f"{len(selected)} available."
        )
    return selected


def select_pilot(
    fasta: str | Path,
    fold_manifest: str | Path,
    output_fasta: str | Path,
    output_manifest: str | Path,
    target_size: int = 250,
    min_length: int = 3,
    max_length: int = 30,
    seed: int = 2026,
    exclude_manifest: str | Path | None = None,
    processed: str | Path | None = None,
    length_bins: list[tuple[int, int]] | None = None,
    plddt_threshold: float = 0.70,
) -> list[dict[str, Any]]:
    records = read_fasta(fasta)
    folds = _read_folds(fold_manifest)
    excluded_indices = _read_excluded_indices(exclude_manifest)
    mean_plddt = _read_mean_plddt(
        processed,
        [record.sequence for record in records],
    )
    strata: dict[tuple[int, int | None, str], list[dict[str, Any]]] = {}
    for source_index, record in enumerate(records):
        fold_row = folds.get(source_index)
        if fold_row is None:
            raise ValueError(f"Missing fold assignment for source_index={source_index}.")
        if source_index in excluded_indices:
            continue
        if not min_length <= len(record.sequence) <= max_length:
            continue
        cysteines = record.sequence.count("C")
        risk = _risk_class(cysteines, len(record.sequence), False, False)
        bin_name = (
            _length_bin(len(record.sequence), length_bins)
            if length_bins is not None
            else f"{min_length}-{max_length}"
        )
        if bin_name is None:
            continue
        confidence_bin = (
            "unknown"
            if mean_plddt[source_index] is None
            else (
                "low"
                if float(mean_plddt[source_index]) < float(plddt_threshold)
                else "high"
            )
        )
        row = {
            "source_index": source_index,
            "sample_id": str(fold_row["sample_id"]),
            "label": record.label,
            "fold": int(fold_row["fold"]),
            "sequence": record.sequence,
            "length": len(record.sequence),
            "cysteines": cysteines,
            "topology_risk": risk,
            "length_bin": bin_name,
            "mean_plddt": mean_plddt[source_index],
            "confidence_bin": confidence_bin,
        }
        strata.setdefault((row["fold"], row["label"], risk), []).append(row)

    selected: list[dict[str, Any]]
    if length_bins is not None:
        fold_values = sorted({int(row["fold"]) for values in strata.values() for row in values})
        label_values = sorted(
            {
                int(row["label"])
                for values in strata.values()
                for row in values
                if row["label"] is not None
            }
        )
        primary_count = len(length_bins) * len(fold_values) * len(label_values)
        if primary_count == 0 or target_size % primary_count != 0:
            raise ValueError(
                "Balanced length-bin selection requires target_size divisible by "
                "length_bins x folds x labels."
            )
        quota = target_size // primary_count
        all_rows = [row for values in strata.values() for row in values]
        selected = []
        for lower, upper in length_bins:
            bin_name = f"{lower}-{upper}"
            for fold in fold_values:
                for label in label_values:
                    candidates = [
                        row
                        for row in all_rows
                        if row["length_bin"] == bin_name
                        and int(row["fold"]) == fold
                        and int(row["label"]) == label
                    ]
                    selected.extend(
                        _round_robin_take(
                            candidates,
                            quota,
                            seed + lower * 100 + fold * 10 + label,
                        )
                    )
    else:
        for values in strata.values():
            values.sort(key=lambda row: _stable_rank(str(row["sample_id"]), seed))
        selected = []
        keys = sorted(
            strata,
            key=lambda key: (key[0], -1 if key[1] is None else int(key[1]), key[2]),
        )
        while len(selected) < target_size:
            progressed = False
            for key in keys:
                values = strata[key]
                if values and len(selected) < target_size:
                    selected.append(values.pop(0))
                    progressed = True
            if not progressed:
                break
    selected.sort(key=lambda row: int(row["source_index"]))
    if not selected:
        raise ValueError("No sequence passed the requested pilot length filters.")

    output_fasta = Path(output_fasta)
    output_manifest = Path(output_manifest)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w", encoding="utf-8") as handle:
        for row in selected:
            label = "" if row["label"] is None else f"|{int(row['label'])}"
            handle.write(f">{row['sample_id']}{label}\n{row['sequence']}\n")
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-fasta", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--target-size", type=int, default=250)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--exclude-manifest",
        help="Optional CSV whose source_index rows have already been generated.",
    )
    parser.add_argument(
        "--processed",
        help="Optional aligned processed cache used to stratify by mean pLDDT.",
    )
    parser.add_argument(
        "--length-bins",
        help="Optional comma-separated inclusive bins, for example 31-35,36-40.",
    )
    parser.add_argument("--plddt-threshold", type=float, default=0.70)
    args = parser.parse_args()
    length_bins = None
    if args.length_bins:
        length_bins = []
        for value in args.length_bins.split(","):
            lower, upper = value.split("-", maxsplit=1)
            length_bins.append((int(lower), int(upper)))
    rows = select_pilot(
        args.fasta,
        args.fold_manifest,
        args.output_fasta,
        args.output_manifest,
        target_size=args.target_size,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
        exclude_manifest=args.exclude_manifest,
        processed=args.processed,
        length_bins=length_bins,
        plddt_threshold=args.plddt_threshold,
    )
    print(
        f"Selected {len(rows)} peptides across "
        f"{len(set((r['fold'], r['label'], r['length_bin'], r['confidence_bin'], r['topology_risk']) for r in rows))} strata."
    )


if __name__ == "__main__":
    main()
