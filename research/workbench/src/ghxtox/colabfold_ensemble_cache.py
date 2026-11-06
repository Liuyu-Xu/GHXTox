"""Collect ColabFold multi-seed PDBs and build GHXTox conformer caches.

ColabFold writes one PDB per model/seed.  The existing GHXTox ensemble
converter consumes one multi-model PDB per peptide, so this module provides a
deterministic bridge between both layouts and then reuses ``conformer_cache``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from ghxtox.conformer_cache import build_conformer_cache
from ghxtox.fasta import read_fasta


_RANK_PATTERN = re.compile(r"(?:^|_)rank_(\d+)(?:_|\.)", re.IGNORECASE)
_SEED_PATTERN = re.compile(r"(?:^|_)seed_(\d+)(?:_|\.)", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"(?:^|_)model_(\d+)(?:_|\.)", re.IGNORECASE)


def _sort_key(path: Path) -> tuple[int, int, int, str]:
    name = path.name

    def value(pattern: re.Pattern[str], fallback: int) -> int:
        match = pattern.search(name)
        return int(match.group(1)) if match else fallback

    return (
        value(_RANK_PATTERN, 10**9),
        value(_SEED_PATTERN, 10**9),
        value(_MODEL_PATTERN, 10**9),
        name,
    )


def _result_roots(results_root: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(results_root, (str, Path)):
        return [Path(results_root)]
    return [Path(root) for root in results_root]


def discover_colabfold_pdbs(
    results_root: str | Path | Iterable[str | Path],
    sample_id: str,
) -> list[Path]:
    """Find a sample's prediction PDBs, preferring unrelaxed outputs.

    ColabFold prefixes outputs with the FASTA identifier.  Matching requires
    the following underscore, so ``peptide_9`` cannot capture ``peptide_90``.
    """

    prefix = f"{sample_id}_"
    candidates = sorted(
        (
            path
            for root in _result_roots(results_root)
            for path in root.rglob("*.pdb")
            if path.name.startswith(prefix) and path.stat().st_size > 0
        ),
        key=_sort_key,
    )
    unrelaxed = [path for path in candidates if "_unrelaxed_" in path.name]
    if unrelaxed:
        return unrelaxed
    relaxed = [path for path in candidates if "_relaxed_" in path.name]
    return relaxed or candidates


def _backbone_coordinates_and_plddt(path: str | Path) -> tuple[np.ndarray, float]:
    coords: list[list[float]] = []
    confidence: list[float] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() not in {"N", "CA", "C"}:
            continue
        if len(line) > 16 and line[16] not in {" ", "A"}:
            continue
        try:
            coords.append(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            )
            if line[12:16].strip() == "CA":
                confidence.append(float(line[60:66]))
        except ValueError as exc:
            raise ValueError(f"Malformed backbone record in {path}: {line}") from exc
    if not coords:
        raise ValueError(f"ColabFold PDB contains no backbone records: {path}")
    if not confidence:
        raise ValueError(f"ColabFold PDB contains no CA confidence records: {path}")
    return np.asarray(coords, dtype=np.float64), float(np.mean(confidence))


def _aligned_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("nan")
    first_centered = first - first.mean(axis=0, keepdims=True)
    second_centered = second - second.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(first_centered.T @ second_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = first_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - second_centered) ** 2, axis=-1))))


def select_colabfold_pdbs(
    paths: list[Path],
    max_conformers: int,
    selection_mode: str = "diversity",
    min_mean_plddt: float = 50.0,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Select a quality-controlled, non-redundant AF2 ensemble.

    Rank-only selection tends to retain near-duplicate high-confidence models.
    Diversity selection starts from the highest-confidence model and then uses
    farthest-point sampling on Kabsch-aligned C-alpha RMSD.  The pLDDT cutoff is
    strict when enough candidates pass it and otherwise degrades gracefully to
    the highest-confidence candidates so short/disordered peptides are not
    silently discarded.
    """

    limit = max(int(max_conformers), 0)
    candidates: list[dict[str, Any]] = []
    for rank, path in enumerate(paths, start=1):
        coords, mean_plddt = _backbone_coordinates_and_plddt(path)
        candidates.append(
            {
                "path": path,
                "rank": rank,
                "mean_plddt": mean_plddt,
                "coords": coords,
                "passes_plddt": mean_plddt >= float(min_mean_plddt),
            }
        )
    if not candidates or limit == 0:
        return [], []

    candidates.sort(key=lambda item: (-item["mean_plddt"], item["rank"], str(item["path"])))
    qualified = [item for item in candidates if item["passes_plddt"]]
    if len(qualified) >= limit:
        pool = qualified
    else:
        # Preserve a fixed ensemble width for downstream tensors, but record
        # that the quality fallback was needed.
        pool = qualified + [item for item in candidates if not item["passes_plddt"]]

    if selection_mode == "rank":
        selected = sorted(pool, key=lambda item: item["rank"])[:limit]
    elif selection_mode == "confidence":
        selected = pool[:limit]
    elif selection_mode == "diversity":
        count = len(pool)
        distances = np.zeros((count, count), dtype=np.float64)
        for first in range(count):
            for second in range(first + 1, count):
                value = _aligned_rmsd(pool[first]["coords"], pool[second]["coords"])
                if not np.isfinite(value):
                    value = 1e6
                distances[first, second] = distances[second, first] = value

        # Highest-confidence initialization followed by farthest-point seeds.
        # A few deterministic PAM updates then replace seeds by within-cluster
        # medoids, so rare outliers do not automatically represent a cluster.
        medoids = [0]
        while len(medoids) < min(limit, count):
            remaining = [index for index in range(count) if index not in medoids]
            chosen = max(
                remaining,
                key=lambda index: (
                    float(np.min(distances[index, medoids])),
                    pool[index]["mean_plddt"],
                    -pool[index]["rank"],
                ),
            )
            medoids.append(chosen)
        assignments = np.argmin(distances[:, medoids], axis=1)
        for _ in range(20):
            updated: list[int] = []
            for cluster_index, current in enumerate(medoids):
                members = np.flatnonzero(assignments == cluster_index).tolist()
                if not members:
                    updated.append(current)
                    continue
                updated.append(
                    min(
                        members,
                        key=lambda index: (
                            float(distances[index, members].sum()),
                            -pool[index]["mean_plddt"],
                            pool[index]["rank"],
                        ),
                    )
                )
            if updated == medoids:
                break
            medoids = updated
            assignments = np.argmin(distances[:, medoids], axis=1)
        selected = [pool[index] for index in medoids]
        cluster_sizes = {
            id(pool[medoid]): int(np.sum(assignments == cluster_index))
            for cluster_index, medoid in enumerate(medoids)
        }
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")

    metadata: list[dict[str, Any]] = []
    for order, item in enumerate(selected):
        previous = selected[:order]
        min_rmsd = None
        if previous:
            distances = [
                _aligned_rmsd(item["coords"], chosen["coords"])
                for chosen in previous
            ]
            finite = [value for value in distances if np.isfinite(value)]
            min_rmsd = min(finite) if finite else None
        metadata.append(
            {
                "path": str(item["path"]),
                "original_rank": int(item["rank"]),
                "mean_plddt": float(item["mean_plddt"]),
                "passes_plddt": bool(item["passes_plddt"]),
                "min_rmsd_to_previous": min_rmsd,
                "cluster_size": (
                    cluster_sizes.get(id(item), 1)
                    if selection_mode == "diversity"
                    else 1
                ),
            }
        )
    return [item["path"] for item in selected], metadata


def _atom_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.startswith(("ATOM", "TER"))]


def write_multimodel_pdb(paths: list[Path], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for model_index, path in enumerate(paths, start=1):
            atoms = _atom_lines(path)
            if not any(line.startswith("ATOM") for line in atoms):
                raise ValueError(f"ColabFold PDB contains no ATOM records: {path}")
            handle.write(f"MODEL     {model_index:4d}\n")
            for line in atoms:
                handle.write(line.rstrip() + "\n")
            handle.write("ENDMDL\n")
        handle.write("END\n")
    temporary.replace(output)


def _read_task_manifest(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_index", "sample_id", "label", "sequence"}
    if rows and not required.issubset(rows[0]):
        missing = sorted(required.difference(rows[0]))
        raise ValueError(f"Task manifest is missing columns: {missing}")
    return {str(row["sample_id"]): row for row in rows}


def collect_colabfold_ensemble(
    fasta: str | Path,
    results_root: str | Path | Iterable[str | Path],
    raw_root: str | Path,
    task_manifest: str | Path | None = None,
    max_conformers: int = 8,
    min_conformers: int = 8,
    selection_mode: str = "diversity",
    min_mean_plddt: float = 50.0,
    on_missing: str = "error",
    overwrite: bool = False,
) -> dict[str, Any]:
    records = read_fasta(fasta)
    manifest = _read_task_manifest(task_manifest)
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        if manifest:
            task = manifest.get(record.sample_id)
            if task is None:
                raise ValueError(f"FASTA sample is absent from task manifest: {record.sample_id}")
            if task["sequence"] != record.sequence or int(task["label"]) != record.label:
                raise ValueError(f"FASTA/task mismatch for {record.sample_id}")
        candidates = discover_colabfold_pdbs(results_root, record.sample_id)
        selected, selection = select_colabfold_pdbs(
            candidates,
            max_conformers=max_conformers,
            selection_mode=selection_mode,
            min_mean_plddt=min_mean_plddt,
        )
        output = raw_root / record.sample_id / f"{record.sequence}.pdb"
        row: dict[str, Any] = {
            "sample_id": record.sample_id,
            "sequence": record.sequence,
            "label": record.label,
            "discovered": len(candidates),
            "selected": len(selected),
            "selection": selection,
            "output_pdb": str(output),
            "source_pdbs": [str(path) for path in selected],
        }
        if len(selected) < int(min_conformers):
            row["status"] = "insufficient_conformers"
            rows.append(row)
            if on_missing == "error":
                raise FileNotFoundError(
                    f"{record.sample_id} has {len(selected)} ColabFold PDBs; "
                    f"at least {min_conformers} are required."
                )
            continue
        if output.exists() and output.stat().st_size > 0 and not overwrite:
            row["status"] = "cached"
        else:
            write_multimodel_pdb(selected, output)
            row["status"] = "collected"
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "method": "ColabFold AlphaFold2 shallow-MSA multi-seed peptide ensemble",
        "fasta": str(Path(fasta)),
        "task_manifest": str(task_manifest) if task_manifest is not None else None,
        "results_roots": [str(path) for path in _result_roots(results_root)],
        "raw_root": str(raw_root),
        "max_conformers": int(max_conformers),
        "min_conformers": int(min_conformers),
        "selection_mode": selection_mode,
        "min_mean_plddt": float(min_mean_plddt),
        "counts": counts,
        "records": rows,
    }
    with (raw_root / "colabfold_collection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def collect_and_cache(
    fasta: str | Path,
    results_root: str | Path | Iterable[str | Path],
    raw_root: str | Path,
    cache_root: str | Path,
    task_manifest: str | Path | None = None,
    max_conformers: int = 8,
    min_conformers: int = 8,
    selection_mode: str = "diversity",
    min_mean_plddt: float = 50.0,
    on_missing: str = "error",
    overwrite: bool = False,
) -> dict[str, Any]:
    collection = collect_colabfold_ensemble(
        fasta=fasta,
        results_root=results_root,
        raw_root=raw_root,
        task_manifest=task_manifest,
        max_conformers=max_conformers,
        min_conformers=min_conformers,
        selection_mode=selection_mode,
        min_mean_plddt=min_mean_plddt,
        on_missing=on_missing,
        overwrite=overwrite,
    )
    cache = build_conformer_cache(
        fasta=fasta,
        pdb_root=raw_root,
        output_root=cache_root,
        pdb_pattern="{sample_id}/{sequence}.pdb",
        source="colabfold_af2_shallow_msa_multiseed",
        confidence_mode="pdb",
        max_conformers=max_conformers,
        on_missing=on_missing,
        on_mismatch=on_missing,
        sample_id_mode="header",
    )
    return {"collection": collection, "cache": cache}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--task-manifest")
    parser.add_argument(
        "--results-root",
        required=True,
        action="append",
        help="ColabFold result directory; repeat for dropout and shallow-MSA pools.",
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--max-conformers", type=int, default=8)
    parser.add_argument("--min-conformers", type=int, default=8)
    parser.add_argument(
        "--selection-mode",
        choices=["rank", "confidence", "diversity"],
        default="diversity",
    )
    parser.add_argument("--min-mean-plddt", type=float, default=50.0)
    parser.add_argument("--on-missing", choices=["error", "skip"], default="error")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = collect_and_cache(
        fasta=args.fasta,
        task_manifest=args.task_manifest,
        results_root=args.results_root,
        raw_root=args.raw_root,
        cache_root=args.cache_root,
        max_conformers=args.max_conformers,
        min_conformers=args.min_conformers,
        selection_mode=args.selection_mode,
        min_mean_plddt=args.min_mean_plddt,
        on_missing=args.on_missing,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "collection_counts": summary["collection"]["counts"],
                "samples_with_conformers": summary["cache"]["samples_with_conformers"],
                "total_conformers": summary["cache"]["total_conformers"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
