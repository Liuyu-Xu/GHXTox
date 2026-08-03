"""Attach one or more conformer-ensemble sources to processed GHXTox records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


CONFORMER_RESIDUE_FEATURE_DIM = 8
CONFORMER_GLOBAL_FEATURE_DIM = 12
CONFORMER_SOURCE_FEATURE_DIM = 3
CONFORMER_SOURCE_NAMES = ("pepflow", "colabfold_af2", "other")
CONFORMER_GRAPH_MAX_CONFORMERS = 8
CONFORMER_GRAPH_MAX_SITE_SLOTS = 2


def _off_diagonal_mean(matrix: np.ndarray) -> np.ndarray:
    length = int(matrix.shape[0])
    if length <= 1:
        return np.zeros(length, dtype=np.float32)
    return (matrix.sum(axis=1) - np.diag(matrix)) / float(length - 1)


def ensemble_feature_tensors(summary_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw ensemble statistics into bounded, scale-normalized tensors."""

    with np.load(summary_path) as payload:
        rmsf = payload["residue_rmsf"].astype(np.float32)
        pair_mean = payload["pair_distance_mean"].astype(np.float32)
        pair_std = payload["pair_distance_std"].astype(np.float32)
        occupancies = [
            payload[f"contact_occupancy_{cutoff}"].astype(np.float32)
            for cutoff in (6, 8, 10)
        ]
        conformer_rmsd = payload["conformer_rmsd"].astype(np.float32)
        radius_gyration = payload["radius_gyration"].astype(np.float32)
        num_conformers = int(payload["aligned_coords"].shape[0])

    residue_features = np.stack(
        [
            rmsf / 10.0,
            _off_diagonal_mean(pair_mean) / 20.0,
            pair_mean.std(axis=1) / 20.0,
            _off_diagonal_mean(pair_std) / 5.0,
            *[_off_diagonal_mean(value) for value in occupancies],
            _off_diagonal_mean(occupancies[1] * (1.0 - occupancies[1])),
        ],
        axis=1,
    )
    global_features = np.asarray(
        [
            min(num_conformers, 8) / 8.0,
            rmsf.mean() / 10.0,
            rmsf.std() / 10.0,
            rmsf.max(initial=0.0) / 10.0,
            conformer_rmsd.mean() / 10.0,
            conformer_rmsd.std() / 10.0,
            conformer_rmsd.max(initial=0.0) / 10.0,
            radius_gyration.mean() / 20.0,
            radius_gyration.std() / 20.0,
            pair_std.mean() / 5.0,
            pair_std.max(initial=0.0) / 10.0,
            np.mean(occupancies[1] * (1.0 - occupancies[1])),
        ],
        dtype=np.float32,
    )
    residue_features = np.nan_to_num(residue_features, nan=0.0, posinf=3.0, neginf=-3.0)
    global_features = np.nan_to_num(global_features, nan=0.0, posinf=3.0, neginf=-3.0)
    return torch.from_numpy(residue_features), torch.from_numpy(global_features)


def conformer_graph_tensors(
    summary_path: str | Path,
    max_conformers: int = CONFORMER_GRAPH_MAX_CONFORMERS,
) -> dict[str, torch.Tensor]:
    """Load aligned per-conformer chemical-site graphs without copying PDB files."""

    paths = sorted(Path(summary_path).parent.glob("conformer_*.npz"))[:max_conformers]
    if not paths:
        raise FileNotFoundError(f"No conformer NPZ files beside {summary_path}.")
    values: dict[str, list[np.ndarray]] = {
        "residue_coords": [],
        "plddt": [],
        "site_coords": [],
        "site_types": [],
        "site_mask": [],
    }
    sequence = None
    for path in paths:
        with np.load(path) as payload:
            current_sequence = str(payload["sequence"].item())
            if sequence is None:
                sequence = current_sequence
            elif current_sequence != sequence:
                raise ValueError(f"Conformer sequence mismatch in {path}.")
            values["residue_coords"].append(payload["coords"].astype(np.float32))
            values["plddt"].append(payload["plddt"].astype(np.float32))
            values["site_coords"].append(
                payload["chemical_site_coords"][:, :CONFORMER_GRAPH_MAX_SITE_SLOTS].astype(
                    np.float32
                )
            )
            values["site_types"].append(
                payload["chemical_site_types"][:, :CONFORMER_GRAPH_MAX_SITE_SLOTS].astype(
                    np.float32
                )
            )
            values["site_mask"].append(
                payload["chemical_site_mask"][:, :CONFORMER_GRAPH_MAX_SITE_SLOTS].astype(
                    np.bool_
                )
            )
    tensors = {
        key: torch.from_numpy(np.stack(items))
        for key, items in values.items()
    }
    tensors["conformer_mask"] = torch.ones(len(paths), dtype=torch.bool)
    return tensors


def _source_feature(source: str) -> torch.Tensor:
    normalized = source.strip().lower()
    if "pepflow" in normalized:
        index = 0
    elif "colabfold" in normalized or "alphafold" in normalized or normalized.startswith("af2"):
        index = 1
    else:
        index = 2
    feature = torch.zeros(CONFORMER_SOURCE_FEATURE_DIM, dtype=torch.float32)
    feature[index] = 1.0
    return feature


def _manifest_summaries(
    manifest: str | Path,
    cache_root: str | Path,
) -> dict[str, tuple[Path, str]]:
    cache_root = Path(cache_root)
    summaries: dict[str, tuple[Path, str]] = {}
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            original_id = str(row.get("original_id") or row["sample_id"])
            indexed_id = str(row["sample_id"])
            source = str(row.get("source") or "unknown_conformer_generator").strip()
            # Full-cohort archives may combine several cache roots.  Prefer a
            # portable manifest cache_path when it resolves locally; legacy
            # server-absolute paths transparently fall back to cache_root.
            cache_path_text = str(row.get("cache_path") or "").strip()
            path = cache_root / indexed_id / "ensemble_summary.npz"
            if cache_path_text:
                cache_path = Path(cache_path_text)
                manifest_summary = cache_path.parent / "ensemble_summary.npz"
                if manifest_summary.exists():
                    path = manifest_summary
            if not path.exists():
                raise FileNotFoundError(f"Missing ensemble summary for {original_id}: {path}")
            previous = summaries.setdefault(original_id, (path, source))
            if previous != (path, source):
                raise ValueError(f"Multiple ensemble summaries were mapped to {original_id}.")
    return summaries


def _manifest_sources(manifest: str | Path) -> list[str]:
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        sources = {
            str(row.get("source") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("source") or "").strip()
        }
    return sorted(sources) or ["unknown_conformer_generator"]


def _as_paths(value: str | Path | Sequence[str | Path]) -> list[str | Path]:
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def attach_conformer_features(
    processed: str | Path,
    conformer_manifest: str | Path | Sequence[str | Path],
    cache_root: str | Path | Sequence[str | Path],
    output: str | Path,
    pilot_output: str | Path | None = None,
    pilot_fold_manifest: str | Path | None = None,
    source_fold_manifest: str | Path | None = None,
    include_conformer_graphs: bool = False,
) -> dict[str, Any]:
    """Attach descriptors by original sample ID and optionally create pilot-only files."""

    payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = payload["records"]
    manifests = _as_paths(conformer_manifest)
    cache_roots = _as_paths(cache_root)
    if len(cache_roots) == 1 and len(manifests) > 1:
        cache_roots *= len(manifests)
    if len(manifests) != len(cache_roots):
        raise ValueError(
            "conformer_manifest and cache_root must have equal counts, or one cache_root "
            "must be supplied for all manifests."
        )
    summaries: dict[str, tuple[Path, str]] = {}
    sources: set[str] = set()
    for manifest, root in zip(manifests, cache_roots):
        current = _manifest_summaries(manifest, root)
        overlap = sorted(set(summaries).intersection(current))
        if overlap:
            raise ValueError(
                f"Conformer manifests overlap on {len(overlap)} original IDs, e.g. {overlap[:3]}"
            )
        summaries.update(current)
        sources.update(_manifest_sources(manifest))
    covered_indices: list[int] = []
    for source_index, record in enumerate(records):
        sample_id = str(record["sample_id"])
        summary_entry = summaries.get(sample_id)
        length = int(record["aa_ids"].shape[0])
        if summary_entry is None:
            record["conformer_residue_features"] = torch.zeros(
                length, CONFORMER_RESIDUE_FEATURE_DIM, dtype=torch.float32
            )
            record["conformer_global_features"] = torch.zeros(
                CONFORMER_GLOBAL_FEATURE_DIM, dtype=torch.float32
            )
            record["conformer_available"] = False
            record["conformer_source_features"] = torch.zeros(
                CONFORMER_SOURCE_FEATURE_DIM, dtype=torch.float32
            )
            record["conformer_source"] = "none"
            if include_conformer_graphs:
                record["conformer_graph_expected"] = True
            continue
        summary_path, source = summary_entry
        residue_features, global_features = ensemble_feature_tensors(summary_path)
        if residue_features.shape[0] != length:
            raise ValueError(
                f"Length mismatch for {sample_id}: processed={length}, ensemble={residue_features.shape[0]}"
            )
        record["conformer_residue_features"] = residue_features
        record["conformer_global_features"] = global_features
        record["conformer_available"] = True
        record["conformer_summary_path"] = str(summary_path)
        record["conformer_source_features"] = _source_feature(source)
        record["conformer_source"] = source
        if include_conformer_graphs:
            graph = conformer_graph_tensors(summary_path)
            if int(graph["residue_coords"].shape[1]) != length:
                raise ValueError(
                    f"Conformer graph length mismatch for {sample_id}: "
                    f"processed={length}, graph={graph['residue_coords'].shape[1]}"
                )
            record["conformer_graph_residue_coords"] = graph["residue_coords"].half()
            record["conformer_graph_plddt"] = graph["plddt"].half()
            record["conformer_graph_site_coords"] = graph["site_coords"].half()
            record["conformer_graph_site_types"] = graph["site_types"].half()
            record["conformer_graph_site_mask"] = graph["site_mask"]
            record["conformer_graph_conformer_mask"] = graph["conformer_mask"]
            record["conformer_graph_expected"] = True
        covered_indices.append(source_index)

    missing_ids = sorted(set(summaries) - {str(record["sample_id"]) for record in records})
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} manifest IDs are absent from processed data, e.g. {missing_ids[:3]}")
    payload.setdefault("metadata", {})["conformer_ensemble"] = {
        "source": sorted(sources)[0] if len(sources) == 1 else sorted(sources),
        "manifest": [str(value) for value in manifests],
        "cache_root": [str(value) for value in cache_roots],
        "covered_records": len(covered_indices),
        "residue_feature_dim": CONFORMER_RESIDUE_FEATURE_DIM,
        "global_feature_dim": CONFORMER_GLOBAL_FEATURE_DIM,
        "source_feature_dim": CONFORMER_SOURCE_FEATURE_DIM,
        "source_feature_names": CONFORMER_SOURCE_NAMES,
        "missing_policy": "zero features with hard availability mask",
        "per_conformer_graphs": bool(include_conformer_graphs),
        "max_conformers": CONFORMER_GRAPH_MAX_CONFORMERS,
        "max_site_slots": CONFORMER_GRAPH_MAX_SITE_SLOTS,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    if pilot_output is not None:
        pilot_records = [records[index] for index in covered_indices]
        pilot_payload = {
            "records": pilot_records,
            "metadata": {
                **payload.get("metadata", {}),
                "source_indices": covered_indices,
                "subset": "conformer-covered training subset",
            },
        }
        pilot_output = Path(pilot_output)
        pilot_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pilot_payload, pilot_output)

        if pilot_fold_manifest is not None:
            if source_fold_manifest is None:
                raise ValueError("source_fold_manifest is required with pilot_fold_manifest.")
            folds: dict[int, dict[str, str]] = {}
            with Path(source_fold_manifest).open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    folds[int(row["source_index"])] = row
            pilot_fold_manifest = Path(pilot_fold_manifest)
            pilot_fold_manifest.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["source_index", "original_source_index", "sample_id", "label", "sequence", "group_id", "fold"]
            with pilot_fold_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for pilot_index, original_index in enumerate(covered_indices):
                    source_row = folds[original_index]
                    writer.writerow(
                        {
                            "source_index": pilot_index,
                            "original_source_index": original_index,
                            "sample_id": records[original_index]["sample_id"],
                            "label": records[original_index]["label"],
                            "sequence": records[original_index]["sequence"],
                            "group_id": source_row.get("group_id", ""),
                            "fold": source_row["fold"],
                        }
                    )

    return {
        "processed": str(processed),
        "output": str(output),
        "total_records": len(records),
        "covered_records": len(covered_indices),
        "coverage_fraction": len(covered_indices) / max(len(records), 1),
        "pilot_output": str(pilot_output) if pilot_output is not None else None,
        "pilot_fold_manifest": str(pilot_fold_manifest) if pilot_fold_manifest is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--conformer-manifest", nargs="+", required=True)
    parser.add_argument("--cache-root", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output")
    parser.add_argument("--pilot-fold-manifest")
    parser.add_argument("--source-fold-manifest")
    parser.add_argument("--include-conformer-graphs", action="store_true")
    args = parser.parse_args()
    summary = attach_conformer_features(
        args.processed,
        args.conformer_manifest,
        args.cache_root,
        args.output,
        pilot_output=args.pilot_output,
        pilot_fold_manifest=args.pilot_fold_manifest,
        source_fold_manifest=args.source_fold_manifest,
        include_conformer_graphs=args.include_conformer_graphs,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
