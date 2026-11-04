"""Convert filtered BioEmu XTC trajectories into GHXTox ensemble summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.conformer_cache import ensemble_descriptors
from ghxtox.features import clean_sequence
from ghxtox.utils import save_json


def bioemu_ensemble_descriptors(ca_coords_nm: np.ndarray) -> dict[str, np.ndarray]:
    """Build the existing Angstrom-based descriptors from BioEmu coordinates."""

    coords = np.asarray(ca_coords_nm, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(
            f"Expected BioEmu C-alpha coordinates [K,L,3], got {coords.shape}."
        )
    return ensemble_descriptors(coords * 10.0)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_single_fasta(path: Path) -> str:
    return clean_sequence(
        "".join(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith(">")
        )
    )


def _expected_valid_counts(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(row["sample_id"]): int(row["valid_conformers"])
        for row in payload["samples"]
    }


def build_bioemu_conformer_cache(
    manifest: str | Path,
    bioemu_root: str | Path,
    output_root: str | Path,
    *,
    feasibility_json: str | Path | None = None,
    source: str = "bioemu-v1.2",
) -> dict[str, Any]:
    """Convert every filtered trajectory while preserving the frozen manifest."""

    try:
        import mdtraj as md
    except ImportError as error:
        raise RuntimeError(
            "BioEmu XTC conversion requires mdtraj (validated with 1.10.3)."
        ) from error

    rows = _read_csv(manifest)
    root = Path(bioemu_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    expected_counts = _expected_valid_counts(feasibility_json)
    manifest_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        sample_id = str(row["sample_id"])
        sequence = clean_sequence(str(row["sequence"]))
        sample_root = root / sample_id
        fasta_path = sample_root / "sequence.fasta"
        topology_path = sample_root / "topology.pdb"
        trajectory_path = sample_root / "samples.xtc"
        if not (
            fasta_path.is_file()
            and topology_path.is_file()
            and trajectory_path.is_file()
        ):
            raise FileNotFoundError(f"Missing BioEmu core output for {sample_id}.")
        if _read_single_fasta(fasta_path) != sequence:
            raise ValueError(f"BioEmu FASTA mismatch for {sample_id}.")

        trajectory = md.load_xtc(str(trajectory_path), top=str(topology_path))
        ca_indices = trajectory.topology.select("name CA")
        if len(ca_indices) != len(sequence):
            raise ValueError(
                f"C-alpha count mismatch for {sample_id}: "
                f"{len(ca_indices)} != {len(sequence)}."
            )
        if trajectory.n_frames <= 0:
            raise ValueError(
                f"BioEmu trajectory has no filtered frames for {sample_id}."
            )
        expected = expected_counts.get(sample_id)
        if expected is not None and trajectory.n_frames != expected:
            raise ValueError(
                f"Filtered-frame mismatch for {sample_id}: "
                f"{trajectory.n_frames} != {expected}."
            )

        descriptors = bioemu_ensemble_descriptors(trajectory.xyz[:, ca_indices, :])
        sample_output = output / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        summary_path = sample_output / "ensemble_summary.npz"
        np.savez_compressed(summary_path, **descriptors)
        manifest_rows.append(
            {
                "source_index": int(row["source_index"]),
                "sample_id": sample_id,
                "original_id": sample_id,
                "sequence": sequence,
                "label": int(row["label"]),
                "fold": int(row["fold"]),
                "length": len(sequence),
                "length_stratum": str(row["length_bin"]),
                "conformer_count": int(trajectory.n_frames),
                "source": source,
                "confidence_imputed": 1,
                "cache_path": str(summary_path),
                "source_pdb": str(topology_path),
                "source_xtc": str(trajectory_path),
            }
        )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "length": len(sequence),
                "conformers": int(trajectory.n_frames),
                "mean_residue_rmsf": float(descriptors["residue_rmsf"].mean()),
                "mean_conformer_rmsd": float(descriptors["conformer_rmsd"].mean()),
                "mean_radius_gyration": float(descriptors["radius_gyration"].mean()),
            }
        )

    manifest_path = output / "conformer_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "method": source,
        "manifest": str(Path(manifest)),
        "bioemu_root": str(root),
        "coordinate_input_units": "nm",
        "descriptor_output_units": "angstrom",
        "samples": len(sample_rows),
        "total_filtered_conformers": int(sum(row["conformers"] for row in sample_rows)),
        "conformer_manifest": str(manifest_path),
        "sample_summaries": sample_rows,
    }
    save_json(summary, output / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bioemu-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feasibility-json")
    parser.add_argument("--source", default="bioemu-v1.2")
    args = parser.parse_args()
    build_bioemu_conformer_cache(
        args.manifest,
        args.bioemu_root,
        args.output_root,
        feasibility_json=args.feasibility_json,
        source=args.source,
    )


if __name__ == "__main__":
    main()
