from pathlib import Path

import numpy as np

from ghxtox.colabfold_ensemble_cache import (
    collect_and_cache,
    discover_colabfold_pdbs,
    select_colabfold_pdbs,
)


def _pdb(offset: float, confidence: float = 80.0) -> str:
    return "\n".join(
        [
            f"ATOM      1  N   ALA A   1    {offset + 0:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           N",
            f"ATOM      2  CA  ALA A   1    {offset + 1:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           C",
            f"ATOM      3  C   ALA A   1    {offset + 2:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           C",
            f"ATOM      4  O   ALA A   1    {offset + 3:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           O",
            f"ATOM      5  CB  ALA A   1    {offset + 1:8.3f}{1:8.3f}{0:8.3f}  1.00{confidence:6.2f}           C",
            f"ATOM      6  N   CYS A   2    {offset + 3:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           N",
            f"ATOM      7  CA  CYS A   2    {offset + 4:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           C",
            f"ATOM      8  C   CYS A   2    {offset + 5:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           C",
            f"ATOM      9  O   CYS A   2    {offset + 6:8.3f}{0:8.3f}{0:8.3f}  1.00{confidence:6.2f}           O",
            f"ATOM     10  CB  CYS A   2    {offset + 4:8.3f}{1:8.3f}{0:8.3f}  1.00{confidence:6.2f}           C",
            f"ATOM     11  SG  CYS A   2    {offset + 4:8.3f}{2:8.3f}{0:8.3f}  1.00{confidence:6.2f}           S",
            "TER",
            "END",
        ]
    )


def test_colabfold_outputs_are_ranked_and_converted(tmp_path: Path) -> None:
    fasta = tmp_path / "pilot.fasta"
    fasta.write_text(">peptide_9|1\nAC\n", encoding="utf-8")
    manifest = tmp_path / "pilot.csv"
    manifest.write_text(
        "source_index,sample_id,label,sequence\n8,peptide_9,1,AC\n",
        encoding="utf-8",
    )
    results = tmp_path / "predictions"
    results.mkdir()
    for rank in (3, 1, 2):
        path = results / (
            f"peptide_9_unrelaxed_rank_{rank:03d}_"
            f"alphafold2_ptm_model_1_seed_{rank:03d}.pdb"
        )
        path.write_text(_pdb(float(rank), confidence=70.0 + rank), encoding="utf-8")
    # A similarly prefixed identifier must not be captured.
    (results / "peptide_90_unrelaxed_rank_001_model_1_seed_001.pdb").write_text(
        _pdb(10.0), encoding="utf-8"
    )
    discovered = discover_colabfold_pdbs(results, "peptide_9")
    assert ["rank_001" in path.name for path in discovered] == [True, False, False]

    summary = collect_and_cache(
        fasta=fasta,
        task_manifest=manifest,
        results_root=results,
        raw_root=tmp_path / "raw",
        cache_root=tmp_path / "cache",
        max_conformers=2,
        min_conformers=2,
    )
    assert summary["cache"]["samples_with_conformers"] == 1
    assert summary["cache"]["total_conformers"] == 2
    first = np.load(tmp_path / "cache" / "peptide_9" / "conformer_0000.npz")
    assert not bool(first["confidence_imputed"])
    # Diversity selection starts from the highest-confidence representative,
    # rather than blindly retaining rank_001.
    assert np.allclose(first["plddt"], 0.73)


def test_colabfold_collection_can_skip_incomplete_samples(tmp_path: Path) -> None:
    fasta = tmp_path / "pilot.fasta"
    fasta.write_text(">peptide_9|1\nAC\n", encoding="utf-8")
    results = tmp_path / "predictions"
    results.mkdir()
    (results / "peptide_9_unrelaxed_rank_001_model_1_seed_001.pdb").write_text(
        _pdb(0.0), encoding="utf-8"
    )
    summary = collect_and_cache(
        fasta=fasta,
        results_root=results,
        raw_root=tmp_path / "raw",
        cache_root=tmp_path / "cache",
        max_conformers=8,
        min_conformers=8,
        on_missing="skip",
    )
    assert summary["collection"]["counts"] == {"insufficient_conformers": 1}
    assert summary["cache"]["samples_with_conformers"] == 0


def _backbone_only_pdb(points: list[tuple[float, float, float]], confidence: float) -> str:
    lines = []
    serial = 1
    for residue, (x, y, z) in enumerate(points, start=1):
        for atom, dx in (("N", -0.5), ("CA", 0.0), ("C", 0.5)):
            element = atom[0]
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} ALA A{residue:4d}    "
                f"{x + dx:8.3f}{y:8.3f}{z:8.3f}  1.00{confidence:6.2f}          {element:>2s}"
            )
            serial += 1
    return "\n".join(lines + ["TER", "END"])


def test_diversity_selection_keeps_distinct_backbone_medoid(tmp_path: Path) -> None:
    traces = [
        ([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], 90.0),
        ([(0, 0, 0), (1, 0.02, 0), (2, 0, 0), (3, 0.02, 0)], 85.0),
        ([(0, 0, 0), (1, 0.5, 0), (1.5, 1.5, 0), (1.5, 2.5, 0)], 70.0),
    ]
    paths = []
    for rank, (trace, confidence) in enumerate(traces, start=1):
        path = tmp_path / f"pep_unrelaxed_rank_{rank:03d}_model_1_seed_001.pdb"
        path.write_text(_backbone_only_pdb(trace, confidence), encoding="utf-8")
        paths.append(path)
    selected, metadata = select_colabfold_pdbs(
        paths,
        max_conformers=2,
        selection_mode="diversity",
        min_mean_plddt=50.0,
    )
    assert paths[2] in selected
    assert len(selected) == 2
    assert sum(item["cluster_size"] for item in metadata) == 3
