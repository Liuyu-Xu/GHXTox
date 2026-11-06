from __future__ import annotations

from pathlib import Path

from ghxtox.surface_3d_short_specialist import (
    SURFACE_DESCRIPTOR_NAMES,
    evaluate_surface_candidate,
    surface_descriptor_vector,
)


def _write_two_model_pdb(path: Path) -> None:
    lines = []
    serial = 1
    residues = (("ALA", 1, 0.0), ("LYS", 2, 3.8), ("ASP", 3, 7.6))
    atoms = (("N", "N", 0.0), ("CA", "C", 1.2), ("C", "C", 2.3), ("O", "O", 3.0), ("CB", "C", 1.2))
    for model in (1, 2):
        lines.append(f"MODEL     {model:4d}")
        for resname, resid, offset in residues:
            for atom_name, element, atom_offset in atoms:
                x = offset + atom_offset
                y = float(model - 1) * 0.2
                lines.append(
                    f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} A{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{0.0:8.3f}  1.00 50.00          {element:>2s}"
                )
                serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def test_surface_descriptor_contract_uses_all_models(tmp_path: Path) -> None:
    pdb_path = tmp_path / "surface.pdb"
    _write_two_model_pdb(pdb_path)
    descriptors, metadata = surface_descriptor_vector(pdb_path, "AKD")
    assert len(SURFACE_DESCRIPTOR_NAMES) == 12
    assert descriptors.shape == (12,)
    assert metadata["models"] == 2
    assert metadata["hydrogens_removed"] == 0
    assert all(value == value for value in descriptors)
    assert descriptors[0] > 0.0


def test_short_specialist_preserves_every_non_target_row() -> None:
    reference_rows = []
    surface_rows = []
    for index in range(50):
        fold = index % 5
        label = index % 2
        target = index < 30
        reference_rows.append(
            {
                "source_index": str(index),
                "sample_id": f"peptide_{index + 1}",
                "fold": str(fold),
                "label": str(label),
                "sequence": "AKDECFILMNP" if target else "AKDECFILMNPQRSTVWYGH",
                "length": "11" if target else "20",
                "length_stratum": "11-15" if target else "16-30",
                "generator_source": "PepFlow" if target else "not_generated",
                "toxicity_probability": "0.75" if label else "0.25",
            }
        )
        if target:
            surface_rows.append(
                {
                    "source_index": index,
                    "sample_id": f"peptide_{index + 1}",
                    "fold": fold,
                    "label": label,
                    "sequence": "AKDECFILMNP",
                    "length": 11,
                    "length_stratum": "11-15",
                    "generator_source": "PepFlow",
                    **{
                        name: float(label) + feature * 0.001
                        for feature, name in enumerate(SURFACE_DESCRIPTOR_NAMES)
                    },
                }
            )
    report, rows = evaluate_surface_candidate(
        reference_rows,
        surface_rows,
        threshold=0.677819,
        residual_weight=0.1,
        regularization_c=0.1,
        random_seed=20260801,
    )
    assert report["external_test_labels_read"] is False
    assert report["target_rows"] == 30
    assert report["fallback_rows"] == 20
    for row in rows[30:]:
        assert row["surface_specialist_available"] == 0
        assert row["candidate_probability"] == row["reference_probability"]
        assert row["residual_logit_delta"] == 0.0
