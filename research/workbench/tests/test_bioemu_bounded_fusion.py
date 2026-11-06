import csv
import json
from pathlib import Path

import numpy as np

from ghxtox.bioemu_bounded_fusion import evaluate_bioemu_bounded_fusion


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, value: float, length: int) -> None:
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        residue_rmsf=np.full(length, value, dtype=np.float32),
        conformer_rmsd=np.asarray([value], dtype=np.float32),
        radius_gyration=np.asarray([value], dtype=np.float32),
        contact_occupancy_8=np.full((length, length), value / 10, dtype=np.float32),
    )


def test_bounded_fusion_preserves_folds_and_rejects_non_improvement(
    tmp_path: Path,
) -> None:
    frozen = []
    manifest = []
    one_d = []
    feasibility = []
    cache_root = tmp_path / "cache"
    source_index = 0
    for fold in (0, 1):
        for length_bin, length in (
            ("11-15", 12),
            ("16-30", 20),
            ("31-50", 40),
        ):
            for label in (0, 1):
                sample_id = f"peptide_{source_index}"
                frozen.append(
                    {
                        "source_index": source_index,
                        "sample_id": sample_id,
                        "sequence": "A" * length,
                        "label": label,
                        "fold": fold,
                        "length": length,
                        "length_bin": length_bin,
                        "requested_conformers": 32,
                    }
                )
                manifest.append(
                    {
                        "source_index": source_index,
                        "sample_id": sample_id,
                        "original_id": sample_id,
                        "sequence": "A" * length,
                        "label": label,
                        "fold": fold,
                        "conformer_count": 32,
                        "cache_path": str(
                            cache_root / sample_id / "ensemble_summary.npz"
                        ),
                    }
                )
                one_d.append(
                    {
                        "source_index": source_index,
                        "label": label,
                        "toxicity_probability": 0.9 if label else 0.1,
                    }
                )
                feasibility.append({"sample_id": sample_id, "usable": True})
                _write_summary(
                    cache_root / sample_id / "ensemble_summary.npz",
                    float(length),
                    length,
                )
                source_index += 1

    frozen_path = tmp_path / "frozen.csv"
    manifest_path = tmp_path / "manifest.csv"
    one_d_path = tmp_path / "one_d.csv"
    feasibility_path = tmp_path / "feasibility.json"
    _write_csv(frozen_path, frozen)
    _write_csv(manifest_path, manifest)
    _write_csv(one_d_path, one_d)
    feasibility_path.write_text(json.dumps({"samples": feasibility}), encoding="utf-8")

    report, rows, decision = evaluate_bioemu_bounded_fusion(
        frozen_path,
        manifest_path,
        cache_root,
        one_d_path,
        feasibility_path,
    )

    assert len(rows) == 12
    assert report["protocol"]["canonical_folds"] == [0, 1]
    assert report["primary_all_200"]["one_d"]["accuracy"] == 1.0
    assert report["retention_gate"]["mcc_strictly_higher"] is False
    assert decision["decision"] == "stop_bounded_bioemu_candidate"
    assert decision["full_generation_authorized"] is False
    assert decision["external_test_labels_read"] is False
