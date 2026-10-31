import csv
from pathlib import Path

import numpy as np

from ghxtox.bioemu_source_analysis import analyze_bioemu_source_shift


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(path: Path, value: float, length: int) -> None:
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        residue_rmsf=np.full(length, value, dtype=np.float32),
        conformer_rmsd=np.asarray([value], dtype=np.float32),
        radius_gyration=np.asarray([value], dtype=np.float32),
        contact_occupancy_8=np.full((length, length), value / 10, dtype=np.float32),
    )


def test_bioemu_source_analysis_uses_paired_canonical_folds(tmp_path: Path) -> None:
    frozen = []
    bio_manifest = []
    legacy_manifest = []
    bio_root = tmp_path / "bio"
    legacy_root = tmp_path / "legacy"
    source_index = 0
    for fold in (0, 1):
        for length_stratum, length, base in (
            ("11-15", 12, 1.0),
            ("16-30", 20, 2.0),
            ("31-50", 40, 3.0),
        ):
            for label in (0, 1):
                sample_id = f"peptide_{source_index}"
                common = {
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "original_id": sample_id,
                    "sequence": "A" * length,
                    "label": label,
                    "fold": fold,
                    "length": length,
                    "length_bin": length_stratum,
                }
                frozen.append(common)
                bio_manifest.append(
                    {
                        **common,
                        "cache_path": str(
                            bio_root / sample_id / "ensemble_summary.npz"
                        ),
                    }
                )
                legacy_manifest.append(
                    {
                        **common,
                        "cache_path": str(
                            legacy_root / sample_id / "ensemble_summary.npz"
                        ),
                    }
                )
                _summary(
                    bio_root / sample_id / "ensemble_summary.npz",
                    base,
                    length,
                )
                _summary(
                    legacy_root / sample_id / "ensemble_summary.npz",
                    base + 4.0,
                    length,
                )
                source_index += 1

    frozen_path = tmp_path / "frozen.csv"
    bio_path = tmp_path / "bio.csv"
    legacy_path = tmp_path / "legacy.csv"
    one_d_path = tmp_path / "one_d.csv"
    _write_csv(frozen_path, frozen)
    _write_csv(bio_path, bio_manifest)
    _write_csv(legacy_path, legacy_manifest)
    _write_csv(
        one_d_path,
        [
            {
                "source_index": row["source_index"],
                "label": row["label"],
                "toxicity_probability": 0.9 if row["label"] else 0.1,
            }
            for row in frozen
        ],
    )
    report, rows = analyze_bioemu_source_shift(
        frozen_path,
        bio_path,
        bio_root,
        [legacy_path],
        [legacy_root],
        ["Legacy"],
        permutations=20,
        one_d_oof=one_d_path,
    )
    assert len(rows) == 12
    assert report["cohort"] == {
        "bioemu_samples": 12,
        "legacy_overlap": 12,
        "legacy_missing": 0,
    }
    assert report["by_length"]["11-15"]["legacy_overlap"] == 4
    assert (
        report["feature_predictability"]["bioemu_vs_legacy_on_paired_samples"][
            "balanced_accuracy"
        ]
        == 1.0
    )
    assert (
        report["feature_predictability"]["toxicity_from_bioemu_features"]["samples"]
        == 12
    )
    assert (
        report["feature_predictability"]["toxicity_from_bioemu_features"][
            "permutation_test"
        ]["permutations"]
        == 20
    )
    assert report["one_d_anchor_comparison"]["overall"]["one_d_accuracy"] == 1.0
    assert report["one_d_anchor_comparison"]["overall"]["net_corrected"] == -6
    assert report["protocol"]["external_test_labels_read"] is False
