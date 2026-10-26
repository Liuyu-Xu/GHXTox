import csv
from pathlib import Path

import numpy as np

from ghxtox.conformer_cohort_analysis import _length_bucket, analyse_conformer_cohort


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_conformer_cohort_analysis_aligns_predictions_and_structure(tmp_path: Path) -> None:
    task_rows = []
    base_rows = []
    candidate_rows = []
    cache_root = tmp_path / "cache"
    for index, (label, base, candidate, length) in enumerate(
        [(1, 0.4, 0.8, 18), (0, 0.7, 0.2, 23), (1, 0.8, 0.9, 28)]
    ):
        sample_id = f"pep_{index}"
        sequence = "A" * length
        task_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "sequence": sequence,
                "length": length,
                "cysteines": 0,
                "topology_risk": "none",
            }
        )
        common = {"sample_id": sample_id, "fold": index, "label": label}
        base_rows.append({**common, "toxicity_probability": base})
        candidate_rows.append({**common, "toxicity_probability": candidate})
        sample_root = cache_root / sample_id
        sample_root.mkdir(parents=True)
        np.savez_compressed(
            sample_root / "ensemble_summary.npz",
            residue_rmsf=np.ones(length, dtype=np.float32),
            conformer_rmsd=np.asarray([0.5, 1.0], dtype=np.float32),
            radius_gyration=np.asarray([3.0, 4.0], dtype=np.float32),
        )
        for conformer in range(2):
            np.savez_compressed(
                sample_root / f"conformer_{conformer:04d}.npz",
                plddt=np.full(length, 0.8, dtype=np.float32),
            )

    task = tmp_path / "task.csv"
    baseline = tmp_path / "base.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(task, task_rows)
    _write_csv(baseline, base_rows)
    _write_csv(candidate, candidate_rows)

    report, rows = analyse_conformer_cohort(
        baseline,
        candidate,
        task,
        cache_root,
        threshold=0.5,
    )

    assert report["cohort"]["samples"] == 3
    assert report["cohort"]["conformer_caches"] == 6
    assert report["classification_transitions"]["net_corrections"] == 2
    assert len(rows) == 3
    assert set(report["length_strata"]) == {"16-20", "21-25", "26-30"}


def test_length_bucket_covers_long_training_peptides() -> None:
    expected = {
        15: "1-15",
        16: "16-20",
        20: "16-20",
        21: "21-25",
        25: "21-25",
        26: "26-30",
        30: "26-30",
        31: "31-35",
        35: "31-35",
        36: "36-40",
        40: "36-40",
        41: "41-45",
        45: "41-45",
        46: "46-50",
        50: "46-50",
        51: "51+",
    }
    assert {length: _length_bucket(length) for length in expected} == expected
