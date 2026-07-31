import csv
from collections import Counter
from pathlib import Path

import pytest

from ghxtox.bioemu_pilot import (
    MODEL_NAME,
    PACKAGE_VERSION,
    build_sample_command,
    decide_feasibility,
    select_bioemu_pilot,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_select_bioemu_pilot_is_exact_and_near_balanced(tmp_path: Path) -> None:
    rows = []
    index = 0
    for length in (12, 20, 40):
        for fold in range(5):
            for label in (0, 1):
                for _ in range(7):
                    rows.append(
                        {
                            "source_index": index,
                            "sample_id": f"peptide_{index + 1}",
                            "label": label,
                            "sequence": "A" * length,
                            "group_id": f"g{index}",
                            "fold": fold,
                        }
                    )
                    index += 1
    folds = tmp_path / "folds.csv"
    _write_csv(folds, rows)
    selected, smoke = select_bioemu_pilot(
        folds,
        tmp_path / "pilot.csv",
        tmp_path / "pilot.fasta",
        tmp_path / "smoke.csv",
        tmp_path / "smoke.fasta",
    )

    assert len(selected) == 200
    assert len(smoke) == 6
    assert {row["cell_quota"] for row in selected} == {6, 7}
    assert Counter(row["length_bin"] for row in selected) == {
        "11-15": 67,
        "16-30": 67,
        "31-50": 66,
    }
    assert Counter(row["label"] for row in selected) == {0: 100, 1: 100}
    assert Counter(row["fold"] for row in selected) == dict.fromkeys(range(5), 40)
    assert {(row["length_bin"], row["label"]) for row in smoke} == {
        (length_bin, label)
        for length_bin in ("11-15", "16-30", "31-50")
        for label in (0, 1)
    }
    assert all(row["bioemu_package"] == PACKAGE_VERSION for row in selected)
    assert all(row["bioemu_model"] == MODEL_NAME for row in selected)


def test_bioemu_command_is_seeded_filtered_and_resumable_by_output_dir() -> None:
    row = {"sample_id": "peptide_1", "sequence": "ACDE", "source_index": 0}
    command = build_sample_command(row, "/tmp/bioemu/peptide_1", python="python3.11")
    assert command[:3] == ["python3.11", "-m", "bioemu.sample"]
    assert f"--model_name={MODEL_NAME}" in command
    assert "--filter_samples=True" in command
    assert any(value.startswith("--base_seed=") for value in command)
    assert "--output_dir=/tmp/bioemu/peptide_1" in command


def _feasible_row(length_bin: str, length: int) -> dict:
    return {
        "sample_id": f"{length_bin}-{length}",
        "length_bin": length_bin,
        "length": length,
        "elapsed_seconds": 20.0,
        "storage_bytes": 10_000_000,
        "returncode": 0,
        "valid_conformers": 16,
        "sequence_match": True,
        "residue_count_match": True,
        "median_ca_rmsd_nm": 0.2,
        "median_radius_gyration_nm": 0.8,
        "max_adjacent_ca_distance_nm": 0.40,
        "error": "",
    }


def test_bioemu_feasibility_never_authorizes_full_generation() -> None:
    rows = [
        _feasible_row("11-15", 12),
        _feasible_row("16-30", 20),
        _feasible_row("31-50", 40),
    ]
    report = decide_feasibility(rows)
    assert report["decision"] == "continue_bioemu_200_pilot"
    assert report["full_generation_authorized"] is False

    failed = [dict(row) for row in rows]
    failed[0]["valid_conformers"] = 2
    report = decide_feasibility(failed)
    assert report["decision"] == "switch_to_esmflow_small_pilot"


def test_bioemu_selection_rejects_undersupplied_cell(tmp_path: Path) -> None:
    folds = tmp_path / "folds.csv"
    _write_csv(
        folds,
        [
            {
                "source_index": 0,
                "sample_id": "peptide_1",
                "label": 0,
                "sequence": "A" * 12,
                "group_id": "g0",
                "fold": 0,
            }
        ],
    )
    with pytest.raises(ValueError, match="needs"):
        select_bioemu_pilot(
            folds,
            tmp_path / "pilot.csv",
            tmp_path / "pilot.fasta",
            tmp_path / "smoke.csv",
            tmp_path / "smoke.fasta",
        )
