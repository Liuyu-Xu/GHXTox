import csv
import importlib.metadata
import json
import sys
from pathlib import Path

import pytest

from ghxtox.boundary_bioemu import (
    FIXED_THRESHOLD,
    TARGET_SIZE,
    prepare_boundary_cohort,
    run_boundary_manifest,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _handoff_rows() -> list[dict]:
    rows = []
    for index in range(6387):
        probability = (index + 1) / 6388
        rows.append(
            {
                "source_index": index,
                "sample_id": f"peptide_{index + 1}",
                "fold": index % 5,
                "label": index % 2,
                "sequence": "A" * (11 + index % 40),
                "toxicity_probability": probability,
                "logit": 0.0,
                "prediction": int(probability >= FIXED_THRESHOLD),
                "decision_threshold": FIXED_THRESHOLD,
                "generator_source": "synthetic",
            }
        )
    return rows


def test_boundary_selection_is_label_blind_and_server_manifest_has_no_labels(
    tmp_path: Path,
) -> None:
    rows = _handoff_rows()
    handoff = tmp_path / "handoff.csv"
    _write_csv(handoff, rows)
    first = tmp_path / "first"
    summary = prepare_boundary_cohort(handoff, first)

    selected = list(csv.DictReader((first / "selected_600.csv").open()))
    assert len(selected) == TARGET_SIZE
    assert summary["selection_was_label_blind"] is True
    assert min(summary["by_fold"].values()) >= 50
    assert "label" not in selected[0]
    assert all(
        float(selected[index]["absolute_probability_margin"])
        <= float(selected[index + 1]["absolute_probability_margin"])
        for index in range(len(selected) - 1)
    )

    for row in rows:
        row["label"] = 1 - int(row["label"])
    flipped = tmp_path / "flipped.csv"
    _write_csv(flipped, rows)
    second = tmp_path / "second"
    prepare_boundary_cohort(flipped, second)
    selected_flipped = list(csv.DictReader((second / "selected_600.csv").open()))
    assert [row["source_index"] for row in selected] == [
        row["source_index"] for row in selected_flipped
    ]


def test_boundary_prepare_reuses_only_verified_matching_pilot_output(
    tmp_path: Path,
) -> None:
    rows = _handoff_rows()
    selected_row = min(
        rows,
        key=lambda row: abs(float(row["toxicity_probability"]) - FIXED_THRESHOLD),
    )
    handoff = tmp_path / "handoff.csv"
    _write_csv(handoff, rows)

    previous_manifest = tmp_path / "pilot.csv"
    _write_csv(previous_manifest, [selected_row])
    previous_root = tmp_path / "pilot_output"
    sample_root = previous_root / selected_row["sample_id"]
    sample_root.mkdir(parents=True)
    (sample_root / "sequence.fasta").write_text(
        f">x\n{selected_row['sequence']}\n", encoding="utf-8"
    )
    (sample_root / "topology.pdb").write_bytes(b"pdb")
    (sample_root / "samples.xtc").write_bytes(b"xtc")
    (previous_root / "protocol.json").write_text(
        json.dumps(
            {
                "package_version": "1.4.1",
                "model_name": "bioemu-v1.2",
                "base_seed": 20260731,
                "num_samples_per_sequence": 32,
                "filter_samples": True,
            }
        ),
        encoding="utf-8",
    )
    previous_report = tmp_path / "pilot_report.json"
    previous_report.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": selected_row["sample_id"],
                        "usable": True,
                        "returncode": 0,
                        "elapsed_seconds": 50.0,
                        "storage_bytes": 300000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    summary = prepare_boundary_cohort(
        handoff,
        output,
        previous_report=previous_report,
        previous_output_root=previous_root,
        previous_manifest=previous_manifest,
    )
    assert summary["reuse_existing"] == 1
    assert summary["new_generation_required"] == TARGET_SIZE - 1
    reused = list(csv.DictReader((output / "reuse_existing.csv").open()))
    assert [row["sample_id"] for row in reused] == [selected_row["sample_id"]]


def test_boundary_runner_dry_run_is_label_free_and_requires_authorization(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        [
            {
                "source_index": 0,
                "sample_id": "peptide_1",
                "fold": 0,
                "sequence": "ACDEFGHIKLM",
                "length": 11,
                "length_bin": "11-15",
            }
        ],
    )
    report = run_boundary_manifest(
        manifest,
        tmp_path / "dry_run",
        dry_run=True,
        python="python3.11",
    )
    assert len(report["samples"]) == 1
    assert report["samples"][0]["returncode"] is None

    with pytest.raises(PermissionError, match="authorized"):
        run_boundary_manifest(manifest, tmp_path / "formal")

    labeled = tmp_path / "labeled.csv"
    _write_csv(labeled, [{"sample_id": "x", "label": 1}])
    with pytest.raises(ValueError, match="must not contain labels"):
        run_boundary_manifest(labeled, tmp_path / "bad", dry_run=True)


def test_boundary_runner_can_continue_after_sample_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.4.1")
    manifest = tmp_path / "manifest.csv"
    rows = []
    for index in range(3):
        rows.append(
            {
                "source_index": index,
                "sample_id": f"peptide_{index + 1}",
                "fold": index % 2,
                "sequence": "ACDEFGHIKLM",
                "length": 11,
                "length_bin": "11-15",
            }
        )
    _write_csv(manifest, rows)

    failing_python = tmp_path / (
        "fail_python.bat" if sys.platform == "win32" else "fail_python"
    )
    if sys.platform == "win32":
        failing_python.write_text("@exit /b 7\n", encoding="utf-8")
    else:
        failing_python.write_text("#!/usr/bin/env sh\nexit 7\n", encoding="utf-8")
        failing_python.chmod(0o755)
    stop_report = run_boundary_manifest(
        manifest,
        tmp_path / "stop",
        authorized=True,
        python=str(failing_python),
    )
    assert len(stop_report["samples"]) == 1
    assert stop_report["samples"][0]["returncode"] == 7

    continue_report = run_boundary_manifest(
        manifest,
        tmp_path / "continue",
        authorized=True,
        continue_on_error=True,
        python=str(failing_python),
    )
    assert len(continue_report["samples"]) == 3
    assert [sample["returncode"] for sample in continue_report["samples"]] == [
        7,
        7,
        7,
    ]
