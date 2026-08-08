#!/usr/bin/env python
"""Verify frozen GHXTox interpretability tables and provenance checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


FUSION_WEIGHT_3D = 0.28
EXPECTED_EXPERT_COUNTS = {
    ("Test1", "both_correct"): 1032,
    ("Test1", "1D_only_correct"): 29,
    ("Test1", "3D_only_correct"): 23,
    ("Test1", "both_incorrect"): 42,
    ("Test2", "both_correct"): 529,
    ("Test2", "1D_only_correct"): 13,
    ("Test2", "3D_only_correct"): 14,
    ("Test2", "both_incorrect"): 26,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "paper_figures" / "data" / "current_interpretability" / "frozen_results",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(first: float, second: float, tolerance: float = 2e-8) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=tolerance)


def _sample_key(row: dict[str, str]) -> tuple[str, str]:
    return row["split_name"], row["sample_id"]


def _verify_output_checksums(result_dir: Path) -> int:
    checked = 0
    for line in (result_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = result_dir / relative
        assert path.exists(), path
        assert _sha256(path) == expected, path
        checked += 1
    return checked


def _verify_input_manifest(root: Path, result_dir: Path) -> int:
    checked = 0
    for row in _read_csv(result_dir / "input_manifest.csv"):
        if row["input_type"] != "local_file":
            continue
        candidate = Path(row["input_identifier"])
        path = candidate if candidate.is_absolute() else root / candidate
        assert path.exists(), path
        assert _sha256(path) == row["sha256"], path
        assert path.stat().st_size == int(row["size_bytes"]), path
        checked += 1
    return checked


def _verify_sequence_tables(result_dir: Path, cohort_keys: set[tuple[str, str]]) -> dict[str, Any]:
    residues = _read_csv(result_dir / "sequence_residue_occlusion.csv")
    samples = _read_csv(result_dir / "sequence_sample_summary.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in residues:
        grouped[_sample_key(row)].append(row)
        baseline = float(row["baseline_probability_1d"])
        occluded = float(row["occluded_probability_1d"])
        delta = float(row["delta_probability_1d_baseline_minus_occluded"])
        assert _close(delta, baseline - occluded)
        agreement = float(row["seed_sign_agreement_fraction"])
        assert -1e-12 <= agreement <= 1.0 + 1e-12
    assert set(grouped) == cohort_keys
    assert {_sample_key(row) for row in samples} == cohort_keys
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["residue_position_1based"]))
        sequence = rows[0]["sequence"]
        assert [int(row["residue_position_1based"]) for row in rows] == list(
            range(1, len(sequence) + 1)
        ), key
        assert "".join(row["residue"] for row in rows) == sequence, key
        normalized_l1 = sum(abs(float(row["signed_l1_normalized_attribution"])) for row in rows)
        raw_l1 = sum(abs(float(row["delta_probability_1d_baseline_minus_occluded"])) for row in rows)
        if raw_l1 > 0:
            assert _close(normalized_l1, 1.0, 2e-7), (key, normalized_l1)
    return {"sample_count": len(grouped), "residue_row_count": len(residues)}


def _verify_gate_tables(result_dir: Path) -> dict[str, Any]:
    samples = _read_csv(result_dir / "plddt_gate_samples.csv")
    residues = _read_csv(result_dir / "plddt_gate_residues.csv")
    sample_map = {_sample_key(row): row for row in samples}
    assert len(sample_map) == len(samples) == 1708
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in residues:
        key = _sample_key(row)
        assert key in sample_map
        confidence = float(row["plddt_scaled_0_to_1"])
        gate = float(row["learned_node_gate"])
        assert -1e-8 <= confidence <= 1.0 + 1e-8
        assert -1e-8 <= gate <= 1.0 + 1e-8
        grouped[key].append(row)
    assert set(grouped) == set(sample_map)
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row["residue_position_1based"]))
        sequence = rows[0]["sequence"]
        assert len(rows) == len(sequence), key
        assert "".join(row["residue"] for row in rows) == sequence, key
        assert int(sample_map[key]["sequence_length"]) == len(sequence), key
    return {"sample_count": len(samples), "residue_row_count": len(residues)}


def _verify_deletion_table(path: Path, *, site_level: bool) -> int:
    rows = _read_csv(path)
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        delta_3d = float(row["delta_probability_3d_baseline_minus_deleted"])
        baseline = float(row["baseline_probability_3d"])
        deleted = float(row["deleted_probability_3d"])
        delta_fused = float(row["delta_fused_probability_holding_1d_fixed"])
        assert _close(delta_3d, baseline - deleted)
        assert _close(delta_fused, FUSION_WEIGHT_3D * delta_3d)
        if site_level:
            baseline_fused = float(row["baseline_fused_probability"])
            deleted_fused = float(row["deleted_fused_probability_holding_1d_fixed"])
            assert _close(deleted_fused, baseline_fused - delta_fused)
            key = (
                *_sample_key(row),
                int(row["residue_position_1based"]),
                int(row["site_slot_1based"]),
            )
        elif "site_category" in row:
            key = (*_sample_key(row), row["site_category"])
        else:
            key = (*_sample_key(row), int(row["residue_position_1based"]))
        assert key not in seen, key
        seen.add(key)
    assert rows
    return len(rows)


def _verify_expert_counts(result_dir: Path) -> dict[str, int]:
    rows = _read_csv(result_dir / "expert_agreement_summary.csv")
    observed = {
        (row["split_name"], row["expert_outcome_group"]): int(row["sample_count"])
        for row in rows
    }
    assert observed == EXPECTED_EXPERT_COUNTS
    return {f"{split_name}/{group}": count for (split_name, group), count in observed.items()}


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    result_dir = args.result_dir.resolve()
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "full"
    assert summary["manuscript_admissible"] is True
    assert summary["one_d_baseline_validation"]["passed"] is True
    assert summary["three_d_baseline_validation"]["passed"] is True
    cohort = _read_csv(result_dir / "representative_cohort.csv")
    cohort_keys = {_sample_key(row) for row in cohort}
    assert len(cohort) == len(cohort_keys) == 16
    assert all(int(row["selection_uses_attribution"]) == 0 for row in cohort)
    assert {
        (row["split_name"], row["expert_outcome_group"], int(row["label"])) for row in cohort
    } == {
        (split_name, group, label)
        for split_name in ("Test1", "Test2")
        for group in ("both_correct", "1D_only_correct", "3D_only_correct", "both_incorrect")
        for label in (0, 1)
    }
    report = {
        "status": "passed",
        "output_checksum_count": _verify_output_checksums(result_dir),
        "input_manifest_checksum_count": _verify_input_manifest(root, result_dir),
        "cohort_sample_count": len(cohort),
        "sequence": _verify_sequence_tables(result_dir, cohort_keys),
        "gate": _verify_gate_tables(result_dir),
        "individual_site_deletion_rows": _verify_deletion_table(
            result_dir / "structure_site_deletion.csv", site_level=True
        ),
        "category_deletion_rows": _verify_deletion_table(
            result_dir / "structure_category_deletion.csv", site_level=False
        ),
        "residue_deletion_rows": _verify_deletion_table(
            result_dir / "structure_residue_deletion.csv", site_level=False
        ),
        "expert_counts": _verify_expert_counts(result_dir),
    }
    expected_counts = summary["result_row_counts"]
    assert report["sequence"]["residue_row_count"] == expected_counts["sequence_residue_occlusion"]
    assert report["gate"]["sample_count"] == expected_counts["plddt_gate_samples"]
    assert report["gate"]["residue_row_count"] == expected_counts["plddt_gate_residues"]
    assert report["individual_site_deletion_rows"] == expected_counts["structure_site_deletion"]
    assert report["category_deletion_rows"] == expected_counts["structure_category_deletion"]
    assert report["residue_deletion_rows"] == expected_counts["structure_residue_deletion"]
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
