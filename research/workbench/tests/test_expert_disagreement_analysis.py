from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "expert_disagreement_analysis_v1.json"
RESULT = ROOT / "runs" / "expert_disagreement_analysis_20260810_v1"
SCRIPT = ROOT / "scripts" / "analyze_expert_disagreement.py"
FIGURE = ROOT / "paper_figures" / "output" / "expert_disagreement_analysis"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _matches_frozen_bytes(path: Path, *, size_bytes: int, sha256: str) -> bool:
    """Validate a frozen input while tolerating Git's LF/CRLF checkout policy."""
    raw = path.read_bytes()
    candidates = [raw]
    if path.suffix.lower() in {".json", ".csv", ".md", ".txt"}:
        lf = raw.replace(b"\r\n", b"\n")
        candidates.extend((lf, lf.replace(b"\n", b"\r\n")))
    return any(
        len(candidate) == size_bytes
        and hashlib.sha256(candidate).hexdigest() == sha256
        for candidate in candidates
    )


def _cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    ordered = np.sort(np.asarray(second, dtype=np.float64))
    first = np.asarray(first, dtype=np.float64)
    less = np.searchsorted(ordered, first, side="left").sum()
    greater = (ordered.size - np.searchsorted(ordered, first, side="right")).sum()
    return float((less - greater) / (first.size * ordered.size))


def _expected_group(one_d_correct: bool, three_d_correct: bool) -> str:
    if one_d_correct and three_d_correct:
        return "both_correct"
    if one_d_correct:
        return "one_d_only_correct"
    if three_d_correct:
        return "three_d_only_correct"
    return "both_wrong"


def test_protocol_is_frozen_and_all_inputs_match() -> None:
    protocol = _json(PROTOCOL)
    assert protocol["protocol_status"] == "frozen_before_outcome_group_computation"
    assert protocol["expert_operating_points"]["one_d_threshold"] == 0.392
    assert protocol["expert_operating_points"]["three_d_threshold"] == 0.677819
    assert protocol["scope"]["retraining"] is False
    assert protocol["scope"]["threshold_search"] is False
    assert protocol["scope"]["pooled_inference_forbidden"] is True
    assert protocol["reporting"]["primary_contrast"] == (
        "three_d_only_correct minus one_d_only_correct"
    )
    assert protocol["resampling"]["bootstrap_replicates"] == 5000
    assert protocol["resampling"]["permutation_replicates"] == 10000
    missing_local_assets: list[str] = []
    for item in protocol["inputs"]:
        path = ROOT / item["path"]
        if not path.is_file():
            # The two multiconformer tensors are intentionally excluded from
            # Git because they are large, local reconstruction assets.  Keep
            # strict size/hash validation whenever they are present, while a
            # clean clone continues to validate all committed inputs and the
            # compact derived result tables in the remaining tests.
            assert path.suffix == ".pt"
            assert item["size_bytes"] >= 100_000_000
            assert len(item["sha256"]) == 64
            missing_local_assets.append(item["path"])
            continue
        assert _matches_frozen_bytes(
            path,
            size_bytes=item["size_bytes"],
            sha256=item["sha256"],
        )
    if missing_local_assets:
        pytest.skip(
            "Optional large local interpretability assets are absent: "
            + ", ".join(missing_local_assets)
        )


def test_all_sample_assignments_reconstruct_without_conformer_use() -> None:
    rows = _csv(RESULT / "sample_assignments.csv")
    assert len(rows) == 1708
    assert Counter(row["split_name"] for row in rows) == {"Test1": 1126, "Test2": 582}
    assert len({(row["split_name"], row["source_index"]) for row in rows}) == len(rows)
    assert len({(row["split_name"], row["sample_id"]) for row in rows}) == len(rows)
    for row in rows:
        label = int(row["label"])
        p_1d = float(row["p_1d"])
        p_3d = float(row["p_3d"])
        one_d_prediction = int(p_1d >= 0.392)
        three_d_prediction = int(p_3d >= 0.677819)
        assert int(row["one_d_prediction"]) == one_d_prediction
        assert int(row["three_d_prediction"]) == three_d_prediction
        assert int(row["one_d_correct"]) == int(one_d_prediction == label)
        assert int(row["three_d_correct"]) == int(three_d_prediction == label)
        assert row["outcome_group"] == _expected_group(
            one_d_prediction == label, three_d_prediction == label
        )
        assert int(row["peptide_length"]) == len(row["sequence"])
        assert int(row["cysteine_count"]) == row["sequence"].count("C")
        assert 0.0 <= float(row["mean_plddt"]) <= 1.0
        assert 0.0 <= float(row["chemical_site_coverage"]) <= 1.0
        availability = int(row["conformer_available"])
        length = int(row["peptide_length"])
        assert bool(availability) == (length <= 30)
        expected_source = (
            "pepflow"
            if length <= 15
            else "colabfold_af2_shallow_msa_multiseed"
            if length <= 30
            else "none"
        )
        assert row["conformer_source"] == expected_source
        assert row["canonical_prediction_uses_conformer"] == "0"


def test_group_counts_rescue_rates_and_primary_effects_recompute() -> None:
    rows = _csv(RESULT / "sample_assignments.csv")
    group_rows = _csv(RESULT / "group_counts.csv")
    observed_counts = {
        (row["split_name"], row["outcome_group"]): (
            int(row["sample_count"]),
            int(row["toxic_count"]),
        )
        for row in group_rows
    }
    expected_counts = {
        ("Test1", "both_correct"): (1032, 277),
        ("Test1", "one_d_only_correct"): (29, 15),
        ("Test1", "three_d_only_correct"): (24, 5),
        ("Test1", "both_wrong"): (41, 23),
        ("Test2", "both_correct"): (529, 33),
        ("Test2", "one_d_only_correct"): (13, 2),
        ("Test2", "three_d_only_correct"): (15, 1),
        ("Test2", "both_wrong"): (25, 10),
    }
    assert observed_counts == expected_counts
    for key, expected in expected_counts.items():
        split, group = key
        subset = [row for row in rows if row["split_name"] == split and row["outcome_group"] == group]
        assert (len(subset), sum(int(row["label"]) for row in subset)) == expected

    rescue = {
        (row["split_name"], row["label_stratum"], row["endpoint"]): row
        for row in _csv(RESULT / "rescue_rates.csv")
    }
    assert (int(rescue[("Test1", "all", "three_d_rescue_among_one_d_errors")]["numerator"]), int(rescue[("Test1", "all", "three_d_rescue_among_one_d_errors")]["denominator"])) == (24, 65)
    assert (int(rescue[("Test2", "all", "three_d_rescue_among_one_d_errors")]["numerator"]), int(rescue[("Test2", "all", "three_d_rescue_among_one_d_errors")]["denominator"])) == (15, 40)
    assert math.isclose(
        float(rescue[("Test1", "non_toxic", "three_d_rescue_among_one_d_errors")]["rate"]),
        19 / 37,
    )
    assert math.isclose(
        float(rescue[("Test2", "toxic", "three_d_rescue_among_one_d_errors")]["rate"]),
        1 / 11,
    )

    contrast_rows = _csv(RESULT / "primary_contrasts.csv")
    assert len(contrast_rows) == 10
    for contrast in contrast_rows:
        split = contrast["split_name"]
        attribute = contrast["attribute"]
        three_d = np.asarray(
            [float(row[attribute]) for row in rows if row["split_name"] == split and row["outcome_group"] == "three_d_only_correct"]
        )
        one_d = np.asarray(
            [float(row[attribute]) for row in rows if row["split_name"] == split and row["outcome_group"] == "one_d_only_correct"]
        )
        if attribute == "conformer_available":
            expected_effect = float(three_d.mean() - one_d.mean())
        else:
            expected_effect = _cliffs_delta(three_d, one_d)
            assert math.isclose(
                float(contrast["median_difference"]),
                float(np.median(three_d) - np.median(one_d)),
                abs_tol=1e-15,
            )
        assert math.isclose(float(contrast["primary_effect"]), expected_effect, abs_tol=1e-15)
        assert 0.0 < float(contrast["permutation_p_value"]) <= 1.0
        assert float(contrast["bh_q_value"]) >= float(contrast["permutation_p_value"])
    assert all(float(row["bh_q_value"]) > 0.05 for row in contrast_rows)
    test2_length = next(
        row
        for row in contrast_rows
        if row["split_name"] == "Test2" and row["attribute"] == "peptide_length"
    )
    assert math.isclose(float(test2_length["primary_effect"]), -0.5230769230769231)
    assert math.isclose(float(test2_length["bh_q_value"]), 0.15598440155984402)


def test_result_checksums_and_no_model_inference_contract() -> None:
    checksum_lines = (RESULT / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 11
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((RESULT / relative).read_bytes()).hexdigest() == expected
    summary = _json(RESULT / "summary.json")
    assert summary["decision"] == "descriptive_analysis_complete_no_model_change"
    assert summary["row_counts"]["sample_assignments"] == 1708
    assert summary["environment"]["torch_used_for"].startswith("read-only deserialization")
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("torch.load(") == 1
    for forbidden in (
        "load_state_dict(",
        ".backward(",
        "torch.optim",
        "optimizer.step(",
        ".train(",
    ):
        assert forbidden not in source


def test_publication_figure_package_is_complete_and_traceable() -> None:
    checksum_lines = (FIGURE / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert len(checksum_lines) == 13
    for line in checksum_lines:
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((FIGURE / relative).read_bytes()).hexdigest() == expected

    assert len(_csv(FIGURE / "source_data" / "sample_assignments.csv")) == 1708
    assert len(_csv(FIGURE / "source_data" / "outcome_group_composition.csv")) == 8
    assert len(_csv(FIGURE / "source_data" / "reciprocal_rescue_rates.csv")) == 12
    assert len(_csv(FIGURE / "source_data" / "attribute_effects.csv")) == 10

    png = (FIGURE / "expert_disagreement_analysis.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png[16:24]) == (4200, 2130)
    svg = (FIGURE / "expert_disagreement_analysis.svg").read_text(encoding="utf-8")
    for internal_name in (
        "both_correct",
        "one_d_only_correct",
        "three_d_only_correct",
        "both_wrong",
    ):
        assert internal_name not in svg
