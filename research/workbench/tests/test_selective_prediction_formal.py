from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "selective_prediction_formal_experiment_v1.json"
RESULT = ROOT / "runs" / "selective_prediction_formal_experiment_20260809_v1"
FIGURE = ROOT / "paper_figures" / "output" / "selective_prediction_formal_comparison"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_formal_protocol_freezes_the_one_feature_policy_and_controls() -> None:
    protocol = _json(PROTOCOL)
    assert protocol["selected_uncertainty_policy"]["required_profile"] == "margin_only"
    assert protocol["selected_uncertainty_policy"]["required_feature_names"] == [
        "decision_closeness"
    ]
    assert protocol["operating_points"]["coverage_targets"] == [1.0, 0.95, 0.9, 0.85, 0.8]
    assert protocol["random_rejection_control"]["replicates"] == 5000
    assert protocol["random_rejection_control"]["seed"] == 20260820
    assert protocol["random_rejection_control"]["same_accepted_count_as_selective"] is True
    assert protocol["paired_bootstrap"]["replicates"] == 5000
    assert protocol["paired_bootstrap"]["seed"] == 20260821
    for key in (
        "allow_policy_refit",
        "allow_feature_reselection",
        "allow_binary_model_change",
        "allow_fusion_or_threshold_change",
        "allow_test1_or_test2_labels",
        "external_validation_claimed",
    ):
        assert protocol["scope"][key] is False


def test_five_coverage_comparison_has_exact_counts_and_all_metrics() -> None:
    rows = _csv(RESULT / "five_coverage_comparison.csv")
    assert len(rows) == 10
    expected_counts = {1.0: 6387, 0.95: 6068, 0.9: 5748, 0.85: 5429, 0.8: 5110}
    required_metrics = (
        "accepted_error_rate",
        "accepted_mcc",
        "accepted_balanced_accuracy",
        "accepted_sensitivity",
        "accepted_specificity",
    )
    by_key = {(row["method"], float(row["coverage_target"])): row for row in rows}
    for coverage, accepted_n in expected_counts.items():
        selective = by_key[("frozen_uncertainty_ranking", coverage)]
        random = by_key[("random_rejection", coverage)]
        assert int(selective["accepted_n"]) == accepted_n
        assert int(random["accepted_n"]) == accepted_n
        assert int(selective["rejected_n"]) == 6387 - accepted_n
        assert int(random["rejected_n"]) == 6387 - accepted_n
        for metric in required_metrics:
            assert 0.0 <= float(selective[metric]) <= 1.0
            assert 0.0 <= float(random[metric]) <= 1.0
    for metric in required_metrics:
        assert float(by_key[("frozen_uncertainty_ranking", 1.0)][metric]) == pytest.approx(
            float(by_key[("random_rejection", 1.0)][metric]), abs=1e-15
        )
    for coverage in (0.95, 0.9, 0.85, 0.8):
        assert float(
            by_key[("frozen_uncertainty_ranking", coverage)]["accepted_error_rate"]
        ) < float(by_key[("random_rejection", coverage)]["accepted_error_rate"])


def test_prespecified_primary_random_control_is_positive() -> None:
    contrasts = _csv(RESULT / "paired_bootstrap_contrasts.csv")
    primary = next(
        row
        for row in contrasts
        if float(row["coverage_target"]) == 0.9 and row["metric"] == "accepted_error_rate"
    )
    assert primary["positive_improvement_definition"] == "random minus selective"
    assert int(primary["bootstrap_replicates"]) == 5000
    assert float(primary["point_estimate"]) == pytest.approx(0.02237686151704941)
    assert float(primary["ci95_lower"]) > 0.0
    assert float(primary["ci95_upper"]) > float(primary["ci95_lower"])
    decision = _json(RESULT / "decision.json")
    assert decision["decision"] == "supports_difficult_sample_identification_on_training_oof"
    assert decision["policy_reselected"] is False
    assert decision["test1_or_test2_labels_used"] is False
    assert decision["canonical_binary_model_changed"] is False
    assert decision["external_generalization_claimed"] is False


def test_formal_and_figure_packages_are_complete_and_test_label_free() -> None:
    _verify_checksums(RESULT)
    _verify_checksums(FIGURE)
    result_inputs = _csv(RESULT / "input_manifest.csv")
    figure_inputs = _csv(FIGURE / "input_manifest.csv")
    for row in result_inputs + figure_inputs:
        source = (row.get("source") or row.get("path") or "").replace("\\", "/").lower()
        assert source
        assert "/test1" not in source
        assert "/test2" not in source
    for name in (
        "selective_prediction_random_rejection_comparison.svg",
        "selective_prediction_random_rejection_comparison.pdf",
        "selective_prediction_random_rejection_comparison.png",
        "five_coverage_table.md",
        "caption.md",
        "panel_manifest.csv",
        "internal_to_publication_names.csv",
        "source_data/five_coverage_metrics.csv",
        "source_data/paired_improvements.csv",
    ):
        assert (FIGURE / name).is_file(), name
