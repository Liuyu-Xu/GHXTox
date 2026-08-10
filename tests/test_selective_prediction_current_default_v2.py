from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runs" / "selective_prediction_current_default_input_20260810_v1"
OOF = ROOT / "runs" / "selective_prediction_current_default_oof_20260810_v2"
FORMAL = ROOT / "runs" / "selective_prediction_formal_current_default_20260810_v2"
TRANSFER = (
    ROOT / "runs" / "selective_prediction_frozen_threshold_transfer_current_default_20260810_v2"
)
FORMAL_FIGURE = ROOT / "paper_figures" / "output" / "selective_prediction_formal_current_default_v2"
TRANSFER_FIGURE = (
    ROOT
    / "paper_figures"
    / "output"
    / "selective_prediction_frozen_threshold_transfer_current_default_v2"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_current_default_input_exactly_reproduces_the_refrozen_classifier() -> None:
    summary = _json(INPUT / "summary.json")
    refreeze = _json(ROOT / "runs" / "lora_mamba_full" / "fusion_refreeze.json")
    assert summary["n"] == 6387
    assert summary["weight_1d"] == 0.72
    assert summary["decision_threshold"] == 0.47022256531284556
    assert summary["reproduced_fusion_oof_metrics"] == refreeze["fusion_oof_metrics"]
    assert set(summary["absolute_metric_differences_vs_refreeze"].values()) == {0.0}
    assert summary["test1_or_test2_used"] is False
    _verify_checksums(INPUT)


def test_v2_selects_the_direct_margin_and_matches_the_teammate_audit() -> None:
    protocol = _json(ROOT / "configs" / "selective_prediction_current_default_oof_v2.json")
    selection = _json(OOF / "summary.json")
    policy = _json(OOF / "selected_policy.json")
    assert protocol["profile_scoring"]["margin_only"] == {
        "type": "direct_feature",
        "feature_name": "decision_closeness",
    }
    assert policy["model"] == {
        "type": "direct_feature_score",
        "feature_name": "decision_closeness",
    }
    assert policy["preprocessing"] == {}
    assert policy["abstention_threshold"] == pytest.approx(0.47361765874159856)
    assert selection["selected_public_policy"] == "margin_only"
    margin = next(row for row in selection["profiles"] if row["profile"] == "margin_only")
    assert margin["aurc"] == pytest.approx(0.015372820597771444)
    assert margin["error_rate_at_90_coverage"] == pytest.approx(0.029227557411273485)
    assert margin["mcc_at_90_coverage"] == pytest.approx(0.925691272326145)
    assert selection["full_coverage"]["mcc"] == pytest.approx(0.8732971560275273)
    assert selection["test1_or_test2_labels_used"] is False
    _verify_checksums(OOF)


def test_v2_random_control_and_fixed_test_transfer_remain_frozen() -> None:
    formal = _json(FORMAL / "decision.json")
    transfer = _json(TRANSFER / "summary.json")
    primary = formal["primary_effect"]
    assert formal["decision"] == "supports_difficult_sample_identification_on_training_oof"
    assert primary["point_estimate"] == pytest.approx(0.021914439805149618)
    assert primary["ci95_lower"] > 0.0
    assert formal["test1_or_test2_labels_used"] is False
    assert transfer["decision"]["decision"] == "strong_support_on_both_benchmarks"
    assert transfer["decision"]["test_specific_cutoff_used"] is False
    assert transfer["decision"]["test_specific_coverage_forced"] is False
    assert transfer["decision"]["test_labels_used_for_selection"] is False
    by_split = {row["split_name"]: row for row in transfer["split_summary"]}
    assert by_split["Test1"]["realized_coverage"] == pytest.approx(0.8854351687388987)
    assert by_split["Test2"]["realized_coverage"] == pytest.approx(0.8694158075601375)
    assert by_split["Test1"]["frozen_accepted_error_rate"] == pytest.approx(
        0.02708124373119358
    )
    assert by_split["Test2"]["frozen_accepted_error_rate"] == pytest.approx(
        0.03162055335968379
    )
    _verify_checksums(FORMAL)
    _verify_checksums(TRANSFER)


def test_v2_publication_packages_are_complete_and_current() -> None:
    _verify_checksums(FORMAL_FIGURE)
    _verify_checksums(TRANSFER_FIGURE)
    formal_caption = (FORMAL_FIGURE / "caption.md").read_text(encoding="utf-8")
    transfer_caption = (TRANSFER_FIGURE / "caption.md").read_text(encoding="utf-8")
    assert "direct normalized proximity" in formal_caption
    assert "0.4736176587" in transfer_caption
    assert "0.719723" not in transfer_caption
    for directory, stem in (
        (FORMAL_FIGURE, "selective_prediction_random_rejection_comparison"),
        (TRANSFER_FIGURE, "selective_prediction_frozen_threshold_transfer"),
    ):
        for suffix in (".svg", ".pdf", ".png"):
            assert (directory / f"{stem}{suffix}").is_file()
