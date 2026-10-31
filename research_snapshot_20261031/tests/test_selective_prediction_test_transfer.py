from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "selective_prediction_frozen_threshold_test_transfer_v1.json"
RESULT = ROOT / "runs" / "selective_prediction_frozen_threshold_test_transfer_20260809_v1"
FIGURE = ROOT / "paper_figures" / "output" / "selective_prediction_frozen_threshold_test_transfer"
POLICY = ROOT / "runs" / "selective_prediction_oof_20260809_v1" / "selected_policy.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected


def test_transfer_protocol_is_frozen_before_test_selection() -> None:
    protocol = _json(PROTOCOL)
    contract = protocol["frozen_policy_contract"]
    assert contract["required_profile"] == "margin_only"
    assert contract["required_feature_names"] == ["decision_closeness"]
    assert contract["binary_decision_threshold"] == 0.47022256531284556
    assert contract["uncertainty_score_cutoff"] == 0.719723910930358
    assert contract["accept_when"] == "uncertainty_score <= uncertainty_score_cutoff"
    assert protocol["random_rejection_control"]["replicates"] == 5000
    assert protocol["paired_bootstrap"]["replicates"] == 5000
    for key in (
        "allow_policy_refit",
        "allow_feature_reselection",
        "allow_test_specific_preprocessing",
        "allow_test_specific_cutoff",
        "allow_test_specific_coverage_target",
        "allow_dataset_identity_in_selection",
        "allow_test_labels_in_selection",
    ):
        assert protocol["scope"][key] is False
    for item in protocol["inputs"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_label_blind_decisions_reconstruct_the_frozen_score_and_mask() -> None:
    policy = _json(POLICY)
    rows = _csv(RESULT / "label_blind_decisions.csv")
    assert len(rows) == 1708
    assert "label" not in rows[0] and "binary_error" not in rows[0]
    counts = {split: sum(row["split_name"] == split for row in rows) for split in ("Test1", "Test2")}
    assert counts == {"Test1": 1126, "Test2": 582}
    assert len({(row["split_name"], row["source_index"]) for row in rows}) == len(rows)
    assert len({(row["split_name"], row["sample_id"]) for row in rows}) == len(rows)

    fused = np.asarray([float(row["toxicity_probability"]) for row in rows])
    observed_closeness = np.asarray([float(row["decision_closeness"]) for row in rows])
    observed_score = np.asarray([float(row["uncertainty_score"]) for row in rows])
    tau = float(policy["decision_threshold"])
    closeness = np.clip(1.0 - np.abs(fused - tau) / max(tau, 1.0 - tau), 0.0, 1.0)
    prep = policy["preprocessing"]
    standardized = (
        np.clip(closeness, prep["lower"][0], prep["upper"][0]) - prep["mean"][0]
    ) / prep["scale"][0]
    logits = policy["model"]["intercept"] + policy["model"]["coefficients"][0] * standardized
    score = 1.0 / (1.0 + np.exp(-logits))
    assert np.max(np.abs(closeness - observed_closeness)) < 1e-12
    assert np.max(np.abs(score - observed_score)) < 1e-12
    accepted = np.asarray([row["accepted"].lower() == "true" for row in rows])
    assert np.array_equal(accepted, score <= policy["abstention_threshold"])
    decision = _json(RESULT / "decision.json")
    assert decision["label_blind_decision_table_sha256"] == hashlib.sha256(
        (RESULT / "label_blind_decisions.csv").read_bytes()
    ).hexdigest()


def test_transfer_metrics_use_natural_coverage_and_beat_random_rejection() -> None:
    points = _csv(RESULT / "method_point_estimates.csv")
    assert len(points) == 6
    by_key = {(row["split_name"], row["method"]): row for row in points}
    expected = {
        "Test1": (997, 0.8854351687388987, 0.02708124373119358),
        "Test2": (506, 0.8694158075601375, 0.03162055335968379),
    }
    for split, (accepted_n, coverage, error) in expected.items():
        full = by_key[(split, "full_binary_predictor")]
        frozen = by_key[(split, "frozen_threshold_accepted_set")]
        random = by_key[(split, "equal_count_random_rejection")]
        assert int(frozen["accepted_n"]) == accepted_n
        assert abs(float(frozen["coverage"]) - coverage) < 1e-15
        assert abs(float(frozen["coverage"]) - 0.9) > 1e-6
        assert abs(float(frozen["accepted_error_rate"]) - error) < 1e-15
        assert float(frozen["accepted_error_rate"]) < float(full["accepted_error_rate"])
        assert float(frozen["accepted_error_rate"]) < float(random["accepted_error_rate"])
        assert float(frozen["accepted_mcc"]) > float(full["accepted_mcc"])
    contrasts = _csv(RESULT / "paired_bootstrap_contrasts.csv")
    error_rows = [row for row in contrasts if row["metric"] == "accepted_error_rate"]
    assert len(error_rows) == 4
    assert all(float(row["ci95_lower"]) > 0.0 for row in error_rows)
    sensitivity = next(
        row
        for row in contrasts
        if row["split_name"] == "Test2"
        and row["comparator"] == "full_binary_predictor"
        and row["metric"] == "accepted_sensitivity"
    )
    assert float(sensitivity["ci95_lower"]) < 0.0 < float(sensitivity["ci95_upper"])


def test_resampling_packages_and_selection_firewall_are_complete() -> None:
    _verify_checksums(RESULT)
    _verify_checksums(FIGURE)
    assert sum(1 for _ in (RESULT / "random_rejection_replicates.csv").open(encoding="utf-8")) - 1 == 10000
    assert (
        sum(1 for _ in (RESULT / "paired_bootstrap_method_replicates.csv").open(encoding="utf-8"))
        - 1
        == 30000
    )
    source = (ROOT / "scripts" / "evaluate_selective_prediction_test_transfer.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_label_blind_split"
    )
    label_access = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "label"
    ]
    assert not label_access
    for name in (
        "selective_prediction_frozen_threshold_transfer.svg",
        "selective_prediction_frozen_threshold_transfer.pdf",
        "selective_prediction_frozen_threshold_transfer.png",
        "exact_results_table.md",
        "caption.md",
        "panel_manifest.csv",
        "input_manifest.csv",
        "source_data/benchmark_transfer_metrics.csv",
        "source_data/paired_improvements.csv",
        "source_data/split_summary.csv",
    ):
        assert (FIGURE / name).is_file(), name
