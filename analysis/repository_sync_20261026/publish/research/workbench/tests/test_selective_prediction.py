from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from ghxtox.multidimensional_default import _write_predictions, build_parser
from ghxtox.selective_prediction import (
    POLICY_ARTIFACT_TYPE,
    apply_selective_policy,
    load_selective_policy,
    validate_selective_policy,
)


def _policy(feature_names: list[str], coefficients: list[float]) -> dict:
    n = len(feature_names)
    return {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "profile": "unit_test",
        "feature_names": feature_names,
        "decision_threshold": 0.5,
        "abstention_threshold": 0.6,
        "abstain_when": "uncertainty_score > abstention_threshold",
        "preprocessing": {
            "median": [0.25] * n,
            "lower": [0.0] * n,
            "upper": [1.0] * n,
            "mean": [0.0] * n,
            "scale": [1.0] * n,
        },
        "model": {"intercept": -1.0, "coefficients": coefficients},
    }


def test_policy_adds_uncertain_without_changing_binary_prediction() -> None:
    fused = np.asarray([0.1, 0.49, 0.9])
    result = apply_selective_policy(
        fused,
        np.asarray([0.1, 0.8, 0.9]),
        np.asarray([0.1, 0.2, 0.9]),
        _policy(["decision_closeness", "expert_disagreement"], [3.0, 3.0]),
    )
    assert result["binary_prediction"].tolist() == [0, 0, 1]
    assert result["selective_prediction"].tolist() == ["Non-toxic", "Uncertain", "Toxic"]
    assert result["abstained"].tolist() == [False, True, False]


def test_direct_margin_policy_uses_the_raw_frozen_feature() -> None:
    policy = {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "profile": "margin_only",
        "feature_names": ["decision_closeness"],
        "decision_threshold": 0.47,
        "abstention_threshold": 0.4,
        "abstain_when": "uncertainty_score > abstention_threshold",
        "preprocessing": {},
        "model": {
            "type": "direct_feature_score",
            "feature_name": "decision_closeness",
        },
    }
    fused = np.asarray([0.05, 0.47, 0.95])
    result = apply_selective_policy(fused, fused, fused, policy)
    expected = 1.0 - np.abs(fused - 0.47) / 0.53
    assert result["uncertainty_score"] == pytest.approx(expected)
    assert result["abstained"].tolist() == [False, True, False]
    assert result["binary_prediction"].tolist() == [0, 1, 1]


def test_direct_margin_policy_rejects_hidden_preprocessing() -> None:
    policy = {
        "artifact_type": POLICY_ARTIFACT_TYPE,
        "profile": "margin_only",
        "feature_names": ["decision_closeness"],
        "decision_threshold": 0.47,
        "abstention_threshold": 0.4,
        "abstain_when": "uncertainty_score > abstention_threshold",
        "preprocessing": {"mean": [0.0]},
        "model": {
            "type": "direct_feature_score",
            "feature_name": "decision_closeness",
        },
    }
    with pytest.raises(ValueError, match="must not define preprocessing"):
        validate_selective_policy(policy)


def test_missing_optional_diagnostic_uses_frozen_policy_median() -> None:
    policy = _policy(["structure_uncertainty", "one_d_seed_std"], [1.0, 1.0])
    result = apply_selective_policy(
        np.asarray([0.2, 0.8]),
        np.asarray([0.2, 0.8]),
        np.asarray([0.2, 0.8]),
        policy,
        mean_plddt=None,
        one_d_seed_std=None,
    )
    assert np.all(np.isfinite(result["uncertainty_score"]))


def test_policy_loader_rejects_an_unrecognized_artifact(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"artifact_type": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="selective-prediction policy"):
        load_selective_policy(path)


def test_prediction_csv_schema_is_unchanged_when_policy_is_disabled(tmp_path: Path) -> None:
    output = tmp_path / "default.csv"
    values = np.asarray([0.2, 0.8])
    _write_predictions(
        output,
        ["a", "b"],
        np.asarray([0, 1]),
        values,
        values,
        values,
        0.5,
        0.72,
    )
    with output.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == [
        "source_index",
        "sample_id",
        "label",
        "p_1d",
        "p_3d",
        "toxicity_probability",
        "prediction",
        "decision_threshold",
        "weight_1d",
    ]


def test_prediction_csv_appends_selective_fields_only_when_enabled(tmp_path: Path) -> None:
    output = tmp_path / "selective.csv"
    values = np.asarray([0.2, 0.8])
    selective = {
        "uncertainty_score": np.asarray([0.1, 0.9]),
        "abstained": np.asarray([False, True]),
        "selective_prediction": np.asarray(["Non-toxic", "Uncertain"]),
        "binary_prediction": np.asarray([0, 1]),
    }
    _write_predictions(
        output,
        ["a", "b"],
        np.asarray([0, 1]),
        values,
        values,
        values,
        0.5,
        0.72,
        selective_result=selective,
    )
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0])[-3:] == ["uncertainty_score", "abstained", "selective_prediction"]
    assert rows[0]["prediction"] == "0"
    assert rows[1]["prediction"] == "1"
    assert rows[1]["selective_prediction"] == "Uncertain"


def test_cli_selective_policy_is_explicitly_opt_in() -> None:
    default = build_parser().parse_args(
        ["assemble", "--cal-1d", "one.csv", "--structure-csv", "three.csv", "--output-csv", "out.csv"]
    )
    enabled = build_parser().parse_args(
        [
            "assemble",
            "--cal-1d",
            "one.csv",
            "--structure-csv",
            "three.csv",
            "--output-csv",
            "out.csv",
            "--selective-policy",
            "policy.json",
        ]
    )
    assert default.selective_policy is None
    assert enabled.selective_policy == "policy.json"
