from __future__ import annotations

import json
from pathlib import Path

from ghxtox.hybrid_default import validate_public_default_config


ROOT = Path(__file__).resolve().parents[1]


def _load_default() -> dict:
    return json.loads(
        (ROOT / "configs" / "default_3d_hybrid.json").read_text(encoding="utf-8")
    )


def _load_expert_config() -> dict:
    return json.loads(
        (ROOT / "configs" / "default.json").read_text(encoding="utf-8")
    )


def test_public_default_is_the_3d_hybrid() -> None:
    config = _load_default()
    validate_public_default_config(config)
    assert config["canonical_public_default"] is True
    assert config["structure_expert"]["required"] is True
    assert config["interaction_adapter"]["required"] is True
    assert config["bacc_residual"]["enabled"] is True
    assert config["bacc_residual"]["decision_only"] is True
    assert config["fusion"]["prott5_weight"] + config["fusion"]["structure_weight"] == 1.0
    artifact_path = ROOT / config["bacc_residual"]["artifact"]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "ghxtox_bacc_decision_residual"
    assert artifact["decision_only"] is True
    assert artifact["teacher_threshold"] == config["bacc_residual"]["teacher_threshold"]
    assert artifact["base_threshold"] == config["bacc_residual"]["base_threshold"]


def test_legacy_default_name_points_to_the_same_3d_public_model() -> None:
    config = _load_default()
    expert = _load_expert_config()
    pointer = expert["public_default"]
    assert pointer["config"] == "configs/default_3d_hybrid.json"
    assert pointer["entrypoint"] == "ghxtox-predict"
    assert pointer["model_name"] == config["public_model_name"]
    assert config["structure_expert"]["checkpoint"] == (
        "runs/3d_v2_conformer_ensemble_candidate/best_model.pt"
    )
    assert config["interaction_adapter"]["required"] is True


def test_public_default_rejects_a_sequence_only_contract() -> None:
    config = _load_default()
    config["inference"]["full_3d_required"] = False
    try:
        validate_public_default_config(config)
    except ValueError as error:
        assert "3D" in str(error)
    else:
        raise AssertionError("A sequence-only route must not pass as the public default.")


def test_public_default_rejects_a_disabled_bacc_residual() -> None:
    config = _load_default()
    config["bacc_residual"]["enabled"] = False
    try:
        validate_public_default_config(config)
    except ValueError as error:
        assert "BACC" in str(error)
    else:
        raise AssertionError("The public default must include the frozen BACC residual.")


def test_bacc_residual_thresholds_match_the_experts() -> None:
    config = _load_default()
    config["bacc_residual"]["teacher_threshold"] = 0.5
    try:
        validate_public_default_config(config)
    except ValueError as error:
        assert "teacher threshold" in str(error)
    else:
        raise AssertionError("Mismatched residual thresholds must fail validation.")
