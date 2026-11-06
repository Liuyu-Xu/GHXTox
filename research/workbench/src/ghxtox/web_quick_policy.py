"""Validation and application for the frozen sequence-only Quick Prediction policy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ARTIFACT_TYPE = "ghxtox_web_quick_sequence_policy"


def load_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("artifact_type") != ARTIFACT_TYPE or payload.get("version") != 1:
        raise ValueError("Not a supported GHXTox web Quick Prediction policy.")
    if payload.get("external_test_labels_used") is not False:
        raise ValueError("Quick policy must be frozen without external test labels.")
    model = payload.get("model", {})
    decision = payload.get("decision", {})
    reject = payload.get("reject", {})
    confidence = payload.get("confidence", {})
    if model.get("seeds") != [42, 123, 2025]:
        raise ValueError("Quick policy must use the frozen three-seed 1D ensemble.")
    calibration = model.get("calibration", {})
    if not all(isinstance(calibration.get(key), (int, float)) for key in ("slope", "intercept")):
        raise ValueError("Quick policy has no valid calibration.")
    threshold = decision.get("threshold")
    band = reject.get("reject_band")
    high_margin = confidence.get("high_margin")
    if not all(isinstance(value, (int, float)) and value >= 0 for value in (threshold, band, high_margin)):
        raise ValueError("Quick policy threshold or margin is invalid.")
    if not 0 < float(threshold) < 1 or not 0 < float(reject.get("target_coverage", 0)) < 1:
        raise ValueError("Quick policy operating point is invalid.")
    if reject.get("rule") != "abs(p_seq - threshold) <= reject_band":
        raise ValueError("Quick policy reject rule is missing or changed.")
    return payload


def apply_policy(probability: float, policy: dict[str, Any]) -> dict[str, Any]:
    """Keep the binary call while returning a frozen reject-band decision."""
    policy = load_policy_payload(policy)
    threshold = float(policy["decision"]["threshold"])
    margin = abs(float(probability) - threshold)
    binary_label = "Toxic" if float(probability) >= threshold else "Non-toxic"
    abstained = margin <= float(policy["reject"]["reject_band"])
    if abstained:
        label, confidence = "Uncertain", "Review"
    elif margin <= float(policy["confidence"]["high_margin"]):
        label, confidence = binary_label, "Moderate"
    else:
        label, confidence = binary_label, "High"
    return {
        "label": label,
        "binary_label": binary_label,
        "confidence": confidence,
        "abstained": abstained,
        "margin": margin,
        "recommend_full_analysis": True,
    }


def load_policy_payload(policy: dict[str, Any]) -> dict[str, Any]:
    """Validate an already-loaded policy without re-reading its file."""
    if not isinstance(policy, dict):
        raise ValueError("Quick policy must be an object.")
    # Reuse the complete file-based validator with a small direct validation path.
    if policy.get("artifact_type") != ARTIFACT_TYPE or policy.get("version") != 1:
        raise ValueError("Not a supported GHXTox web Quick Prediction policy.")
    model, decision, reject, confidence = policy.get("model", {}), policy.get("decision", {}), policy.get("reject", {}), policy.get("confidence", {})
    if policy.get("external_test_labels_used") is not False or model.get("seeds") != [42, 123, 2025]:
        raise ValueError("Quick policy provenance is invalid.")
    if reject.get("rule") != "abs(p_seq - threshold) <= reject_band":
        raise ValueError("Quick policy reject rule is invalid.")
    for value in (decision.get("threshold"), reject.get("reject_band"), confidence.get("high_margin")):
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError("Quick policy contains an invalid numeric operating point.")
    return policy
