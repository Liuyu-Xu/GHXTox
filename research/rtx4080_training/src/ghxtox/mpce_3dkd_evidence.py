"""Post-prediction evidence analysis for the frozen MPCE-3DKD product.

This layer never changes the MPCE-3DKD prediction math.  It consumes frozen
checkpoint outputs and adds:
- cross-view counterfactual evidence (already returned by predict_sequences);
- sequence perturbation stability (single conservative substitution per position);
- evidence-consistency / review-priority record;
- privileged-knowledge provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.mpce_3dkd_predict import predict_sequences


EVIDENCE_SCHEMA_VERSION = "1.0"
STABILITY_DECISION_RULES = {
    "high": {"rule": "mean_abs_probability_delta < 0.02"},
    "moderate": {"rule": "0.02 <= mean_abs_probability_delta < 0.08"},
    "low": {"rule": "mean_abs_probability_delta >= 0.08"},
}
REVIEW_DECISION_RULES = {
    "evidence_consistency_high": "margin >= 0.25 AND cross_view_support_fraction >= 0.75 AND perturbation stability != low",
    "evidence_consistency_moderate": "margin >= 0.10 AND cross_view_support_fraction >= 0.50 AND perturbation stability != low",
    "review_recommended": "evidence_consistency == low OR margin < 0.15 OR (margin < 0.25 AND evidence_consistency == moderate)",
}


CONSERVATIVE_SUBSTITUTIONS: dict[str, str] = {
    "A": "G",
    "G": "A",
    "V": "I",
    "I": "V",
    "L": "I",
    "M": "L",
    "F": "Y",
    "Y": "F",
    "W": "Y",
    "S": "T",
    "T": "S",
    "C": "S",
    "D": "E",
    "E": "D",
    "N": "Q",
    "Q": "N",
    "K": "R",
    "R": "K",
    "H": "K",
    "P": "A",
}


def _conservative_mutants(sequence: str) -> list[dict[str, Any]]:
    mutants = []
    for position, residue in enumerate(sequence, start=1):
        substitution = CONSERVATIVE_SUBSTITUTIONS.get(residue)
        if substitution is None or substitution == residue:
            continue
        mutated = sequence[: position - 1] + substitution + sequence[position:]
        mutants.append(
            {
                "position": position,
                "original_aa": residue,
                "substituted_aa": substitution,
                "sequence": mutated,
            }
        )
    return mutants


def _stability_label(mean_abs_delta: float) -> str:
    if mean_abs_delta < 0.02:
        return "high"
    if mean_abs_delta < 0.08:
        return "moderate"
    return "low"


def _review_record(probability: float, stability_label: str | None, cross_support_fraction: float) -> dict[str, Any]:
    margin = abs(probability - 0.5)
    perturbation_penalized = stability_label == "low"
    if margin >= 0.25 and cross_support_fraction >= 0.75 and not perturbation_penalized:
        consistency = "high"
    elif margin >= 0.10 and cross_support_fraction >= 0.5 and not perturbation_penalized:
        consistency = "moderate"
    else:
        consistency = "low"
    reasons = []
    if margin < 0.10:
        reasons.append("probability is close to the 0.5 decision threshold")
    if cross_support_fraction < 0.5:
        reasons.append("fewer than half of interaction groups support the prediction")
    if perturbation_penalized:
        reasons.append("prediction changes materially under conservative single substitutions")
    review_recommended = consistency == "low" or margin < 0.15 or (
        margin < 0.25 and consistency == "moderate"
    )
    return {
        "probability_margin": margin,
        "cross_view_support_fraction": cross_support_fraction,
        "perturbation_stability": stability_label or "not_computed",
        "evidence_consistency": consistency,
        "review_recommended": bool(review_recommended),
        "review_priority": "high" if review_recommended else "routine",
        "reason": reasons,
        "note": "evidence consistency is a model-output property, not experimental toxicity confidence",
        "decision_rules": REVIEW_DECISION_RULES,
    }


def analyze_sequence(
    sequence: str,
    manifest_path: str | Path,
    *,
    model_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 16,
    include_perturbation_stability: bool = True,
    max_perturbation_positions: int | None = None,
) -> dict[str, Any]:
    mutants = _conservative_mutants(sequence)
    if max_perturbation_positions is not None:
        mutants = mutants[: max_perturbation_positions]
    if not include_perturbation_stability:
        mutants = []
    query_sequences = [sequence] + [mutant["sequence"] for mutant in mutants]
    predictions = predict_sequences(
        query_sequences,
        manifest_path,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )
    original = predictions[0]
    original_label = str(original["label"])
    position_rows = []
    label_flip_count = 0
    for mutant, prediction in zip(mutants, predictions[1:], strict=True):
        delta_probability = float(prediction["probability"] - original["probability"])
        if str(prediction["label"]) != original_label:
            label_flip_count += 1
        position_rows.append(
            {
                "position": mutant["position"],
                "original_aa": mutant["original_aa"],
                "substituted_aa": mutant["substituted_aa"],
                "mutant_probability": float(prediction["probability"]),
                "delta_probability": delta_probability,
            }
        )
    if include_perturbation_stability:
        mean_abs_delta = (
            float(np.mean([abs(row["delta_probability"]) for row in position_rows]))
            if position_rows
            else 0.0
        )
        stability_label = _stability_label(mean_abs_delta)
        stability_score = max(0.0, 1.0 - mean_abs_delta / 0.15)
        max_abs_shift = max(
            (abs(row["delta_probability"]) for row in position_rows),
            default=0.0,
        )
        ranked = sorted(position_rows, key=lambda row: abs(row["delta_probability"]), reverse=True)
        most_sensitive = ranked[:3]
        largest = ranked[0] if ranked else None
    else:
        mean_abs_delta = None
        stability_label = None
        stability_score = None
        max_abs_shift = None
        ranked = []
        most_sensitive = []
        largest = None
    stability = {
        "analysis_performed": include_perturbation_stability,
        "prediction_stability_score": stability_score,
        "prediction_stability_label": stability_label or "not_computed",
        "mean_abs_probability_delta": mean_abs_delta,
        "max_abs_probability_shift": max_abs_shift,
        "label_flip_count": label_flip_count,
        "position_count": len(position_rows),
        "most_sensitive_positions": [
            {
                "position": row["position"],
                "mutation": f"{row['original_aa']}{row['position']}->{row['substituted_aa']}",
                "delta_probability": row["delta_probability"],
            }
            for row in most_sensitive
        ],
        "largest_probability_shift": (
            {
                "position": largest["position"],
                "mutation": f"{largest['original_aa']}{largest['position']}->{largest['substituted_aa']}",
                "delta_probability": largest["delta_probability"],
            }
            if largest
            else None
        ),
        "direction": (
            "increases_toxicity_probability"
            if largest and largest["delta_probability"] > 0
            else "decreases_toxicity_probability"
            if largest
            else "no_change"
        ),
        "position_analysis": position_rows,
        "interpretation_note": "single conservative substitutions; model-output response only, not experimental mutagenesis",
        "decision_rules": STABILITY_DECISION_RULES,
    }
    cross_support_fraction = float(original["cross_view_evidence"]["support_fraction"])
    review = _review_record(float(original["probability"]), stability_label, cross_support_fraction)
    provenance = dict(original["provenance"])
    provenance.update(
        {
            "teacher_used_during_inference": False,
            "structural_input_used_during_inference": False,
            "esmfold_used_during_inference": False,
            "evidence_layers": [
                "prototype_evidence",
                "cross_view_counterfactual",
                "sequence_perturbation_stability",
            ],
        }
    )
    profile_map_path = Path(manifest_path).parent / "prototype_profile_map.json"
    prototype_profile_version = None
    prototype_profile_sha256 = None
    if profile_map_path.is_file():
        profile_payload = json.loads(profile_map_path.read_text(encoding="utf-8"))
        prototype_profile_version = profile_payload.get("prototype_profile_version")
        prototype_profile_sha256 = profile_payload.get("prototype_profile_sha256")
    return {
        "sequence": sequence,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "model_architecture": provenance.get("model_name"),
        "model_version": provenance.get("model_version"),
        "prototype_profile_version": prototype_profile_version,
        "prototype_profile_sha256": prototype_profile_sha256,
        "analysis_level": "detailed" if include_perturbation_stability else "base",
        "prediction": original["label"],
        "probability": original["probability"],
        "calibrated_probability": original["calibrated_probability"],
        "decision_threshold": original["decision_threshold"],
        "prototype_evidence": original["prototype_evidence"],
        "cross_view_evidence": original["cross_view_evidence"],
        "perturbation_stability": stability,
        "evidence_record": review,
        "physchem": original["physchem"],
        "provenance": provenance,
    }


def predict_evidence(
    sequence: str,
    manifest_path: str | Path,
    *,
    analysis_level: str = "base",
    model_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, Any]:
    """Single-sequence evidence entry point with a frozen analysis-level policy.

    ``base`` returns prototype + cross-view evidence without perturbation
    budget; ``detailed`` additionally runs the conservative-substitution
    stability screen.  Batch consumers should default to ``base``.
    """
    if analysis_level not in {"base", "detailed"}:
        raise ValueError("analysis_level must be 'base' or 'detailed'")
    return analyze_sequence(
        sequence,
        manifest_path,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        include_perturbation_stability=analysis_level == "detailed",
    )


def predict_batch_base(
    sequences: list[str],
    manifest_path: str | Path,
    *,
    model_path: str | None = None,
    device: str = "cuda",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Batch policy: base predictions + prototype/cross-view evidence only.

    Perturbation stability is intentionally disabled by default; callers may
    send a single sequence to ``predict_evidence(analysis_level='detailed')``
    when the user requests a detailed view.
    """
    results = predict_sequences(
        sequences,
        manifest_path,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
    )
    profile_map_path = Path(manifest_path).parent / "prototype_profile_map.json"
    prototype_profile_version = None
    prototype_profile_sha256 = None
    if profile_map_path.is_file():
        profile_payload = json.loads(profile_map_path.read_text(encoding="utf-8"))
        prototype_profile_version = profile_payload.get("prototype_profile_version")
        prototype_profile_sha256 = profile_payload.get("prototype_profile_sha256")
    for result in results:
        result["evidence_schema_version"] = EVIDENCE_SCHEMA_VERSION
        result["analysis_level"] = "base"
        result["prototype_profile_version"] = prototype_profile_version
        result["prototype_profile_sha256"] = prototype_profile_sha256
        result["model_version"] = result["provenance"].get("model_version")
        result["perturbation_stability"] = {
            "analysis_performed": False,
            "budget_policy": "batch_default_disabled",
            "note": "call detailed evidence endpoint for a single sequence to enable conservative-substitution stability analysis",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = analyze_sequence(
        args.sequence,
        args.manifest,
        model_path=args.model_path,
        device=args.device,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
