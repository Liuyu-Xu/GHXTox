"""Phase 29 fixed-threshold short-peptide 3D specialist utilities.

The training run is intentionally narrow: five fold-specific conformer residual
heads are trained on the existing PepFlow pilot250 only.  Aggregation replaces
exactly those 250 OOF probabilities in the retained Phase 25 handoff and copies
all other probabilities unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from ghxtox.metrics import binary_metrics
from ghxtox.utils import save_json


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(rows: list[dict[str, Any]], probability_key: str, threshold: float) -> dict[str, Any]:
    probabilities = torch.tensor(
        [float(row[probability_key]) for row in rows], dtype=torch.float64
    ).clamp(1e-12, 1.0 - 1e-12)
    labels = torch.tensor([float(row["label"]) for row in rows], dtype=torch.float64)
    return binary_metrics(torch.logit(probabilities), labels, threshold=threshold)


def verify_protocol(protocol_path: str | Path) -> dict[str, Any]:
    """Verify the frozen protocol and every immutable local input."""

    path = Path(protocol_path)
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_candidate_oof":
        raise ValueError("Phase 29 protocol is not marked frozen before candidate OOF.")
    if protocol.get("external_test_labels_read") is not False:
        raise ValueError("Phase 29 protocol must keep external test labels sealed.")

    expected_files = {
        protocol["reference"]["oof_path"]: protocol["reference"]["oof_sha256"],
        protocol["training"]["processed_path"]: protocol["training"]["processed_sha256"],
        protocol["training"]["fold_manifest"]: protocol["training"]["fold_manifest_sha256"],
    }
    if "config_path" in protocol["training"]:
        expected_files[protocol["training"]["config_path"]] = protocol["training"][
            "config_sha256"
        ]
    evaluation = protocol.get("evaluation", {})
    if "candidate_full_processed_sha256" in evaluation:
        expected_files[evaluation["candidate_full_processed"]] = evaluation[
            "candidate_full_processed_sha256"
        ]
    if "candidate_full_fold_manifest_sha256" in evaluation:
        expected_files[evaluation["candidate_full_fold_manifest"]] = evaluation[
            "candidate_full_fold_manifest_sha256"
        ]
    checkpoint_pattern = protocol["training"]["initial_checkpoint_pattern"]
    for fold, expected in protocol["training"]["initial_checkpoint_sha256"].items():
        expected_files[checkpoint_pattern.format(fold=fold)] = expected

    verified: dict[str, str] = {}
    for file_path, expected in expected_files.items():
        actual = _sha256(file_path)
        if actual != expected:
            raise ValueError(
                f"Frozen input hash mismatch for {file_path}: expected {expected}, got {actual}."
            )
        verified[file_path] = actual

    return {
        "status": "verified",
        "protocol": str(path),
        "verified_files": verified,
        "external_test_labels_read": False,
    }


def _update_probability_fields(row: dict[str, Any], probability: float, threshold: float) -> None:
    probability = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    logit = math.log(probability / (1.0 - probability))
    entropy = -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    ) / math.log(2.0)
    row["toxicity_probability"] = f"{probability:.17g}"
    row["logit"] = f"{logit:.17g}"
    row["prediction"] = str(int(probability >= threshold))
    row["decision_threshold"] = f"{threshold:.17g}"
    row["probability_margin"] = f"{abs(2.0 * probability - 1.0):.17g}"
    row["probability_entropy"] = f"{entropy:.17g}"
    if "mc_mean_probability" in row:
        row["mc_mean_probability"] = f"{probability:.17g}"
    if "mc_std_probability" in row:
        row["mc_std_probability"] = "0"


def aggregate_candidate(
    *,
    protocol_path: str | Path,
    candidate_oof: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Merge the frozen pilot candidate into Phase 25 and apply the fixed gate."""

    verify_protocol(protocol_path)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    threshold = float(protocol["evaluation"]["fixed_threshold"])
    reference_rows = _read_csv(protocol["reference"]["oof_path"])
    candidate_rows = _read_csv(candidate_oof)
    pilot_rows = _read_csv(protocol["training"]["fold_manifest"])

    if len(reference_rows) != int(protocol["reference"]["rows"]):
        raise ValueError("Reference OOF row count does not match the frozen protocol.")
    reference_by_id = {row["sample_id"]: row for row in reference_rows}
    candidate_by_id = {row["sample_id"]: row for row in candidate_rows}
    pilot_by_id = {row["sample_id"]: row for row in pilot_rows}
    if len(reference_by_id) != len(reference_rows) or len(candidate_by_id) != len(candidate_rows):
        raise ValueError("Reference and candidate OOF sample IDs must be unique.")
    if len(pilot_by_id) != int(protocol["training"]["rows"]):
        raise ValueError("Pilot fold manifest does not contain the frozen 250 unique rows.")
    missing = sorted(set(pilot_by_id) - set(candidate_by_id))
    if missing:
        raise ValueError(f"Candidate OOF is missing pilot samples: {missing[:5]}")

    merged_rows: list[dict[str, Any]] = []
    for reference in reference_rows:
        sample_id = reference["sample_id"]
        merged = dict(reference)
        if sample_id in pilot_by_id:
            candidate = candidate_by_id[sample_id]
            pilot = pilot_by_id[sample_id]
            expected_index = int(pilot["original_source_index"])
            checks = {
                "source_index": int(candidate["source_index"]) == expected_index,
                "reference_index": int(reference["source_index"]) == expected_index,
                "fold": int(candidate["fold"]) == int(pilot["fold"]) == int(reference["fold"]),
                "label": int(candidate["label"]) == int(reference["label"]) == int(pilot["label"]),
                "sequence": candidate["sequence"] == reference["sequence"] == pilot["sequence"],
            }
            if not all(checks.values()):
                failed = [name for name, passed in checks.items() if not passed]
                raise ValueError(f"Pilot alignment failed for {sample_id}: {failed}")
            _update_probability_fields(
                merged, float(candidate["toxicity_probability"]), threshold
            )
        merged_rows.append(merged)

    fallback_rows = [row for row in merged_rows if row["sample_id"] not in pilot_by_id]
    fallback_exact = all(
        row["toxicity_probability"]
        == reference_by_id[row["sample_id"]]["toxicity_probability"]
        for row in fallback_rows
    )
    if len(fallback_rows) != len(reference_rows) - len(pilot_rows):
        raise RuntimeError("Unexpected Phase 29 fallback row count.")

    reference_pilot = [row for row in reference_rows if row["sample_id"] in pilot_by_id]
    candidate_pilot = [row for row in merged_rows if row["sample_id"] in pilot_by_id]
    global_reference = _metrics(reference_rows, "toxicity_probability", threshold)
    global_candidate = _metrics(merged_rows, "toxicity_probability", threshold)
    pilot_reference = _metrics(reference_pilot, "toxicity_probability", threshold)
    pilot_candidate = _metrics(candidate_pilot, "toxicity_probability", threshold)

    fold_metrics = []
    fold_regressions = []
    for fold in range(int(protocol["training"]["num_folds"])):
        reference_fold = [row for row in reference_pilot if int(row["fold"]) == fold]
        candidate_fold = [row for row in candidate_pilot if int(row["fold"]) == fold]
        reference_metrics = _metrics(reference_fold, "toxicity_probability", threshold)
        candidate_metrics = _metrics(candidate_fold, "toxicity_probability", threshold)
        regression = float(reference_metrics["mcc"] - candidate_metrics["mcc"])
        fold_regressions.append(regression)
        fold_metrics.append(
            {
                "fold": fold,
                "samples": len(reference_fold),
                "reference": reference_metrics,
                "candidate": candidate_metrics,
                "mcc_delta": float(candidate_metrics["mcc"] - reference_metrics["mcc"]),
            }
        )

    corrections = regressions = changed = 0
    for reference, candidate in zip(reference_pilot, candidate_pilot, strict=True):
        label = int(reference["label"])
        reference_prediction = int(float(reference["toxicity_probability"]) >= threshold)
        candidate_prediction = int(float(candidate["toxicity_probability"]) >= threshold)
        if reference_prediction != candidate_prediction:
            changed += 1
            if reference_prediction != label and candidate_prediction == label:
                corrections += 1
            elif reference_prediction == label and candidate_prediction != label:
                regressions += 1

    gate_spec = protocol["promotion_gate"]
    gates = {
        "exact_fallback_for_6137_rows": fallback_exact and len(fallback_rows) == 6137,
        "global_mcc_non_decreasing": float(global_candidate["mcc"])
        >= float(global_reference["mcc"]),
        "pilot250_mcc_gain_at_least_0_005": float(pilot_candidate["mcc"])
        - float(pilot_reference["mcc"])
        >= float(gate_spec["pilot250_mcc_gain_at_least"]),
        "global_auprc_regression_at_most_0_0005": float(global_reference["auprc"])
        - float(global_candidate["auprc"])
        <= float(gate_spec["global_auprc_regression_at_most"]),
        "pilot250_auprc_regression_at_most_0_005": float(pilot_reference["auprc"])
        - float(pilot_candidate["auprc"])
        <= float(gate_spec["pilot250_auprc_regression_at_most"]),
        "worst_pilot_fold_mcc_regression_at_most_0_02": max(fold_regressions)
        <= float(gate_spec["worst_pilot_fold_mcc_regression_at_most"]),
        "net_corrected_threshold_decisions_strictly_positive": corrections - regressions > 0,
    }
    gates["all_conditions_passed"] = all(gates.values())
    decision = (
        "promote_phase29_short_margin_3d"
        if gates["all_conditions_passed"]
        else "retain_phase25_3d_handoff"
    )
    summary = {
        "protocol": str(protocol_path),
        "candidate_oof": str(candidate_oof),
        "fixed_threshold": threshold,
        "rows": len(merged_rows),
        "pilot_rows_replaced": len(candidate_pilot),
        "fallback_rows": len(fallback_rows),
        "fallback_probability_exact": fallback_exact,
        "global": {"reference": global_reference, "candidate": global_candidate},
        "pilot250": {"reference": pilot_reference, "candidate": pilot_candidate},
        "folds": fold_metrics,
        "decision_transitions": {
            "changed": changed,
            "corrections": corrections,
            "regressions": regressions,
            "net_corrections": corrections - regressions,
        },
        "gate": gates,
        "decision": decision,
        "external_test_labels_read": False,
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0]))
        writer.writeheader()
        writer.writerows(merged_rows)
    save_json(summary, output / "summary.json")
    save_json(
        {
            "decision": decision,
            "all_conditions_passed": gates["all_conditions_passed"],
            "gate": gates,
            "fixed_threshold": threshold,
            "external_test_labels_read": False,
        },
        output / "decision.json",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Verify all frozen Phase 29 inputs.")
    verify.add_argument("--protocol", default="runs/short_margin_3d/protocol.json")
    aggregate = subparsers.add_parser("aggregate", help="Merge and gate candidate OOF.")
    aggregate.add_argument("--protocol", default="runs/short_margin_3d/protocol.json")
    aggregate.add_argument("--candidate-oof", required=True)
    aggregate.add_argument("--output-dir", default="runs/short_margin_3d/aggregate")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "verify":
        print(json.dumps(verify_protocol(args.protocol), indent=2))
    else:
        summary = aggregate_candidate(
            protocol_path=args.protocol,
            candidate_oof=args.candidate_oof,
            output_dir=args.output_dir,
        )
        print(json.dumps({"decision": summary["decision"], "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
