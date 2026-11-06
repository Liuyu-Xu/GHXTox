"""Frozen OOF-only promotion gate for the structure-quality-guided student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ghxtox.prottox_struct_guided import MODEL_NAME, MODEL_VERSION
from ghxtox.prottox_tsk import write_manifest


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed(manifest: dict[str, Any]) -> int:
    """Recover the documented run seed from early retained MLP manifests."""
    if "seed" in manifest:
        return int(manifest["seed"])
    checkpoint_path = manifest.get("final_checkpoint")
    if not checkpoint_path:
        raise ValueError("Manifest has no seed and no final checkpoint for legacy seed recovery")
    metadata = dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("metadata", {}))
    if "run_seed" in metadata:
        return int(metadata["run_seed"])
    if "refit_seed" in metadata:
        return int(metadata["refit_seed"]) - 1000
    if "seed" in metadata:
        return int(metadata["seed"]) - 1000
    raise ValueError("Cannot recover the run seed from the legacy checkpoint metadata")


def _paired(paths: list[str | Path], *, name: str, version: str, variant: str | None = None) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        manifest = _read(path)
        if manifest.get("model_name") != name or manifest.get("model_version") != version:
            raise ValueError(f"Unexpected manifest: {path}")
        if variant is not None and manifest.get("variant") != variant:
            raise ValueError(f"Unexpected variant: {path}")
        seed = _seed(manifest)
        if seed in result:
            raise ValueError(f"Duplicate seed {seed}")
        result[seed] = manifest
    return result


def compare(baseline_paths: list[str | Path], control_paths: list[str | Path], candidate_paths: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    baseline = _paired(baseline_paths, name="ProtTox-TSK", version="prottox-tsk-v1")
    control = _paired(control_paths, name=MODEL_NAME, version=MODEL_VERSION, variant="toxicity_only_control")
    candidate = _paired(candidate_paths, name=MODEL_NAME, version=MODEL_VERSION, variant="structure_quality_guided")
    if set(baseline) != set(control) or set(baseline) != set(candidate) or set(baseline) != {20260905, 20260906}:
        raise ValueError("Baseline, control and candidate must contain the same two seeds")
    rows: list[dict[str, float | int]] = []
    for seed in sorted(baseline):
        base, control_run, candidate_run = baseline[seed], control[seed], candidate[seed]
        if base["fold_manifest"] != candidate_run["fold_manifest"]["path"]:
            raise ValueError("Candidate fold manifest does not match baseline manifest")
        if base["cache"]["sha256"] != candidate_run["base_cache"]["sha256"]:
            raise ValueError("Candidate was not derived from the retained baseline cache")
        if control_run["cache"]["sha256"] != candidate_run["cache"]["sha256"]:
            raise ValueError("Control and candidate must use the same structure-guided cache")
        base_metrics, candidate_metrics = base["oof_metrics"], candidate_run["oof_metrics"]
        rows.append({"seed": seed, "mcc_delta": float(candidate_metrics["mcc"]) - float(base_metrics["mcc"]), "auprc_delta": float(candidate_metrics["auprc"]) - float(base_metrics["auprc"]), "brier_delta": float(candidate_metrics["brier"]) - float(base_metrics["brier"]), "candidate_minus_control_mcc": float(candidate_metrics["mcc"]) - float(control_run["oof_metrics"]["mcc"])})
    mean = {key: sum(float(row[key]) for row in rows) / len(rows) for key in ("mcc_delta", "auprc_delta", "brier_delta", "candidate_minus_control_mcc")}
    promote = all(float(row["mcc_delta"]) >= 0.0 for row in rows) and mean["mcc_delta"] >= 0.005 and mean["auprc_delta"] >= -0.002 and mean["brier_delta"] <= 0.002
    report = {"artifact_type": "prottox_struct_guide_promotion_report", "selection_scope": "paired training OOF only; Test1/Test2 excluded", "frozen_rule": {"both_seeds_mcc_not_lower": True, "minimum_mean_mcc_advantage": 0.005, "maximum_mean_auprc_decline": 0.002, "maximum_mean_brier_increase": 0.002}, "paired_deltas": rows, "mean_delta": mean, "decision": "promote_for_one_time_test1_audit" if promote else "retain_as_ablation", "controls": {"toxicity_only_control": "same architecture and parameter count; auxiliary Huber target loss disabled", "baseline": "retained TriStat-MLP"}}
    write_manifest(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.control, args.candidate, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
