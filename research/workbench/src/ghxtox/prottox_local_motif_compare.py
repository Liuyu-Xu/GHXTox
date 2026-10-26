"""Frozen paired OOF promotion gate for the ProtTox LocalMotif candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ghxtox.prottox_local_motif import MODEL_NAME as LOCAL_NAME, MODEL_VERSION as LOCAL_VERSION
from ghxtox.prottox_spectral_compare import _seed
from ghxtox.prottox_tsk import MODEL_NAME as BASE_NAME, MODEL_VERSION as BASE_VERSION, write_manifest


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    location = Path(path)
    return location, json.loads(location.read_text(encoding="utf-8"))


def _pair(paths: list[str | Path], *, name: str, version: str) -> dict[int, tuple[Path, dict[str, Any]]]:
    if len(paths) != 2:
        raise ValueError("Frozen LocalMotif comparison requires exactly two seed manifests")
    rows = [_load(path) for path in paths]
    if any(manifest.get("model_name") != name or manifest.get("model_version") != version for _, manifest in rows):
        raise ValueError("Comparison received an incompatible manifest")
    result = {_seed(manifest): (path, manifest) for path, manifest in rows}
    if set(result) != {20260905, 20260906}:
        raise ValueError("Comparison requires frozen seeds 20260905 and 20260906")
    return result


def compare(baseline_manifests: list[str | Path], control_manifests: list[str | Path], candidate_manifests: list[str | Path], output_path: str | Path) -> dict[str, Any]:
    baseline = _pair(baseline_manifests, name=BASE_NAME, version=BASE_VERSION)
    control = _pair(control_manifests, name=LOCAL_NAME, version=LOCAL_VERSION)
    candidate = _pair(candidate_manifests, name=LOCAL_NAME, version=LOCAL_VERSION)
    deltas, control_rows = [], []
    for seed in sorted(baseline):
        _, base = baseline[seed]
        _, matched_control = control[seed]
        _, current = candidate[seed]
        for manifest, expected_variant in ((matched_control, "nonlocal_projection_control"), (current, "multiscale_cnn_attention")):
            if manifest.get("variant") != expected_variant or manifest.get("base_cache", {}).get("sha256") != base.get("cache", {}).get("sha256") or manifest.get("fold_manifest") != base.get("fold_manifest"):
                raise ValueError("LocalMotif run is not paired to the retained MLP cache/fold protocol")
        if matched_control.get("cache", {}).get("sha256") != current.get("cache", {}).get("sha256"):
            raise ValueError("LocalMotif control and candidate must use the exact same residue cache")
        candidate_local = int(current["local_branch_parameter_count"])
        control_local = int(matched_control["local_branch_parameter_count"])
        if abs(candidate_local - control_local) / max(candidate_local, 1) > 0.002:
            raise ValueError("Nonlocal control is not parameter-matched within the frozen 0.2% tolerance")
        deltas.append({"seed": seed, "mcc": float(current["oof_metrics"]["mcc"] - base["oof_metrics"]["mcc"]), "auprc": float(current["oof_metrics"]["auprc"] - base["oof_metrics"]["auprc"]), "brier": float(current["oof_metrics"]["brier"] - base["oof_metrics"]["brier"])})
        control_rows.append({"seed": seed, "local_branch_parameters": control_local, "total_parameters": int(matched_control["parameter_count"]), "oof_metrics": matched_control["oof_metrics"]})
    mean = {name: sum(row[name] for row in deltas) / len(deltas) for name in ("mcc", "auprc", "brier")}
    promote = all(row["mcc"] >= 0.0 for row in deltas) and mean["mcc"] >= 0.005 and mean["auprc"] >= -0.002
    report = {"artifact_type": "prottox_local_motif_promotion_report", "selection_scope": "paired training OOF only; Test1/Test2 excluded", "frozen_rule": {"both_seeds_mcc_not_lower": True, "minimum_mean_mcc_advantage": 0.005, "maximum_mean_auprc_decline": 0.002, "parameter_matched_control_tolerance": 0.002}, "paired_deltas": deltas, "mean_delta": mean, "decision": "promote_for_one_time_test1_audit" if promote else "retain_as_ablation", "control": control_rows}
    write_manifest(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs=2, required=True)
    parser.add_argument("--control", nargs=2, required=True)
    parser.add_argument("--candidate", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.control, args.candidate, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
