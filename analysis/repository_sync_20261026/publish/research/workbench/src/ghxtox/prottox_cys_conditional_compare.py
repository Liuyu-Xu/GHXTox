"""Frozen paired OOF promotion rule for the Cys-conditioned candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from ghxtox.prottox_cys_conditional import MODEL_NAME as CANDIDATE_NAME, MODEL_VERSION as CANDIDATE_VERSION
from ghxtox.prottox_spectral_compare import _seed
from ghxtox.prottox_tsk import MODEL_NAME as BASE_NAME, MODEL_VERSION as BASE_VERSION, file_sha256, write_manifest


def _manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pair(paths: list[str | Path], *, name: str, version: str) -> dict[int, tuple[Path, dict[str, Any]]]:
    if len(paths) != 2:
        raise ValueError("Frozen comparison requires exactly two paired seed manifests")
    rows = [(Path(path), _manifest(path)) for path in paths]
    if any(row.get("model_name") != name or row.get("model_version") != version for _, row in rows):
        raise ValueError("Comparison received an incompatible manifest")
    result = {_seed(row): (path, row) for path, row in rows}
    if set(result) != {20260905, 20260906}:
        raise ValueError("Comparison requires frozen seeds 20260905 and 20260906")
    return result


def _low_cys_sensitivity(manifest_path: Path, manifest: dict[str, Any]) -> float:
    cache_path = Path(manifest["cache"]["path"])
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cys = {str(record["sample_id"]): int(round(float(torch.as_tensor(record["physchem"])[6]))) for record in cache["records"]}
    oof_path = manifest_path.parent / "oof_predictions.csv"
    with oof_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if int(row["label"]) == 1 and cys.get(str(row["sample_id"]), 99) <= 1]
    if not selected or any(str(row["sample_id"]) not in cys for row in rows):
        raise ValueError("OOF and cache do not support low-Cys sensitivity calculation")
    return sum(float(row["probability"]) >= 0.5 for row in selected) / len(selected)


def compare(baseline_manifests: list[str | Path], candidate_manifests: list[str | Path], output_path: str | Path, *, control_manifests: list[str | Path]) -> dict[str, Any]:
    baseline = _pair(baseline_manifests, name=BASE_NAME, version=BASE_VERSION)
    candidate = _pair(candidate_manifests, name=CANDIDATE_NAME, version=CANDIDATE_VERSION)
    deltas = []
    for seed in sorted(baseline):
        base_path, base = baseline[seed]
        candidate_path, current = candidate[seed]
        if current.get("variant") != "cys_conditional_residual" or base["cache"]["sha256"] != current["cache"]["sha256"] or base["fold_manifest"] != current["fold_manifest"]:
            raise ValueError("Candidate is not a paired Cys-conditioned run under the baseline protocol")
        deltas.append({"seed": seed, "mcc": float(current["oof_metrics"]["mcc"] - base["oof_metrics"]["mcc"]), "auprc": float(current["oof_metrics"]["auprc"] - base["oof_metrics"]["auprc"]), "brier": float(current["oof_metrics"]["brier"] - base["oof_metrics"]["brier"]), "low_cys_toxic_sensitivity": _low_cys_sensitivity(candidate_path, current), "low_cys_toxic_sensitivity_delta": _low_cys_sensitivity(candidate_path, current) - _low_cys_sensitivity(base_path, base)})
    mean = {metric: sum(row[metric] for row in deltas) / len(deltas) for metric in ("mcc", "auprc", "brier", "low_cys_toxic_sensitivity_delta")}
    promote = all(row["mcc"] >= 0 and row["low_cys_toxic_sensitivity_delta"] >= 0.05 for row in deltas) and mean["mcc"] >= 0.005 and mean["auprc"] >= -0.002
    controls = []
    for path in control_manifests:
        item = _manifest(path)
        if item.get("model_name") != CANDIDATE_NAME or item.get("variant") != "shared_residual":
            raise ValueError("Controls must be shared_residual CysConditional manifests")
        controls.append({"path": str(path), "seed": item.get("seed"), "oof_metrics": item["oof_metrics"], "low_cys_toxic_sensitivity": _low_cys_sensitivity(Path(path), item)})
    report = {"artifact_type": "prottox_cys_conditional_promotion_report", "selection_scope": "paired training OOF only; Test1/Test2 excluded", "frozen_rule": {"both_seeds_mcc_not_lower": True, "minimum_mean_mcc_advantage": 0.005, "maximum_mean_auprc_decline": 0.002, "minimum_each_seed_low_cys_sensitivity_improvement": 0.05}, "paired_deltas": deltas, "mean_delta": mean, "decision": "promote_for_one_time_test1_audit" if promote else "retain_as_ablation", "controls": controls}
    write_manifest(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs=2, required=True)
    parser.add_argument("--candidate", nargs=2, required=True)
    parser.add_argument("--controls", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.output, control_manifests=args.controls), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
