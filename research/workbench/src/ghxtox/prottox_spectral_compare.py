"""Apply the frozen OOF-only promotion rule for the spectral candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ghxtox.prottox_spectral_refine import MODEL_NAME as SPECTRAL_MODEL_NAME, MODEL_VERSION as SPECTRAL_MODEL_VERSION
from ghxtox.prottox_tsk import MODEL_NAME as BASE_MODEL_NAME, MODEL_VERSION as BASE_MODEL_VERSION, write_manifest


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _seed(manifest: dict[str, Any]) -> int:
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
        # Early v1 manifests recorded the final-refit seed only. The trainer's
        # fixed full-data refit convention is `run_seed + 1000`.
        return int(metadata["seed"]) - 1000
    raise ValueError("Cannot recover the run seed from the legacy checkpoint metadata")


def _paired(manifests: list[str | Path], *, expected_name: str, expected_version: str) -> dict[int, dict[str, Any]]:
    if len(manifests) != 2:
        raise ValueError("Frozen comparison requires exactly two paired seed manifests")
    rows = [_read(path) for path in manifests]
    if any(row.get("model_name") != expected_name or row.get("model_version") != expected_version for row in rows):
        raise ValueError("Comparison received an incompatible model manifest")
    by_seed = {_seed(row): row for row in rows}
    if len(by_seed) != 2 or set(by_seed) != {20260905, 20260906}:
        raise ValueError("Comparison requires the frozen paired seeds 20260905 and 20260906")
    return by_seed


def compare(baseline_manifests: list[str | Path], gated_manifests: list[str | Path], output_path: str | Path, *, control_manifests: list[str | Path] | None = None) -> dict[str, Any]:
    baseline = _paired(baseline_manifests, expected_name=BASE_MODEL_NAME, expected_version=BASE_MODEL_VERSION)
    gated = _paired(gated_manifests, expected_name=SPECTRAL_MODEL_NAME, expected_version=SPECTRAL_MODEL_VERSION)
    deltas: list[dict[str, float]] = []
    for seed in sorted(baseline):
        base, candidate = baseline[seed], gated.get(seed)
        if candidate is None or candidate.get("variant") != "gated_residual":
            raise ValueError("Each candidate manifest must be the gated_residual variant")
        if base["cache"]["sha256"] != candidate["cache"]["sha256"] or base["fold_manifest"] != candidate["fold_manifest"]:
            raise ValueError("Baseline and candidate do not share the same audited cache/fold protocol")
        deltas.append({"seed": seed, "mcc": float(candidate["oof_metrics"]["mcc"] - base["oof_metrics"]["mcc"]), "auprc": float(candidate["oof_metrics"]["auprc"] - base["oof_metrics"]["auprc"]), "brier": float(candidate["oof_metrics"]["brier"] - base["oof_metrics"]["brier"])})
    mean_mcc = sum(row["mcc"] for row in deltas) / len(deltas)
    mean_auprc = sum(row["auprc"] for row in deltas) / len(deltas)
    both_mcc_not_lower = all(row["mcc"] >= 0.0 for row in deltas)
    promoted = both_mcc_not_lower and mean_mcc >= 0.005 and mean_auprc >= -0.002
    controls: list[dict[str, Any]] = []
    for path in control_manifests or []:
        row = _read(path)
        if row.get("model_name") != SPECTRAL_MODEL_NAME or row.get("variant") not in {"spectral_only", "concat"}:
            raise ValueError("Control manifests must be spectral_only or concat runs")
        controls.append({"path": str(path), "seed": row.get("seed"), "variant": row["variant"], "oof_metrics": row["oof_metrics"]})
    report = {"artifact_type": "prottox_spectral_refine_promotion_report", "selection_scope": "paired training OOF only; Test1/Test2 excluded", "frozen_rule": {"both_seeds_mcc_not_lower": True, "minimum_mean_mcc_advantage": 0.005, "maximum_mean_auprc_decline": 0.002}, "paired_deltas": deltas, "mean_delta": {"mcc": mean_mcc, "auprc": mean_auprc, "brier": sum(row["brier"] for row in deltas) / len(deltas)}, "decision": "promote_for_one_time_test1_audit" if promoted else "retain_as_ablation", "controls": controls}
    write_manifest(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs=2, required=True)
    parser.add_argument("--gated", nargs=2, required=True)
    parser.add_argument("--controls", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.gated, args.output, control_manifests=args.controls), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
