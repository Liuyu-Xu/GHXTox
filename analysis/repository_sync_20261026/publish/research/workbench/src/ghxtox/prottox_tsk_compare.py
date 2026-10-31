"""Audit paired same-protocol ProtTox-TSK MLP/KAN run manifests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from ghxtox.prottox_tsk import write_manifest

_METRICS = ("mcc", "auroc", "auprc", "brier")


def _load_run(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path) / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(Path(path) / "final_model.pt", map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    # Older v1 artifacts stored only the final-refit seed (= run seed + 1000).
    # Keep that legacy interpretation explicit while new manifests store seed.
    seed = int(payload.get("seed", metadata.get("run_seed", int(metadata.get("seed", -1)) - 1000)))
    return {"path": str(Path(path)), "manifest": payload, "seed": seed}


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std_population": pstdev(values) if len(values) > 1 else 0.0, "n": len(values)}


def compare(mlp_paths: list[str], kan_paths: list[str], output_dir: str | Path) -> dict[str, Any]:
    if len(mlp_paths) != len(kan_paths) or not mlp_paths:
        raise ValueError("Provide equal non-empty ordered lists of MLP and KAN run directories")
    mlp_runs, kan_runs = [_load_run(path) for path in mlp_paths], [_load_run(path) for path in kan_paths]
    pairs: list[dict[str, Any]] = []
    for mlp, kan in zip(mlp_runs, kan_runs):
        left, right = mlp["manifest"], kan["manifest"]
        if left["head_kind"] != "mlp" or right["head_kind"] != "kan":
            raise ValueError("Run directories are not ordered as MLP then KAN")
        for field in ("cache", "fold_manifest", "epochs", "learning_rate"):
            if left[field] != right[field]:
                raise ValueError(f"Paired runs differ in {field}")
        if left["cache"]["sha256"] != right["cache"]["sha256"]:
            raise ValueError("Paired runs use different frozen feature caches")
        deltas = {metric: float(right["oof_metrics"][metric]) - float(left["oof_metrics"][metric]) for metric in _METRICS}
        pairs.append({"mlp_run": mlp["path"], "kan_run": kan["path"], "mlp_seed": mlp["seed"], "kan_seed": kan["seed"], "deltas_kan_minus_mlp": deltas})
    deltas = {metric: _summary([pair["deltas_kan_minus_mlp"][metric] for pair in pairs]) for metric in _METRICS}
    # Promotion requires no MCC regression in every same-protocol run.  Ranking
    # gains or a smaller model alone are not enough because the project uses MCC
    # as its primary class-imbalance decision metric.
    mcc_deltas = [pair["deltas_kan_minus_mlp"]["mcc"] for pair in pairs]
    promote_kan = all(delta > 0.0 for delta in mcc_deltas)
    decision = "retain_kan_candidate" if promote_kan else "retain_mlp_default_keep_kan_ablation"
    reason = "KAN improved MCC in every paired run." if promote_kan else "KAN did not improve the primary MCC metric in every paired same-protocol run; it remains an ablation candidate despite any ranking/calibration gains."
    report = {"comparison": "ProtTox-TSK TriStat-KAN versus TriStat-MLP", "paired_runs": pairs, "delta_summary_kan_minus_mlp": deltas, "decision": decision, "reason": reason, "scope": "training OOF comparison only; no held-out Test1/Test2 result is claimed", "deployment": {"mlp_parameter_counts": [run["manifest"]["parameter_count"] for run in mlp_runs], "kan_parameter_counts": [run["manifest"]["parameter_count"] for run in kan_runs], "mlp_elapsed_seconds": [run["manifest"]["elapsed_seconds"] for run in mlp_runs], "kan_elapsed_seconds": [run["manifest"]["elapsed_seconds"] for run in kan_runs]}}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output / "comparison.json", report)
    lines = ["# ProtTox-TSK MLP vs KAN OOF audit", "", f"Decision: **{decision}**", "", reason, "", "| Metric | Mean KAN − MLP | Population SD |", "|---|---:|---:|"]
    for metric in _METRICS:
        lines.append(f"| {metric.upper()} | {deltas[metric]['mean']:.6f} | {deltas[metric]['std_population']:.6f} |")
    lines.extend(["", "This audit compares only the listed same-protocol training OOF runs. It is not a Test1/Test2 claim."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-runs", nargs="+", required=True)
    parser.add_argument("--kan-runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = compare(args.mlp_runs, args.kan_runs, args.output_dir)
    print(json.dumps({"decision": result["decision"], "reason": result["reason"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
