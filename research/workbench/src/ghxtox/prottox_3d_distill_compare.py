"""Compare ProtTox-3D-Distill OOF arms against retained controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("mcc", "auprc", "brier")


def load_oof_metrics(manifest_path: str | Path) -> dict[str, float]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    metrics = manifest.get("oof_metrics") or manifest.get("summary", {}).get("oof_metrics")
    if metrics is None:
        raise ValueError(f"No oof_metrics in {manifest_path}")
    return {key: float(metrics[key]) for key in METRICS}


def deltas(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {key: candidate[key] - reference[key] for key in METRICS}


def mean_by_seed(values: dict[str, dict[str, float]]) -> dict[str, float]:
    return {key: sum(value[key] for value in values.values()) / len(values) for key in METRICS}


def evaluate_gate(per_seed: dict[str, dict[str, float]], mean: dict[str, float]) -> dict[str, bool]:
    checks = {
        "both_seeds_mcc_not_lower": all(value["mcc"] >= 0 for value in per_seed.values()),
        "minimum_mean_mcc_advantage": mean["mcc"] >= 0.005,
        "maximum_mean_auprc_decline": mean["auprc"] >= -0.002,
        "maximum_mean_brier_increase": mean["brier"] <= 0.002,
    }
    return checks


def compare(seed_dirs: dict[str, str]) -> None:
    baseline_paths = {
        20260905: "runs/prottox_tsk_v1/mlp_fivefold_20epochs_v1/manifest.json",
        20260906: "runs/prottox_tsk_v1/mlp_fivefold_20epochs_seed20260906_v1/manifest.json",
    }
    structguide_paths = {
        20260905: "runs/prottox_tsk_v1/structguide_candidate_seed20260905_v1/manifest.json",
        20260906: "runs/prottox_tsk_v1/structguide_candidate_seed20260906_v1/manifest.json",
    }
    baseline = {seed: load_oof_metrics(path) for seed, path in baseline_paths.items()}
    structguide = {seed: load_oof_metrics(path) for seed, path in structguide_paths.items()}
    report: dict[str, object] = {
        "artifact_type": "prottox_3d_distill_promotion_report",
        "decision_target": "retained TriStat-MLP",
        "controls": {
            "mlp": baseline,
            "proxy_no_kd_structguide": structguide,
        },
        "arms": {},
        "primary_candidate": "proxy_teacher_kd",
        "selection_scope": "training OOF only; Test1/Test2 excluded",
    }
    for variant, template in seed_dirs.items():
        per_seed = {}
        per_seed_delta_vs_mlp = {}
        per_seed_delta_vs_structguide = {}
        for seed_text, directory in template.items():
            seed = int(seed_text)
            candidate = load_oof_metrics(Path(directory) / "manifest.json")
            per_seed[seed] = candidate
            per_seed_delta_vs_mlp[seed] = deltas(candidate, baseline[seed])
            per_seed_delta_vs_structguide[seed] = deltas(candidate, structguide[seed])
        mean_delta_vs_mlp = mean_by_seed(per_seed_delta_vs_mlp)
        mean_delta_vs_structguide = mean_by_seed(per_seed_delta_vs_structguide)
        report["arms"][variant] = {
            "oof_metrics_by_seed": per_seed,
            "delta_vs_retained_mlp": per_seed_delta_vs_mlp,
            "mean_delta_vs_retained_mlp": mean_delta_vs_mlp,
            "delta_vs_structguide_proxy_no_kd": per_seed_delta_vs_structguide,
            "mean_delta_vs_structguide_proxy_no_kd": mean_delta_vs_structguide,
            "gate_vs_retained_mlp": evaluate_gate(per_seed_delta_vs_mlp, mean_delta_vs_mlp),
        }
    primary_gate = report["arms"]["proxy_teacher_kd"]["gate_vs_retained_mlp"]
    report["decision"] = "promote_proxy_teacher_kd" if all(primary_gate.values()) else "retain_as_ablation"
    report["gate_checks"] = primary_gate
    output = Path(seed_dirs["proxy_teacher_kd"]["20260905"]).parent / "promotion_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proxy-kd",
        action="append",
        nargs=2,
        metavar=("SEED", "DIR"),
        required=True,
    )
    parser.add_argument(
        "--mlp-kd",
        action="append",
        nargs=2,
        metavar=("SEED", "DIR"),
        required=True,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_dirs = {
        "proxy_teacher_kd": {str(seed): directory for seed, directory in args.proxy_kd},
        "mlp_teacher_kd": {str(seed): directory for seed, directory in args.mlp_kd},
    }
    compare(seed_dirs)


if __name__ == "__main__":
    main()
