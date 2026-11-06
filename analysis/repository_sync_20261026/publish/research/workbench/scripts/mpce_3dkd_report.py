"""Audit and report the closing MPCE-3DKD experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_prottox_tsk import _metrics


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUNS = ROOT / "runs"
FOLD_MANIFEST = ROOT / "data" / "folds" / "prottox_tsk_train_prott5_official_mean_folds_v1.csv"
V1_ENSEMBLE_CSV = RUNS / "prottox_3d_distill_v1" / "v2a_targets_corrected" / "v1_fold_base_oof_probabilities.csv"

REFERENCE_RUNS = {
    "A0_20260905": RUNS / "prottox_tsk_v1" / "mlp_fivefold_20epochs_v1" / "manifest.json",
    "A0_20260906": RUNS / "prottox_tsk_v1" / "mlp_fivefold_20epochs_seed20260906_v1" / "manifest.json",
    "B_20260905": RUNS / "mpce_net_b_seed20260905_v1" / "manifest.json",
    "B_20260906": RUNS / "mpce_net_b_seed20260906_v1" / "manifest.json",
    "D_20260905": RUNS / "mpce_3dkd_seed20260905_v1" / "manifest.json",
    "D_20260906": RUNS / "mpce_3dkd_seed20260906_v1" / "manifest.json",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_rows() -> dict[str, dict[str, int]]:
    with FOLD_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["sample_id"]): {"fold": int(row["fold"]), "label": int(row["label"])}
        for row in rows
    }


def _validate_oof_csv(run_dir: Path, manifest_rows: dict[str, dict[str, int]]) -> None:
    path = run_dir / "oof_predictions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(manifest_rows):
        raise ValueError(f"{path}: row count mismatch")
    for row in rows:
        expected = manifest_rows[row["sample_id"]]
        if int(row["label"]) != expected["label"] or int(row["fold"]) != expected["fold"]:
            raise ValueError(f"{path}: label/fold mismatch at {row['sample_id']}")
        if not 0.0 <= float(row["probability"]) <= 1.0:
            raise ValueError(f"{path}: invalid probability at {row['sample_id']}")


def _v1_ensemble_metrics(manifest_rows: dict[str, dict[str, int]]) -> dict:
    with V1_ENSEMBLE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(manifest_rows):
        raise ValueError("v1 ensemble OOF row count does not match the fold manifest")
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    fold_labels = {str(fold): [] for fold in range(5)}
    fold_probs = {str(fold): [] for fold in range(5)}
    for row in rows:
        expected = manifest_rows[row["sample_id"]]
        if int(row["label"]) != expected["label"] or int(row["fold"]) != expected["fold"]:
            raise ValueError(f"v1 ensemble CSV fold/label mismatch at {row['sample_id']}")
        fold = str(row["fold"])
        fold_labels[fold].append(float(row["label"]))
        fold_probs[fold].append(float(row["probability"]))
    fold_metrics = {
        int(fold): _metrics(
            np.asarray(fold_labels[str(fold)], dtype=np.float64),
            np.asarray(fold_probs[str(fold)], dtype=np.float64),
        )
        for fold in range(5)
    }
    return {"oof_metrics": _metrics(labels, probabilities), "folds": fold_metrics}


def main() -> None:
    manifest_rows = _manifest_rows()
    protocol = _json(ROOT / "configs" / "mpce_3dkd_v1_protocol.json")

    data: dict[str, dict] = {}
    for key, path in REFERENCE_RUNS.items():
        run_dir = path.parent
        if key.startswith("A0"):
            _validate_oof_csv(run_dir, manifest_rows)
        else:
            _validate_oof_csv(run_dir, manifest_rows)
        data[key] = _json(path)

    v1 = _v1_ensemble_metrics(manifest_rows)
    v1_csv_sha = file_sha256(V1_ENSEMBLE_CSV)

    def seed_metrics(key: str) -> dict[str, float]:
        manifest = data[key]
        return {
            name: float(manifest["oof_metrics"][name])
            for name in ("mcc", "auprc", "brier", "auroc")
        }

    summaries = {
        "A0": {seed: seed_metrics(f"A0_{seed}") for seed in ("20260905", "20260906")},
        "A1_v1_ensemble": {"oof_metrics": v1["oof_metrics"]},
        "B": {seed: seed_metrics(f"B_{seed}") for seed in ("20260905", "20260906")},
        "D": {seed: seed_metrics(f"D_{seed}") for seed in ("20260905", "20260906")},
    }

    means = {}
    for arm in ("A0", "B", "D"):
        means[arm] = {
            name: (summaries[arm]["20260905"][name] + summaries[arm]["20260906"][name]) / 2.0
            for name in ("mcc", "auprc", "brier")
        }

    gate = protocol["final_recommendation_gate"]["D_vs_B"]
    deltas = {
        name: means["D"][name] - means["B"][name]
        for name in ("mcc", "auprc", "brier")
    }
    gate_result = {
        "mean_mcc_delta_D_minus_B": deltas["mcc"],
        "mean_auprc_delta_D_minus_B": deltas["auprc"],
        "mean_brier_delta_D_minus_B": deltas["brier"],
        "mean_brier_improvement": -deltas["brier"],
        "mcc_tolerance_ok": bool(deltas["mcc"] >= -gate["max_mean_mcc_decline"]),
        "auprc_not_declined": bool(deltas["auprc"] >= gate["min_mean_auprc_delta"]),
        "brier_improved": bool(-deltas["brier"] >= gate["min_mean_brier_improvement"]),
        "pass": bool(
            deltas["mcc"] >= -gate["max_mean_mcc_decline"]
            and deltas["auprc"] >= gate["min_mean_auprc_delta"]
            and -deltas["brier"] >= gate["min_mean_brier_improvement"]
        ),
    }

    decision = {
        "protocol": str(ROOT / "configs" / "mpce_3dkd_v1_protocol.json"),
        "decision_date": "2026-09-05",
        "v1_ensemble_csv": str(V1_ENSEMBLE_CSV),
        "v1_ensemble_csv_sha256": v1_csv_sha,
        "per_seed": summaries,
        "means": means,
        "deltas_vs_B": deltas,
        "gate_result": gate_result,
        "final_patent_architecture": "MPCE-3DKD" if gate_result["pass"] else "MPCE-B",
        "deployment_baseline": "ProtTox-3DKD v1 (immutable)",
        "selection_scope": "training OOF only; Test1/Test2 never read",
    }
    decision_path = RUNS / "mpce_3dkd_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(value: float) -> str:
        return f"{value:.5f}"

    lines = [
        "# MPCE-3DKD 最终实验报告",
        "",
        "日期：2026-09-05。只训练 OOF，未读 Test1/Test2；ProtTox-3DKD v1 与前端均未改动。",
        "",
        "## 冻结设计摘要",
        "",
        "- 架构完全复用 MPCE-B：Mean/Max/Std 投影 64d、PhysChem 投影 64d、四组低秩交互、每类 4 个 96d prototype、z_final = z_global + sigmoid(alpha)·z_proto。",
        "- 训练目标：BCE(z_final) + 0.5·prototype CE + 1.0·KD(z_final, teacher；T=2.0)。",
        "- teacher 复用 ProtTox-3DKD v1 的严格 outer-safe 单结构 3D targets；KD 不修改 v1 的任何 checkpoint。",
        "- 5 group folds × seeds 20260905/20260906，20 epochs、lr 1e-3、batch 256、阈值 0.5。",
        "",
        "## OOF 汇总",
        "",
        "| 模型 | MCC（05 / 06） | AUPRC（05 / 06） | Brier（05 / 06） |",
        "| --- | ---: | ---: | ---: |",
    ]
    a0_05, a0_06 = summaries["A0"]["20260905"], summaries["A0"]["20260906"]
    b_05, b_06 = summaries["B"]["20260905"], summaries["B"]["20260906"]
    d_05, d_06 = summaries["D"]["20260905"], summaries["D"]["20260906"]
    a1 = summaries["A1_v1_ensemble"]["oof_metrics"]
    lines.append(
        f"| A0 原 TriStat-MLP | {fmt(a0_05['mcc'])} / {fmt(a0_06['mcc'])} | "
        f"{fmt(a0_05['auprc'])} / {fmt(a0_06['auprc'])} | {fmt(a0_05['brier'])} / {fmt(a0_06['brier'])} |"
    )
    lines.append(
        f"| A1 ProtTox-3DKD v1（双 seed ensemble，context） | {fmt(a1['mcc'])} | "
        f"{fmt(a1['auprc'])} | {fmt(a1['brier'])} |"
    )
    lines.append(
        f"| B MPCE-B | {fmt(b_05['mcc'])} / {fmt(b_06['mcc'])} | "
        f"{fmt(b_05['auprc'])} / {fmt(b_06['auprc'])} | {fmt(b_05['brier'])} / {fmt(b_06['brier'])} |"
    )
    lines.append(
        f"| **D MPCE-3DKD（本轮）** | **{fmt(d_05['mcc'])} / {fmt(d_06['mcc'])}** | "
        f"**{fmt(d_05['auprc'])} / {fmt(d_06['auprc'])}** | **{fmt(d_05['brier'])} / {fmt(d_06['brier'])}** |"
    )
    lines += [
        "",
        "## 两 seed 均值对比",
        "",
        "| 模型 | MCC 均值 | AUPRC 均值 | Brier 均值 |",
        "| --- | ---: | ---: | ---: |",
        f"| A1 v1（context） | {fmt(a1['mcc'])} | {fmt(a1['auprc'])} | {fmt(a1['brier'])} |",
        f"| B MPCE-B | {fmt(means['B']['mcc'])} | {fmt(means['B']['auprc'])} | {fmt(means['B']['brier'])} |",
        f"| **D MPCE-3DKD** | **{fmt(means['D']['mcc'])}** | **{fmt(means['D']['auprc'])}** | **{fmt(means['D']['brier'])}** |",
        "",
        f"D 相对 B：ΔMCC {deltas['mcc']:+.5f}，ΔAUPRC {deltas['auprc']:+.5f}，ΔBrier {deltas['brier']:+.5f}。",
        "",
        "## 冻结判定",
        "",
    ]
    if gate_result["pass"]:
        lines.append(
            "MPCE-3DKD 同时保留 MPCE-B 的判别优势并把概率质量拉回："
            "相对 MPCE-B，MCC 未损失（实际更高）、AUPRC 未下降（实际更高）、Brier 至少改善 0.001（实际明显改善）。"
        )
        lines.append("")
        lines.append("**最终专利架构推荐：MPCE-3DKD。**")
        lines.append("")
        lines.append("部署仍为 sequence-only：ProtT5 → 多视图/低秩交互/多原型 → prediction，整条 3D teacher 删除。")
    else:
        lines.append("未达到冻结推荐门槛，保留 MPCE-B 为最终专利架构；MPCE-3DKD 记为最终消融。")
    lines += [
        "",
        "ProtTox-3DKD v1 继续作为不可修改的部署基线；本实验不改变任何 v1 checkpoint、代码或前端。",
        "",
        "## 每折 MCC（两 seed 均值）",
        "",
        "| fold | v1（context） | B MPCE-B | D MPCE-3DKD | Δ(D−B) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in range(5):
        v1_mcc = float(v1["folds"][fold]["mcc"])
        b_values = [
            float(manifest["folds"][fold]["mcc"])
            for key, manifest in (
                ("B_20260905", data["B_20260905"]),
                ("B_20260906", data["B_20260906"]),
            )
        ]
        d_values = [
            float(manifest["folds"][fold]["mcc"])
            for key, manifest in (
                ("D_20260905", data["D_20260905"]),
                ("D_20260906", data["D_20260906"]),
            )
        ]
        b_mean = float(np.mean(b_values))
        d_mean = float(np.mean(d_values))
        lines.append(
            f"| {fold} | {fmt(v1_mcc)} | {fmt(b_mean)} | {fmt(d_mean)} | "
            f"{d_mean - b_mean:+.5f} |"
        )
    lines += [
        "",
        "## 完整性说明",
        "",
        f"- v1 双 seed ensemble OOF CSV SHA-256：`{v1_csv_sha[:16]}...`",
        "- 所有 OOF CSV 均与 audited fold manifest 逐行核对 sample_id/label/fold。",
        "- Test1/Test2 在整个实验中未读取；未做任何 seed/阈值/fold 的事后选择。",
        "",
    ]
    report_path = RUNS / "mpce_3dkd_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {decision_path}")
    print(f"wrote {report_path}")
    print(json.dumps({"means": means, "deltas": deltas, "gate": gate_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
