"""Summarize and audit the frozen PepBERT representation shootout."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUN_ROOT = ROOT / "runs" / "representation_shootout_v1"
FOLD_MANIFEST = ROOT / "data" / "folds" / "prottox_tsk_train_prott5_official_mean_folds_v1.csv"

ARMS = {
    "A": {
        "label": "A0 ProtT5 TriStat-MLP (no KD)",
        "seeds": {
            20260905: ROOT / "runs" / "prottox_tsk_v1" / "mlp_fivefold_20epochs_v1",
            20260906: ROOT / "runs" / "prottox_tsk_v1" / "mlp_fivefold_20epochs_seed20260906_v1",
        },
    },
    "B": {
        "label": "B PepBERT-small-UniParc TriStat-MLP (no KD)",
        "seeds": {
            20260905: RUN_ROOT / "pepbert_small_uniparc_seed20260905_v1",
            20260906: RUN_ROOT / "pepbert_small_uniparc_seed20260906_v1",
        },
    },
    "C": {
        "label": "C PepBERT-large-UniParc TriStat-MLP (no KD)",
        "seeds": {
            20260905: RUN_ROOT / "pepbert_large_uniparc_seed20260905_v1",
            20260906: RUN_ROOT / "pepbert_large_uniparc_seed20260906_v1",
        },
    },
}

FEATURE_MANIFESTS = {
    "B": RUN_ROOT / "pepbert_small_uniparc_features.pt.manifest.json",
    "C": RUN_ROOT / "pepbert_large_uniparc_features.pt.manifest.json",
}

PRIMARY = ("mcc", "auprc", "brier")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fold_rows() -> dict[str, dict[str, int]]:
    with FOLD_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row["sample_id"]): {
            "fold": int(row["fold"]),
            "label": int(row["label"]),
        }
        for row in rows
    }


def _validate_oof(run_dir: Path, manifest_rows: dict[str, dict[str, int]]) -> None:
    path = run_dir / "oof_predictions.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(manifest_rows):
        raise ValueError(f"{run_dir}: OOF row count {len(rows)} != manifest {len(manifest_rows)}")
    for row in rows:
        expected = manifest_rows[row["sample_id"]]
        if int(row["fold"]) != expected["fold"] or int(row["label"]) != expected["label"]:
            raise ValueError(f"{run_dir}: OOF row {row['sample_id']} fold/label mismatch")
        probability = float(row["probability"])
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{run_dir}: non-probability value in OOF CSV")


def main() -> None:
    manifest_rows = _fold_rows()
    protocol = _json(ROOT / "configs" / "representation_shootout_v1.json")
    gate = protocol["continue_gate"]["per_candidate_requirements"]

    summaries: dict[str, dict[str, dict]] = {}
    for arm, definition in ARMS.items():
        summaries[arm] = {}
        for seed, run_dir in definition["seeds"].items():
            _validate_oof(run_dir, manifest_rows)
            manifest = _json(run_dir / "manifest.json")
            metrics = {
                key: float(manifest["oof_metrics"][key])
                for key in (*PRIMARY, "auroc")
            }
            fold_mcc = [float(row["mcc"]) for row in manifest["folds"]]
            summaries[arm][str(seed)] = {
                "run_dir": str(run_dir),
                "metrics": metrics,
                "fold_mcc": fold_mcc,
                "parameter_count": int(manifest["parameter_count"]),
                "cache": manifest["cache"],
            }

    means: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        seed_values = summaries[arm]
        means[arm] = {
            key: sum(seed_values[seed]["metrics"][key] for seed in seed_values) / 2.0
            for key in PRIMARY
        }

    def delta(candidate: str, metric: str) -> float:
        return means[candidate][metric] - means["A"][metric]

    gates = {}
    for candidate in ("B", "C"):
        mcc_ok = delta(candidate, "mcc") >= 0.0
        auprc_ok = delta(candidate, "auprc") >= -gate["max_mean_auprc_decline_vs_A"]
        brier_ok = delta(candidate, "brier") <= gate["max_mean_brier_increase_vs_A"]
        improvement = max(
            delta(candidate, "mcc"),
            delta(candidate, "auprc"),
            -delta(candidate, "brier"),
        )
        improvement_ok = improvement >= gate["minimum_primary_improvement"]
        gates[candidate] = {
            "delta_mcc": delta(candidate, "mcc"),
            "delta_auprc": delta(candidate, "auprc"),
            "delta_brier": delta(candidate, "brier"),
            "max_improvement": improvement,
            "mean_mcc_not_lower": mcc_ok,
            "auprc_within_tolerance": auprc_ok,
            "brier_within_tolerance": brier_ok,
            "has_minimum_improvement": improvement_ok,
            "continue": bool(mcc_ok and auprc_ok and brier_ok and improvement_ok),
        }

    comparison = {
        "protocol": str(ROOT / "configs" / "representation_shootout_v1.json"),
        "decision_date": "2026-09-05",
        "per_seed_summaries": summaries,
        "two_seed_means": means,
        "candidate_deltas_vs_A": {key: value for key, value in gates.items()},
        "fold_mcc_table": {
            arm: [
                summaries[arm][seed]["fold_mcc"]
                for seed in ("20260905", "20260906")
            ]
            for arm in ARMS
        },
        "decision": "stop_pepbert_representation_line" if not (gates["B"]["continue"] or gates["C"]["continue"]) else "continue_to_amt",
        "note": "ProtTox-3DKD v1 remains the immutable deployment baseline; no model/checkpoint changed.",
    }
    comparison_path = RUN_ROOT / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def fmt(value: float) -> str:
        return f"{value:.5f}"

    lines = [
        "# PepBERT Representation Shootout（V1 冻结协议）结果",
        "",
        "日期：2026-09-05。只训练 OOF，未读 Test1/Test2；ProtTox-3DKD v1 与现有前端均未改动。",
        "",
        "## OOF 汇总",
        "",
        "| 臂 | MCC（20260905 / 20260906） | AUPRC（05 / 06） | Brier（05 / 06） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in ("A", "B", "C"):
        s05, s06 = summaries[arm]["20260905"], summaries[arm]["20260906"]
        lines.append(
            f"| {ARMS[arm]['label']} | {fmt(s05['metrics']['mcc'])} / {fmt(s06['metrics']['mcc'])} "
            f"| {fmt(s05['metrics']['auprc'])} / {fmt(s06['metrics']['auprc'])} "
            f"| {fmt(s05['metrics']['brier'])} / {fmt(s06['metrics']['brier'])} |"
        )
    lines += [
        "",
        "两 seed 均值差（相对 A0）：",
        "",
        "| 候选 | ΔMCC | ΔAUPRC | ΔBrier | 判定 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for candidate in ("B", "C"):
        result = gates[candidate]
        lines.append(
            f"| {ARMS[candidate]['label']} | {result['delta_mcc']:+.5f} | "
            f"{result['delta_auprc']:+.5f} | {result['delta_brier']:+.5f} | "
            f"{'通过' if result['continue'] else '不通过'} |"
        )
    lines += [
        "",
        "## 每折 MCC",
        "",
        "| 臂/seed | fold0 | fold1 | fold2 | fold3 | fold4 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("A", "B", "C"):
        for seed in ("20260905", "20260906"):
            fold_mcc = summaries[arm][seed]["fold_mcc"]
            values = " | ".join(fmt(value) for value in fold_mcc)
            lines.append(f"| {arm} {seed} | {values} |")
    lines += [
        "",
        "## 运行/规模信息",
        "",
        "| 臂 | encoder | 参数量 | MLP 参数 | TriStat 维度 | 全量 6387 特征生成 | 峰值显存 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("B", "C"):
        feature = _json(FEATURE_MANIFESTS[arm])
        encoder = feature["encoder"]
        mlp_params = summaries[arm]["20260905"]["parameter_count"]
        lines.append(
            f"| {arm} | {encoder['identifier']} | {encoder['parameters']:,} | {mlp_params:,} | "
            f"{3 * encoder['embedding_dim'] + 9} | "
            f"{feature['timing']['total_elapsed_seconds']:.2f}s | "
            f"{feature['peak_cuda_memory_bytes'] / 1024 / 1024:.1f} MiB |"
        )
    lines += [
        "",
        "## 判定",
        "",
        "两个 PepBERT 臂的双 seed MCC、AUPRC、Brier 全部稳定差于现有 ProtT5 TriStat-MLP。",
        "",
        "- B（PepBERT-small-UniParc）：MCC −0.097 左右，AUPRC −0.066，Brier +0.025；",
        "- C（PepBERT-large-UniParc）：MCC −0.053，AUPRC −0.035，Brier +0.014；",
        "- 没有任何候选满足冻结的 continue gate，且不是单折/单 seed 偶然，而是 5 折 2 seed 一致落后。",
        "",
        "**结论：按冻结协议停止 PepBERT representation 线，不进入 AMT-PepTox 设计。**",
        "",
        "现有 ProtTox-3DKD v1 继续作为不可修改的部署保底；后续不会把本次失败作为借口回退交付。",
        "",
    ]
    (RUN_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {comparison_path}")
    print(f"wrote {RUN_ROOT / 'REPORT.md'}")
    print(json.dumps(gates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
