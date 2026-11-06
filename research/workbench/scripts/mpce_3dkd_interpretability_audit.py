"""Prototype interpretability and low-rank interaction audit for MPCE-3DKD.

This is a read-only audit of the frozen full MPCE-3DKD OOF checkpoints.  It
does not retrain or modify the candidate.  For every held-out fold the
fold-specific checkpoint is applied to that fold's records only.

Outputs:
- runs/mpce_3dkd_prototype_audit.md / .json
- runs/mpce_3dkd_interaction_audit.md / .json
- runs/mpce_3dkd_prototype_assignment_ledger.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from ghxtox.models import PHYS_CHEM_FEATURE_NAMES
from ghxtox.train_mpce_net import MPCEB, PROTOTYPES_PER_CLASS
from ghxtox.train_prottox_tsk import _metrics


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUNS = ROOT / "runs"
CACHE = RUNS / "prottox_tsk_v1" / "structure_quality_proxy_v1.pt"
SEEDS = ("20260905", "20260906")
BRANCH_NAMES = ("mean_max", "mean_std", "max_std", "sequence_physchem")


def _load_cache():
    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    records = payload["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    return records, tri, phys, labels, folds


def _forward(model: torch.nn.Module, tri: torch.Tensor, phys: torch.Tensor, zero_branch: int | None = None):
    h_mean = model.proj_mean(tri[:, :1024])
    h_max = model.proj_max(tri[:, 1024:2048])
    h_std = model.proj_std(tri[:, 2048:])
    h_phys = model.proj_phys(phys)
    branches = [
        h_mean * h_max,
        h_mean * h_std,
        h_max * h_std,
        h_mean * h_phys,
    ]
    interaction_blocks = list(branches)
    if zero_branch is not None:
        interaction_blocks[zero_branch] = torch.zeros_like(interaction_blocks[zero_branch])
    interactions = torch.cat(interaction_blocks, dim=-1)
    vector = torch.cat([h_mean, h_max, h_std, h_phys, interactions], dim=-1)
    z_global = model.global_mlp(vector).squeeze(-1)
    latent = model.latent(interactions)
    latent = torch.nn.functional.normalize(latent, dim=-1)
    prototypes = torch.nn.functional.normalize(model.prototypes, dim=-1)
    cosine = latent @ prototypes.t()
    toxic_scores = cosine[:, :PROTOTYPES_PER_CLASS]
    nontoxic_scores = cosine[:, PROTOTYPES_PER_CLASS:]
    affinity_toxic = torch.logsumexp(toxic_scores, dim=-1)
    affinity_nontoxic = torch.logsumexp(nontoxic_scores, dim=-1)
    z_proto = affinity_toxic - affinity_nontoxic
    z_final = z_global + torch.sigmoid(model.alpha) * z_proto
    return {
        "h_mean": h_mean,
        "h_max": h_max,
        "h_std": h_std,
        "h_phys": h_phys,
        "branches": branches,
        "interactions": interactions,
        "latent": latent,
        "cosine": cosine,
        "z_global": z_global,
        "z_proto": z_proto,
        "z_final": z_final,
    }


def _load_model(fold_checkpoint: Path, device: torch.device) -> tuple[MPCEB, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(fold_checkpoint, map_location="cpu", weights_only=False)
    model = MPCEB()
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return (
        model,
        torch.as_tensor(payload["tri_mean"], dtype=torch.float32),
        torch.as_tensor(payload["tri_std"], dtype=torch.float32),
        torch.as_tensor(payload["phys_mean"], dtype=torch.float32),
        torch.as_tensor(payload["phys_std"], dtype=torch.float32),
    )


def main() -> None:
    records, tri, phys, labels, folds = _load_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_records = len(records)
    sample_index = {str(record["sample_id"]): index for index, record in enumerate(records)}
    branch_magnitudes = {name: [] for name in BRANCH_NAMES}
    full_probability = {seed: np.full(n_records, np.nan, dtype=np.float64) for seed in SEEDS}
    variant_probability = {
        seed: {name: np.full(n_records, np.nan, dtype=np.float64) for name in BRANCH_NAMES}
        for seed in SEEDS
    }
    sample_rows = []

    for seed in SEEDS:
        for fold in range(5):
            fold_dir = RUNS / f"mpce_3dkd_seed{seed}_v1"
            model, tri_mean, tri_std, phys_mean, phys_std = _load_model(fold_dir / f"fold_{fold}.pt", device)
            test_idx = np.flatnonzero(folds == fold)
            test_tri = ((tri[test_idx] - tri_mean) / tri_std).to(device)
            test_phys = ((phys[test_idx] - phys_mean) / phys_std).to(device)
            batch_size = 256
            with torch.inference_mode():
                for start in range(0, len(test_idx), batch_size):
                    stop = min(start + batch_size, len(test_idx))
                    chunk_tri = test_tri[start:stop]
                    chunk_phys = test_phys[start:stop]
                    outputs = _forward(model, chunk_tri, chunk_phys)
                    full_probability[seed][test_idx[start:stop]] = torch.sigmoid(outputs["z_final"]).cpu().numpy()
                    for branch_index, branch_name in enumerate(BRANCH_NAMES):
                        variant = _forward(model, chunk_tri, chunk_phys, zero_branch=branch_index)
                        variant_probability[seed][branch_name][test_idx[start:stop]] = (
                            torch.sigmoid(variant["z_final"]).cpu().numpy()
                        )
                    for sample_position, row_index in enumerate(test_idx[start:stop]):
                        cosine = outputs["cosine"][sample_position].cpu().numpy()
                        toxic_part = cosine[:PROTOTYPES_PER_CLASS]
                        nontoxic_part = cosine[PROTOTYPES_PER_CLASS:]
                        toxic_proto = int(np.argmax(toxic_part))
                        nontoxic_proto = int(np.argmax(nontoxic_part)) + PROTOTYPES_PER_CLASS
                        global_proto = int(np.argmax(cosine))
                        sample_rows.append(
                            {
                                "sample_id": records[row_index]["sample_id"],
                                "historical_sample_id": records[row_index]["historical_sample_id"],
                                "sequence": records[row_index]["sequence"],
                                "label": int(records[row_index]["label"]),
                                "fold": fold,
                                "seed": seed,
                                "probability": float(torch.sigmoid(outputs["z_final"][sample_position]).item()),
                                "z_proto": float(outputs["z_proto"][sample_position].item()),
                                "global_closest_prototype": global_proto,
                                "toxic_side_prototype": toxic_proto,
                                "nontoxic_side_prototype": nontoxic_proto,
                                "toxic_side_top_cosine": float(toxic_part.max()),
                                "nontoxic_side_top_cosine": float(nontoxic_part.max()),
                            }
                        )
                    for branch_index, branch_name in enumerate(BRANCH_NAMES):
                        branch_magnitudes[branch_name].append(
                            float(outputs["branches"][branch_index].abs().mean().item())
                        )

    # Interaction audit aggregate metrics.
    def overall(probabilities: np.ndarray) -> dict[str, float]:
        metrics = _metrics(labels, probabilities)
        return {key: float(metrics[key]) for key in ("mcc", "auprc", "brier")}

    interaction_summary = {
        "full_two_seed_mean": {
            key: float(np.mean([overall(full_probability[seed])[key] for seed in SEEDS]))
            for key in ("mcc", "auprc", "brier")
        },
        "branch": {},
    }
    for branch_name in BRANCH_NAMES:
        zeroed_metrics = {
            seed: overall(variant_probability[seed][branch_name])
            for seed in SEEDS
        }
        branch_mean = {
            key: float(np.mean([zeroed_metrics[seed][key] for seed in SEEDS]))
            for key in ("mcc", "auprc", "brier")
        }
        interaction_summary["branch"][branch_name] = {
            "mean_abs_activation": float(np.mean(branch_magnitudes[branch_name])),
            "zeroed_two_seed_mean": branch_mean,
            "full_minus_zeroed": {
                key: interaction_summary["full_two_seed_mean"][key] - branch_mean[key]
                for key in ("mcc", "auprc", "brier")
            },
        }

    # Prototype interpretability aggregate.
    ledger_path = RUNS / "mpce_3dkd_prototype_assignment_ledger.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "historical_sample_id",
                "sequence",
                "label",
                "fold",
                "seed",
                "probability",
                "z_proto",
                "global_closest_prototype",
                "toxic_side_prototype",
                "nontoxic_side_prototype",
                "toxic_side_top_cosine",
                "nontoxic_side_top_cosine",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    global_counts = np.zeros(2 * PROTOTYPES_PER_CLASS, dtype=np.int64)
    toxic_counts = np.zeros(PROTOTYPES_PER_CLASS, dtype=np.int64)
    nontoxic_counts = np.zeros(PROTOTYPES_PER_CLASS, dtype=np.int64)
    label_toxic_fraction = np.zeros(2 * PROTOTYPES_PER_CLASS, dtype=np.float64)
    label_toxic_fraction[:] = np.nan
    for row in sample_rows:
        global_counts[row["global_closest_prototype"]] += 1
        toxic_counts[row["toxic_side_prototype"]] += 1
        nontoxic_counts[row["nontoxic_side_prototype"] - PROTOTYPES_PER_CLASS] += 1
    for prototype in range(2 * PROTOTYPES_PER_CLASS):
        selected = [row for row in sample_rows if row["global_closest_prototype"] == prototype]
        if selected:
            label_toxic_fraction[prototype] = float(np.mean([row["label"] for row in selected]))

    # Note ledger contains both seeds, so each sample appears twice. For
    # prototype usage statistics we report per-seed-normalized shares below.
    n_rows_per_seed = len(sample_rows) // 2
    per_seed_global_counts = global_counts.astype(np.float64) / 2.0
    distribution = {
        prototype: round(float(count), 1)
        for prototype, count in enumerate(per_seed_global_counts)
    }
    entropy = -sum(
        (count / n_rows_per_seed) * math.log2(count / n_rows_per_seed)
        for count in per_seed_global_counts
        if count > 0
    )
    max_entropy = math.log2(2 * PROTOTYPES_PER_CLASS)

    prototype_summary = {
        "n_prototypes": 2 * PROTOTYPES_PER_CLASS,
        "assignment_scope": "each sample appears once per seed; fold-specific checkpoint used on its own outer fold",
        "global_closest_counts_per_seed": {int(k): v for k, v in distribution.items()},
        "toxic_side_counts_per_seed": {int(k): int(v) / 2 for k, v in enumerate(toxic_counts)},
        "nontoxic_side_counts_per_seed": {int(k): int(v) / 2 for k, v in enumerate(nontoxic_counts)},
        "global_assignment_entropy_bits": entropy,
        "max_entropy_bits": max_entropy,
        "normalized_entropy": entropy / max_entropy if max_entropy else float("nan"),
        "label_toxic_fraction_by_global_closest": {
            int(k): float(v) for k, v in enumerate(label_toxic_fraction)
        },
    }

    # PhysChem means by globally closest prototype.
    physchem_by_prototype = np.zeros((2 * PROTOTYPES_PER_CLASS, len(PHYS_CHEM_FEATURE_NAMES)), dtype=np.float64)
    counts_by_prototype = np.zeros(2 * PROTOTYPES_PER_CLASS, dtype=np.int64)
    for row in sample_rows:
        record_index = sample_index[row["sample_id"]]
        phys_values = phys[record_index].numpy()
        prototype = row["global_closest_prototype"]
        physchem_by_prototype[prototype] += phys_values
        counts_by_prototype[prototype] += 1
    for prototype in range(2 * PROTOTYPES_PER_CLASS):
        if counts_by_prototype[prototype]:
            physchem_by_prototype[prototype] /= counts_by_prototype[prototype]
    prototype_summary["physchem_mean_by_global_closest"] = {
        int(prototype): {
            str(name): float(value)
            for name, value in zip(PHYS_CHEM_FEATURE_NAMES, physchem_by_prototype[prototype], strict=True)
        }
        for prototype in range(2 * PROTOTYPES_PER_CLASS)
    }
    prototype_summary["collapse_check"] = (
        "no_collapse" if all(prototype_summary["global_closest_counts_per_seed"][p] > 20 for p in range(8)) else "check"
    )

    (RUNS / "mpce_3dkd_interaction_audit.json").write_text(
        json.dumps(interaction_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (RUNS / "mpce_3dkd_prototype_audit.json").write_text(
        json.dumps(prototype_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def fmt(value: float) -> str:
        return f"{value:.5f}"

    interaction_lines = [
        "# MPCE-3DKD low-rank interaction branch audit",
        "",
        "方法：对冻结 fold checkpoint 的各自 outer fold，逐一将该 64d 交互分支置零并重算最终概率。",
        "",
        "| branch | mean abs activation | MCC（zeroed 均值） | AUPRC（zeroed） | Brier（zeroed） | full−zeroed MCC | full−zeroed AUPRC | full−zeroed Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for branch_name in BRANCH_NAMES:
        branch = interaction_summary["branch"][branch_name]
        zeroed = branch["zeroed_two_seed_mean"]
        delta = branch["full_minus_zeroed"]
        interaction_lines.append(
            f"| {branch_name} | {branch['mean_abs_activation']:.5f} | {fmt(zeroed['mcc'])} | "
            f"{fmt(zeroed['auprc'])} | {fmt(zeroed['brier'])} | {delta['mcc']:+.5f} | "
            f"{delta['auprc']:+.5f} | {delta['brier']:+.5f} |"
        )
    interaction_lines += [
        "",
        f"完整 MPCE-3DKD 两 seed 均值：MCC {fmt(interaction_summary['full_two_seed_mean']['mcc'])}、"
        f"AUPRC {fmt(interaction_summary['full_two_seed_mean']['auprc'])}、"
        f"Brier {fmt(interaction_summary['full_two_seed_mean']['brier'])}。",
        "",
        "结论：若某一分支的置零几乎不改变指标，则该分支仍可能只提供边际信息；若四个分支均有非零影响，则交互模块不是退化空转。",
        "",
    ]
    (RUNS / "mpce_3dkd_interaction_audit.md").write_text("\n".join(interaction_lines), encoding="utf-8")

    proto_lines = [
        "# MPCE-3DKD prototype interpretability audit",
        "",
        "prototype 共 8 个：0–3 对应 toxic affinity 侧，4–7 对应 non-toxic affinity 侧；统计基于各自 fold checkpoint 对自身 outer fold 的输出，样本按 seed 各出现一次。",
        "",
        "## 使用/坍缩检查",
        "",
        f"- global closest 每 seed 计数：{prototype_summary['global_closest_counts_per_seed']}",
        f"- toxic 侧计数：{prototype_summary['toxic_side_counts_per_seed']}",
        f"- non-toxic 侧计数：{prototype_summary['nontoxic_side_counts_per_seed']}",
        f"- 分配熵 {prototype_summary['global_assignment_entropy_bits']:.3f} / {prototype_summary['max_entropy_bits']:.3f} bits"
        f"（normalized {prototype_summary['normalized_entropy']:.3f}）",
        f"- collapse check：{prototype_summary['collapse_check']}",
        "",
        "## global-closest prototype 的 label toxic fraction 与 PhysChem 均值",
        "",
        "| prototype | label=1 fraction | length | net charge | hydropathy | basic frac | hydrophobic frac | aromatic frac | Cys count |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for prototype in range(8):
        label_frac = prototype_summary["label_toxic_fraction_by_global_closest"][prototype]
        physchem = prototype_summary["physchem_mean_by_global_closest"][prototype]
        proto_lines.append(
            f"| {prototype} | {label_frac:.3f} | {physchem['length']:.1f} | "
            f"{physchem['net_charge_ph7']:.2f} | {physchem['mean_kyte_doolittle_hydropathy']:.2f} | "
            f"{physchem['basic_fraction']:.3f} | {physchem['hydrophobic_fraction']:.3f} | "
            f"{physchem['aromatic_fraction']:.3f} | {physchem['cysteine_count']:.2f} |"
        )
    proto_lines += [
        "",
        "完整逐样本 assignment 见 `runs/mpce_3dkd_prototype_assignment_ledger.csv`。",
        "",
    ]
    (RUNS / "mpce_3dkd_prototype_audit.md").write_text("\n".join(proto_lines), encoding="utf-8")
    print(f"wrote {ledger_path}")
    print(f"wrote {RUNS / 'mpce_3dkd_interaction_audit.md'}")
    print(f"wrote {RUNS / 'mpce_3dkd_prototype_audit.md'}")
    print(json.dumps({"interaction": interaction_summary, "prototype": prototype_summary}, ensure_ascii=False, indent=2)[:5000])


if __name__ == "__main__":
    main()
