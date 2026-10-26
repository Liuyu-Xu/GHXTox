"""Lightweight metrics for the frozen MPCE-3DKD product ensemble.

The deployment contract is sequence-only and embeds every sequence in its own
ProtT5 forward pass to reproduce the canonical batch-size-1 product cache.
This benchmark reports the resulting per-query throughput and GPU memory on
the local RTX 4060.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ghxtox.models import physchem_features, tristat_pool
from ghxtox.mpce_3dkd_predict import load_ensemble_manifest, load_product_checkpoint
from ghxtox.plm_embed import _format_prott5_sequence, _load_prott5


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUNS = ROOT / "runs"
MANIFEST = RUNS / "mpce_3dkd_product_single_refit_v1" / "ensemble_manifest.json"
CACHE = RUNS / "mpce_3dkd_product_single_cache.pt"
PROTT5_SNAPSHOT = Path(
    r"D:\dachuang2026\ziliao\huggingface_cache\hub\models--Rostlab--prot_t5_xl_half_uniref50-enc"
    r"\snapshots\94a6abc029ae13029317b140b7424e012bf8dfbf"
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_ensemble_manifest(MANIFEST)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}
    members = {
        seed: load_product_checkpoint(checkpoint_paths[seed], device)
        for seed in (20260905, 20260906)
    }
    head_parameters = sum(
        sum(parameter.numel() for parameter in model.parameters())
        for model, *_rest in members.values()
    )
    head_parameters_single = head_parameters // 2

    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    benchmark_rows = payload["records"][:100]
    sequences = [str(row["sequence"]) for row in benchmark_rows]

    encoder, tokenizer, embedding_dim = _load_prott5(
        "prot_t5_xl_half_uniref50-enc",
        device,
        model_path=str(PROTT5_SNAPSHOT),
    )
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    encoder.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    def run_once(rows: list) -> tuple[float, torch.Tensor]:
        started = time.perf_counter()
        tri_list = []
        phys = physchem_features([str(row["sequence"]) for row in rows])
        with torch.inference_mode():
            for row in rows:
                sequence = str(row["sequence"])
                encoded = tokenizer(
                    [_format_prott5_sequence(sequence)],
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                hidden = encoder(**encoded).last_hidden_state.detach().cpu().float()
                residue = hidden[0, : len(sequence)].contiguous()
                if residue.shape[-1] != embedding_dim:
                    raise ValueError("Unexpected ProtT5 embedding dimension")
                tri_list.append(
                    tristat_pool(
                        residue.unsqueeze(0),
                        torch.ones((1, len(sequence)), dtype=torch.bool),
                    ).squeeze(0)
                )
            tri = torch.stack(tri_list)
            probabilities = []
            for seed, (model, tri_mean, tri_std, phys_mean, phys_std, metadata) in members.items():
                normalized_tri = (tri.to(device) - tri_mean) / tri_std
                normalized_phys = (phys.to(device) - phys_mean) / phys_std
                z_final, _z_global, _z_proto, _cosine = model(normalized_tri, normalized_phys)
                probabilities.append(torch.sigmoid(z_final).cpu())
            averaged = torch.stack(probabilities).mean(dim=0)
        elapsed = time.perf_counter() - started
        return elapsed, averaged

    # Warm-up and two timed runs.
    _warm, _ = run_once(benchmark_rows[:5])
    elapsed_first, _ = run_once(benchmark_rows)
    elapsed_second, _ = run_once(benchmark_rows)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available()
        else None
    )
    query_count = len(benchmark_rows)
    per_query_seconds = elapsed_second / query_count
    throughput = query_count / elapsed_second
    result = {
        "model": "MPCE-3DKD-Product (dual-seed ensemble)",
        "encoder": "Frozen ProtT5 prot_t5_xl_half_uniref50-enc",
        "prot_t5_parameters": encoder_parameters,
        "mpce_head_parameters_single_seed": head_parameters_single,
        "mpce_head_parameters_two_seed": head_parameters,
        "embedding_dim": 1024,
        "feature_dim": 3081,
        "per_sequence_forward": {
            "correctness_mode": "one sequence per ProtT5 forward (required for exact train/API parity)",
            "warmup_queries": 5,
            "timed_queries": query_count,
            "first_run_seconds": elapsed_first,
            "second_run_seconds": elapsed_second,
            "per_query_seconds": per_query_seconds,
            "queries_per_second": throughput,
        },
        "gpu_peak_memory_bytes": peak_memory,
        "gpu_peak_memory_mib": (peak_memory / 1024 / 1024) if peak_memory else None,
        "device": str(device),
        "esmfold_required": False,
        "3d_expert_required": False,
        "remote_gpu_required": False,
        "note": "ProtT5 model files occupy about 2.4 GB on disk; MPCE ensemble is a few MB. Throughput is measured on a local RTX 4060 Laptop GPU.",
    }
    (RUNS / "mpce_3dkd_lightweight_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def fmt(value: float, suffix: str = "") -> str:
        return f"{value:.4f}{suffix}"

    lines = [
        "# MPCE-3DKD lightweight metrics",
        "",
        "设备：本地 RTX 4060 Laptop GPU；只做训练 OOF 后的产品审计，未读 Test1/Test2。",
        "",
        "## 规模",
        "",
        "| 项目 | 数值 |",
        "| --- | ---: |",
        f"| ProtT5 encoder 参数 | {encoder_parameters:,} |",
        f"| MPCE head（单 seed） | {head_parameters_single:,} |",
        f"| MPCE head（双 seed ensemble） | {head_parameters:,} |",
        "| 输入特征 | 3,072 TriStat + 9 PhysChem |",
        "",
        "## 推理",
        "",
        "为保证任意新序列与 canonical batch-size-1 训练缓存完全一致，ProtT5 按单序列独立前向。",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 100 条 warm 后耗时（第二次） | {elapsed_second:.2f}s |",
        f"| 单条端到端 | {fmt(per_query_seconds * 1000, ' ms')} |",
        f"| 吞吐 | {fmt(throughput, ' seq/s')} |",
        f"| GPU 峰值显存 | {(peak_memory / 1024 / 1024):.0f} MiB（含 encoder 与双 seed） |",
        "",
        "## 部署结论",
        "",
        "- 不运行 ESMFold；不运行 GHXTox 3D/EGNN/chemical-site；不需要 pLDDT、AutoDL 或远程 GPU。",
        "- 最终推理链：sequence → Frozen ProtT5 → TriStat+PhysChem → MPCE dual-seed ensemble → probability。",
        "- 展示概率可按已冻结 temperature T=2.6918 校准，模型输出本身不变。",
        "",
    ]
    (RUNS / "mpce_3dkd_lightweight_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
