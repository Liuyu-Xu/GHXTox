"""Deployment-consistency audit for the frozen MPCE-3DKD product ensemble.

The audit compares:
1. feature-cache inference (the exact training-time feature contract) with a
   standalone sequence runner that recomputes ProtT5 residue embeddings;
2. repeated standalone runner calls for deterministic equivalence.

The fixed audit set is 200 training sequences selected deterministically
(40 per fold, RNG seed 20260905) before running; labels are not used for
model selection and no Test1/Test2 sequence is included.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ghxtox.mpce_3dkd_predict import load_ensemble_manifest, load_product_checkpoint, predict_sequences
from ghxtox.prottox_tsk import file_sha256


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUNS = ROOT / "runs"
CACHE = RUNS / "mpce_3dkd_product_single_cache.pt"
PRODUCT_DIR = RUNS / "mpce_3dkd_product_single_refit_v1"
MANIFEST = PRODUCT_DIR / "ensemble_manifest.json"
PROTT5_SNAPSHOT = Path(
    r"D:\dachuang2026\ziliao\huggingface_cache\hub\models--Rostlab--prot_t5_xl_half_uniref50-enc"
    r"\snapshots\94a6abc029ae13029317b140b7424e012bf8dfbf"
)
AUDIT_PER_FOLD = 40
AUDIT_RNG_SEED = 20260905
TOLERANCE_RUNNER_REPEAT = 1e-9
TOLERANCE_CACHE_VS_RUNNER = 1e-5


def _load_cache():
    payload = torch.load(CACHE, map_location="cpu", weights_only=False)
    records = payload["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    return records, tri, phys


def main() -> None:
    records, tri, phys = _load_cache()
    fold_indices = {fold: [] for fold in range(5)}
    for index, record in enumerate(records):
        fold_indices[int(record["fold"])].append(index)
    rng = np.random.default_rng(AUDIT_RNG_SEED)
    selected: list[int] = []
    for fold in range(5):
        indices = np.asarray(fold_indices[fold], dtype=np.int64)
        if len(indices) < AUDIT_PER_FOLD:
            raise ValueError(f"Fold {fold} has fewer than {AUDIT_PER_FOLD} records")
        chosen = rng.choice(indices, size=AUDIT_PER_FOLD, replace=False)
        selected.extend(int(value) for value in chosen)
    if len(selected) != 5 * AUDIT_PER_FOLD or len(set(selected)) != len(selected):
        raise ValueError("Audit selection must be unique and exactly 200 records")
    selected_ids = [str(records[index]["sample_id"]) for index in selected]
    audit_manifest = {
        "audit_rng_seed": AUDIT_RNG_SEED,
        "audit_per_fold": AUDIT_PER_FOLD,
        "count": len(selected),
        "sample_ids": selected_ids,
        "source": "training fold manifest records; Test1/Test2 excluded",
        "model": "MPCE-3DKD product full refit; no metrics are computed from these labels",
    }
    audit_path = RUNS / "mpce_3dkd_deployment_audit_manifest.json"
    audit_path.write_text(json.dumps(audit_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = load_ensemble_manifest(MANIFEST)
    checkpoint_paths = {int(item["seed"]): Path(item["path"]) for item in manifest["checkpoints"]}
    sequences = [str(records[index]["sequence"]) for index in selected]
    selected_tri = tri[selected]
    selected_phys = phys[selected]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cache_inference() -> tuple[np.ndarray, dict[str, np.ndarray]]:
        members = {
            seed: load_product_checkpoint(checkpoint_paths[seed], device)
            for seed in checkpoint_paths
        }
        probabilities = {}
        with torch.inference_mode():
            for seed, (model, tri_mean, tri_std, phys_mean, phys_std, metadata) in members.items():
                normalized_tri = (selected_tri.to(device) - tri_mean) / tri_std
                normalized_phys = (selected_phys.to(device) - phys_mean) / phys_std
                z_final, _z_global, _z_proto, _cosine = model(normalized_tri, normalized_phys)
                probabilities[seed] = torch.sigmoid(z_final).cpu().numpy()
        stacked = np.stack(list(probabilities.values()), axis=0)
        return stacked.mean(axis=0), probabilities

    cache_average, cache_members = cache_inference()
    runner_once = predict_sequences(
        sequences,
        MANIFEST,
        model_path=str(PROTT5_SNAPSHOT),
        device="cuda",
        batch_size=1,
    )
    runner_twice = predict_sequences(
        sequences,
        MANIFEST,
        model_path=str(PROTT5_SNAPSHOT),
        device="cuda",
        batch_size=1,
    )
    runner_average = np.asarray([float(item["probability"]) for item in runner_once], dtype=np.float64)
    runner_repeat = np.asarray([float(item["probability"]) for item in runner_twice], dtype=np.float64)
    runner_members = {
        seed: np.asarray([float(item["member_probabilities"][str(seed)]) for item in runner_once], dtype=np.float64)
        for seed in (20260905, 20260906)
    }
    runner_repeat_diff = float(np.max(np.abs(runner_average - runner_repeat)))
    cache_vs_runner_diff = float(np.max(np.abs(cache_average - runner_average)))
    member_diffs = {
        str(seed): float(np.max(np.abs(cache_members[seed] - runner_members[seed])))
        for seed in (20260905, 20260906)
    }
    result = {
        "audit_manifest": str(audit_path),
        "audit_manifest_sha256": file_sha256(audit_path),
        "count": len(sequences),
        "cache_vs_runner_max_abs_diff": cache_vs_runner_diff,
        "member_cache_vs_runner_max_abs_diff": member_diffs,
        "runner_repeat_max_abs_diff": runner_repeat_diff,
        "tolerances": {
            "runner_repeat": TOLERANCE_RUNNER_REPEAT,
            "cache_vs_runner": TOLERANCE_CACHE_VS_RUNNER,
        },
        "passed": bool(
            cache_vs_runner_diff <= TOLERANCE_CACHE_VS_RUNNER
            and runner_repeat_diff <= TOLERANCE_RUNNER_REPEAT
            and all(value <= TOLERANCE_CACHE_VS_RUNNER for value in member_diffs.values())
        ),
        "prot_t5_snapshot": str(PROTT5_SNAPSHOT),
        "selection_scope": "training records only; Test1/Test2 excluded",
    }
    (RUNS / "mpce_3dkd_deployment_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MPCE-3DKD deployment-consistency audit",
        "",
        f"audit set：200 条训练序列（每 fold 40 条，RNG seed {AUDIT_RNG_SEED}），只比较推理一致性，不使用 label 或 Test1/Test2。",
        "",
        "| check | max abs diff | tolerance | pass |",
        "| --- | ---: | ---: | --- |",
        f"| cache inference vs standalone runner（ensemble） | {cache_vs_runner_diff:.3e} | {TOLERANCE_CACHE_VS_RUNNER:.0e} | {'PASS' if cache_vs_runner_diff <= TOLERANCE_CACHE_VS_RUNNER else 'FAIL'} |",
    ]
    for seed in (20260905, 20260906):
        value = member_diffs[str(seed)]
        lines.append(
            f"| cache vs runner（seed {seed}） | {value:.3e} | {TOLERANCE_CACHE_VS_RUNNER:.0e} | "
            f"{'PASS' if value <= TOLERANCE_CACHE_VS_RUNNER else 'FAIL'} |"
        )
    lines.append(
        f"| runner repeat determinism | {runner_repeat_diff:.3e} | {TOLERANCE_RUNNER_REPEAT:.0e} | "
        f"{'PASS' if runner_repeat_diff <= TOLERANCE_RUNNER_REPEAT else 'FAIL'} |"
    )
    lines += [
        "",
        f"整体判定：{'PASS' if result['passed'] else 'FAIL'}。",
        "",
    ]
    (RUNS / "mpce_3dkd_deployment_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {audit_path}")
    print(f"wrote {RUNS / 'mpce_3dkd_deployment_audit.json'}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
