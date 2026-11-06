"""Select a display-probability calibration policy for frozen MPCE-3DKD.

Only the two-seed averaged training OOF is used.  Candidates are identity
and temperature scaling.  A single scalar temperature is fit once; it is a
display calibration policy and never changes model predictions or the frozen
architecture.  Test1/Test2 are never read.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ghxtox.prottox_tsk import file_sha256
from ghxtox.train_prottox_tsk import _metrics


ROOT = Path(r"D:\dachuang2026\GHXTox")
RUNS = ROOT / "runs"
RUN_PATHS = {
    "20260905": RUNS / "mpce_3dkd_seed20260905_v1" / "oof_predictions.csv",
    "20260906": RUNS / "mpce_3dkd_seed20260906_v1" / "oof_predictions.csv",
}


def _read_probabilities(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6387:
        raise ValueError(f"{path}: expected 6387 OOF rows")
    labels = np.asarray([float(row["label"]) for row in rows], dtype=np.float64)
    probabilities = np.asarray([float(row["probability"]) for row in rows], dtype=np.float64)
    return labels, probabilities


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total_error = 0.0
    total_count = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (probabilities > left) & (probabilities <= right)
        if not selected.any():
            continue
        mean_probability = float(probabilities[selected].mean())
        label_fraction = float(labels[selected].mean())
        total_error += float(selected.sum()) * abs(mean_probability - label_fraction)
        total_count += float(selected.sum())
    return total_error / max(total_count, 1.0)


def main() -> None:
    per_seed: list[np.ndarray] = []
    labels = None
    for seed, path in RUN_PATHS.items():
        seed_labels, seed_probabilities = _read_probabilities(path)
        if labels is None:
            labels = seed_labels
        elif not np.array_equal(labels, seed_labels):
            raise ValueError("Seed OOF label order mismatch")
        per_seed.append(seed_probabilities)
    probabilities = np.mean(per_seed, axis=0)
    logits = np.log(np.clip(probabilities, 1e-7, 1.0 - 1e-7) / np.clip(1.0 - probabilities, 1e-7, 1.0 - 1e-7))
    identity_metrics = _metrics(labels, probabilities)
    identity = {
        key: float(identity_metrics[key])
        for key in ("mcc", "auprc", "brier", "auroc")
    }
    identity["ece"] = _ece(labels, probabilities)

    target = torch.tensor(labels, dtype=torch.float32)
    z = torch.tensor(logits, dtype=torch.float32)
    temperature = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([temperature], lr=0.01)
    for _ in range(3000):
        optimizer.zero_grad(set_to_none=True)
        t = temperature.clamp_min(0.05)
        loss = F.binary_cross_entropy_with_logits(z / t, target)
        loss.backward()
        optimizer.step()
    temperature_value = float(temperature.detach().clamp_min(0.05))
    calibrated = torch.sigmoid(z / temperature_value).numpy()
    calibrated_metrics = _metrics(labels, calibrated)
    calibrated_summary = {
        key: float(calibrated_metrics[key])
        for key in ("mcc", "auprc", "brier", "auroc")
    }
    calibrated_summary["ece"] = _ece(labels, calibrated)
    calibrated_summary["temperature"] = temperature_value

    identity_brier_gain = identity["brier"] - calibrated_summary["brier"]
    identity_ece_gain = identity["ece"] - calibrated_summary["ece"]
    if identity_brier_gain >= 0.0005 or identity_ece_gain >= 0.002:
        selected = "temperature_scaling"
        selected_summary = calibrated_summary
    else:
        selected = "identity"
        selected_summary = identity

    result = {
        "selection_scope": "training OOF only; Test1/Test2 never read",
        "model": "MPCE-3DKD two-seed averaged OOF",
        "identity": identity,
        "temperature_scaling": calibrated_summary,
        "identity_minus_temperature": {
            "brier_gain": identity_brier_gain,
            "ece_gain": identity_ece_gain,
        },
        "selected": selected,
        "selected_summary": selected_summary,
        "decision_rule": "choose temperature scaling only if OOF Brier gain >= 0.0005 or ECE gain >= 0.002; otherwise identity",
    }
    (RUNS / "mpce_3dkd_calibration.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def fmt(value: float) -> str:
        return f"{value:.5f}"

    lines = [
        "# MPCE-3DKD display calibration selection",
        "",
        "范围：仅两 seed 平均训练 OOF；Test1/Test2 未读取；不改变模型与冻结阈值。",
        "",
        "| policy | MCC | AUPRC | Brier | ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| identity | {fmt(identity['mcc'])} | {fmt(identity['auprc'])} | {fmt(identity['brier'])} | {fmt(identity['ece'])} |",
        f"| temperature (T={temperature_value:.4f}) | {fmt(calibrated_summary['mcc'])} | "
        f"{fmt(calibrated_summary['auprc'])} | {fmt(calibrated_summary['brier'])} | "
        f"{fmt(calibrated_summary['ece'])} |",
        "",
        f"Brier gain {identity_brier_gain:+.5f}；ECE gain {identity_ece_gain:+.5f}。",
        "",
        f"**选定：{selected}。**",
        "",
    ]
    (RUNS / "mpce_3dkd_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RUNS / 'mpce_3dkd_calibration.json'}")
    print(f"wrote {RUNS / 'mpce_3dkd_calibration.md'}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
