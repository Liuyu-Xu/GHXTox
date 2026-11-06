"""V2-C component-effect diagnostic (口径 A)."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DELTAS = ("delta_chem", "delta_structure_total", "delta_geo_conditional")


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, len(DELTAS)))

    def forward(self, x):
        return self.net(x)


def _r2(target, prediction):
    residual = float(np.sum((target - prediction) ** 2))
    total = float(np.sum((target - target.mean(axis=0)) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def _spearman(target, prediction):
    return float(spearmanr(target, prediction).statistic)


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_v2c_diagnostic(effect_dir, cache_path, v1_oof_csv, output):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    tri = torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records])
    phys = torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records])
    seq_features = torch.cat((tri, phys), dim=1).numpy()
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.float64)
    folds = np.asarray([int(record["fold"]) for record in records], dtype=np.int64)
    historical_index = {str(record["historical_sample_id"]): index for index, record in enumerate(records)}
    v1_rows = _read_csv(v1_oof_csv)
    v1_probability = np.full(len(records), np.nan)
    for row in v1_rows:
        v1_probability[historical_index[str(row["historical_sample_id"])]] = float(row["probability"])
    if not np.isfinite(v1_probability).all():
        raise ValueError("V1 OOF probabilities incomplete")

    payloads = {}
    for fold in range(5):
        payload = torch.load(Path(effect_dir) / f"v2c_component_effects_outer{fold}.pt", map_location="cpu", weights_only=False)
        sample_ids = [str(value) for value in payload["sample_id"]]
        aligned = {}
        for key in ("z_full", "z_no_chem", "z_no_geo", "z_no_chem_no_geo"):
            vector = payload[key].numpy().reshape(-1)
            values = np.full(len(records), np.nan)
            for position, sample_id in enumerate(sample_ids):
                values[historical_index[sample_id]] = vector[position]
            aligned[key] = values
        payloads[fold] = aligned

    delta_all = {name: np.full(len(records), np.nan) for name in DELTAS}
    teacher_full_probability = np.full(len(records), np.nan)
    for index in range(len(records)):
        fold = int(folds[index])
        z_full = payloads[fold]["z_full"][index]
        z_no_chem = payloads[fold]["z_no_chem"][index]
        z_no_geo = payloads[fold]["z_no_geo"][index]
        z_no_chem_no_geo = payloads[fold]["z_no_chem_no_geo"][index]
        delta_all["delta_chem"][index] = z_full - z_no_chem
        delta_all["delta_structure_total"][index] = z_full - z_no_geo
        delta_all["delta_geo_conditional"][index] = z_no_chem - z_no_chem_no_geo
        teacher_full_probability[index] = float(1.0 / (1.0 + np.exp(-z_full)))

    existence = {}
    for name in DELTAS:
        values = delta_all[name]
        abs_values = np.abs(values)
        existence[name] = {
            "mean": float(np.mean(values)),
            "abs_mean": float(np.mean(abs_values)),
            "abs_median": float(np.median(abs_values)),
            "abs_p75": float(np.quantile(abs_values, 0.75)),
            "abs_p90": float(np.quantile(abs_values, 0.90)),
            "abs_max": float(np.max(abs_values)),
            "nonzero_gt_1e-4": int(np.sum(abs_values > 1e-4)),
        }
    v1_binary = (v1_probability >= 0.5).astype(int)
    categories = {
        "TP": (v1_binary == 1) & (labels == 1),
        "FP": (v1_binary == 1) & (labels == 0),
        "TN": (v1_binary == 0) & (labels == 0),
        "FN": (v1_binary == 0) & (labels == 1),
    }
    relation = {}
    for name in DELTAS:
        abs_values = np.abs(delta_all[name])
        by_category = {key: {"n": int(np.sum(mask)), "abs_mean": float(np.mean(abs_values[mask])) if np.any(mask) else float("nan")} for key, mask in categories.items()}
        error = categories["FP"] | categories["FN"]
        correct = categories["TP"] | categories["TN"]
        relation[name] = {
            "by_category": by_category,
            "error_abs_mean": float(np.mean(abs_values[error])),
            "correct_abs_mean": float(np.mean(abs_values[correct])),
            "spearman_with_teacher_v1_disagreement": float(spearmanr(abs_values, np.abs(teacher_full_probability - v1_probability)).statistic),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_fold_probes = {}
    for eval_fold in range(5):
        train_mask = folds != eval_fold
        test_mask = folds == eval_fold
        scaler = StandardScaler().fit(seq_features[train_mask])
        x_train = scaler.transform(seq_features[train_mask])
        x_test = scaler.transform(seq_features[test_mask])
        targets_train = np.stack([delta_all[name][train_mask] for name in DELTAS], axis=1)
        targets_test = np.stack([delta_all[name][test_mask] for name in DELTAS], axis=1)
        ridge = Ridge(alpha=10.0)
        ridge.fit(x_train, targets_train)
        ridge_pred = ridge.predict(x_test)
        # Small MLP with standardized targets.
        target_mean = targets_train.mean(axis=0)
        target_std = targets_train.std(axis=0).clip(min=1e-6)
        normalized_y = (targets_train - target_mean) / target_std
        x_tensor = torch.as_tensor(x_train, dtype=torch.float32)
        y_tensor = torch.as_tensor(normalized_y, dtype=torch.float32)
        model = SmallMLP(int(x_train.shape[1])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(20260905 + eval_fold)
        random.seed(20260905 + eval_fold)
        np.random.seed(20260905 + eval_fold)
        torch.manual_seed(20260905 + eval_fold)
        for _ in range(30):
            order = torch.randperm(len(x_train), generator=generator)
            for start in range(0, len(order), 256):
                rows = order[start : start + 256]
                loss = nn.functional.mse_loss(model(x_tensor[rows].to(device)), y_tensor[rows].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            mlp_pred = model(torch.as_tensor(x_test, dtype=torch.float32).to(device)).cpu().numpy() * target_std + target_mean
        fold_metrics = {}
        for column, name in enumerate(DELTAS):
            fold_metrics[name] = {
                "ridge_r2": _r2(targets_test[:, column], ridge_pred[:, column]),
                "ridge_spearman": _spearman(targets_test[:, column], ridge_pred[:, column]),
                "ridge_sign_accuracy": float(np.mean(np.sign(targets_test[:, column]) == np.sign(ridge_pred[:, column]))),
                "mlp_r2": _r2(targets_test[:, column], mlp_pred[:, column]),
                "mlp_spearman": _spearman(targets_test[:, column], mlp_pred[:, column]),
                "mlp_sign_accuracy": float(np.mean(np.sign(targets_test[:, column]) == np.sign(mlp_pred[:, column]))),
            }
        per_fold_probes[eval_fold] = fold_metrics
        print(f"probe eval fold {eval_fold} done", flush=True)

    result = {
        "artifact_type": "prottox_3dkd_v2c_diagnostic",
        "version": "v2c-1",
        "definition": "口径 A component intervention effects",
        "existence": existence,
        "relation_to_v1": relation,
        "predictability_probes": per_fold_probes,
        "selection_scope": "training diagnostic only; Test1/Test2 excluded",
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--v1-oof-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_v2c_diagnostic(args.effect_dir, args.cache, args.v1_oof_csv, args.output)


if __name__ == "__main__":
    main()
