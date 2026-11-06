"""Platt calibration for the combination-A 1D probabilities (fit on training OOF only).

The 1D classifier outputs slightly over-confident positive probabilities
(SN >> SP at the median threshold).  A monotone sigmoid re-scaling is fit on the
out-of-fold training probabilities and applied to the test ensembles.  Because
the transform is strictly monotone, ranking metrics (AUROC/AUPRC) are unchanged;
only the probability scale and decision thresholds improve.

Usage:
  python -B scripts/calibrate_lora_mamba.py \
    --oof-jsons runs/lora_mamba_full/oof_seed42.json [--oof-jsons ...] \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --test1-csv runs/lora_mamba_full/test1_lora_mamba_ensemble.csv \
    --test2-csv runs/lora_mamba_full/test2_lora_mamba_ensemble.csv \
    --output-oof runs/lora_mamba_full/oof_calibrated.json \
    --output-test1 runs/lora_mamba_full/test1_lora_mamba_calibrated.csv \
    --output-test2 runs/lora_mamba_full/test2_lora_mamba_calibrated.csv
"""
import argparse
import csv
import json
import os
import numpy as np

from ghxtox.lora_mamba_1d import PUBLIC_1D_EXPERIMENT, PUBLIC_1D_SEEDS

EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def fit_platt(logits: np.ndarray, labels: np.ndarray):
    """Fit 1-D logistic regression on logits; returns (a, b) with p=sigmoid(a*logit+b)."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(logits.reshape(-1, 1), labels.astype(int))
    a = float(model.coef_[0][0])
    b = float(model.intercept_[0])
    if not (np.isfinite(a) and np.isfinite(b)):
        return 1.0, 0.0
    return a, b


def read_probs_csv(path: str) -> tuple[list[str], list[str], list[int], np.ndarray]:
    sample_ids, sequences, labels, probs = [], [], [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row_number, row in enumerate(csv.DictReader(f), start=2):
            sample_id = str(row["sample_id"])
            if not sample_id or sample_id in sample_ids:
                raise ValueError(f"{path}:{row_number}: empty or duplicate sample_id {sample_id!r}")
            sample_ids.append(sample_id)
            sequences.append(row.get("sequence", ""))
            labels.append(int(row["label"]))
            probability = float(row["toxicity_probability"])
            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{path}:{row_number}: invalid probability {probability!r}")
            probs.append(probability)
    return sample_ids, sequences, labels, np.asarray(probs, dtype=np.float64)


def write_probs_csv(path: str, sample_ids, sequences, labels, probs) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
        writer.writeheader()
        for sid, seq, lab, prob in zip(sample_ids, sequences, labels, probs):
            writer.writerow({"sample_id": sid, "sequence": seq, "label": lab,
                             "toxicity_probability": f"{float(prob):.9g}"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-jsons", nargs="+", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--test1-csv", required=True)
    parser.add_argument("--test2-csv", required=True)
    parser.add_argument("--output-oof", required=True)
    parser.add_argument("--output-test1", required=True)
    parser.add_argument("--output-test2", required=True)
    args = parser.parse_args()

    manifest = list(csv.DictReader(open(args.fold_manifest, encoding="utf-8-sig")))
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    manifest_labels = np.asarray([int(row["label"]) for row in manifest], dtype=int)
    manifest_folds = [int(row["fold"]) for row in manifest]
    manifest_indices = [int(row["source_index"]) for row in manifest]
    if manifest_indices != list(range(len(manifest))) or len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("fold manifest is not uniquely and contiguously row-aligned")
    if len(args.oof_jsons) != len(PUBLIC_1D_SEEDS):
        raise ValueError(f"exactly {len(PUBLIC_1D_SEEDS)} seed OOF files are required")

    # Average only the three aligned, same-generation seed OOFs.
    oofs = []
    seeds = []
    for path in args.oof_jsons:
        payload = json.load(open(path, encoding="utf-8"))
        seed = int(payload.get("seed", -1))
        if seed in seeds:
            raise ValueError(f"{path}: duplicate seed {seed}")
        if payload.get("config_name") != PUBLIC_1D_EXPERIMENT:
            raise ValueError(f"{path}: stale or unknown 1D OOF generation")
        if [str(value) for value in payload.get("sample_ids", [])] != manifest_ids:
            raise ValueError(f"{path}: sample IDs/order differ from the fold manifest")
        if not np.array_equal(np.asarray(payload.get("labels", []), dtype=int), manifest_labels):
            raise ValueError(f"{path}: labels differ from the fold manifest")
        if [int(value) for value in payload.get("folds", [])] != manifest_folds:
            raise ValueError(f"{path}: folds differ from the fold manifest")
        values = np.asarray(payload["oof"], dtype=np.float64)
        if values.shape != manifest_labels.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: invalid OOF probability vector")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{path}: OOF probabilities lie outside [0, 1]")
        oofs.append(values)
        seeds.append(seed)
    if tuple(sorted(seeds)) != PUBLIC_1D_SEEDS:
        raise ValueError(f"OOF seeds must be exactly {PUBLIC_1D_SEEDS}; found {sorted(seeds)}")
    p_oof = np.mean(oofs, axis=0)
    labels = manifest_labels
    if len(p_oof) != len(labels):
        raise ValueError("OOF probs/labels length mismatch")

    a, b = fit_platt(_logit(p_oof), labels)
    cal_oof = _sigmoid(a * _logit(p_oof) + b)
    print(f"platt: a={a:.4f} b={b:.4f}")

    # write calibrated OOF json (same structure refreeze expects)
    sample_ids = manifest_ids
    if len(sample_ids) != len(cal_oof):
        raise ValueError("manifest rows != calibrated OOF length")
    os.makedirs(os.path.dirname(args.output_oof), exist_ok=True)
    json.dump({
        "artifact_type": "ghxtox_lora_bimamba_platt_oof",
        "schema_version": 1,
        "source_seeds": sorted(seeds),
        "config_name": PUBLIC_1D_EXPERIMENT,
        "platt": {"slope": a, "intercept": b},
        "oof": cal_oof.tolist(),
        "labels": labels.tolist(),
        "sample_ids": sample_ids,
        "folds": manifest_folds,
    }, open(args.output_oof, "w", encoding="utf-8"), ensure_ascii=False)

    for src, out in ((args.test1_csv, args.output_test1), (args.test2_csv, args.output_test2)):
        ids, seqs, labs, probs = read_probs_csv(src)
        cal = _sigmoid(a * _logit(probs) + b)
        write_probs_csv(out, ids, seqs, labs, cal)
        print(f"calibrated {src} -> {out} (n={len(ids)})")

    print("CALIBRATION DONE")


if __name__ == "__main__":
    main()
