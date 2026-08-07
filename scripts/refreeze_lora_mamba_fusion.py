"""Refreeze the 1D/3D fusion weight and decision threshold for combination A.

Alignment: the new-1D OOF JSON (oof/labels arrays) is in fold-manifest order
(source_index 0..N-1).  The 3D OOF CSV is aligned by sample_id via the fold
manifest, so the script does NOT depend on a source_index column existing in
the 3D OOF file.

Inputs:
  --new-1d-oof     JSON from ghxtox.lora_mamba_1d cv (oof, labels, folds)
  --fold-manifest  CD-HIT fold manifest CSV (source_index, sample_id, ...)
  --three-d-oof    CSV with sample_id and toxicity_probability (row order may differ)
  --baseline-oof   CSV (optional) old-default fused OOF for comparison
Grid-searches w in [0, 1]; for each w the best threshold is found with a
vectorized MCC scan.  Selection uses training OOF only.
"""
import argparse
import csv
import json
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score

from ghxtox.lora_mamba_1d import PUBLIC_1D_EXPERIMENT, PUBLIC_1D_SEEDS


def metrics(labels, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1))); tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0))); fn = int(np.sum((pred == 0) & (labels == 1)))
    sn = tp/(tp+fn) if tp+fn else 0.0; sp = tn/(tn+fp) if tn+fp else 0.0
    return {"accuracy": float(np.mean(pred == labels)), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "sensitivity": sn, "specificity": sp, "mcc": float(matthews_corrcoef(labels, pred)),
            "auroc": float(roc_auc_score(labels, prob)), "auprc": float(average_precision_score(labels, prob))}


def _mcc_scan(labels: np.ndarray, prob: np.ndarray):
    """Vectorized MCC over all unique thresholds.  Returns (best_threshold, best_mcc)."""
    n = len(labels)
    order = np.argsort(-prob, kind="stable")
    sorted_labels = labels[order]
    probs_sorted = prob[order]
    cum_pos = np.cumsum(sorted_labels)
    cum_neg = np.cumsum(1 - sorted_labels)
    total_pos = int(cum_pos[-1]); total_neg = n - total_pos
    tp = cum_pos.astype(np.float64)
    fp = cum_neg.astype(np.float64)
    fn = total_pos - tp
    tn = total_neg - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.where(denom > 0, denom, 1.0), 0.0)
    # threshold = probs_sorted[i] must include every sample with prob == threshold,
    # so only evaluate at the last index of each tied group.
    change = np.ones(n, dtype=bool)
    change[1:] = probs_sorted[1:] != probs_sorted[:-1]
    candidate_idx = np.concatenate([np.where(change)[0] - 1, [n - 1]])
    candidate_idx = np.unique(candidate_idx)
    candidate_idx = candidate_idx[candidate_idx >= 0]
    best_pos = int(candidate_idx[np.argmax(mcc[candidate_idx])])
    return float(probs_sorted[best_pos]), float(mcc[best_pos])


def load_prob_by_sample_id(path, prob_key="toxicity_probability", *, require_pure_3d=False):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        required = {"sample_id", "label", prob_key}
        if require_pure_3d:
            required |= {"source_index", "fold", "sequence"}
            forbidden = {"baseline_probability", "expert_probability"}
            if fields & forbidden:
                raise ValueError(
                    f"{path} is a mixed/fused OOF, not a pure 3D expert OOF; "
                    f"forbidden columns={sorted(fields & forbidden)}"
                )
        if required - fields:
            raise ValueError(f"{path} missing required columns {sorted(required - fields)}")
        rows = list(reader)
    output = {}
    for row_number, row in enumerate(rows, start=2):
        sample_id = str(row["sample_id"])
        if not sample_id or sample_id in output:
            raise ValueError(f"{path}:{row_number}: empty or duplicate sample_id {sample_id!r}")
        probability = float(row[prob_key])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{path}:{row_number}: invalid probability {probability!r}")
        output[sample_id] = {
            "probability": probability,
            "label": int(row["label"]),
            "sequence": row.get("sequence", ""),
            "fold": int(row["fold"]) if row.get("fold", "") != "" else None,
            "source_index": int(row["source_index"]) if row.get("source_index", "") != "" else None,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-1d-oof", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--three-d-oof", required=True)
    parser.add_argument("--baseline-oof", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="mcc", choices=["mcc", "bacc"])
    args = parser.parse_args()

    oof = json.load(open(args.new_1d_oof, encoding="utf-8"))
    if oof.get("artifact_type") != "ghxtox_lora_bimamba_platt_oof":
        raise ValueError("1D OOF must be the strict three-seed Platt-calibrated artifact")
    if tuple(oof.get("source_seeds", ())) != PUBLIC_1D_SEEDS:
        raise ValueError(f"1D OOF source seeds must be exactly {PUBLIC_1D_SEEDS}")
    if oof.get("config_name") != PUBLIC_1D_EXPERIMENT:
        raise ValueError("1D OOF has stale or unknown model-generation provenance")
    p1d = np.asarray(oof["oof"], dtype=np.float64)
    labels = np.asarray(oof["labels"], dtype=np.int64)

    manifest = list(csv.DictReader(open(args.fold_manifest, encoding="utf-8-sig")))
    manifest_ids = [str(r["sample_id"]) for r in manifest]
    manifest_labels = np.asarray([int(r["label"]) for r in manifest], dtype=np.int64)
    manifest_indices = [int(r["source_index"]) for r in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("fold manifest contains duplicate sample_ids")
    if manifest_indices != list(range(len(manifest))):
        raise ValueError("fold manifest source_index is not contiguous and row-aligned")
    if len(manifest_ids) != len(p1d):
        raise ValueError(f"fold manifest rows ({len(manifest_ids)}) != 1D OOF length ({len(p1d)})")
    if [str(value) for value in oof.get("sample_ids", [])] != manifest_ids:
        raise ValueError("1D OOF sample IDs/order differ from the fold manifest")
    if not np.array_equal(labels, manifest_labels):
        raise ValueError("1D OOF labels differ from the fold manifest")
    if [int(value) for value in oof.get("folds", [])] != [int(r["fold"]) for r in manifest]:
        raise ValueError("1D OOF folds differ from the fold manifest")
    if not np.all(np.isfinite(p1d)) or np.any((p1d < 0.0) | (p1d > 1.0)):
        raise ValueError("1D OOF probabilities are invalid")

    p3d_by_id = load_prob_by_sample_id(args.three_d_oof, require_pure_3d=True)
    missing = [sid for sid in manifest_ids if sid not in p3d_by_id]
    if missing:
        raise ValueError(f"{len(missing)} sample_ids missing from 3D OOF (e.g. {missing[:3]})")
    for row in manifest:
        record = p3d_by_id[str(row["sample_id"])]
        if record["source_index"] != int(row["source_index"]):
            raise ValueError(f"3D OOF source_index mismatch for {row['sample_id']}")
        if record["fold"] != int(row["fold"]):
            raise ValueError(f"3D OOF fold mismatch for {row['sample_id']}")
        if record["label"] != int(row["label"]):
            raise ValueError(f"3D OOF label mismatch for {row['sample_id']}")
        if record["sequence"] != row["sequence"]:
            raise ValueError(f"3D OOF sequence mismatch for {row['sample_id']}")
    p3d = np.asarray([p3d_by_id[sid]["probability"] for sid in manifest_ids])

    best = None
    for w in np.arange(0.0, 1.001, 0.02):
        fused = w * p1d + (1 - w) * p3d
        thr, score = _mcc_scan(labels, fused) if args.metric == "mcc" else _bacc_scan(labels, fused)
        if best is None or score > best[0] or (abs(score - best[0]) < 1e-9 and thr > best[2]):
            best = (score, w, thr)

    _, w_best, thr_best = best
    fused_best = w_best * p1d + (1 - w_best) * p3d
    result = {
        "artifact_type": "ghxtox_lora_bimamba_pure3d_oof_refreeze",
        "schema_version": 1,
        "metric": args.metric,
        "best_weight_1d": float(w_best),
        "best_threshold": float(thr_best),
        "fusion_oof_metrics": metrics(labels, fused_best, thr_best),
        "new_1d_metrics": metrics(labels, p1d, float(np.median(p1d))),
        "provenance": {
            "new_1d_oof": args.new_1d_oof,
            "fold_manifest": args.fold_manifest,
            "pure_3d_oof": args.three_d_oof,
            "mixed_3d_oof_rejected": True,
        },
    }
    if args.baseline_oof:
        p_base_by_id = load_prob_by_sample_id(args.baseline_oof)
        p_base = np.asarray([p_base_by_id[sid]["probability"] for sid in manifest_ids])
        result["baseline_fusion_oof_metrics"] = metrics(labels, p_base, float(np.median(p_base)))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(result, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("refreeze:", json.dumps(result, indent=2))


def _bacc_scan(labels: np.ndarray, prob: np.ndarray):
    """Vectorized balanced-accuracy scan (analogous to _mcc_scan)."""
    n = len(labels)
    order = np.argsort(-prob, kind="stable")
    sorted_labels = labels[order]
    probs_sorted = prob[order]
    cum_pos = np.cumsum(sorted_labels)
    total_pos = int(cum_pos[-1]); total_neg = n - total_pos
    sn = cum_pos / max(total_pos, 1)
    fp = (np.arange(1, n + 1) - cum_pos).astype(np.float64)
    tn = total_neg - fp
    sp = tn / max(total_neg, 1)
    bacc = 0.5 * (sn + sp)
    change = np.ones(n, dtype=bool)
    change[1:] = probs_sorted[1:] != probs_sorted[:-1]
    candidate_idx = np.unique(np.concatenate([np.where(change)[0] - 1, [n - 1]]))
    candidate_idx = candidate_idx[candidate_idx >= 0]
    best_pos = int(candidate_idx[np.argmax(bacc[candidate_idx])])
    return float(probs_sorted[best_pos]), float(bacc[best_pos])


if __name__ == "__main__":
    main()
