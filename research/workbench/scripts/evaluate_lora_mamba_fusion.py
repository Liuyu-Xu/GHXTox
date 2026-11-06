"""Apply the refrozen 1D/3D fusion to Test1/Test2 and report metrics vs the old default.

Inputs:
  --refreeze        JSON from refreeze_lora_mamba_fusion.py (best_weight_1d, best_threshold)
  --test1-1d        new 1D ensemble probability CSV (sample_id, toxicity_probability)
  --test2-1d        new 1D ensemble probability CSV
  --hybrid-test1    runs/default_3d_hybrid/test1_predictions.csv (structure_probability column)
  --hybrid-test2    runs/default_3d_hybrid/test2_predictions.csv
  --old-test1-metrics / --old-test2-metrics  old-default metrics JSON
  --output          output JSON
"""
import argparse
import csv
import json
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score

from ghxtox.multidimensional_default import _aligned_components


def metrics(labels, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1))); tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0))); fn = int(np.sum((pred == 0) & (labels == 1)))
    sn = tp/(tp+fn) if tp+fn else 0.0; sp = tn/(tn+fp) if tn+fp else 0.0
    return {"accuracy": float(np.mean(pred == labels)), "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "sensitivity": sn, "specificity": sp, "mcc": float(matthews_corrcoef(labels, pred)),
            "auroc": float(roc_auc_score(labels, prob)), "auprc": float(average_precision_score(labels, prob))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refreeze", required=True)
    parser.add_argument("--test1-1d", required=True)
    parser.add_argument("--test2-1d", required=True)
    parser.add_argument("--hybrid-test1", required=True)
    parser.add_argument("--hybrid-test2", required=True)
    parser.add_argument("--old-test1-metrics", required=True)
    parser.add_argument("--old-test2-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    refreeze = json.load(open(args.refreeze, encoding="utf-8"))
    if refreeze.get("artifact_type") != "ghxtox_lora_bimamba_pure3d_oof_refreeze":
        raise ValueError("refreeze artifact is stale or has mixed/unknown provenance")
    if refreeze.get("provenance", {}).get("mixed_3d_oof_rejected") is not True:
        raise ValueError("refreeze artifact does not certify rejection of mixed 3D OOF")
    w = float(refreeze["best_weight_1d"])
    thr = float(refreeze["best_threshold"])

    out = {"weight_1d": w, "threshold": thr}
    for ds, f1d, fhyb, oldm in (("test1", args.test1_1d, args.hybrid_test1, args.old_test1_metrics),
                                ("test2", args.test2_1d, args.hybrid_test2, args.old_test2_metrics)):
        ids, labels, p1d, p3d = _aligned_components(f1d, fhyb)
        fused = w * p1d + (1 - w) * p3d
        new = metrics(labels, fused, thr)
        old = json.load(open(oldm, encoding="utf-8"))["metrics"]
        out[ds] = {
            "n": len(ids),
            "old_default": {k: old.get(k) for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "mcc", "auroc", "auprc")},
            "new_lora_mamba_fusion": new,
            "delta": {k: round(new[k] - old.get(k, 0.0), 6) for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "mcc", "auroc", "auprc")},
        }
        print(f"=== {ds} ===")
        print("  old_default: ACC %.4f BACC %.4f SN %.4f SP %.4f MCC %.4f AUROC %.4f AUPRC %.4f"
              % tuple(out[ds]["old_default"][k] for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "mcc", "auroc", "auprc")))
        print("  new fusion  : ACC %.4f BACC %.4f SN %.4f SP %.4f MCC %.4f AUROC %.4f AUPRC %.4f"
              % tuple(new[k] for k in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "mcc", "auroc", "auprc")))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
