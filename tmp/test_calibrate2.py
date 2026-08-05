import json, csv, sys, os
import numpy as np
sys.path.insert(0, os.path.abspath("scripts"))
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss

manifest = list(csv.DictReader(open("data/folds/train_cdhit080_fallback_folds.csv", encoding="utf-8-sig")))
rows3d = list(csv.DictReader(open("runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv", encoding="utf-8-sig")))
ids = [r["sample_id"] for r in manifest]
p3d = np.array([float(r["toxicity_probability"]) for r in rows3d])
labels = np.array([int(r["label"]) for r in rows3d])
def logit(p): return np.log(np.clip(p, 1e-7, 1-1e-7) / (1 - np.clip(p, 1e-7, 1-1e-7)))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))

# miscalibrated 1D (overconfident)
p1d_mis = sigmoid(2.2 * logit(p3d) + 1.3)
json.dump({"oof": p1d_mis.tolist(), "labels": labels.tolist(), "sample_ids": ids}, open("tmp/oof_mis.json", "w"))

# Test with DIFFERENT shift than OOF (simulates distribution shift)
rng = np.random.RandomState(0)
for ds, n, shift in (("test1", 1126, 0.6), ("test2", 582, 1.8)):
    idx = rng.choice(len(p3d), n, replace=False)
    p_test = sigmoid(2.2 * logit(p3d[idx]) + 1.3 + shift + rng.normal(0, 0.05, n))
    with open(f"tmp/{ds}_mis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
        w.writeheader()
        for i, k in enumerate(idx):
            w.writerow({"sample_id": ids[k], "sequence": "ACD", "label": labels[k],
                        "toxicity_probability": f"{p_test[i]:.9g}"})

sys.argv = ["calibrate",
            "--oof-jsons", "tmp/oof_mis.json",
            "--fold-manifest", "data/folds/train_cdhit080_fallback_folds.csv",
            "--test1-csv", "tmp/test1_mis.csv",
            "--test2-csv", "tmp/test2_mis.csv",
            "--output-oof", "tmp/oof_cal.json",
            "--output-test1", "tmp/test1_cal.csv",
            "--output-test2", "tmp/test2_cal.csv"]
import calibrate_lora_mamba as cl
cl.main()

def read(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["toxicity_probability"]) for r in rows]), np.array([int(r["label"]) for r in rows])

print("\n-- fixed threshold 0.5 decisions --")
for name, p, lab in [("OOF-mis", p1d_mis, labels)] + \
    [(f"test1-{'mis' if False else 'cal'}",) + (lambda: read("tmp/test1_cal.csv"))()] + \
    [(f"test2-cal",) + (lambda: read("tmp/test2_cal.csv"))()]:
    pass

# OOF: fixed-threshold comparison
for name, p, lab in [("OOF before cal", p1d_mis, labels), ("OOF after cal", np.array(json.load(open("tmp/oof_cal.json"))["oof"]), labels)]:
    print("%s: BACC@0.5=%.4f  Brier=%.4f  AUROC=%.4f" % (name,
          balanced_accuracy_score(lab, (p >= 0.5).astype(int)),
          brier_score_loss(lab, p),
          roc_auc_score(lab, p)))

# Test: does calibration (fit on OOF) help under distribution shift at the OOF-selected fusion threshold?
p_t1_cal, l_t1_cal = read("tmp/test1_cal.csv")
p_t1_mis, l_t1_mis = read("tmp/test1_mis.csv")
p3d_idx = None
with open("tmp/test1_mis.csv", encoding="utf-8-sig") as f:
    t1_ids = [r["sample_id"] for r in csv.DictReader(f)]
p3d_map = {sid: p for sid, p in zip(ids, p3d)}
p3d_t1 = np.array([p3d_map[s] for s in t1_ids])
for w in (0.18, 0.5):
    fused_mis = w*p_t1_mis + (1-w)*p3d_t1
    fused_cal = w*p_t1_cal + (1-w)*p3d_t1
    print("test1 w=%.2f: mis BACC@0.5=%.4f | cal BACC@0.5=%.4f" % (w,
          balanced_accuracy_score(l_t1_mis, (fused_mis >= 0.5).astype(int)),
          balanced_accuracy_score(l_t1_cal, (fused_cal >= 0.5).astype(int))))
print("CALIBRATE VERIFY2 DONE")
