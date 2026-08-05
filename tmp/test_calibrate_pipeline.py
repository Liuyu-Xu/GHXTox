import json, csv, sys, os
import numpy as np
sys.path.insert(0, os.path.abspath("scripts"))

# ---- real base: 3D OOF probs + manifest (calibrated reference) ----
manifest = list(csv.DictReader(open("data/folds/train_cdhit080_fallback_folds.csv", encoding="utf-8-sig")))
rows3d = list(csv.DictReader(open("runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv", encoding="utf-8-sig")))
ids = [r["sample_id"] for r in manifest]
p3d = np.array([float(r["toxicity_probability"]) for r in rows3d])  # same order as manifest (verified earlier)
labels = np.array([int(r["label"]) for r in rows3d])

def logit(p): return np.log(np.clip(p, 1e-7, 1-1e-7) / (1 - np.clip(p, 1e-7, 1-1e-7)))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))

# ---- miscalibrated 1D: overconfident (SN>>SP at median) ----
p1d_mis = sigmoid(2.2 * logit(p3d) + 1.3)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef
med = np.median(p1d_mis)
print("before calibration: BACC=%.4f SN=%.3f SP=%.3f AUROC=%.4f"
      % (balanced_accuracy_score(labels, (p1d_mis >= med).astype(int)),
         (p1d_mis[labels==1] >= med).mean(), (p1d_mis[labels==0] < med).mean(),
         roc_auc_score(labels, p1d_mis)))

# synthetic OOF json
json.dump({"oof": p1d_mis.tolist(), "labels": labels.tolist(), "sample_ids": ids}, open("tmp/oof_mis.json", "w"))

# synthetic test CSVs (same transform + tiny noise)
rng = np.random.RandomState(0)
for ds, n in (("test1", 1126), ("test2", 582)):
    idx = rng.choice(len(p3d), n, replace=False)
    p_test = sigmoid(2.2 * logit(p3d[idx]) + 1.3 + rng.normal(0, 0.05, n))
    with open(f"tmp/{ds}_mis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
        w.writeheader()
        for i, k in enumerate(idx):
            w.writerow({"sample_id": ids[k], "sequence": "ACD", "label": labels[k],
                        "toxicity_probability": f"{p_test[i]:.9g}"})

# ---- run calibrate ----
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

# ---- check calibration ----
cal = np.array(json.load(open("tmp/oof_cal.json"))["oof"])
print("after  calibration: BACC=%.4f SN=%.3f SP=%.3f AUROC=%.4f (AUROC should be ~unchanged)"
      % (balanced_accuracy_score(labels, (cal >= np.median(cal)).astype(int)),
         (cal[labels==1] >= np.median(cal)).mean(), (cal[labels==0] < np.median(cal)).mean(),
         roc_auc_score(labels, cal)))
assert abs(roc_auc_score(labels, cal) - roc_auc_score(labels, p1d_mis)) < 1e-6, "AUROC changed!"
print("AUROC preserved exactly. CALIBRATE PIPELINE OK")
