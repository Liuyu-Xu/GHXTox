import json, csv, sys, os
import numpy as np
sys.path.insert(0, os.path.abspath("scripts"))

manifest = list(csv.DictReader(open("data/folds/train_cdhit080_fallback_folds.csv", encoding="utf-8-sig")))
rows3d = list(csv.DictReader(open("runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv", encoding="utf-8-sig")))
ids = [r["sample_id"] for r in manifest]
p3d = np.array([float(r["toxicity_probability"]) for r in rows3d])
labels = np.array([int(r["label"]) for r in rows3d])
def logit(p): return np.log(np.clip(p, 1e-7, 1-1e-7) / (1 - np.clip(p, 1e-7, 1-1e-7)))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))

p1d_mis = sigmoid(2.2 * logit(p3d) + 1.3)
json.dump({"oof": p1d_mis.tolist(), "labels": labels.tolist(), "sample_ids": ids}, open("tmp/s3_oof.json", "w"))

def read_hybrid(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {r["sample_id"]: (float(r["structure_probability"]), int(r["label"])) for r in rows}
hy1 = read_hybrid("runs/default_3d_hybrid/test1_predictions.csv")
hy2 = read_hybrid("runs/default_3d_hybrid/test2_predictions.csv")

rng = np.random.RandomState(0)
def make_1d_csv(path, hy, shift):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
        w.writeheader()
        for sid in hy:
            p3 = hy[sid][0]
            p1 = sigmoid(2.2 * logit(p3) + 1.3 + shift + rng.normal(0, 0.05))
            w.writerow({"sample_id": sid, "sequence": "ACD", "label": hy[sid][1], "toxicity_probability": f"{p1:.9g}"})
make_1d_csv("tmp/s3_test1.csv", hy1, 0.6)
make_1d_csv("tmp/s3_test2.csv", hy2, 1.8)

sys.argv = ["calibrate", "--oof-jsons", "tmp/s3_oof.json",
            "--fold-manifest", "data/folds/train_cdhit080_fallback_folds.csv",
            "--test1-csv", "tmp/s3_test1.csv", "--test2-csv", "tmp/s3_test2.csv",
            "--output-oof", "tmp/s3_oof_cal.json",
            "--output-test1", "tmp/s3_test1_cal.csv", "--output-test2", "tmp/s3_test2_cal.csv"]
import calibrate_lora_mamba as cl
cl.main()

sys.argv = ["refreeze",
            "--new-1d-oof", "tmp/s3_oof_cal.json",
            "--fold-manifest", "data/folds/train_cdhit080_fallback_folds.csv",
            "--three-d-oof", "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv",
            "--output", "tmp/s3_refreeze.json"]
import refreeze_lora_mamba_fusion as rf
rf.main()

sys.argv = ["eval",
            "--refreeze", "tmp/s3_refreeze.json",
            "--test1-1d", "tmp/s3_test1_cal.csv", "--test2-1d", "tmp/s3_test2_cal.csv",
            "--hybrid-test1", "runs/default_3d_hybrid/test1_predictions.csv",
            "--hybrid-test2", "runs/default_3d_hybrid/test2_predictions.csv",
            "--old-test1-metrics", "runs/default_3d_hybrid/test1_metrics.json",
            "--old-test2-metrics", "runs/default_3d_hybrid/test2_metrics.json",
            "--output", "tmp/s3_eval.json"]
import evaluate_lora_mamba_fusion as ef
ef.main()
print("STAGE3 FULL CHAIN OK")
