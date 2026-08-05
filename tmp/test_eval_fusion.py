import json, csv, os, sys, tempfile
import numpy as np
tmp = tempfile.mkdtemp()
rng = np.random.RandomState(1)
# fake refreeze
json.dump({"best_weight_1d": 0.6, "best_threshold": 0.5}, open(os.path.join(tmp, "refreeze.json"), "w"))
ids = [f"p{i}" for i in range(50)]
labels = [1]*15 + [0]*35
rng.shuffle(labels)

def write_csv(path, prob_col, probs):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "label", prob_col])
        w.writeheader()
        for i, sid in enumerate(ids):
            w.writerow({"sample_id": sid, "sequence": "ACD", "label": labels[i], prob_col: f"{probs[i]:.9g}"})

p1d = rng.rand(50)
p3d = rng.rand(50)
write_csv(os.path.join(tmp, "test1_1d.csv"), "toxicity_probability", p1d)
write_csv(os.path.join(tmp, "test1_hyb.csv"), "structure_probability", p3d)
json.dump({"metrics": {"accuracy": 0.9, "balanced_accuracy": 0.85, "sensitivity": 0.8, "specificity": 0.9, "mcc": 0.7, "auroc": 0.95, "auprc": 0.9}},
          open(os.path.join(tmp, "old1.json"), "w"))
write_csv(os.path.join(tmp, "test2_1d.csv"), "toxicity_probability", rng.rand(50))
write_csv(os.path.join(tmp, "test2_hyb.csv"), "structure_probability", rng.rand(50))
json.dump({"metrics": {"accuracy": 0.8, "balanced_accuracy": 0.75, "sensitivity": 0.7, "specificity": 0.8, "mcc": 0.5, "auroc": 0.9, "auprc": 0.8}},
          open(os.path.join(tmp, "old2.json"), "w"))

sys.argv = ["eval", "--refreeze", os.path.join(tmp, "refreeze.json"),
            "--test1-1d", os.path.join(tmp, "test1_1d.csv"), "--test2-1d", os.path.join(tmp, "test2_1d.csv"),
            "--hybrid-test1", os.path.join(tmp, "test1_hyb.csv"), "--hybrid-test2", os.path.join(tmp, "test2_hyb.csv"),
            "--old-test1-metrics", os.path.join(tmp, "old1.json"), "--old-test2-metrics", os.path.join(tmp, "old2.json"),
            "--output", os.path.join(tmp, "out.json")]
sys.path.insert(0, os.path.abspath("scripts"))
import evaluate_lora_mamba_fusion as ef
ef.main()
print("EVAL FUSION LOGIC OK")
