import json, csv, os, sys, tempfile
import numpy as np
tmp = tempfile.mkdtemp()
rng = np.random.RandomState(0)
n = 300
labels = np.array([1]*90 + [0]*210)
p1d = rng.rand(n)
p3d = rng.rand(n)
json.dump({"oof": p1d.tolist(), "labels": labels.tolist(), "folds": [i % 5 for i in range(n)]},
          open(os.path.join(tmp, "oof.json"), "w"))
with open(os.path.join(tmp, "3d.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["source_index", "sample_id", "sequence", "label", "fold", "toxicity_probability"])
    w.writeheader()
    for i in range(n):
        w.writerow({"source_index": i, "sample_id": f"p{i}", "sequence": "ACDEFGHIKLMNPQRSTVWY", "label": labels[i],
                    "fold": i % 5, "toxicity_probability": f"{p3d[i]:.9g}"})
sys.argv = ["refreeze", "--new-1d-oof", os.path.join(tmp, "oof.json"), "--three-d-oof", os.path.join(tmp, "3d.csv"),
            "--output", os.path.join(tmp, "out.json")]
sys.path.insert(0, os.path.abspath("scripts"))
import refreeze_lora_mamba_fusion as rf
rf.main()
print("REFREEZE LOGIC OK")
