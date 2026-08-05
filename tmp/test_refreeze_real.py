import json, csv, sys, os
sys.path.insert(0, os.path.abspath("scripts"))
import numpy as np

# real files
manifest = list(csv.DictReader(open("data/folds/train_cdhit080_fallback_folds.csv", encoding="utf-8-sig")))
rows3d = list(csv.DictReader(open("runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv", encoding="utf-8-sig")))
print("manifest rows:", len(manifest), "3D rows:", len(rows3d))

ids_fold = [r["sample_id"] for r in manifest]
p3d_by_id = {r["sample_id"]: float(r["toxicity_probability"]) for r in rows3d}
labels_by_id = {r["sample_id"]: int(r["label"]) for r in rows3d}

# synthetic 1D OOF identical to 3D (deterministic: fusion must equal 3D metrics)
p1d = np.array([p3d_by_id[sid] for sid in ids_fold])
labels = np.array([labels_by_id[sid] for sid in ids_fold])
oof_json = {"oof": p1d.tolist(), "labels": labels.tolist(), "sample_ids": ids_fold, "folds": [int(r["fold"]) for r in manifest]}
json.dump(oof_json, open("tmp/test_oof_real.json", "w"))

sys.argv = ["refreeze",
            "--new-1d-oof", "tmp/test_oof_real.json",
            "--fold-manifest", "data/folds/train_cdhit080_fallback_folds.csv",
            "--three-d-oof", "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv",
            "--output", "tmp/test_refreeze_real_out.json"]
import refreeze_lora_mamba_fusion as rf
rf.main()

out = json.load(open("tmp/test_refreeze_real_out.json"))
print("best_weight_1d:", out["best_weight_1d"])
print("fusion_oof_metrics:", out["fusion_oof_metrics"])
# sanity: since 1D == 3D, fused AUROC must equal 3D OOF AUROC (check vs 3D metrics.json)
m3d = json.load(open("runs/3d_v2_conformer_prott5_frozen_oof/metrics.json", encoding="utf-8"))
print("3D OOF metrics.json:", json.dumps(m3d, ensure_ascii=False)[:400])
print("TEST REFREEZE REAL DONE")
