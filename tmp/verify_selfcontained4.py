import sys, os, zipfile, shutil, subprocess, json
tmp = os.path.join("tmp", "extract_test4")
if os.path.exists(tmp):
    shutil.rmtree(tmp)
os.makedirs(tmp)
with zipfile.ZipFile("lora_mamba_autodl_upload.zip") as z:
    z.extractall(tmp)

# 1) module import + config strict utf-8 (server path)
code = r'''
import os, sys, json
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ghxtox.lora_mamba_1d as m
cfg = json.loads(open("configs/lora_mamba_1d_autodl.json", encoding="utf-8").read())
print("module+config OK:", cfg["plm"])
recs = m.load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
print("records:", len(recs))
print("VERIFY4-A PASS")
'''
open(os.path.join(tmp, "verify_a.py"), "w", encoding="utf-8").write(code)
env = dict(os.environ); env["PYTHONPATH"] = os.path.join(tmp, "src")
r = subprocess.run([sys.executable, "-u", "-B", "verify_a.py"], cwd=tmp, env=env, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR-A:", r.stderr[-2000:]); sys.exit(1)

# 2) run the FIXED refreeze inside the extracted package (real bundled data)
code2 = r'''
import os, sys, json
sys.path.insert(0, os.path.abspath("scripts"))
sys.argv = ["refreeze",
            "--new-1d-oof", "tmp/test_oof_real.json",
            "--fold-manifest", "data/folds/train_cdhit080_fallback_folds.csv",
            "--three-d-oof", "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv",
            "--output", "runs/lora_mamba_full/fusion_refreeze.json"]
import refreeze_lora_mamba_fusion as rf
rf.main()
out = json.load(open("runs/lora_mamba_full/fusion_refreeze.json"))
print("weight:", out["best_weight_1d"], "thr:", out["best_threshold"], "mcc:", out["fusion_oof_metrics"]["mcc"])
print("VERIFY4-B PASS")
'''
# copy the real OOF test json into the extracted tmp tree
os.makedirs(os.path.join(tmp, "tmp"), exist_ok=True)
import shutil as _s
_s.copyfile("tmp/test_oof_real.json", os.path.join(tmp, "tmp", "test_oof_real.json"))
open(os.path.join(tmp, "verify_b.py"), "w", encoding="utf-8").write(code2)
env2 = dict(os.environ); env2["PYTHONPATH"] = os.path.join(tmp, "scripts")
r2 = subprocess.run([sys.executable, "-u", "-B", "verify_b.py"], cwd=tmp, env=env2, capture_output=True, text=True)
print(r2.stdout)
if r2.returncode != 0:
    print("STDERR-B:", r2.stderr[-2000:]); sys.exit(1)
print("ALL VERIFY4 PASS")
