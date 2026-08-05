import sys, os, zipfile, shutil, subprocess
root = os.path.abspath(".")
tmp = os.path.join("tmp", "extract_test2")
if os.path.exists(tmp):
    shutil.rmtree(tmp)
os.makedirs(tmp)
with zipfile.ZipFile("lora_mamba_autodl_upload.zip") as z:
    z.extractall(tmp)

code = r'''
import os, sys, json
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ghxtox.lora_mamba_1d as m

# 1) config read with STRICT utf-8 (this failed on the server before the fix)
cfg = json.loads(open("configs/lora_mamba_1d_autodl.json", encoding="utf-8").read())
print("config strict-utf8 OK:", cfg["plm"], "epochs", cfg["epochs"])

# 2) module reads config via utf-8-sig (server path)
import argparse, types
args = types.SimpleNamespace(config="configs/lora_mamba_1d_autodl.json")
config = json.loads(open(args.config, encoding="utf-8-sig").read())
print("config utf-8-sig OK")

# 3) folds CSV strict utf-8 (module load_train_records path)
recs = m.load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
print("records:", len(recs))

# 4) no BOM on any bundled text file
for f in ["configs/lora_mamba_1d_autodl.json",
          "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv",
          "runs/default_3d_hybrid/test1_predictions.csv",
          "runs/default_3d_hybrid/test2_predictions.csv",
          "runs/default_3d_hybrid/test1_metrics.json",
          "runs/default_3d_hybrid/test2_metrics.json"]:
    head = open(f, "rb").read(3)
    print("BOM" if head == b"\xef\xbb\xbf" else "no-BOM", f)

# 5) fold split
tr, va = m._load_fold_indices("data/folds/train_cdhit080_fallback_folds.csv", 0, len(recs))
print("fold0 train/val:", len(tr), len(va))
print("VERIFY2 PASS")
'''
script = os.path.join(tmp, "verify.py")
open(script, "w", encoding="utf-8").write(code)
env = dict(os.environ)
env["PYTHONPATH"] = os.path.join(tmp, "src")
r = subprocess.run([sys.executable, "-u", "-B", "verify.py"], cwd=tmp, env=env, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:])
    sys.exit(1)
