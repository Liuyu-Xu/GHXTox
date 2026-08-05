import sys, os, zipfile, shutil, subprocess
root = os.path.abspath(".")
tmp = os.path.join("tmp", "extract_test")
if os.path.exists(tmp):
    shutil.rmtree(tmp)
os.makedirs(tmp)
with zipfile.ZipFile("lora_mamba_autodl_upload.zip") as z:
    z.extractall(tmp)
print("extracted to", tmp)
print("extracted files:", sorted(os.listdir(tmp)))

# simulate the server: PYTHONPATH=<extracted>/src only, run a real import + fold load
code = r'''
import os, sys
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ghxtox.lora_mamba_1d as m
print("module import OK (self-contained)")
recs = m.load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
print("records:", len(recs))
tr, va = m._load_fold_indices("data/folds/train_cdhit080_fallback_folds.csv", 0, len(recs))
print("fold0 train/val:", len(tr), len(va))
# light adapter load
import torch
for ds in ("test1", "test2"):
    p = torch.load(f"data/processed/{ds}_lora_mamba_light.pt", map_location="cpu", weights_only=False)
    print(ds, "light adapter n:", len(p["sample_ids"]))
print("VERIFY PASS")
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
