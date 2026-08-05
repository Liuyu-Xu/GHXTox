import sys, os, zipfile, shutil, subprocess
tmp = os.path.join("tmp", "extract_test3")
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
cfg = json.loads(open("configs/lora_mamba_1d_autodl.json", encoding="utf-8").read())
print("config strict-utf8 OK; plm =", cfg["plm"])
print("has _make_scaler:", hasattr(m, "_make_scaler"))
print("has _amp_autocast:", hasattr(m, "_amp_autocast"))
recs = m.load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
tr, va = m._load_fold_indices("data/folds/train_cdhit080_fallback_folds.csv", 0, len(recs))
print("records:", len(recs), "fold0 train/val:", len(tr), len(va))
print("VERIFY3 PASS")
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
