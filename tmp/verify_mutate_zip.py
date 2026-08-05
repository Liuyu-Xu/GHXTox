import sys, os, zipfile, shutil, subprocess, json
tmp = os.path.join("tmp", "mutate_extract")
if os.path.exists(tmp): shutil.rmtree(tmp)
os.makedirs(tmp)
with zipfile.ZipFile("lora_mamba_mutate_upload.zip") as z: z.extractall(tmp)
code = r'''
import os, sys, json
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import ghxtox.lora_mamba_1d as m
print("module OK; has mutation_augment_records:", hasattr(m, "mutation_augment_records"))
cfg = json.loads(open("configs/lora_mamba_mutate.json", encoding="utf-8").read())
print("config OK; mutation_aug:", cfg["mutation_aug"])
recs = m.load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
print("records:", len(recs))
import numpy as np
rng = np.random.RandomState(0)
out = m.mutation_augment_records(recs[:200], recs[200:260], cfg, rng)
print("augment fn runs; added:", len(out) - 200)
print("VERIFY MUTATE PASS")
'''
open(os.path.join(tmp, "verify.py"), "w", encoding="utf-8").write(code)
env = dict(os.environ); env["PYTHONPATH"] = os.path.join(tmp, "src")
r = subprocess.run([sys.executable, "-u", "-B", "verify.py"], cwd=tmp, env=env, capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr[-1500:]); sys.exit(1)
shutil.rmtree(tmp)
