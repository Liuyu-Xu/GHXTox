import json, sys, os
sys.path.insert(0, os.path.abspath("scripts"))
import numpy as np
from sklearn.metrics import matthews_corrcoef
import refreeze_lora_mamba_fusion as rf

oof = json.load(open("tmp/test_oof_real.json"))
p = np.asarray(oof["oof"])
labels = np.asarray(oof["labels"])
w = 0.5
fused = w * p + (1 - w) * p

# brute force
candidates = np.sort(np.unique(fused))
best_t, best_m = None, -1
for t in candidates:
    m = matthews_corrcoef(labels, (fused >= t).astype(int))
    if m > best_m:
        best_m = m; best_t = t
print("brute: thr=%.9f mcc=%.9f" % (best_t, best_m))

# vectorized
vt, vm = rf._mcc_scan(labels, fused)
print("vect : thr=%.9f mcc=%.9f" % (vt, vm))
assert abs(vt - best_t) < 1e-9 and abs(vm - best_m) < 1e-9, "MISMATCH"
print("MCC SCAN MATCH OK")

# bacc scan vs brute
from sklearn.metrics import balanced_accuracy_score
bt, bm = None, -1
for t in candidates:
    m = balanced_accuracy_score(labels, (fused >= t).astype(int))
    if m > bm:
        bm = m; bt = t
vt2, vm2 = rf._bacc_scan(labels, fused)
print("bacc brute: %.9f %.9f | vect: %.9f %.9f" % (bt, bm, vt2, vm2))
assert abs(vt2 - bt) < 1e-9 and abs(vm2 - bm) < 1e-9, "BACC MISMATCH"
print("BACC SCAN MATCH OK")
