import json, sys, os
sys.path.insert(0, os.path.abspath("scripts"))
import numpy as np
import refreeze_lora_mamba_fusion as rf

oof = json.load(open("tmp/test_oof_real.json"))
p = np.asarray(oof["oof"])
labels = np.asarray(oof["labels"])
fused = p  # w=1

n = len(labels)
order = np.argsort(-fused, kind="stable")
sorted_labels = labels[order]
probs_sorted = fused[order]
cum_pos = np.cumsum(sorted_labels)
total_pos = int(cum_pos[-1]); total_neg = n - total_pos
sn = cum_pos / max(total_pos, 1)
sp = (np.arange(1, n + 1) - cum_pos) / max(total_neg, 1)
bacc = 0.5 * (sn + sp)
change = np.ones(n, dtype=bool)
change[1:] = probs_sorted[1:] != probs_sorted[:-1]
candidate_idx = np.unique(np.concatenate([np.where(change)[0] - 1, [n - 1]]))
candidate_idx = candidate_idx[candidate_idx >= 0]
print("n:", n, "total_pos:", total_pos, "total_neg:", total_neg)
print("candidate count:", len(candidate_idx), "first 5:", candidate_idx[:5])
best_pos = int(candidate_idx[np.argmax(bacc[candidate_idx])])
print("vect best idx:", best_pos, "thr:", probs_sorted[best_pos], "bacc:", bacc[best_pos])
print("at best idx: cum_pos:", cum_pos[best_pos], "pred_pos:", best_pos+1, "sn:", sn[best_pos], "sp:", sp[best_pos])
# brute force at this exact threshold via sklearn
from sklearn.metrics import balanced_accuracy_score
m = balanced_accuracy_score(labels, (fused >= probs_sorted[best_pos]).astype(int))
print("sklearn bacc at that thr:", m)
# where does sklearn's brute-force best live?
candidates = np.sort(np.unique(fused))
bt, bm = None, -1
for t in candidates:
    mm = balanced_accuracy_score(labels, (fused >= t).astype(int))
    if mm > bm:
        bm = mm; bt = t
print("brute best thr:", bt, "bacc:", bm)
