import sys, os, json
sys.path.insert(0, os.path.abspath("src"))
import numpy as np
from ghxtox.lora_mamba_1d import load_train_records, mutation_augment_records

config = json.load(open("configs/lora_mamba_mutate.json", encoding="utf-8"))
records = load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
print("total records:", len(records), "pos:", sum(r.label == 1 for r in records))

# simulate a fold: train ~80%, val ~20% (split by index)
rng0 = np.random.RandomState(42)
idx = np.arange(len(records))
rng0.shuffle(idx)
train = [records[i] for i in idx[:5100]]
val = [records[i] for i in idx[5100:]]
print("train:", len(train), "pos:", sum(r.label == 1 for r in train), "| val:", len(val))

rng = np.random.RandomState(42 * 1000 + 0)
aug = mutation_augment_records(train, train + val, config, rng)
added = [r for r in aug if "__mut" in r.sample_id]
print("augmented count:", len(added), "-> train size:", len(aug), "pos:", sum(r.label == 1 for r in aug))
print("all augmented labels == 1:", all(r.label == 1 for r in added))
print("all lengths positive:", all(len(r.sequence) == r.length for r in aug))

# leakage check: no augmented seq exactly in val, and none >=0.8 identity to val
val_seqs = {r.sequence for r in val}
leak_exact = sum(1 for r in added if r.sequence in val_seqs)
print("exact leak into val:", leak_exact)
# identity >= 0.8 check (same length hamming)
from ghxtox.lora_mamba_1d import _identity_at_least
leak_sim = 0
for r in added:
    if any(_identity_at_least(r.sequence, v.sequence, 0.8) for v in val):
        leak_sim += 1
print(">=0.8 identity leak into val:", leak_sim)

# reproducibility
rng2 = np.random.RandomState(42 * 1000 + 0)
aug2 = mutation_augment_records(train, train + val, config, rng2)
added2 = [r for r in aug2 if "__mut" in r.sample_id]
print("reproducible:", [r.sequence for r in added] == [r.sequence for r in added2])
print("TEST MUTATION OK")
