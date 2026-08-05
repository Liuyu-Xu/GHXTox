import sys, os, json, time
sys.path.insert(0, os.path.abspath("src"))
import numpy as np
from ghxtox.lora_mamba_1d import load_train_records, mutation_augment_records, _identity_at_least

config = json.load(open("configs/lora_mamba_mutate.json", encoding="utf-8"))
records = load_train_records("data/folds/train_cdhit080_fallback_folds.csv")
rng0 = np.random.RandomState(42)
idx = np.arange(len(records)); rng0.shuffle(idx)
train = [records[i] for i in idx[:5100]]
val = [records[i] for i in idx[5100:]]

t0 = time.time()
rng = np.random.RandomState(42 * 1000 + 0)
aug = mutation_augment_records(train, val, config, rng)
added = [r for r in aug if "__mut" in r.sample_id]
dt = time.time() - t0
print("augmented:", len(added), "of", sum(r.label == 1 for r in train), "positives, in %.1fs" % dt)
print("train size:", len(aug), "pos:", sum(r.label == 1 for r in aug))
print("all labels==1:", all(r.label == 1 for r in added))
print("all mutated length == len(seq):", all(len(r.sequence) == r.length for r in added))
# mutation actually differs from parent
par = {r.sample_id: r.sequence for r in train}
diff = sum(1 for r in added if par.get(r.sample_id.replace("__mut", "").rsplit("__", 1)[0], r.sequence) == r.sequence)
# simpler: parent id is before __mut
diff = sum(1 for r in added if par[r.sample_id.split("__")[0]] != r.sequence)
print("all differ from parent:", diff == len(added))
# leak into val
leak = sum(1 for r in added if any(_identity_at_least(r.sequence, v.sequence, 0.8) for v in val))
print(">=0.8 identity leak into val:", leak)
# reproducibility
rng2 = np.random.RandomState(42 * 1000 + 0)
aug2 = mutation_augment_records(train, val, config, rng2)
added2 = [r for r in aug2 if "__mut" in r.sample_id]
print("reproducible:", [r.sequence for r in added] == [r.sequence for r in added2])
print("TEST MUTATION2 OK")
