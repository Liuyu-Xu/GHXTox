p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()

# 1) kmer size 2 -> 3 in mutation_augment_records
old = '''    filter_seqs = [record.sequence for record in filter_records]
    exact = set(filter_seqs)
    kmer_index: dict[str, list[int]] = {}
    for idx, seq in enumerate(filter_seqs):
        for k in {seq[i:i + 2] for i in range(max(len(seq) - 1, 0))}:
            kmer_index.setdefault(k, []).append(idx)
'''
new = '''    filter_seqs = [record.sequence for record in filter_records]
    exact = set(filter_seqs)
    kmer_size = 3
    kmer_index: dict[str, list[int]] = {}
    for idx, seq in enumerate(filter_seqs):
        for k in {seq[i:i + kmer_size] for i in range(max(len(seq) - kmer_size + 1, 0))}:
            kmer_index.setdefault(k, []).append(idx)
'''
assert old in s
s = s.replace(old, new, 1)

old = '''        candidates: set[int] = set()
        for i in range(len(mutated) - 1):
            candidates.update(kmer_index.get(mutated[i:i + 2], ()))
'''
new = '''        candidates: set[int] = set()
        for i in range(len(mutated) - kmer_size + 1):
            candidates.update(kmer_index.get(mutated[i:i + kmer_size], ()))
'''
assert old in s
s = s.replace(old, new, 1)

# 2) run_cv: filter against validation only (leakage guard), not train (parent)
old = "        train_records = mutation_augment_records(train_records, train_records + val_records, config, fold_rng)"
new = "        train_records = mutation_augment_records(train_records, val_records, config, fold_rng)"
assert old in s
s = s.replace(old, new, 1)

# 3) train_full: no filter (no validation available; test sequences are never used)
old = "    records = mutation_augment_records(records, records, config, full_rng)"
new = "    records = mutation_augment_records(records, [], config, full_rng)"
assert old in s
s = s.replace(old, new, 1)

# 4) docstring clarification
old = '''    """Point-mutate positive training sequences inside the fold; drop any augmented
    sequence whose identity to a filter (train+val) sequence reaches the threshold.

    This is the sequence-space analogue of the leakage-controlled Borderline-SMOTE
    protocol: synthetic samples are generated only from the training fold and are
    never allowed to become near-duplicates of validation sequences.
    """'''
new = '''    """Point-mutate positive training sequences; drop any augmented sequence whose
    identity to a *filter* sequence (the fold validation set) reaches the threshold.

    This is the sequence-space analogue of the leakage-controlled Borderline-SMOTE
    protocol: synthetic samples are generated only from the training fold and are
    never allowed to become near-duplicates of validation sequences.  The parent
    sequence itself is intentionally NOT filtered against, since a 1-2 point
    mutation is by construction a near-copy of its parent.
    """'''
assert old in s
s = s.replace(old, new, 1)

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched mutation filter semantics")
