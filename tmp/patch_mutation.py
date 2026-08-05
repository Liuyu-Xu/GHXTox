p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()

# 1) insert mutation-augmentation helpers after load_test_records (before class LoraMambaDataset)
anchor = "class LoraMambaDataset(Dataset):"
helpers = '''AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _point_mutate(sequence: str, rng, max_mutations: int) -> str:
    if not sequence:
        return sequence
    positions = rng.choice(len(sequence), size=int(rng.randint(1, max_mutations + 1)), replace=False)
    chars = list(sequence)
    for pos in positions:
        options = [c for c in AA_ALPHABET if c != chars[pos]]
        chars[pos] = options[int(rng.randint(0, len(options) - 1))]
    return "".join(chars)


def _identity_at_least(first: str, second: str, threshold: float) -> bool:
    la, lb = len(first), len(second)
    if min(la, lb) / max(la, lb) < threshold:
        return False
    if la == lb:
        same = sum(1 for a, b in zip(first, second) if a == b)
        return same / la >= threshold
    prev = list(range(lb + 1))
    for i, ca in enumerate(first, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(second, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return (min(la, lb) - prev[lb]) / min(la, lb) >= threshold


def mutation_augment_records(
    train_records: list[SequenceRecord],
    filter_records: list[SequenceRecord],
    config: dict[str, Any],
    rng,
) -> list[SequenceRecord]:
    """Point-mutate positive training sequences inside the fold; drop any augmented
    sequence whose identity to a filter (train+val) sequence reaches the threshold.

    This is the sequence-space analogue of the leakage-controlled Borderline-SMOTE
    protocol: synthetic samples are generated only from the training fold and are
    never allowed to become near-duplicates of validation sequences.
    """
    cfg = config.get("mutation_aug") or {}
    if not cfg.get("enabled", False):
        return train_records
    rate = float(cfg.get("rate", 0.2))
    max_mutations = int(cfg.get("max_mutations", 2))
    threshold = float(cfg.get("identity_threshold", 0.8))

    filter_seqs = [record.sequence for record in filter_records]
    exact = set(filter_seqs)
    kmer_index: dict[str, list[int]] = {}
    for idx, seq in enumerate(filter_seqs):
        for k in {seq[i:i + 2] for i in range(max(len(seq) - 1, 0))}:
            kmer_index.setdefault(k, []).append(idx)

    added: list[SequenceRecord] = []
    counter = 0
    for record in train_records:
        if record.label != 1 or not record.sequence:
            continue
        if rng.random() >= rate:
            continue
        mutated = _point_mutate(record.sequence, rng, max_mutations)
        if mutated in exact:
            continue
        candidates: set[int] = set()
        for i in range(len(mutated) - 1):
            candidates.update(kmer_index.get(mutated[i:i + 2], ()))
        if any(_identity_at_least(mutated, filter_seqs[idx], threshold) for idx in candidates):
            continue
        new_record = SequenceRecord()
        new_record.sample_id = f"{record.sample_id}__mut{counter}"
        new_record.sequence = mutated
        new_record.label = 1
        new_record.length = len(mutated)
        added.append(new_record)
        counter += 1
    return train_records + added


'''
assert anchor in s
s = s.replace(anchor, helpers + anchor, 1)

# 2) hook into run_cv (after train_records/val_records are built)
old = '''        train_idx, val_idx = _load_fold_indices(fold_manifest, fold, count)
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        n_pos = sum(1 for record in train_records if record.label == 1)'''
new = '''        train_idx, val_idx = _load_fold_indices(fold_manifest, fold, count)
        train_records = [records[i] for i in train_idx]
        val_records = [records[i] for i in val_idx]
        fold_rng = np.random.RandomState(seed * 1000 + fold)
        train_records = mutation_augment_records(train_records, train_records + val_records, config, fold_rng)
        n_pos = sum(1 for record in train_records if record.label == 1)'''
assert old in s
s = s.replace(old, new, 1)

# 3) hook into train_full (after records arrive, before n_pos)
old = '''    torch.manual_seed(seed)
    np.random.seed(seed)
    plm = _load_plm(config, device)
    model = build_model(plm, config, device)
    n_pos = sum(1 for record in records if record.label == 1)'''
new = '''    torch.manual_seed(seed)
    np.random.seed(seed)
    full_rng = np.random.RandomState(seed * 1000)
    records = mutation_augment_records(records, records, config, full_rng)
    plm = _load_plm(config, device)
    model = build_model(plm, config, device)
    n_pos = sum(1 for record in records if record.label == 1)'''
assert old in s
s = s.replace(old, new, 1)

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched mutation augmentation")
