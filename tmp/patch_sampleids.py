p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = '''    labels = np.asarray([record.label for record in records])
    count = len(records)
    oof = np.zeros(count, dtype=np.float64)'''
new = '''    labels = np.asarray([record.label for record in records])
    sample_ids = [record.sample_id for record in records]
    count = len(records)
    oof = np.zeros(count, dtype=np.float64)'''
assert old in s
s = s.replace(old, new, 1)
old = '''        "oof": oof.tolist(),
        "labels": labels.tolist(),
        "folds": fold_assignments,'''
new = '''        "oof": oof.tolist(),
        "labels": labels.tolist(),
        "sample_ids": sample_ids,
        "folds": fold_assignments,'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("cv now saves sample_ids")
