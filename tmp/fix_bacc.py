p = "scripts/refreeze_lora_mamba_fusion.py"
s = open(p, encoding="utf-8").read()
old = '''    sn = cum_pos / max(total_pos, 1)
    sp = (np.arange(1, n + 1) - cum_pos) / max(total_neg, 1)
    bacc = 0.5 * (sn + sp)'''
new = '''    sn = cum_pos / max(total_pos, 1)
    fp = (np.arange(1, n + 1) - cum_pos).astype(np.float64)
    tn = total_neg - fp
    sp = tn / max(total_neg, 1)
    bacc = 0.5 * (sn + sp)'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("fixed _bacc_scan specificity")
