# 1) module: read config with utf-8-sig (tolerates BOM)
p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = '    config = json.loads(Path(args.config).read_text(encoding="utf-8"))'
new = '    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))'
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("module config read -> utf-8-sig")

# 2) refreeze script: CSV read with utf-8-sig
p2 = "scripts/refreeze_lora_mamba_fusion.py"
s2 = open(p2, encoding="utf-8").read()
old2 = 'rows = sorted(list(csv.DictReader(open(args.three_d_oof, encoding="utf-8"))), key=lambda r: int(r["source_index"]))'
new2 = 'rows = sorted(list(csv.DictReader(open(args.three_d_oof, encoding="utf-8-sig"))), key=lambda r: int(r["source_index"]))'
assert old2 in s2
s2 = s2.replace(old2, new2, 1)
old2b = 'base_rows = sorted(list(csv.DictReader(open(args.baseline_oof, encoding="utf-8"))), key=lambda r: int(r["source_index"]))'
new2b = 'base_rows = sorted(list(csv.DictReader(open(args.baseline_oof, encoding="utf-8-sig"))), key=lambda r: int(r["source_index"]))'
if old2b in s2:
    s2 = s2.replace(old2b, new2b, 1)
open(p2, "w", encoding="utf-8", newline="\n").write(s2)
print("refreeze CSV read -> utf-8-sig")
