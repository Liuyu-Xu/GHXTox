p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = '    p_cv.add_argument("--max-samples", type=int, default=None, help="Pilot: only keep the first N records.")'
new = '    p_cv.add_argument("--max-samples", type=int, default=None, help="Pilot: only keep the first N records.")\n    p_cv.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")'
assert old in s
s = s.replace(old, new, 1)
old = '    p_full.add_argument("--max-epochs", type=int, default=None)'
new = '    p_full.add_argument("--max-epochs", type=int, default=None)\n    p_full.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")'
assert old in s
s = s.replace(old, new, 1)
old = '    p_pred.add_argument("--device", default="auto")\n    return parser'
new = '    p_pred.add_argument("--device", default="auto")\n    p_pred.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")\n    return parser'
assert old in s
s = s.replace(old, new, 1)
old = '''    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    from transformers import T5Tokenizer'''
new = '''    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if getattr(args, "batch_size", None):
        config["batch_size"] = args.batch_size
    from transformers import T5Tokenizer'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched batch-size override")
