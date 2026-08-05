p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = '''        records = load_train_records(args.fold_manifest)
        if getattr(args, "max_samples", None):
            records = records[: args.max_samples]
        result = run_cv(records, args.fold_manifest, config, args.seed, device, tokenizer, args.max_epochs, args.output_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(result, out_dir / f"oof_seed{args.seed}.json")'''
new = '''        records = load_train_records(args.fold_manifest)
        if getattr(args, "max_samples", None):
            records = records[: args.max_samples]
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_cv(records, args.fold_manifest, config, args.seed, device, tokenizer, args.max_epochs, args.output_dir)
        save_json(result, out_dir / f"oof_seed{args.seed}.json")'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched mkdir")
