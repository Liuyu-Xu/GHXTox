p = "scripts/run_lora_mamba_full.sh"
s = open(p, encoding="utf-8").read()
old = """  "$PY" -u -B scripts/refreeze_lora_mamba_fusion.py \\
    --new-1d-oof "$RUN/oof_seed${FIRST_SEED}.json" \\
    --three-d-oof runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv \\
    --output "$RUN/fusion_refreeze.json\""""
new = """  "$PY" -u -B scripts/refreeze_lora_mamba_fusion.py \\
    --new-1d-oof "$RUN/oof_seed${FIRST_SEED}.json" \\
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \\
    --three-d-oof runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv \\
    --output "$RUN/fusion_refreeze.json\""""
assert old in s, "anchor not found"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("run script patched")
