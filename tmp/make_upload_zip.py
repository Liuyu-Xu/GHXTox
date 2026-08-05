import zipfile, os
out = "lora_mamba_autodl_upload.zip"
files = [
    "src/ghxtox/lora_mamba_1d.py",
    "configs/lora_mamba_1d_autodl.json",
    "scripts/run_lora_mamba_full.sh",
    "scripts/refreeze_lora_mamba_fusion.py",
    "scripts/calibrate_lora_mamba.py",
    "scripts/evaluate_lora_mamba_fusion.py",
    "scripts/ensemble_lora_mamba_predictions.py",
    "data/folds/train_cdhit080_fallback_folds.csv",
    "data/processed/test1_lora_mamba_light.pt",
    "data/processed/test2_lora_mamba_light.pt",
    "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv",
    "runs/default_3d_hybrid/test1_predictions.csv",
    "runs/default_3d_hybrid/test2_predictions.csv",
    "runs/default_3d_hybrid/test1_metrics.json",
    "runs/default_3d_hybrid/test2_metrics.json",
    "docs/README_lora_mamba_autodl.md",
]
missing = [f for f in files if not os.path.exists(f)]
if missing:
    print("MISSING:", missing); raise SystemExit(1)
total = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for arc in files:
        z.write(arc, arc)
        total += os.path.getsize(arc)
        print("added", arc, os.path.getsize(arc))
print("wrote", out, round(os.path.getsize(out)/1e6, 2), "MB (raw", round(total/1e6, 2), "MB)")
