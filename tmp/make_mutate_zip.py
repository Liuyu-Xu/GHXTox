import zipfile, os
out = "lora_mamba_mutate_upload.zip"
files = [
    "src/ghxtox/lora_mamba_1d.py",
    "configs/lora_mamba_mutate.json",
    "scripts/run_lora_mamba_mutate.sh",
    "scripts/calibrate_lora_mamba.py",
    "scripts/refreeze_lora_mamba_fusion.py",
    "scripts/evaluate_lora_mamba_fusion.py",
    "scripts/ensemble_lora_mamba_predictions.py",
    "scripts/length_stratified_ablation.py",
    "scripts/paired_bootstrap_fusion.py",
    "scripts/calibration_report.py",
    "data/folds/train_cdhit080_fallback_folds.csv",
    "data/processed/test1_lora_mamba_light.pt",
    "data/processed/test2_lora_mamba_light.pt",
    "runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv",
    "runs/default_3d_hybrid/test1_predictions.csv",
    "runs/default_3d_hybrid/test2_predictions.csv",
    "runs/default_3d_hybrid/test1_metrics.json",
    "runs/default_3d_hybrid/test2_metrics.json",
    "docs/README_lora_mamba_mutate_autodl.md",
]
missing = [f for f in files if not os.path.exists(f)]
if missing:
    print("MISSING:", missing); raise SystemExit(1)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f)
        print("added", f, os.path.getsize(f))
print("wrote", out, round(os.path.getsize(out)/1e6, 2), "MB")
