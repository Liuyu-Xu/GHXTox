# Local Metric Optimization Workflow

These steps do not require ProtT5 generation or ESMFold generation. They use existing checkpoints and processed tensors.

## 1. Search Ensemble Weights And Threshold On Validation

Use the current seven ESM2/geometry/3D-v2 checkpoints and select both weights and threshold on the no-leak validation split.

```bash
python -m ghxtox.ensemble_evaluate \
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt \
  --checkpoints \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_3d_v2_augmix_noleak/best_model.pt \
  --optimize-weights-metric mcc \
  --optimize-threshold-metric mcc \
  --weight-search-trials 512 \
  --weight-search-seed 42 \
  --output runs/hypeptox_fuse_ensemble_local_optimized/val_weight_threshold_metrics.json \
  --predictions runs/hypeptox_fuse_ensemble_local_optimized/val_predictions.csv \
  --device cuda
```

Copy `weight_search.best_weights` and `threshold` from the JSON before evaluating test.

## 2. Apply The Selected Weights And Threshold To Test

```bash
python -m ghxtox.ensemble_evaluate \
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt \
  --checkpoints \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_esm2_bilstm_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed123/best_model.pt \
    runs/hypeptox_fuse_plm_fusion_esm2_geometry_confidence_augmix_noleak_seed2025/best_model.pt \
    runs/hypeptox_fuse_3d_v2_augmix_noleak/best_model.pt \
  --weights WEIGHT_1 WEIGHT_2 WEIGHT_3 WEIGHT_4 WEIGHT_5 WEIGHT_6 WEIGHT_7 \
  --threshold THRESHOLD_FROM_VALIDATION \
  --output runs/hypeptox_fuse_ensemble_local_optimized/test_metrics.json \
  --predictions runs/hypeptox_fuse_ensemble_local_optimized/test_predictions.csv \
  --device cuda
```

## 3. Fit Probability Calibration On Validation Predictions

Temperature scaling is the safest first calibration method for Brier and ECE-10.

```bash
python -m ghxtox.calibrate_predictions \
  --validation runs/hypeptox_fuse_ensemble_local_optimized/val_predictions.csv \
  --apply runs/hypeptox_fuse_ensemble_local_optimized/test_predictions.csv \
  --method temperature \
  --optimize-threshold-metric mcc \
  --output-dir runs/hypeptox_fuse_ensemble_local_optimized/calibrated \
  --output-json runs/hypeptox_fuse_ensemble_local_optimized/calibration_summary.json
```

If temperature scaling underfits, try Platt scaling as a second pass:

```bash
python -m ghxtox.calibrate_predictions \
  --validation runs/hypeptox_fuse_ensemble_local_optimized/val_predictions.csv \
  --apply runs/hypeptox_fuse_ensemble_local_optimized/test_predictions.csv \
  --method platt \
  --optimize-threshold-metric mcc \
  --output-dir runs/hypeptox_fuse_ensemble_local_optimized/platt_calibrated \
  --output-json runs/hypeptox_fuse_ensemble_local_optimized/platt_calibration_summary.json
```

## What Each Step Targets

- Weight search: ACC, BACC, MCC, sometimes AUROC/AUPRC.
- Threshold search: ACC, BACC, SN, SP, MCC.
- Calibration: Brier and ECE-10; AUROC/AUPRC usually stay unchanged.
