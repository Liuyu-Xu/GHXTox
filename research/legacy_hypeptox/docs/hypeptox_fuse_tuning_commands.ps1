# Auto-generated HyPepTox-Fuse tuning commands.
# Run from the project root after activating the Python/PyTorch environment.
$env:PYTHONPATH='src'

# bilstm_focal_mcc seed 42: Increase sequence expert capacity and use focal BCE to improve hard-sample MCC/SN.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/bilstm_focal_mcc_seed42.json `
  --output-dir runs/hypeptox_tuning/bilstm_focal_mcc_seed42 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_focal_mcc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_focal_mcc_seed42/val_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_focal_mcc_seed42/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_focal_mcc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_focal_mcc_seed42/test_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_focal_mcc_seed42/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/bilstm_focal_mcc_seed42/val_predictions.csv `
  --apply runs/hypeptox_tuning/bilstm_focal_mcc_seed42/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/bilstm_focal_mcc_seed42/thresholded `
  --output-json runs/hypeptox_tuning/bilstm_focal_mcc_seed42/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# bilstm_focal_mcc seed 123: Increase sequence expert capacity and use focal BCE to improve hard-sample MCC/SN.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/bilstm_focal_mcc_seed123.json `
  --output-dir runs/hypeptox_tuning/bilstm_focal_mcc_seed123 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_focal_mcc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_focal_mcc_seed123/val_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_focal_mcc_seed123/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_focal_mcc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_focal_mcc_seed123/test_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_focal_mcc_seed123/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/bilstm_focal_mcc_seed123/val_predictions.csv `
  --apply runs/hypeptox_tuning/bilstm_focal_mcc_seed123/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/bilstm_focal_mcc_seed123/thresholded `
  --output-json runs/hypeptox_tuning/bilstm_focal_mcc_seed123/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# bilstm_focal_mcc seed 2025: Increase sequence expert capacity and use focal BCE to improve hard-sample MCC/SN.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/bilstm_focal_mcc_seed2025.json `
  --output-dir runs/hypeptox_tuning/bilstm_focal_mcc_seed2025 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/val_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/test_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/val_predictions.csv `
  --apply runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/thresholded `
  --output-json runs/hypeptox_tuning/bilstm_focal_mcc_seed2025/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# bilstm_smooth_auprc seed 42: Use mild label smoothing and AUPRC checkpointing to improve ranking stability.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/bilstm_smooth_auprc_seed42.json `
  --output-dir runs/hypeptox_tuning/bilstm_smooth_auprc_seed42 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/val_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/test_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/val_predictions.csv `
  --apply runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/thresholded `
  --output-json runs/hypeptox_tuning/bilstm_smooth_auprc_seed42/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# bilstm_smooth_auprc seed 123: Use mild label smoothing and AUPRC checkpointing to improve ranking stability.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/bilstm_smooth_auprc_seed123.json `
  --output-dir runs/hypeptox_tuning/bilstm_smooth_auprc_seed123 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/val_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/test_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/val_predictions.csv `
  --apply runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/thresholded `
  --output-json runs/hypeptox_tuning/bilstm_smooth_auprc_seed123/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# bilstm_smooth_auprc seed 2025: Use mild label smoothing and AUPRC checkpointing to improve ranking stability.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/bilstm_smooth_auprc_seed2025.json `
  --output-dir runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/val_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/test_metrics.json `
  --predictions runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/val_predictions.csv `
  --apply runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/thresholded `
  --output-json runs/hypeptox_tuning/bilstm_smooth_auprc_seed2025/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# geometry_focal_mcc seed 42: Tune geometry regularization and graph neighborhood for MCC while retaining 3D signal.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/geometry_focal_mcc_seed42.json `
  --output-dir runs/hypeptox_tuning/geometry_focal_mcc_seed42 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_focal_mcc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_focal_mcc_seed42/val_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_focal_mcc_seed42/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_focal_mcc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_focal_mcc_seed42/test_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_focal_mcc_seed42/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/geometry_focal_mcc_seed42/val_predictions.csv `
  --apply runs/hypeptox_tuning/geometry_focal_mcc_seed42/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/geometry_focal_mcc_seed42/thresholded `
  --output-json runs/hypeptox_tuning/geometry_focal_mcc_seed42/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# geometry_focal_mcc seed 123: Tune geometry regularization and graph neighborhood for MCC while retaining 3D signal.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/geometry_focal_mcc_seed123.json `
  --output-dir runs/hypeptox_tuning/geometry_focal_mcc_seed123 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_focal_mcc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_focal_mcc_seed123/val_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_focal_mcc_seed123/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_focal_mcc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_focal_mcc_seed123/test_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_focal_mcc_seed123/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/geometry_focal_mcc_seed123/val_predictions.csv `
  --apply runs/hypeptox_tuning/geometry_focal_mcc_seed123/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/geometry_focal_mcc_seed123/thresholded `
  --output-json runs/hypeptox_tuning/geometry_focal_mcc_seed123/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# geometry_focal_mcc seed 2025: Tune geometry regularization and graph neighborhood for MCC while retaining 3D signal.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/geometry_focal_mcc_seed2025.json `
  --output-dir runs/hypeptox_tuning/geometry_focal_mcc_seed2025 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_focal_mcc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_focal_mcc_seed2025/val_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_focal_mcc_seed2025/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_focal_mcc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_focal_mcc_seed2025/test_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_focal_mcc_seed2025/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/geometry_focal_mcc_seed2025/val_predictions.csv `
  --apply runs/hypeptox_tuning/geometry_focal_mcc_seed2025/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/geometry_focal_mcc_seed2025/thresholded `
  --output-json runs/hypeptox_tuning/geometry_focal_mcc_seed2025/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# geometry_low_noise_auprc seed 42: Reduce coordinate noise and select by AUPRC to target the remaining AUROC/AUPRC gap.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/geometry_low_noise_auprc_seed42.json `
  --output-dir runs/hypeptox_tuning/geometry_low_noise_auprc_seed42 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/val_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/test_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/val_predictions.csv `
  --apply runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/thresholded `
  --output-json runs/hypeptox_tuning/geometry_low_noise_auprc_seed42/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# geometry_low_noise_auprc seed 123: Reduce coordinate noise and select by AUPRC to target the remaining AUROC/AUPRC gap.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/geometry_low_noise_auprc_seed123.json `
  --output-dir runs/hypeptox_tuning/geometry_low_noise_auprc_seed123 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/val_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/test_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/val_predictions.csv `
  --apply runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/thresholded `
  --output-json runs/hypeptox_tuning/geometry_low_noise_auprc_seed123/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# geometry_low_noise_auprc seed 2025: Reduce coordinate noise and select by AUPRC to target the remaining AUROC/AUPRC gap.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/geometry_low_noise_auprc_seed2025.json `
  --output-dir runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/val_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/test_metrics.json `
  --predictions runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/val_predictions.csv `
  --apply runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/thresholded `
  --output-json runs/hypeptox_tuning/geometry_low_noise_auprc_seed2025/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# chemical_branch_focal_mcc seed 42: Tune only the chemical-site expert capacity around the current 3D-v2 branch.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/chemical_branch_focal_mcc_seed42.json `
  --output-dir runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/val_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/test_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/val_predictions.csv `
  --apply runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/thresholded `
  --output-json runs/hypeptox_tuning/chemical_branch_focal_mcc_seed42/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# chemical_branch_focal_mcc seed 123: Tune only the chemical-site expert capacity around the current 3D-v2 branch.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/chemical_branch_focal_mcc_seed123.json `
  --output-dir runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/val_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/test_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/val_predictions.csv `
  --apply runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/thresholded `
  --output-json runs/hypeptox_tuning/chemical_branch_focal_mcc_seed123/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# chemical_branch_focal_mcc seed 2025: Tune only the chemical-site expert capacity around the current 3D-v2 branch.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025.json `
  --output-dir runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/val_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/test_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/val_predictions.csv `
  --apply runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/thresholded `
  --output-json runs/hypeptox_tuning/chemical_branch_focal_mcc_seed2025/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91

# chemical_full_finetune_auprc seed 42: Low-LR full fine-tune from the 3D-v2 initialization to test whether frozen-base training is limiting ranking metrics.
python -m ghxtox.train `
  --train data/external/hypeptox_fuse/processed/train_fit_chemical_sites_esm2_augmix.pt `
  --val data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --config configs/hypeptox_tuning/chemical_full_finetune_auprc_seed42.json `
  --output-dir runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42 `
  --device cuda
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/train_val_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/val_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/val_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.evaluate `
  --checkpoint runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/best_model.pt `
  --processed data/external/hypeptox_fuse/processed/test_chemical_sites_esm2.pt `
  --output runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/test_metrics.json `
  --predictions runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/test_predictions.csv `
  --device cuda `
  --threshold 0.5
python -m ghxtox.threshold_predictions `
  --validation runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/val_predictions.csv `
  --apply runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/test_predictions.csv `
  --output-dir runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/thresholded `
  --output-json runs/hypeptox_tuning/chemical_full_finetune_auprc_seed42/thresholded/threshold_summary.json `
  --objective mcc `
  --min-sn 0.875 `
  --min-sp 0.91
