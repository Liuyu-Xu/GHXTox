# HyPepTox-Fuse Balanced Benchmark Optimization Plan

## Current Focus

The immediate priority is model and algorithm improvement.  Paper tables and
figures are already reproducible, but manuscript writing should wait until the
model margin over HyPepTox-Fuse is larger and more stable.

New algorithm switches available in `train` configs:

- `ema_decay` and `ema_start_epoch`: validate and save exponential moving
  average weights for smoother generalization.
- `scheduler: "cosine"` and `min_learning_rate`: use cosine learning-rate decay.
- `selection_score`: select checkpoints by a composite validation score, for
  example `{"mcc": 1.0, "auroc": 0.2, "auprc": 0.2}`.
- `selection_constraints`: softly penalize checkpoints whose validation `sn` or
  `sp` falls below a target.
- `save_auxiliary_checkpoints`: keep extra best-MCC and best-AUPRC checkpoints
  for later ensemble search.

New threshold tool:

```bash
python -m ghxtox.threshold_predictions \
  --validation runs/example/val_predictions.csv \
  --apply runs/example/test_predictions.csv \
  --output-dir runs/example/thresholded \
  --output-json runs/example/thresholded/threshold_summary.json \
  --objective mcc \
  --min-sn 0.875 \
  --min-sp 0.91
```

This selects a threshold only on validation predictions, then rewrites applied
prediction CSVs and reports their metrics.

## Reporting Policy

The HyPepTox-Fuse independent test split used in this project is balanced
(`1104` toxic and `1104` non-toxic samples).  For this reason, the headline
comparison table should omit BACC.  The main balanced-set table should report:

- ACC
- SN
- SP
- F1
- MCC
- AUROC
- AUPRC
- Brier
- ECE-10

HyPepTox-Fuse reports SEN, SPE, ACC, F1, MCC, and AUC in its paper table.  It
does not report AUPRC, Brier, or ECE-10, and it does not provide per-sample
probabilities.  Therefore, direct comparison with HyPepTox-Fuse is a
point-estimate comparison.  Paired bootstrap or randomization tests should be
used only among our own saved prediction CSVs unless HyPepTox-Fuse per-sample
outputs are reproduced.

## Reproducible Analysis Command

After saving model metrics and prediction CSV files, regenerate all comparison
tables and figures with:

```powershell
python -m ghxtox.hypeptox_paper_analysis `
  --root D:\dachuang2026\GHXTox `
  --output-dir reports/hypeptox_fuse_paper_analysis `
  --bootstrap-iterations 200 `
  --seed 2026
```

For the final paper version, run a longer paired bootstrap on the server:

```powershell
python -m ghxtox.hypeptox_paper_analysis `
  --root D:\dachuang2026\GHXTox `
  --output-dir reports/hypeptox_fuse_paper_analysis_final `
  --bootstrap-iterations 1000 `
  --seed 2026
```

The command writes:

- `headline_comparison_no_bacc.csv`
- `ablation_table_no_bacc.csv`
- `paired_bootstrap_vs_final.csv`
- `summary.json`
- `headline_delta_vs_hypeptox_fuse.png`
- `ablation_progression.png`
- `calibration_brier_ece.png`
- `final_roc_pr_calibration.png`

## Ablation Story

The current paper-safe ablation order is:

1. `3D-v2 baseline`
2. `ESM2-BiLSTM + AugMix`
3. `ESM2-Geometry + AugMix`
4. `3D-v2 + AugMix`
5. `ESM2-BiLSTM 3-seed ensemble`
6. `ESM2-Geometry 3-seed ensemble`
7. `ESM2 6-model ensemble`
8. `ESM2+3D 7-model ensemble`
9. `ESM2+3D 7-model ensemble + validation threshold`
10. Calibration variants for Brier and ECE-10 discussion

This supports a discussion that the performance gain comes from training-only
feature-space augmentation, multi-seed variance reduction, heterogeneous expert
complementarity, validation-only threshold selection, and post-hoc calibration.

## Parameter Tuning Priority

Do not tune on the independent test labels.  Select parameters on the held-out
validation split, then evaluate the unchanged independent set once per saved
configuration.

Generate the current tuning matrix and command files with:

```powershell
python -m ghxtox.hypeptox_tuning_matrix --device cuda
```

This writes:

- `configs/hypeptox_tuning/*.json`
- `configs/hypeptox_tuning/manifest.json`
- `docs/hypeptox_fuse_tuning_commands.sh`
- `docs/hypeptox_fuse_tuning_commands.ps1`

On AutoDL/Linux, run selected blocks from:

```bash
bash docs/hypeptox_fuse_tuning_commands.sh
```

Highest-impact sweeps:

1. Ensemble weight search on validation predictions/checkpoint probabilities.
2. Threshold selection using validation MCC or ACC/F1 joint score.
3. AugMix strength and synthetic ratio: `alpha`, nearest-neighbor count, and
   synthetic multiplier.
4. Training regularization: dropout, weight decay, learning rate, batch size,
   early-stopping patience.
5. Expert diversity: add ProtT5/TextCNN seeds when server features are ready.
6. Calibration method selection: Platt for Brier, temperature scaling for ECE.

Every tuning run should save the config JSON, random seed, train/validation/test
processed file paths, checkpoint path, prediction CSV, metric JSON, and command.
