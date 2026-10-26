# Prediction-Level Ensemble Search On AutoDL

This workflow does not retrain checkpoints. It searches validation-selected
model combinations, weights, and thresholds from saved prediction CSV files.

Run it after the 16 HyPepTox tuning runs have produced:

- `runs/hypeptox_tuning/*/val_predictions.csv`
- `runs/hypeptox_tuning/*/test_predictions.csv`
- `configs/hypeptox_tuning/manifest.json`

## MCC-First Search

```bash
cd /root/autodl-tmp/GHXTox
export PYTHONPATH=/root/autodl-tmp/GHXTox/src:$PYTHONPATH

python -m ghxtox.prediction_ensemble_search \
  --manifest configs/hypeptox_tuning/manifest.json \
  --output-dir runs/hypeptox_tuning/prediction_ensemble_search_mcc \
  --objective mcc \
  --min-sn 0.875 \
  --min-sp 0.91 \
  --min-k 2 \
  --max-k 10 \
  --refine-top 64 \
  --weight-trials 512 \
  --weight-seed 2026 \
  --weight-alpha 0.8 \
  --output-top 100
```

Key outputs:

- `ensemble_search_summary.json`
- `ensemble_search_ranking.csv`
- `best_val_predictions.csv`
- `best_test_predictions.csv`

## AUPRC-Oriented Search

Use this if MCC is stable but ranking metrics need improvement.

```bash
python -m ghxtox.prediction_ensemble_search \
  --manifest configs/hypeptox_tuning/manifest.json \
  --output-dir runs/hypeptox_tuning/prediction_ensemble_search_auprc \
  --objective auprc \
  --min-sn 0.875 \
  --min-sp 0.91 \
  --min-k 2 \
  --max-k 10 \
  --refine-top 64 \
  --weight-trials 512 \
  --weight-seed 2027 \
  --weight-alpha 0.8 \
  --output-top 100
```

## Calibration After Search

Calibrate the best MCC-first search result on validation predictions and apply
it to the independent test predictions.

```bash
python -m ghxtox.calibrate_predictions \
  --validation runs/hypeptox_tuning/prediction_ensemble_search_mcc/best_val_predictions.csv \
  --apply runs/hypeptox_tuning/prediction_ensemble_search_mcc/best_test_predictions.csv \
  --method platt \
  --optimize-threshold-metric mcc \
  --output-dir runs/hypeptox_tuning/prediction_ensemble_search_mcc/calibrated_platt \
  --output-json runs/hypeptox_tuning/prediction_ensemble_search_mcc/calibration_platt_summary.json
```

## Inspect Results

```bash
cat runs/hypeptox_tuning/prediction_ensemble_search_mcc/ensemble_search_summary.json
head -n 20 runs/hypeptox_tuning/prediction_ensemble_search_mcc/ensemble_search_ranking.csv
cat runs/hypeptox_tuning/prediction_ensemble_search_mcc/calibration_platt_summary.json
```
