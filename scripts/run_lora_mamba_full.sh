#!/usr/bin/env bash
# Combination-A full run: LoRA-ProtT5 + BiMamba 1D expert (self-contained package).
#   Stage 1: 5-fold OOF for the frozen public seed set 42/123/2025
#   Stage 2: full-train checkpoints + Test1/Test2 prediction + seed ensemble
#   Stage 3: refreeze against the pure current 3D OOF + Test1/Test2 metrics
# Run from the repo root on AutoDL:
#   bash scripts/run_lora_mamba_full.sh
set -euo pipefail

PY="${PY:-python}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

SEEDS="${SEEDS:-42 123 2025}"
if [ "$SEEDS" != "42 123 2025" ]; then
  echo "ERROR: public full run requires SEEDS='42 123 2025'; use a separate script/output for ablations." >&2
  exit 2
fi
RUN=runs/lora_mamba_full
CFG=configs/lora_mamba_1d_autodl.json
mkdir -p "$RUN"

echo "== [1/3] 5-fold OOF (seeds: $SEEDS) =="
for seed in $SEEDS; do
  echo "--- seed $seed OOF ---"
  "$PY" -u -B -m ghxtox.lora_mamba_1d cv \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --config "$CFG" \
    --seed "$seed" \
    --output-dir "$RUN" \
    --device cuda
done

echo "== [2/3] full training + Test1/Test2 prediction =="
for seed in $SEEDS; do
  echo "--- seed $seed full ---"
  "$PY" -u -B -m ghxtox.lora_mamba_1d full \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --config "$CFG" \
    --seed "$seed" \
    --output "$RUN/full_seed${seed}.pt" \
    --device cuda
done
for ds in test1 test2; do
  for seed in $SEEDS; do
    "$PY" -u -B -m ghxtox.lora_mamba_1d predict \
      --adapter "data/processed/${ds}_lora_mamba_light.pt" \
      --checkpoint "$RUN/full_seed${seed}.pt" \
      --config "$CFG" \
      --output-csv "$RUN/${ds}_lora_mamba_seed${seed}.csv" \
      --device cuda
  done
  "$PY" -u -B scripts/ensemble_lora_mamba_predictions.py \
    --inputs $(for seed in $SEEDS; do echo "$RUN/${ds}_lora_mamba_seed${seed}.csv"; done) \
    --output "$RUN/${ds}_lora_mamba_ensemble.csv"
done

echo "== [3/3] required three-seed calibration + pure-3D fusion refreeze + Test metrics =="
FIRST_SEED=$(echo $SEEDS | awk '{print $1}')
OOF_IN="$RUN/oof_seed${FIRST_SEED}.json"
T1_IN="$RUN/test1_lora_mamba_ensemble.csv"
T2_IN="$RUN/test2_lora_mamba_ensemble.csv"
if [ "${CALIBRATE:-1}" != "1" ]; then
  echo "ERROR: the public default requires three-seed OOF Platt calibration; CALIBRATE=0 is ablation-only." >&2
  exit 2
fi
if [ "${CALIBRATE:-1}" = "1" ]; then
  if "$PY" -u -B scripts/calibrate_lora_mamba.py \
      --oof-jsons $(for seed in $SEEDS; do echo "$RUN/oof_seed${seed}.json"; done) \
      --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
      --test1-csv "$T1_IN" \
      --test2-csv "$T2_IN" \
      --output-oof "$RUN/oof_calibrated.json" \
      --output-test1 "$RUN/test1_lora_mamba_calibrated.csv" \
      --output-test2 "$RUN/test2_lora_mamba_calibrated.csv"; then
    OOF_IN="$RUN/oof_calibrated.json"
    T1_IN="$RUN/test1_lora_mamba_calibrated.csv"
    T2_IN="$RUN/test2_lora_mamba_calibrated.csv"
    echo "using calibrated probabilities"
  else
    echo "ERROR: calibration failed; refusing to fall back to a single/raw OOF generation" >&2
    exit 2
  fi
fi
PURE_3D_OOF=runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv
if [ -f "$PURE_3D_OOF" ]; then
  "$PY" -u -B scripts/refreeze_lora_mamba_fusion.py \
    --new-1d-oof "$OOF_IN" \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --three-d-oof "$PURE_3D_OOF" \
    --output "$RUN/fusion_refreeze.json"
else
  echo "ERROR: required pure 3D OOF is missing: $PURE_3D_OOF" >&2
  exit 2
fi
if [ -f "$RUN/fusion_refreeze.json" ] && [ -f runs/default_3d_hybrid/test1_predictions.csv ]; then
  "$PY" -u -B scripts/evaluate_lora_mamba_fusion.py \
    --refreeze "$RUN/fusion_refreeze.json" \
    --test1-1d "$T1_IN" \
    --test2-1d "$T2_IN" \
    --hybrid-test1 runs/default_3d_hybrid/test1_predictions.csv \
    --hybrid-test2 runs/default_3d_hybrid/test2_predictions.csv \
    --old-test1-metrics runs/default_3d_hybrid/test1_metrics.json \
    --old-test2-metrics runs/default_3d_hybrid/test2_metrics.json \
    --output "$RUN/test_fusion_metrics.json"
fi

echo "== DONE. See $RUN/ =="
