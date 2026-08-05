p = "scripts/run_lora_mamba_full.sh"
s = open(p, encoding="utf-8").read()

old = '''echo "== [3/3] fusion refreeze + Test fusion metrics =="
FIRST_SEED=$(echo $SEEDS | awk '{print $1}')
if [ -f runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv ]; then
  "$PY" -u -B scripts/refreeze_lora_mamba_fusion.py \\
    --new-1d-oof "$RUN/oof_seed${FIRST_SEED}.json" \\
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \\
    --three-d-oof runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv \\
    --output "$RUN/fusion_refreeze.json"
fi
if [ -f "$RUN/fusion_refreeze.json" ] && [ -f runs/default_3d_hybrid/test1_predictions.csv ]; then
  "$PY" -u -B scripts/evaluate_lora_mamba_fusion.py \\
    --refreeze "$RUN/fusion_refreeze.json" \\
    --test1-1d "$RUN/test1_lora_mamba_ensemble.csv" \\
    --test2-1d "$RUN/test2_lora_mamba_ensemble.csv" \\
    --hybrid-test1 runs/default_3d_hybrid/test1_predictions.csv \\
    --hybrid-test2 runs/default_3d_hybrid/test2_predictions.csv \\
    --old-test1-metrics runs/default_3d_hybrid/test1_metrics.json \\
    --old-test2-metrics runs/default_3d_hybrid/test2_metrics.json \\
    --output "$RUN/test_fusion_metrics.json"
fi'''

new = '''echo "== [3/3] calibration (optional) + fusion refreeze + Test fusion metrics =="
FIRST_SEED=$(echo $SEEDS | awk '{print $1}')
OOF_IN="$RUN/oof_seed${FIRST_SEED}.json"
T1_IN="$RUN/test1_lora_mamba_ensemble.csv"
T2_IN="$RUN/test2_lora_mamba_ensemble.csv"
if [ "${CALIBRATE:-1}" = "1" ]; then
  if "$PY" -u -B scripts/calibrate_lora_mamba.py \\
      --oof-jsons $(for seed in $SEEDS; do echo "$RUN/oof_seed${seed}.json"; done) \\
      --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \\
      --test1-csv "$T1_IN" \\
      --test2-csv "$T2_IN" \\
      --output-oof "$RUN/oof_calibrated.json" \\
      --output-test1 "$RUN/test1_lora_mamba_calibrated.csv" \\
      --output-test2 "$RUN/test2_lora_mamba_calibrated.csv"; then
    OOF_IN="$RUN/oof_calibrated.json"
    T1_IN="$RUN/test1_lora_mamba_calibrated.csv"
    T2_IN="$RUN/test2_lora_mamba_calibrated.csv"
    echo "using calibrated probabilities"
  else
    echo "calibration failed; falling back to raw probabilities"
  fi
fi
if [ -f runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv ]; then
  "$PY" -u -B scripts/refreeze_lora_mamba_fusion.py \\
    --new-1d-oof "$OOF_IN" \\
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \\
    --three-d-oof runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv \\
    --output "$RUN/fusion_refreeze.json"
fi
if [ -f "$RUN/fusion_refreeze.json" ] && [ -f runs/default_3d_hybrid/test1_predictions.csv ]; then
  "$PY" -u -B scripts/evaluate_lora_mamba_fusion.py \\
    --refreeze "$RUN/fusion_refreeze.json" \\
    --test1-1d "$T1_IN" \\
    --test2-1d "$T2_IN" \\
    --hybrid-test1 runs/default_3d_hybrid/test1_predictions.csv \\
    --hybrid-test2 runs/default_3d_hybrid/test2_predictions.csv \\
    --old-test1-metrics runs/default_3d_hybrid/test1_metrics.json \\
    --old-test2-metrics runs/default_3d_hybrid/test2_metrics.json \\
    --output "$RUN/test_fusion_metrics.json"
fi'''

assert old in s, "anchor not found"
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("run script patched with calibration")
