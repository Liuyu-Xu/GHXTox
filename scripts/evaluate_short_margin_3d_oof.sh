#!/usr/bin/env bash
set -euo pipefail

stage="${1:-all}"
resume="${2:-}"
device="${GHXTOX_DEVICE:-cuda}"
python_bin="${GHXTOX_PYTHON:-python}"

case "${stage}" in
  verify|train|evaluate|aggregate|all) ;;
  *) echo "Stage must be verify, train, evaluate, aggregate, or all." >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH=src

output_root="runs/short_margin_3d"
protocol="${output_root}/protocol.json"
config="configs/conformer_ensemble_short_margin_fixed12.json"

"${python_bin}" -B -m ghxtox.short_margin_3d verify --protocol "${protocol}"
if [[ "${stage}" == "verify" ]]; then
  exit 0
fi

if [[ "${stage}" == "train" || "${stage}" == "all" ]]; then
  for fold in 0 1 2 3 4; do
    fold_output="${output_root}/fold${fold}"
    checkpoint="${fold_output}/best_model.pt"
    if [[ "${resume}" == "--resume" && -f "${checkpoint}" ]]; then
      echo "Reusing ${checkpoint}"
      continue
    fi
    "${python_bin}" -u -B -m ghxtox.train \
      --train data/processed/train_chemical_sites_final_esm2_pepflow_pilot250.pt \
      --fold-manifest data/folds/pepflow_pilot250_folds.csv \
      --fold "${fold}" --config "${config}" \
      --output-dir "${fold_output}" --device "${device}"
  done
fi

if [[ "${stage}" == "evaluate" || "${stage}" == "all" ]]; then
  checkpoints=()
  for fold in 0 1 2 3 4; do
    checkpoint="${output_root}/fold${fold}/best_model.pt"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing Phase 29 fold checkpoint: ${checkpoint}" >&2
      exit 3
    fi
    checkpoints+=("${checkpoint}")
  done
  mkdir -p "${output_root}/candidate"
  "${python_bin}" -u -B -m ghxtox.oof \
    --checkpoints "${checkpoints[@]}" \
    --processed data/processed/train_chemical_sites_final_esm2_pepflow.pt \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --output-csv "${output_root}/candidate/oof_predictions.csv" \
    --output-json "${output_root}/candidate/oof_summary.json" \
    --fixed-threshold 0.677819 --device "${device}"
fi

if [[ "${stage}" == "aggregate" || "${stage}" == "all" ]]; then
  candidate="${output_root}/candidate/oof_predictions.csv"
  if [[ ! -f "${candidate}" ]]; then
    echo "Missing Phase 29 candidate OOF: ${candidate}" >&2
    exit 4
  fi
  "${python_bin}" -B -m ghxtox.short_margin_3d aggregate \
    --protocol "${protocol}" --candidate-oof "${candidate}" \
    --output-dir "${output_root}/aggregate"
fi
