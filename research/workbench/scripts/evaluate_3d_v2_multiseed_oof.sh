#!/usr/bin/env bash
set -euo pipefail

seed="${1:-42}"
stage="${2:-all}"
resume="${3:-}"
device="${GHXTOX_DEVICE:-cuda}"
python_bin="${GHXTOX_PYTHON:-python}"

case "${seed}" in
  42|123|2025) ;;
  *) echo "Seed must be one of 42, 123, 2025." >&2; exit 2 ;;
esac
case "${stage}" in
  prepare|base|residual|all|aggregate) ;;
  *) echo "Stage must be prepare, base, residual, all, or aggregate." >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH=src

output_root="runs/3d_v2_multiseed"
protocol_path="${output_root}/protocol.json"

prepare_protocol() {
  "${python_bin}" -B -m ghxtox.three_d_multiseed prepare --output-root "${output_root}"
}

if [[ "${stage}" == "prepare" || ! -f "${protocol_path}" ]]; then
  prepare_protocol
fi
if [[ "${stage}" == "prepare" ]]; then
  exit 0
fi

if [[ "${seed}" != "42" && "${stage}" != "aggregate" ]]; then
  gate_path="${output_root}/seed_42/reference_gate.json"
  if [[ ! -f "${gate_path}" ]]; then
    echo "Seed 42 reference gate is missing; seeds 123/2025 are not authorized." >&2
    exit 3
  fi
  "${python_bin}" -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p["new_seeds_authorized"] else 1)' "${gate_path}" || {
    echo "Seed 42 reproduction did not pass; seeds 123/2025 are not authorized." >&2
    exit 3
  }
fi

seed_root="${output_root}/seed_${seed}"
base_config="${output_root}/configs/base_seed${seed}.json"
residual_config="${output_root}/configs/residual_seed${seed}.json"

if [[ "${stage}" == "base" || "${stage}" == "all" ]]; then
  for fold in 0 1 2 3 4; do
    fold_output="${seed_root}/base/fold${fold}"
    checkpoint="${fold_output}/best_model.pt"
    if [[ "${resume}" == "--resume" && -f "${checkpoint}" ]]; then
      echo "Reusing ${checkpoint}"
      continue
    fi
    "${python_bin}" -u -B -m ghxtox.train \
      --train data/processed/train_chemical_sites_final_esm2.pt \
      --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
      --fold "${fold}" --config "${base_config}" \
      --output-dir "${fold_output}" --device "${device}"
  done
  base_checkpoints=()
  for fold in 0 1 2 3 4; do
    base_checkpoints+=("${seed_root}/base/fold${fold}/best_model.pt")
  done
  "${python_bin}" -u -B -m ghxtox.oof \
    --checkpoints "${base_checkpoints[@]}" \
    --processed data/processed/train_chemical_sites_final_esm2.pt \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --output-csv "${seed_root}/base/oof_predictions.csv" \
    --output-json "${seed_root}/base/oof_summary.json" \
    --metric mcc --device "${device}"
fi

if [[ "${stage}" == "residual" || "${stage}" == "all" ]]; then
  for fold in 0 1 2 3 4; do
    base_checkpoint="${seed_root}/base/fold${fold}/best_model.pt"
    if [[ ! -f "${base_checkpoint}" ]]; then
      echo "Missing matching seed-${seed} base checkpoint: ${base_checkpoint}" >&2
      exit 4
    fi
    fold_output="${seed_root}/residual/fold${fold}"
    checkpoint="${fold_output}/best_model.pt"
    if [[ "${resume}" == "--resume" && -f "${checkpoint}" ]]; then
      echo "Reusing ${checkpoint}"
      continue
    fi
    "${python_bin}" -u -B -m ghxtox.train \
      --train data/processed/train_chemical_sites_final_esm2_pepflow_pilot250.pt \
      --fold-manifest data/folds/pepflow_pilot250_folds.csv \
      --fold "${fold}" --config "${residual_config}" \
      --output-dir "${fold_output}" --device "${device}"
  done
  residual_checkpoints=()
  for fold in 0 1 2 3 4; do
    residual_checkpoints+=("${seed_root}/residual/fold${fold}/best_model.pt")
  done
  candidate_oof="${seed_root}/residual/oof_predictions.csv"
  "${python_bin}" -u -B -m ghxtox.oof \
    --checkpoints "${residual_checkpoints[@]}" \
    --processed data/processed/train_chemical_sites_final_esm2_pepflow.pt \
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
    --output-csv "${candidate_oof}" \
    --output-json "${seed_root}/residual/oof_summary.json" \
    --metric mcc --device "${device}"
  if [[ "${seed}" == "42" ]]; then
    "${python_bin}" -B -m ghxtox.three_d_multiseed compare-reference \
      --reference-oof runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv \
      --candidate-oof "${candidate_oof}" \
      --output-json "${seed_root}/reference_gate.json"
  fi
fi

if [[ "${stage}" == "aggregate" ]]; then
  "${python_bin}" -B -m ghxtox.three_d_multiseed aggregate \
    --reference-oof runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv \
    --quality-csv runs/three_d_team_handoff/oof_3d_quality.csv \
    --seed-oof \
      42=runs/3d_v2_multiseed/seed_42/residual/oof_predictions.csv \
      123=runs/3d_v2_multiseed/seed_123/residual/oof_predictions.csv \
      2025=runs/3d_v2_multiseed/seed_2025/residual/oof_predictions.csv \
    --output-csv "${output_root}/ensemble/oof_predictions.csv" \
    --output-summary "${output_root}/ensemble/summary.json" \
    --output-decision "${output_root}/ensemble/decision.json"
fi
