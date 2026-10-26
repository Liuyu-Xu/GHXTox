#!/usr/bin/env bash
# Strict five-fold topology OOF runner for one AutoDL GPU. Run stages serially.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${GHXTOX_PYTHON:-python}"
DEVICE="${GHXTOX_DEVICE:-cuda}"
RUN_ROOT="${GHXTOX_TOPOLOGY_RUN_ROOT:-${PROJECT_ROOT}/runs/topology_strict_oof_v1}"
PROTOCOL="${PROJECT_ROOT}/configs/topology_strict_oof_v1.json"
BASE_CONFIG="${PROJECT_ROOT}/configs/topology_strict_base_v1.json"
TRAIN_CACHE="${GHXTOX_TRAIN_CACHE:?Set GHXTOX_TRAIN_CACHE to the transferred training .pt cache}"
OUTER_MANIFEST="${GHXTOX_OUTER_MANIFEST:?Set GHXTOX_OUTER_MANIFEST to the transferred outer-fold CSV}"
NESTED_MANIFEST="${GHXTOX_NESTED_MANIFEST:-${PROJECT_ROOT}/data/folds/topology_strict_nested_v1.csv}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

require_file() {
  [[ -s "$1" ]] || { echo "Required file is missing or empty: $1" >&2; exit 2; }
}

prepare() {
  require_file "${PROTOCOL}"
  require_file "${BASE_CONFIG}"
  require_file "${TRAIN_CACHE}"
  require_file "${OUTER_MANIFEST}"
  require_file "${NESTED_MANIFEST}"
  require_file "${NESTED_MANIFEST%.csv}.json"
  mkdir -p "${RUN_ROOT}/base"
  echo "Inputs present. Hash verification runs before each provenance/finalization step."
}

base_fold() {
  local fold="$1"
  local fold_dir="${RUN_ROOT}/base/fold${fold}"
  if [[ -f "${fold_dir}/best_model.pt" && -f "${fold_dir}/provenance.json" ]]; then
    echo "Fold ${fold} base and provenance already exist; leaving them untouched."
    return
  fi
  if [[ -e "${fold_dir}" ]]; then
    echo "Partial fold directory exists: ${fold_dir}. Inspect it; this script will not overwrite it." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m ghxtox.train \
    --train "${TRAIN_CACHE}" --config "${BASE_CONFIG}" \
    --nested-manifest "${NESTED_MANIFEST}" --outer-fold "${fold}" \
    --output-dir "${fold_dir}" --device "${DEVICE}"
  "${PYTHON_BIN}" scripts/prepare_strict_topology_oof.py base-provenance \
    --protocol "${PROTOCOL}" --processed "${TRAIN_CACHE}" \
    --outer-manifest "${OUTER_MANIFEST}" --nested-manifest "${NESTED_MANIFEST}" \
    --fold "${fold}" --base-dir "${fold_dir}" --frozen-base-config "${BASE_CONFIG}"
}

all_base() {
  for fold in 0 1 2 3 4; do base_fold "${fold}"; done
}

topology() {
  local execution_protocol="${RUN_ROOT}/topology_oof_protocol.json"
  local output_dir="${RUN_ROOT}/topology_oof"
  [[ ! -e "${execution_protocol}" ]] || { echo "Frozen execution protocol exists: ${execution_protocol}" >&2; exit 2; }
  [[ ! -e "${output_dir}" ]] || { echo "Topology output already exists: ${output_dir}" >&2; exit 2; }
  "${PYTHON_BIN}" scripts/prepare_strict_topology_oof.py finalize \
    --protocol "${PROTOCOL}" --processed "${TRAIN_CACHE}" \
    --outer-manifest "${OUTER_MANIFEST}" --nested-manifest "${NESTED_MANIFEST}" \
    --base-root "${RUN_ROOT}/base" --output "${execution_protocol}"
  "${PYTHON_BIN}" -m ghxtox.topology_oof --protocol "${execution_protocol}" --output-dir "${output_dir}" --device "${DEVICE}"
  "${PYTHON_BIN}" scripts/evaluate_topology_aware_oof.py \
    --oof "${output_dir}/oof_predictions.csv" --protocol "${execution_protocol}" \
    --output "${output_dir}/evaluation.json"
  sha256sum "${PROTOCOL}" "${execution_protocol}" "${output_dir}/run_audit.json" "${output_dir}/evaluation.json" \
    > "${RUN_ROOT}/final_checksums.sha256"
}

case "${1:-all}" in
  prepare) prepare ;;
  base) prepare; [[ $# -eq 2 ]] || { echo "Usage: $0 base FOLD" >&2; exit 2; }; base_fold "$2" ;;
  all-base) prepare; all_base ;;
  topology) prepare; topology ;;
  all) prepare; all_base; topology ;;
  *) echo "Usage: $0 {prepare|base FOLD|all-base|topology|all}" >&2; exit 2 ;;
esac
