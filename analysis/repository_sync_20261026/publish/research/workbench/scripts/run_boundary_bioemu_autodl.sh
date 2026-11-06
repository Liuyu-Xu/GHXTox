#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BIOEMU_VERSION="1.4.1"
BIOEMU_PYTHON="${BIOEMU_PYTHON:-${PROJECT_ROOT}/.venv-bioemu-${BIOEMU_VERSION}/bin/python}"
MANIFEST="${PROJECT_ROOT}/data/bioemu_boundary_600/generate_missing.csv"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/bioemu_boundary_600_server}"
MODEL_ARGS=()

if [[ "${GHXTOX_ALLOW_BOUNDARY_BIOEMU:-0}" != "1" ]]; then
  echo "Phase 33 generation requires GHXTOX_ALLOW_BOUNDARY_BIOEMU=1." >&2
  exit 2
fi
if [[ ! -s "${MANIFEST}" ]]; then
  echo "Missing Phase 33 generation manifest: ${MANIFEST}" >&2
  exit 2
fi

BIOEMU_CKPT_PATH="${BIOEMU_CKPT_PATH:-}"
BIOEMU_MODEL_CONFIG_PATH="${BIOEMU_MODEL_CONFIG_PATH:-}"
if [[ -z "${BIOEMU_CKPT_PATH}" && -z "${BIOEMU_MODEL_CONFIG_PATH}" ]]; then
  for HF_CACHE_ROOT in \
    "${HF_HOME:-${HOME}/.cache/huggingface}" \
    "/root/autodl-tmp/cache/huggingface"; do
    SNAPSHOT_ROOT="${HF_CACHE_ROOT}/hub/models--microsoft--bioemu/snapshots"
    [[ -d "${SNAPSHOT_ROOT}" ]] || continue
    BIOEMU_CKPT_PATH="$(
      find -L "${SNAPSHOT_ROOT}" \
        -path "*/checkpoints/bioemu-v1.2/checkpoint.ckpt" \
        -type f -print -quit 2>/dev/null || true
    )"
    BIOEMU_MODEL_CONFIG_PATH="$(
      find -L "${SNAPSHOT_ROOT}" \
        -path "*/checkpoints/bioemu-v1.2/config.yaml" \
        -type f -print -quit 2>/dev/null || true
    )"
    [[ -n "${BIOEMU_CKPT_PATH}" ]] && break
  done
fi
if [[ -n "${BIOEMU_CKPT_PATH}" || -n "${BIOEMU_MODEL_CONFIG_PATH}" ]]; then
  if [[ ! -s "${BIOEMU_CKPT_PATH}" || ! -s "${BIOEMU_MODEL_CONFIG_PATH}" ]]; then
    echo "Local BioEmu checkpoint and config must both exist and be non-empty." >&2
    exit 2
  fi
  MODEL_ARGS+=(
    --ckpt-path "${BIOEMU_CKPT_PATH}"
    --model-config-path "${BIOEMU_MODEL_CONFIG_PATH}"
  )
  echo "Using local BioEmu checkpoint: ${BIOEMU_CKPT_PATH}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

"${BIOEMU_PYTHON}" -m ghxtox.boundary_bioemu run \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --authorized \
  --continue-on-error \
  --python "${BIOEMU_PYTHON}" \
  "${MODEL_ARGS[@]}"

"${BIOEMU_PYTHON}" -m ghxtox.boundary_bioemu assess \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --output-json "${OUTPUT_ROOT}/generation_feasibility.json" \
  --output-csv "${OUTPUT_ROOT}/generation_sample_metrics.csv"

sha256sum \
  "${MANIFEST}" \
  "${OUTPUT_ROOT}/protocol.json" \
  "${OUTPUT_ROOT}/generation_feasibility.json" \
  "${OUTPUT_ROOT}/generation_sample_metrics.csv" \
  > "${OUTPUT_ROOT}/generation_checksums.sha256"

echo "Phase 33 generation finished; inspect ${OUTPUT_ROOT}/generation_feasibility.json."
