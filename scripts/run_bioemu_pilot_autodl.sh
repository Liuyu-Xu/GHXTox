#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
BIOEMU_VERSION="1.4.1"
BIOEMU_PYTHON="${BIOEMU_PYTHON:-${PROJECT_ROOT}/.venv-bioemu-${BIOEMU_VERSION}/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/bioemu_pilot_200}"
MODE="${1:-smoke}"
LIMIT=6
ALLOW_ARGS=()

if [[ "${MODE}" == "pilot" ]]; then
  if [[ "${GHXTOX_ALLOW_BIOEMU_PILOT:-0}" != "1" ]]; then
    echo "Pilot mode requires GHXTOX_ALLOW_BIOEMU_PILOT=1 after smoke decision passes." >&2
    exit 2
  fi
  MANIFEST="${PROJECT_ROOT}/data/bioemu_pilot/train_pilot_200.csv"
  LIMIT=200
  ALLOW_ARGS+=(--allow-pilot)
elif [[ "${MODE}" != "smoke" ]]; then
  echo "Usage: $0 [smoke|pilot]. There is intentionally no full-dataset mode." >&2
  exit 2
else
  MANIFEST="${PROJECT_ROOT}/data/bioemu_pilot/smoke_6.csv"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

"${BIOEMU_PYTHON}" -m ghxtox.bioemu_pilot run \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --limit "${LIMIT}" \
  --python "${BIOEMU_PYTHON}" \
  --seed 20260731 \
  --num-samples 32 \
  --cache-embeds-dir "${OUTPUT_ROOT}/shared_cache/embeds" \
  --cache-so3-dir "${OUTPUT_ROOT}/shared_cache/so3" \
  "${ALLOW_ARGS[@]}"

"${BIOEMU_PYTHON}" -m ghxtox.bioemu_pilot assess \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --limit "${LIMIT}" \
  --output-json "${OUTPUT_ROOT}/${MODE}_feasibility.json" \
  --output-csv "${OUTPUT_ROOT}/${MODE}_sample_metrics.csv"

sha256sum \
  "${MANIFEST}" \
  "${OUTPUT_ROOT}/protocol.json" \
  "${OUTPUT_ROOT}/${MODE}_feasibility.json" \
  "${OUTPUT_ROOT}/${MODE}_sample_metrics.csv" \
  > "${OUTPUT_ROOT}/${MODE}_checksums.sha256"

echo "Completed ${MODE}; inspect ${OUTPUT_ROOT}/${MODE}_feasibility.json."
