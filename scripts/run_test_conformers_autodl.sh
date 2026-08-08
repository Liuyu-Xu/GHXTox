#!/usr/bin/env bash
set -euo pipefail

# Generate Test1/Test2 conformer caches without changing the frozen predictor.
#
# Usage:
#   bash scripts/run_test_conformers_autodl.sh pepflow test1 smoke
#   bash scripts/run_test_conformers_autodl.sh pepflow test1 full
#   bash scripts/run_test_conformers_autodl.sh af2 test1 smoke
#   bash scripts/run_test_conformers_autodl.sh af2 test1 full
#
# Resume a partial AF2 job with, for example:
#   CF_CHUNK_START=3 bash scripts/run_test_conformers_autodl.sh af2 test1 full

GENERATOR="${1:-}"
DATASET="${2:-}"
SCOPE="${3:-smoke}"

if [[ "${GENERATOR}" != "pepflow" && "${GENERATOR}" != "af2" ]]; then
  echo "generator must be pepflow or af2" >&2
  exit 2
fi
if [[ "${DATASET}" != "test1" && "${DATASET}" != "test2" ]]; then
  echo "dataset must be test1 or test2" >&2
  exit 2
fi
if [[ "${SCOPE}" != "smoke" && "${SCOPE}" != "full" ]]; then
  echo "scope must be smoke or full" >&2
  exit 2
fi

GHX_ROOT="${GHX_ROOT:-/root/autodl-tmp/GHXTox}"
GHX_PY="${GHX_PY:-/root/miniconda3/bin/python}"
TASK_ROOT="${GHX_ROOT}/data/test_conformers/tasks"
OUTPUT_ROOT="${GHX_ROOT}/data/test_conformers/${GENERATOR}/${DATASET}"
RUN_ROOT="${GHX_ROOT}/runs/test_conformers/${GENERATOR}_${DATASET}_${SCOPE}"
mkdir -p "${OUTPUT_ROOT}" "${RUN_ROOT}"
LENGTH_RANGE=""

export PYTHONPATH="${GHX_ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

if [[ "${GENERATOR}" == "pepflow" ]]; then
  LENGTH_RANGE="3-15"
  PEPFLOW_ROOT="${PEPFLOW_ROOT:-/root/autodl-tmp/PepFlow}"
  PF_PY="${PF_PY:-/root/autodl-tmp/venvs/pepflow_pip/bin/python}"
  PEPFLOW_MODEL="${PEPFLOW_MODEL:-${PEPFLOW_ROOT}/params/full_model.pth}"
  if [[ "${SCOPE}" == "smoke" ]]; then
    FASTA="${TASK_ROOT}/${DATASET}_pepflow_smoke.fasta"
    TASK_MANIFEST="${TASK_ROOT}/${DATASET}_pepflow_smoke.csv"
    RAW_ROOT="${OUTPUT_ROOT}/raw_smoke"
    CACHE_ROOT="${OUTPUT_ROOT}/cache_smoke"
  else
    FASTA="${TASK_ROOT}/${DATASET}_pepflow_03_15.fasta"
    TASK_MANIFEST="${TASK_ROOT}/${DATASET}_pepflow_03_15.csv"
    RAW_ROOT="${OUTPUT_ROOT}/raw"
    CACHE_ROOT="${OUTPUT_ROOT}/cache"
  fi
  for path in "${GHX_PY}" "${PF_PY}" "${PEPFLOW_MODEL}" "${FASTA}" "${TASK_MANIFEST}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required path: ${path}" >&2
      exit 1
    fi
  done
  mkdir -p "${RAW_ROOT}" "${CACHE_ROOT}"
  "${PF_PY}" -u -B -m ghxtox.pepflow_batch \
    --fasta "${FASTA}" \
    --pepflow-root "${PEPFLOW_ROOT}" \
    --full-model "${PEPFLOW_MODEL}" \
    --output-root "${RAW_ROOT}" \
    --num-samples 8 \
    --chunk-size "${PEPFLOW_CHUNK_SIZE:-4}" \
    --min-length 3 \
    --max-length 15 \
    --python "${PF_PY}" \
    --no-energies \
    --on-error continue \
    --sample-id-mode header \
    2>&1 | tee "${RUN_ROOT}/generate.log"

  "${GHX_PY}" -u -B -m ghxtox.conformer_cache \
    --fasta "${FASTA}" \
    --pdb-root "${RAW_ROOT}" \
    --output-root "${CACHE_ROOT}" \
    --source pepflow \
    --confidence-mode imputed \
    --imputed-confidence 0.7 \
    --max-conformers 8 \
    --on-missing skip \
    --on-mismatch error \
    --sample-id-mode header \
    2>&1 | tee "${RUN_ROOT}/cache.log"
else
  # Match the retained training-time source routing: PepFlow covers 3--15 aa,
  # while AF2/ColabFold covers 16--30 aa.  The rejected 31--50 pilot is not
  # expanded onto the independent tests.
  LENGTH_RANGE="16-30"
  RUN_ROOT="${GHX_ROOT}/runs/test_conformers/af2_16_30_${DATASET}_${SCOPE}"
  mkdir -p "${RUN_ROOT}"
  LOCALCOLABFOLD_ROOT="${LOCALCOLABFOLD_ROOT:-/root/autodl-tmp/localcolabfold}"
  CF_BIN="${CF_BIN:-${LOCALCOLABFOLD_ROOT}/.pixi/envs/default/bin/colabfold_batch}"
  if [[ "${SCOPE}" == "smoke" ]]; then
    FASTA="${TASK_ROOT}/${DATASET}_af2_smoke.fasta"
    TASK_MANIFEST="${TASK_ROOT}/${DATASET}_af2_smoke.csv"
    CHUNK_ROOT="${TASK_ROOT}/chunks/${DATASET}_af2_smoke"
    WORK_ROOT="${OUTPUT_ROOT}/work_smoke_16_30"
    CACHE_ROOT="${OUTPUT_ROOT}/cache_smoke_16_30"
  else
    FASTA="${TASK_ROOT}/${DATASET}_af2_16_30.fasta"
    TASK_MANIFEST="${TASK_ROOT}/${DATASET}_af2_16_30.csv"
    CHUNK_ROOT="${TASK_ROOT}/chunks/${DATASET}_af2_16_30"
    WORK_ROOT="${OUTPUT_ROOT}/work_16_30"
    CACHE_ROOT="${OUTPUT_ROOT}/cache_16_30"
  fi
  for path in "${GHX_PY}" "${CF_BIN}" "${FASTA}" "${TASK_MANIFEST}" "${CHUNK_ROOT}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Missing required path: ${path}" >&2
      exit 1
    fi
  done
  mkdir -p "${WORK_ROOT}" "${CACHE_ROOT}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="false"
  export TF_FORCE_UNIFIED_MEMORY="1"
  export JAX_COMPILATION_CACHE_DIR="${GHX_ROOT}/.jax_cache/colabfold"
  export TMPDIR="${GHX_ROOT}/tmp/colabfold"
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

  mapfile -t CHUNKS < <(find "${CHUNK_ROOT}" -maxdepth 1 -type f -name 'chunk_*.fasta' | sort)
  TOTAL_CHUNKS="${#CHUNKS[@]}"
  START_CHUNK="${CF_CHUNK_START:-1}"
  END_CHUNK="${CF_CHUNK_END:-${TOTAL_CHUNKS}}"
  if (( TOTAL_CHUNKS == 0 )); then
    echo "No AF2 chunk FASTAs found in ${CHUNK_ROOT}" >&2
    exit 1
  fi
  echo "AF2 chunks: total=${TOTAL_CHUNKS}, requested=${START_CHUNK}-${END_CHUNK}"

  for index in "${!CHUNKS[@]}"; do
    number=$((index + 1))
    if (( number < START_CHUNK || number > END_CHUNK )); then
      continue
    fi
    chunk="${CHUNKS[$index]}"
    chunk_name="$(basename "${chunk}" .fasta)"
    chunk_work="${WORK_ROOT}/${chunk_name}"
    msa_root="${chunk_work}/msa"
    result_root="${chunk_work}/results"
    mkdir -p "${msa_root}" "${result_root}"
    if [[ ! -f "${chunk_work}/.msa_complete" ]]; then
      "${CF_BIN}" "${chunk}" "${msa_root}" \
        --msa-only \
        --msa-mode mmseqs2_uniref_env \
        2>&1 | tee "${RUN_ROOT}/${chunk_name}_msa.log"
      touch "${chunk_work}/.msa_complete"
    else
      echo "Skip completed MSA: ${chunk_name}"
    fi
    if [[ ! -f "${chunk_work}/.predict_complete" ]]; then
      "${CF_BIN}" "${msa_root}" "${result_root}" \
        --model-type alphafold2_ptm \
        --num-models 5 \
        --num-seeds "${CF_NUM_SEEDS:-4}" \
        --random-seed 2026 \
        --max-seq 16 \
        --max-extra-seq 32 \
        --num-recycle 3 \
        --num-relax 0 \
        --rank plddt \
        --stop-at-score 100 \
        --sort-queries-by length \
        2>&1 | tee "${RUN_ROOT}/${chunk_name}_predict.log"
      touch "${chunk_work}/.predict_complete"
    else
      echo "Skip completed prediction: ${chunk_name}"
    fi
  done

  completed="$(find "${WORK_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '.predict_complete' | wc -l)"
  if (( completed != TOTAL_CHUNKS )); then
    echo "Partial AF2 run saved: ${completed}/${TOTAL_CHUNKS} chunks complete." >&2
    echo "Resume remaining chunks, then rerun this command to build the cache." >&2
    exit 3
  fi
  RESULT_ARGS=()
  while IFS= read -r result_root; do
    RESULT_ARGS+=(--results-root "${result_root}")
  done < <(find "${WORK_ROOT}" -mindepth 2 -maxdepth 2 -type d -name results | sort)
  RAW_ROOT="${OUTPUT_ROOT}/$([[ "${SCOPE}" == "smoke" ]] && echo raw_smoke_16_30 || echo raw_16_30)"
  mkdir -p "${RAW_ROOT}"
  "${GHX_PY}" -u -B -m ghxtox.colabfold_ensemble_cache \
    --fasta "${FASTA}" \
    --task-manifest "${TASK_MANIFEST}" \
    "${RESULT_ARGS[@]}" \
    --raw-root "${RAW_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --max-conformers 8 \
    --min-conformers 8 \
    --selection-mode diversity \
    --min-mean-plddt "${CF_MIN_MEAN_PLDDT:-50}" \
    --on-missing error \
    2>&1 | tee "${RUN_ROOT}/cache.log"
fi

"${GHX_PY}" -u -B "${GHX_ROOT}/scripts/normalize_test_conformer_manifest.py" \
  --task-manifest "${TASK_MANIFEST}" \
  --conformer-manifest "${CACHE_ROOT}/conformer_manifest.csv" \
  --output "${CACHE_ROOT}/conformer_manifest.csv" \
  2>&1 | tee "${RUN_ROOT}/normalize.log"

"${GHX_PY}" -u -B "${GHX_ROOT}/scripts/audit_test_conformer_cache.py" \
  --task-manifest "${TASK_MANIFEST}" \
  --cache-root "${CACHE_ROOT}" \
  --expected-conformers 8 \
  --strict \
  2>&1 | tee "${RUN_ROOT}/audit.log"

cat > "${RUN_ROOT}/protocol.json" <<JSON
{
  "dataset": "${DATASET}",
  "generator": "${GENERATOR}",
  "scope": "${SCOPE}",
  "task_manifest": "${TASK_MANIFEST}",
  "cache_root": "${CACHE_ROOT}",
  "retained_conformers": 8,
  "length_range": "${LENGTH_RANGE}",
  "routing_policy": "source-matched to retained training conformer branches",
  "labels_used_by_generator": false,
  "id_policy": "dataset-scoped server ID mapped to frozen processed-record ID only after sequence audit"
}
JSON

echo "COMPLETE: ${GENERATOR} ${DATASET} ${SCOPE}"
