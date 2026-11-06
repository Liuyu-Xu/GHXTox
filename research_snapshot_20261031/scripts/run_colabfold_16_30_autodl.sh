#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_colabfold_16_30_autodl.sh smoke
#   bash scripts/run_colabfold_16_30_autodl.sh pilot
#   bash scripts/run_colabfold_16_30_autodl.sh full

SCOPE="${1:-smoke}"
GHX_ROOT="${GHX_ROOT:-/root/autodl-tmp/GHXTox}"
LOCALCOLABFOLD_ROOT="${LOCALCOLABFOLD_ROOT:-/root/autodl-tmp/localcolabfold}"
GHX_PY="${GHX_PY:-/root/miniconda3/bin/python}"
CF_BIN="${CF_BIN:-${LOCALCOLABFOLD_ROOT}/.pixi/envs/default/bin/colabfold_batch}"
TASK_ROOT="${GHX_ROOT}/data/af2_ensemble_16_30"
RUN_ROOT="${GHX_ROOT}/runs/af2_ensemble_16_30_${SCOPE}"
MSA_ROOT="${TASK_ROOT}/msa_${SCOPE}"
RESULT_ROOT="${TASK_ROOT}/colabfold_${SCOPE}"
RAW_ROOT="${TASK_ROOT}/raw_${SCOPE}"
CACHE_ROOT="${TASK_ROOT}/cache_${SCOPE}"
NUM_SEEDS="${CF_NUM_SEEDS:-4}"
MIN_MEAN_PLDDT="${CF_MIN_MEAN_PLDDT:-50}"

case "${SCOPE}" in
  smoke)
    FASTA="${TASK_ROOT}/smoke_2.fasta"
    MANIFEST="${TASK_ROOT}/smoke_2.csv"
    ;;
  pilot)
    FASTA="${TASK_ROOT}/train_pilot_400.fasta"
    MANIFEST="${TASK_ROOT}/train_pilot_400.csv"
    ;;
  full)
    FASTA="${TASK_ROOT}/train_full_2034.fasta"
    MANIFEST="${TASK_ROOT}/train_full_2034.csv"
    ;;
  *)
    echo "scope must be smoke, pilot, or full" >&2
    exit 2
    ;;
esac

for path in "${FASTA}" "${MANIFEST}" "${GHX_PY}" "${CF_BIN}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}" "${MSA_ROOT}" "${RESULT_ROOT}" "${RAW_ROOT}" "${CACHE_ROOT}"
export PYTHONPATH="${GHX_ROOT}/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export TF_FORCE_UNIFIED_MEMORY="1"
export JAX_COMPILATION_CACHE_DIR="${GHX_ROOT}/.jax_cache/colabfold"
export TMPDIR="${GHX_ROOT}/tmp/colabfold"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${TMPDIR}"

"${CF_BIN}" --help > "${RUN_ROOT}/colabfold_batch_help.txt"
"${CF_BIN}" "${FASTA}" "${MSA_ROOT}" \
  --msa-only \
  --msa-mode mmseqs2_uniref_env \
  2>&1 | tee "${RUN_ROOT}/msa.log"

"${CF_BIN}" "${MSA_ROOT}" "${RESULT_ROOT}" \
  --model-type alphafold2_ptm \
  --num-models 5 \
  --num-seeds "${NUM_SEEDS}" \
  --random-seed 2026 \
  --max-seq 16 \
  --max-extra-seq 32 \
  --num-recycle 3 \
  --num-relax 0 \
  --rank plddt \
  --stop-at-score 100 \
  --sort-queries-by length \
  2>&1 | tee "${RUN_ROOT}/predict.log"

"${GHX_PY}" -u -B -m ghxtox.colabfold_ensemble_cache \
  --fasta "${FASTA}" \
  --task-manifest "${MANIFEST}" \
  --results-root "${RESULT_ROOT}" \
  --raw-root "${RAW_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --max-conformers 8 \
  --min-conformers 8 \
  --selection-mode diversity \
  --min-mean-plddt "${MIN_MEAN_PLDDT}" \
  --on-missing error \
  2>&1 | tee "${RUN_ROOT}/cache.log"

"${GHX_PY}" -B - "${CACHE_ROOT}" "${MANIFEST}" <<'PY'
import csv
import json
from pathlib import Path
import sys

cache_root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
expected = sum(1 for _ in csv.DictReader(manifest.open(encoding="utf-8-sig")))
summary = json.loads((cache_root / "summary.json").read_text(encoding="utf-8"))
assert summary["samples_with_conformers"] == expected, summary["status_counts"]
assert summary["total_conformers"] == expected * 8, summary["total_conformers"]
print(f"Audit OK: samples={expected}, conformers={expected * 8}")
PY

cat > "${RUN_ROOT}/protocol.json" <<JSON
{
  "generator": "ColabFold AlphaFold2-ptm",
  "scope": "${SCOPE}",
  "num_models": 5,
  "num_seeds": ${NUM_SEEDS},
  "raw_candidates_per_sample": $((5 * NUM_SEEDS)),
  "max_msa": "16:32",
  "num_recycles": 3,
  "dropout": false,
  "templates": false,
  "retained_conformers": 8,
  "selection": "backbone-RMSD k-medoids after mean-pLDDT quality floor",
  "min_mean_plddt": ${MIN_MEAN_PLDDT},
  "interpretation": "AF2 prediction-uncertainty ensemble; not a thermodynamic equilibrium ensemble"
}
JSON

STAMP="$(date +%Y%m%d)"
ARCHIVE="/root/autodl-tmp/af2_ensemble_16_30_${SCOPE}_${STAMP}.tar.gz"
tar -C "${GHX_ROOT}" -czf "${ARCHIVE}" \
  "data/af2_ensemble_16_30/$(basename "${FASTA}")" \
  "data/af2_ensemble_16_30/$(basename "${MANIFEST}")" \
  "data/af2_ensemble_16_30/raw_${SCOPE}" \
  "data/af2_ensemble_16_30/cache_${SCOPE}" \
  "runs/af2_ensemble_16_30_${SCOPE}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
ls -lh "${ARCHIVE}" "${ARCHIVE}.sha256"
