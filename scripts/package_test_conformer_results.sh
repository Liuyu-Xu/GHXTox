#!/usr/bin/env bash
set -euo pipefail

GHX_ROOT="${GHX_ROOT:-/root/autodl-tmp/GHXTox}"
GHX_PY="${GHX_PY:-/root/miniconda3/bin/python}"
export PYTHONPATH="${GHX_ROOT}/src"

for dataset in test1 test2; do
  "${GHX_PY}" -B "${GHX_ROOT}/scripts/audit_test_conformer_cache.py" \
    --task-manifest "${GHX_ROOT}/data/test_conformers/tasks/${dataset}_pepflow_03_15.csv" \
    --cache-root "${GHX_ROOT}/data/test_conformers/pepflow/${dataset}/cache" \
    --expected-conformers 8 --strict
  "${GHX_PY}" -B "${GHX_ROOT}/scripts/audit_test_conformer_cache.py" \
    --task-manifest "${GHX_ROOT}/data/test_conformers/tasks/${dataset}_af2_16_30.csv" \
    --cache-root "${GHX_ROOT}/data/test_conformers/af2/${dataset}/cache_16_30" \
    --expected-conformers 8 --strict
done

STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="/root/autodl-tmp/ghxtox_test1_test2_multiconformer_${STAMP}.tar.gz"
ITEMS=(
  data/test_conformers/README_AUTODL.md
  data/test_conformers/tasks/task_summary.json
  data/test_conformers/tasks/test1_pepflow_03_15.csv
  data/test_conformers/tasks/test1_pepflow_03_15.fasta
  data/test_conformers/tasks/test2_pepflow_03_15.csv
  data/test_conformers/tasks/test2_pepflow_03_15.fasta
  data/test_conformers/tasks/test1_af2_16_30.csv
  data/test_conformers/tasks/test1_af2_16_30.fasta
  data/test_conformers/tasks/test2_af2_16_30.csv
  data/test_conformers/tasks/test2_af2_16_30.fasta
  data/test_conformers/tasks/chunks/test1_af2_16_30
  data/test_conformers/tasks/chunks/test2_af2_16_30
  data/test_conformers/pepflow/test1/cache
  data/test_conformers/pepflow/test2/cache
  data/test_conformers/af2/test1/cache_16_30
  data/test_conformers/af2/test2/cache_16_30
  runs/test_conformers/pepflow_test1_full
  runs/test_conformers/pepflow_test2_full
  runs/test_conformers/af2_16_30_test1_full
  runs/test_conformers/af2_16_30_test2_full
)
if [[ "${INCLUDE_RAW:-0}" == "1" ]]; then
  ITEMS+=(
    data/test_conformers/pepflow/test1/raw
    data/test_conformers/pepflow/test2/raw
    data/test_conformers/af2/test1/raw_16_30
    data/test_conformers/af2/test2/raw_16_30
  )
fi
tar -C "${GHX_ROOT}" -czf "${ARCHIVE}" "${ITEMS[@]}"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
ls -lh "${ARCHIVE}" "${ARCHIVE}.sha256"
echo "RESULT_ARCHIVE=${ARCHIVE}"
