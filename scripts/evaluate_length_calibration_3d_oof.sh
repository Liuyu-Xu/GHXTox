#!/usr/bin/env bash
set -euo pipefail

python_bin="${GHXTOX_PYTHON:-python}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH=src

"${python_bin}" -u -B -m ghxtox.length_conditioned_calibration_3d \
  --protocol runs/length_calibration_3d/protocol.json \
  --output-root runs/length_calibration_3d
