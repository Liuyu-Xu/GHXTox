#!/usr/bin/env bash
set -euo pipefail

device="${GHXTOX_DEVICE:-cuda}"
python_bin="${GHXTOX_PYTHON:-python}"
resume="${1:-}"
if [[ -n "${resume}" && "${resume}" != "--resume" ]]; then
  echo "Usage: $0 [--resume]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH=src

args=(
  -u -B -m ghxtox.ensemble_domain_ssl_3d
  --protocol runs/ensemble_domain_ssl_3d/protocol.json
  --output-root runs/ensemble_domain_ssl_3d
  --device "${device}"
)
if [[ "${resume}" == "--resume" ]]; then
  args+=(--resume)
fi

"${python_bin}" "${args[@]}"
