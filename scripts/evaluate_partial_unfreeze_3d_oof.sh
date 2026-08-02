#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON_BIN="${GHXTOX_PYTHON:-python}"
DEVICE="${GHXTOX_DEVICE:-auto}"

exec "$PYTHON_BIN" -m ghxtox.partial_unfreeze_3d \
  --protocol runs/partial_unfreeze_3d/protocol.json \
  --device "$DEVICE" \
  --resume "$@"
