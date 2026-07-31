#!/usr/bin/env bash
set -euo pipefail

BIOEMU_VERSION="1.4.1"
HUGGINGFACE_HUB_VERSION="0.36.0"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ENV_DIR="${BIOEMU_ENV_DIR:-$PWD/.venv-bioemu-${BIOEMU_VERSION}}"

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip
"${ENV_DIR}/bin/python" -m pip install \
  "bioemu[cuda]==${BIOEMU_VERSION}" \
  "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"

mkdir -p "${ENV_DIR}/reproducibility"
"${ENV_DIR}/bin/python" --version > "${ENV_DIR}/reproducibility/python-version.txt"
"${ENV_DIR}/bin/python" -m pip freeze > "${ENV_DIR}/reproducibility/requirements-lock.txt"
"${ENV_DIR}/bin/python" - <<'PY' > "${ENV_DIR}/reproducibility/runtime-check.json"
import importlib.metadata
import json
import torch

print(json.dumps({
    "bioemu_version": importlib.metadata.version("bioemu"),
    "huggingface_hub_version": importlib.metadata.version("huggingface-hub"),
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
PY

"${ENV_DIR}/bin/python" -m bioemu.sample --help > "${ENV_DIR}/reproducibility/bioemu-sample-help.txt"
echo "BioEmu environment ready: ${ENV_DIR}"
echo "The first smoke run downloads BioEmu and bundled AlphaFold2/ColabFold weights."
