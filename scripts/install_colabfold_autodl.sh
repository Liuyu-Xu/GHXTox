#!/usr/bin/env bash
set -euo pipefail

# Install LocalColabFold in an isolated pixi environment on the AutoDL data
# disk.  This intentionally does not modify the GHXTox conda environment.

INSTALL_ROOT="${LOCALCOLABFOLD_ROOT:-/root/autodl-tmp/localcolabfold}"
PIXI_BIN="${PIXI_BIN:-${HOME}/.pixi/bin/pixi}"

if [[ ! -x "${PIXI_BIN}" ]]; then
  curl -fsSL https://pixi.sh/install.sh | bash
fi
export PATH="$(dirname "${PIXI_BIN}"):${PATH}"

if [[ ! -d "${INSTALL_ROOT}/.git" ]]; then
  git clone --depth 1 \
    https://github.com/YoshitakaMo/localcolabfold.git \
    "${INSTALL_ROOT}"
fi

cd "${INSTALL_ROOT}"
"${PIXI_BIN}" install
"${PIXI_BIN}" run setup

CF_BIN="${INSTALL_ROOT}/.pixi/envs/default/bin/colabfold_batch"
CF_PY="${INSTALL_ROOT}/.pixi/envs/default/bin/python"
test -x "${CF_BIN}"
"${CF_BIN}" --help >/dev/null
"${CF_PY}" - <<'PY'
import importlib.metadata
import jax

print("colabfold:", importlib.metadata.version("colabfold"))
print("jax:", jax.__version__)
print("devices:", jax.devices())
PY
echo "LocalColabFold ready: ${CF_BIN}"
