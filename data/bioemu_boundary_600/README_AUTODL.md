# Phase 33 AutoDL generation

This package generates only the 574 frozen Phase 33 decision-boundary rows
that are not already present in the verified local BioEmu-200 pilot. It does
not contain toxicity labels and does not authorize full 6,387-row generation.

## 1. Extract into the existing project

Upload the zip to `/root/autodl-tmp/`, then run:

```bash
cd /root/autodl-tmp/GHXTox
python -m zipfile -e \
  /root/autodl-tmp/GHXTox_phase33_boundary_bioemu_autodl.zip \
  /root/autodl-tmp/GHXTox
chmod +x scripts/run_boundary_bioemu_autodl.sh
```

The existing `.venv-bioemu-1.4.1` and downloaded BioEmu/AlphaFold weights from
the completed 200-row run are reused. Do not delete those caches.

## 2. Start the resumable generation

```bash
cd /root/autodl-tmp/GHXTox
export HF_HOME=/root/autodl-tmp/cache/huggingface
export GHXTOX_ALLOW_BOUNDARY_BIOEMU=1
mkdir -p runs/bioemu_boundary_600_server

nohup bash scripts/run_boundary_bioemu_autodl.sh \
  > runs/bioemu_boundary_600_server/console.log 2>&1 &

echo $! | tee runs/bioemu_boundary_600_server/pid
tail -f runs/bioemu_boundary_600_server/console.log
```

Press `Ctrl+C` only to leave `tail`; the background job continues. The launcher
skips sample directories that already contain a successful `run_status.json`,
`sequence.fasta`, `topology.pdb`, and `samples.xtc`, so the same command is the
correct recovery command after a disconnect or instance restart.

Expected RTX 4090 generation time is approximately 8.1 GPU hours, extrapolated
from the verified 200-row pilot. MSA/embedding stages can leave GPU utilization
temporarily low.

## 3. Check completion

```bash
cd /root/autodl-tmp/GHXTox
cat runs/bioemu_boundary_600_server/generation_feasibility.json
```

Continue only when `decision` is
`ready_to_import_phase33_boundary_structures` and
`boundary_specialist_training_authorized` is `true`.

## 4. Package the results for download

```bash
cd /root/autodl-tmp/GHXTox

tar \
  --exclude='runs/bioemu_boundary_600_server/shared_cache' \
  -czf /root/autodl-tmp/bioemu_boundary_600_results_20260801.tar.gz \
  runs/bioemu_boundary_600_server \
  data/bioemu_boundary_600

sha256sum /root/autodl-tmp/bioemu_boundary_600_results_20260801.tar.gz \
  > /root/autodl-tmp/bioemu_boundary_600_results_20260801.tar.gz.sha256
```

Download both the `.tar.gz` and `.sha256` files. The local import stage will
merge these 574 outputs with the 26 verified reusable BioEmu pilot rows before
any Phase 33 toxicity model is trained.
