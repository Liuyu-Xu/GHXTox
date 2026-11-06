#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
MODEL_DIR="${MODEL_DIR:-/root/autodl_cys_supcon_20261029}"
for file in config.json pytorch_model.bin spiece.model; do
  if [ ! -f "$MODEL_DIR/$file" ]; then echo "Missing $MODEL_DIR/$file" >&2; exit 1; fi
done
python validate_inputs.py
python -c "import torch; assert torch.cuda.is_available(), 'CUDA GPU is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
for arm in A; do
  for seed in 42 123 2025; do
    for fold in 0 1 2 3 4 full; do
      folder="runs/$arm/seed$seed/fold$fold"
      if [ -f "$folder/result.json" ]; then
        python - "$folder" "$arm" "$seed" "$fold" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'result.json').read_text(encoding='utf-8'))
assert r['success'] and not r['smoke'] and r['arm']==sys.argv[2] and r['seed']==int(sys.argv[3]) and str(r['fold'])==sys.argv[4]
assert (p/'checkpoint.pt').is_file()
assert sys.argv[4]=='full' or (p/'oof.json').is_file()
print('Reusing completed:', p)
PY
      else
        echo "Training arm=$arm seed=$seed fold=$fold"
        python train.py --model-dir "$MODEL_DIR" --arm "$arm" --seed "$seed" --fold "$fold"
      fi
    done
  done
done
for arm in A; do
  echo "Evaluating arm=$arm"
  python evaluate.py --model-dir "$MODEL_DIR" --arm "$arm"
done
echo 'A training and evaluation completed. B/C/D were not run.'
