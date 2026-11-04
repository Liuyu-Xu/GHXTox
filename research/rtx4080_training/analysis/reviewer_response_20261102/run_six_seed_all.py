"""Sequential, resumable 6 x 5 experiment. Fail closed on changed inputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'analysis/reviewer_response_20261102/six_seed_1d'
TRAIN=ROOT/'analysis/reviewer_response_20261102/train_six_seed_1d.py'
COMPARE=ROOT/'analysis/reviewer_response_20261102/compare_six_seed_1d.py'
MODEL=Path(os.environ.get('GHX_PROTT5_DIR',str(ROOT/'data/models/prot_t5_xl_half_uniref50-enc'))).resolve()
SEEDS=(42,123,2025,314,2718,2026)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'runner.lock').open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RuntimeError('Another six-seed runner is active') from exc
    protocol={'seeds':list(SEEDS),'folds':list(range(5)),'epochs':15,'batch_size':16,'lora_rank':0,
        'initialization':'fold_seed = seed*1000+fold; random/NumPy/PyTorch/CUDA seeded',
        'trainer_sha256':sha(TRAIN),'implementation_sha256':sha(ROOT/'src/ghxtox/lora_mamba_1d.py'),
        'model_sha256':sha(MODEL/'pytorch_model.bin'),'fold_manifest_sha256':sha(ROOT/'data/folds/train_cdhit080_fallback_folds.csv'),
        'config_sha256':sha(ROOT/'configs/no_lora_mamba_1d_autodl.json'),
        'comparison_sha256':sha(COMPARE),'structure_oof_sha256':sha(ROOT/'runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv'),
        'comparison':'3 newly trained 1D + 1 historical 3D vs 6 newly trained 1D; OOF meta-fit is exploratory and not end-to-end nested',
        'test_sets_used':False}
    path=RUN/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Existing protocol does not match current inputs; refusing to mix runs')
    if not path.exists(): path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for seed in SEEDS:
        for fold in range(5):
            result=RUN/'runs'/f'seed{seed}_fold{fold}'/'result.json'
            if result.exists():
                d=json.loads(result.read_text(encoding='utf-8'))
                if (d['seed'],d['fold'],d['epochs'],d['smoke'],d['trainer_sha256'],d['implementation_sha256'],d['fold_manifest_sha256'],d['config_sha256']) != (seed,fold,15,False,protocol['trainer_sha256'],protocol['implementation_sha256'],protocol['fold_manifest_sha256'],protocol['config_sha256']):
                    raise ValueError(f'Incompatible existing result {result}')
                if sha(result.parent/'model.pt') != d['checkpoint_sha256']: raise ValueError(f'Bad checkpoint {result}')
                print(f'skip verified seed={seed} fold={fold}',flush=True)
                continue
            print(f'start seed={seed} fold={fold} time={time.strftime("%Y-%m-%d %H:%M:%S")}',flush=True)
            subprocess.run([sys.executable,'-u',str(TRAIN),'--seed',str(seed),'--fold',str(fold),'--model-dir',str(MODEL),'--output-dir',str(RUN/'runs')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-u',str(COMPARE)],cwd=ROOT,check=True)
    print('ALL_COMPLETED',flush=True)

if __name__=='__main__': main()
