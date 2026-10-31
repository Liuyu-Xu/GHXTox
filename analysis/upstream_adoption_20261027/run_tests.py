"""Run the handoff's frozen public test protocol; no training or tuning.

Compatibility adapter loads the supplied trusted T5 .bin with weights_only=True
and strict assignment because local torch 2.3 predates transformers' bin loader.
Source/config/checkpoints remain unmodified. NumPy 1.26 is isolated via sys.path.
"""
from pathlib import Path
import os, sys, json, time, gc, hashlib

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HANDOFF = OUT / 'handoff/GHXTox_team_handoff_20261027'
sys.path.insert(0, str(ROOT / 'analysis/repository_sync_20261026/validation_numpy1_clean'))
sys.path.insert(0, str(HANDOFF / 'src'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.chdir(HANDOFF)

import numpy as np
import torch
torch.set_num_threads(2)
import ghxtox.lora_mamba_1d as lm
import ghxtox.multidimensional_default as public

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_supplied_t5(config, device):
    from transformers import T5Config, T5EncoderModel
    path = Path(config['plm'])
    tc = T5Config.from_pretrained(path, local_files_only=True)
    with torch.device('meta'):
        model = T5EncoderModel(tc)
    state = torch.load(path / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    if 'shared.weight' not in state:
        state['shared.weight'] = state['encoder.embed_tokens.weight']
    if 'encoder.embed_tokens.weight' not in state:
        state['encoder.embed_tokens.weight'] = state['shared.weight']
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    state = {k: v.to(device=device, dtype=dtype) for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.tie_weights()
    model.eval()
    print('Supplied T5 loaded strictly', flush=True)
    return model

lm._load_plm = load_supplied_t5
original_seed = public._predict_1d_seed
def logged_seed(*args, **kwargs):
    seed = kwargs.get('expected_seed', args[-1] if args else None)
    print(f'Sequence seed {seed}: start', flush=True)
    t = time.time()
    result = original_seed(*args, **kwargs)
    gc.collect()
    torch.cuda.empty_cache()
    print(f'Sequence seed {seed}: done {time.time()-t:.1f}s', flush=True)
    return result
public._predict_1d_seed = logged_seed

def main():
    from transformers import __version__ as tv
    cfg = json.loads(Path('configs/default_multidimensional_fusion.json').read_text(encoding='utf-8-sig'))
    public.validate_public_default_config(cfg)
    checkpoint = Path('runs/default_no_inner_transformer_20261023/best_model.pt')
    public._validate_structure_checkpoint_architecture(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert payload['config']['model']['num_sequence_layers'] == 0
    from ghxtox.models import GHXToxModel
    m = GHXToxModel(payload['config'])
    m.load_state_dict(payload['model_state'], strict=True)
    params = sum(p.numel() for p in m.parameters())
    del payload, m
    provenance = {'source_commit': '60854abd61c4f1fd8b65c2eb586a7d03d7588130',
                  'torch': torch.__version__, 'numpy': np.__version__, 'transformers': tv,
                  'gpu': torch.cuda.get_device_name(0), 'structure_parameters': params,
                  'loader_adapter': 'supplied .bin, torch.load weights_only=True, strict assignment, tied embeddings',
                  'no_training': True, 'no_test_tuning': True,
                  'structure_checkpoint_sha256': sha(checkpoint),
                  'config_sha256': sha('configs/default_multidimensional_fusion.json')}
    (OUT/'test_run_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance), flush=True)
    for name in ['test1', 'test2']:
        dest = OUT/'new_tests'
        dest.mkdir(exist_ok=True)
        print(f'{name}: full public inference start', flush=True)
        t = time.time()
        result = public.predict(
            f'data/processed/{name}_chemical_sites_final_esm2.pt',
            f'data/processed/{name}_lora_mamba_light.pt',
            public.DEFAULT_1D_CONFIG, list(public.DEFAULT_CHECKPOINTS),
            list(public.DEFAULT_OOF_JSONS), public.DEFAULT_MANIFEST,
            public.DEFAULT_STRUCTURE_CHECKPOINT,
            cfg['fusion']['weight_sequence'], cfg['fusion']['threshold'],
            'cuda', 16, str(dest/f'{name}_predictions.csv'),
            str(dest/f'{name}_metrics.json'))
        print(json.dumps({'dataset':name,'seconds':time.time()-t,'metrics':result['metrics']}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
