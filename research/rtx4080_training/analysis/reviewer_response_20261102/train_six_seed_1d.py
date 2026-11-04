"""Resumable frozen-ProtT5/BiMamba grouped OOF; no external test data."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import ghxtox.lora_mamba_1d as lm

ROOT = Path(__file__).resolve().parents[2]
FOLD = ROOT/'data/folds/train_cdhit080_fallback_folds.csv'
BASE = ROOT/'configs/no_lora_mamba_1d_autodl.json'
DEFAULT_MODEL = Path(os.environ.get('GHX_PROTT5_DIR', str(ROOT / 'data/models/prot_t5_xl_half_uniref50-enc')))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--smoke', action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not silently train on CPU')
    if args.smoke and args.output_dir.name != 'smoke':
        raise ValueError('Smoke output directory must be named smoke')
    cfg=json.loads(BASE.read_text(encoding='utf-8-sig'))
    cfg['plm']=str(args.model_dir.resolve())
    assert cfg['lora_rank']==0 and not cfg.get('grad_checkpoint')
    records=lm.load_train_records(FOLD)
    assert len(records)==6387
    train_idx,val_idx=lm._load_fold_indices(FOLD,args.fold,len(records))
    if args.smoke:
        train_idx=([i for i in train_idx if records[i].label == 0][:8] +
                   [i for i in train_idx if records[i].label == 1][:8])
        val_idx=val_idx[:8]
    fold_seed=args.seed*1000+args.fold
    random.seed(fold_seed); np.random.seed(fold_seed); torch.manual_seed(fold_seed); torch.cuda.manual_seed_all(fold_seed)
    device=torch.device('cuda')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    out=args.output_dir/f'seed{args.seed}_fold{args.fold}'
    out.mkdir(exist_ok=True)
    result_path=out/'result.json'
    if result_path.exists():
        raise FileExistsError(f'Complete result already exists: {result_path}')
    tokenizer=T5Tokenizer.from_pretrained(str(args.model_dir),do_lower_case=False,local_files_only=True)
    plm=lm._load_plm(cfg,device)
    model=lm.build_model(plm,cfg,device)
    initial_head = {k: v.detach().clone() for k,v in model.named_parameters() if v.requires_grad}
    train_records=[records[i] for i in train_idx]
    val_records=[records[i] for i in val_idx]
    n_pos=sum(x.label==1 for x in train_records)
    if n_pos==0 or n_pos==len(train_records):
        if not args.smoke: raise ValueError('Training fold must contain both labels')
    loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_records)-n_pos)/max(n_pos,1),device=device))
    opt=lm.make_training_optimizer(model,cfg)
    collator=lm.LoraMambaCollator(tokenizer,max_length=cfg['max_length'])
    loader=DataLoader(lm.LoraMambaDataset(train_records),batch_size=cfg['batch_size'],shuffle=True,collate_fn=collator)
    scaler=lm._make_scaler(True)
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    epochs=1 if args.smoke else cfg['epochs']
    for epoch in range(epochs):
        lm.train_epoch(model,loader,opt,loss,device,True,scaler)
        print(json.dumps({'seed':args.seed,'fold':args.fold,'epoch':epoch+1,'seconds':round(time.time()-started,1)}),flush=True)
    if not any(not torch.equal(v, dict(model.named_parameters())[k].detach()) for k,v in initial_head.items()):
        raise RuntimeError('No trainable parameters changed')
    probs=lm.predict_probs(model,lm.LoraMambaDataset(val_records),collator,device,cfg['batch_size'],True)
    if not np.isfinite(probs).all(): raise ValueError('Non-finite probabilities')
    state=lm._trainable_state_dict(model)
    if any('.lora_' in key for key in state): raise ValueError('Unexpected LoRA parameter')
    ckpt=out/'model.pt'
    torch.save({'model_state':state,'config':cfg,'seed':args.seed,'fold':args.fold,'fold_seed':fold_seed},ckpt)
    result={'seed':args.seed,'fold':args.fold,'fold_seed':fold_seed,'smoke':args.smoke,'epochs':epochs,
            'source_indices':val_idx,'sample_ids':[x.sample_id for x in val_records],
            'labels':[int(x.label) for x in val_records],'probabilities':probs.tolist(),
            'train_n':len(train_records),'seconds':time.time()-started,'peak_mib':torch.cuda.max_memory_allocated()/1024**2,
            'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'config':cfg,
            'fold_manifest_sha256':sha(FOLD),'config_sha256':sha(BASE),'checkpoint_sha256':sha(ckpt),
            'trainer_sha256':sha(Path(__file__)),'implementation_sha256':sha(Path(lm.__file__))}
    save_json(result_path,result)
    print(json.dumps({'completed':str(result_path),'seconds':result['seconds'],'peak_mib':result['peak_mib']}),flush=True)

if __name__=='__main__': main()
