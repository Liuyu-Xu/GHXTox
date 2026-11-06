"""Reproduce every saved prediction; retain logits for stable mean-probability ranking."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from torch import nn
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
z=np.load(O/'frozen_features.npz');ids=z['sample_ids'].tolist();index={s:i for i,s in enumerate(ids)};groups=z['groups'];folds=z['folds'];base=z['base'];zero=np.zeros((6387,480),dtype=np.float32)
raw={'baseline_BCE':np.concatenate([base,zero],axis=1),'plus_SaProt_masked_BCE':np.concatenate([base,z['saprot_masked']],axis=1),'plus_SaProt_3Di_BCE':np.concatenate([base,z['saprot_struct']],axis=1)};raw['baseline_GCE_q07']=raw['baseline_BCE']
pred=read(O/'oof_predictions.json');pp={(r['seed'],r['fold'],r['arm'],r['sample_id']):r['probability'] for r in pred};logs=read(O/'fit_logs.json');assert len(logs)==40
class Model(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
torch.set_num_threads(2);checks=[];logits=[]
for row in logs:
 seed,fold,arm=row['seed'],row['fold'],row['arm'];path=O/'checkpoints'/f'{seed}_fold{fold}_{arm}.pt';assert sha(path)==row['checkpoint_sha256'];c=torch.load(path,map_location='cpu',weights_only=False)
 tr=[index[s] for s in c['train_ids']];te=[index[s] for s in c['test_ids']];assert not set(groups[tr])&set(groups[te]);assert all(folds[i]==fold for i in te)
 x=raw[arm][te].copy();x-=np.array(c['scaler_mean']);x/=np.array(c['scaler_scale']);model=Model();model.load_state_dict(c['state_dict']);model.eval()
 with torch.inference_mode():lo=model(torch.tensor(x.tolist(),dtype=torch.float32));p=torch.sigmoid(lo).tolist();ll=lo.tolist()
 difference=max(abs(v-pp[(seed,fold,arm,s)]) for s,v in zip(c['test_ids'],p));assert difference<1e-7
 checks.append(dict(seed=seed,fold=fold,arm=arm,max_prediction_difference=difference,group_disjoint=True))
 for s,l in zip(c['test_ids'],ll):logits.append(dict(seed=seed,fold=fold,arm=arm,sample_id=s,logit=l))
for seed in [20261026,20261027]:
 for f in range(5):
  a=[x for x in logs if x['seed']==seed and x['fold']==f]
  for key in ['steps','batch_hash','initial_weights_hash','parameters']:assert len({x[key] for x in a})==1
frozen=read(O/'before_fitting_manifest.json')['input_hashes'];assert all(sha(Path(p))==h for p,h in frozen.items())
for r in pred:assert int(z['labels'][index[r['sample_id']]])==r['label']
(O/'oof_logits.json').write_text(json.dumps(logits),encoding='utf-8')
v=dict(n_checkpoints=40,n_logits=len(logits),checks=checks,max_prediction_difference=max(x['max_prediction_difference'] for x in checks),all_group_splits_verified=True,all_frozen_inputs_unchanged=True,all_arms_batches_steps_initialization_capacity_matched=True,no_new_fits=True,original_probability_file_unchanged=True,logits_reason='avoid float32 probability saturation ties; evaluate mathematically same mean-of-two-seed probabilities via stable logodds, allarms identically')
(O/'independent_verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print('40checkpoints reproduced and51096logits saved; no retraining.')
