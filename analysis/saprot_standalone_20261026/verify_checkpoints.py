"""Reload the 30 new checkpoints; never refit models."""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
S=R/'analysis/saprot_and_robust_classifier_pilot_20261026'
z=np.load(S/'frozen_features.npz'); ids=z['sample_ids'].tolist(); index={s:i for i,s in enumerate(ids)}
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
results=[]
for name in ['saprot_standalone_20261026','saprot_ensemble_size_control_20261026']:
 O=R/'analysis'/name; standalone='standalone' in name
 records=read(O/('oof_predictions.json' if standalone else 'oof_logits.json'))
 lookup={(r['seed'],r['fold'],r.get('arm','baseline'),r['sample_id']):r for r in records}
 logs=read(O/'fit_logs.json'); errors=[]
 for log in logs:
  seed,f=log['seed'],log['fold']; arm=log.get('arm','baseline')
  p=O/'checkpoints'/(f'{seed}_fold{f}_{arm}.pt' if standalone else f'{seed}_fold{f}.pt')
  assert sha(p)==log['checkpoint_sha256']; c=torch.load(p,map_location='cpu')
  tr=[index[s] for s in c['train_ids']]; te=[index[s] for s in c['test_ids']]
  assert set(tr).isdisjoint(te) and len(tr)+len(te)==len(ids)
  assert not set(z['groups'][tr]) & set(z['groups'][te])
  assert np.all(z['folds'][te]==f) and np.all(z['folds'][tr]!=f)
  if standalone:
   features=z['saprot_masked' if arm.endswith('masked') else 'saprot_struct']
   raw=np.concatenate([np.zeros((len(te),3081),dtype=np.float32),features[te]],axis=1)
  else: raw=np.concatenate([z['base'][te],np.zeros((len(te),480),dtype=np.float32)],axis=1)
  raw-=np.array(c['scaler_mean']); raw/=np.array(c['scaler_scale'])
  model=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
  model.load_state_dict({k.removeprefix('net.'):v for k,v in c['state_dict'].items()}); model.eval()
  with torch.inference_mode(): ll=model(torch.tensor(raw.tolist(),dtype=torch.float32)).squeeze(-1).tolist()
  expected=[]
  for i in te:
   r=lookup[(seed,f,arm,ids[i])]; assert r['label']==int(z['labels'][i]); expected.append(r['logit'])
  errors.append(float(np.max(np.abs(np.array(ll)-expected))))
 manifest=read(O/'before_fitting_manifest.json'); manifest=manifest.get('input_hashes',manifest)
 assert all(sha(Path(p))==h for p,h in manifest.items())
 result=dict(directory=name,checkpoints=len(logs),records=len(records),max_absolute_logit_difference=max(errors),all_checkpoint_hashes_match=True,all_input_hashes_match=True,labels_and_folds_match=True,group_disjoint=True)
 assert max(errors)==0
 (O/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); results.append(result)
print(json.dumps(results,indent=2))
