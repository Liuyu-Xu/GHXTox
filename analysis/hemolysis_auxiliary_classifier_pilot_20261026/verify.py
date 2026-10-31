"""Reload every fitted head and independently reproduce saved heldout predictions."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import torch
from torch import nn
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;A=R/'analysis/hemolysis_auxiliary_feasibility_20261026'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
z=np.load(O/'frozen_mean_features.npz');X=z['main_x'];mi={str(s):i for i,s in enumerate(z['main_ids'])};aseq={str(s):str(t) for s,t in zip(z['aux_ids'],z['aux_sequences'])}
component=read(A/'joint_sequence_components.json');protected=read(A/'protected_sequence_manifest.json');protected_components={component[s] for v in protected.values() for s in v}
pred=read(O/'oof_predictions.json');pm={(r['seed'],r['fold'],r['arm'],r['sample_id']):r['probability'] for r in pred};assert len(pm)==6387*6
main=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');mseq={r['sample_id']:r['sequence'] for r in main};logs=read(O/'fit_logs.json');assert len(logs)==30
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.trunk=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.tox=nn.Linear(64,1);self.aux=nn.Linear(64,1)
 def forward(self,x):return self.tox(self.trunk(x)).squeeze(-1)
torch.set_num_threads(2);checks=[]
for row in logs:
 seed,fold,arm=row['seed'],row['fold'],row['arm'];path=O/'checkpoints'/f'{seed}_fold{fold}_{arm}.pt'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['checkpoint_sha256']
 c=torch.load(path,map_location='cpu',weights_only=False);model=Model();model.load_state_dict(c['state_dict']);model.eval()
 tr={component[mseq[s]] for s in c['train_main_ids']};te={component[mseq[s]] for s in c['test_ids']};auxc={component[aseq[s]] for s in c['train_aux_ids']}
 assert not tr&te and not auxc&te and not auxc&protected_components
 xx=(X[[mi[s] for s in c['test_ids']]].astype(np.float64)-np.array(c['scaler_mean']))/np.array(c['scaler_scale'])
 # StandardScaler operates on float32 and rounds between subtraction/division.
 xx2=X[[mi[s] for s in c['test_ids']]].copy();xx2-=np.array(c['scaler_mean']);xx2/=np.array(c['scaler_scale'])
 with torch.inference_mode():pp=torch.sigmoid(model(torch.tensor(xx2.tolist(),dtype=torch.float32))).tolist()
 diff=max(abs(float(p)-pm[(seed,fold,arm,s)]) for s,p in zip(c['test_ids'],pp));assert diff<1e-7
 checks.append(dict(seed=seed,fold=fold,arm=arm,max_prediction_difference=diff,main_and_aux_components_disjoint_from_eval=True,aux_disjoint_from_reserved=True))
for seed in [20261026,20261027]:
 for fold in range(5):
  a=[r for r in logs if r['seed']==seed and r['fold']==fold]
  for k in ['initial_weights_hash','batch_hash','optimizer_steps','parameters']:assert len({r[k] for r in a})==1
freeze=read(O/'before_fitting_manifest.json')['input_hashes']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in freeze.items())
feasibility=read(A/'verification.json')
assert all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in feasibility['inputs'].values())
(O/'independent_verification.json').write_text(json.dumps(dict(checks=checks,max_prediction_difference=max(x['max_prediction_difference'] for x in checks),all30_checkpoints_reproduced=True,all_source_and_frozen_inputs_unchanged=True,all_arm_initialization_and_batch_steps_matched=True,paper_disjointness_certified=False,no_new_fits_in_verification=True),indent=2),encoding='utf-8')
print('30 checkpoints independently reproduced; all sequence-component purges and source hashes pass.')
