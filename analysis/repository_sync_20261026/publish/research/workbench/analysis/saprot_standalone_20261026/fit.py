"""Complete missing SaProt-only controls; reuse prior baseline and encodings."""
from pathlib import Path
import json,hashlib,time,copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
O=Path(__file__).resolve().parent;R=O.parents[1];S=R/'analysis/saprot_and_robust_classifier_pilot_20261026';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=read(O/'protocol.json');z=np.load(S/'frozen_features.npz');ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];groups=z['groups'];zeros=np.zeros((6387,3081),dtype=np.float32)
raw={'SaProt_only_masked':np.concatenate([zeros,z['saprot_masked']],axis=1),'SaProt_only_3Di':np.concatenate([zeros,z['saprot_struct']],axis=1)}
prior={(r['seed'],r['fold']):r for r in read(S/'fit_logs.json') if r['arm']=='baseline_BCE'}
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
class Model(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def tensor(x):return torch.tensor(x.tolist(),dtype=torch.float32)
def whash(model):
 h=hashlib.sha256()
 for a in model.state_dict().values():h.update(np.array(a.tolist(),dtype=np.float32).tobytes())
 return h.hexdigest()
frozen=[O/'protocol.json',S/'frozen_features.npz',S/'oof_logits.json',S/'fit_logs.json'];hashes={str(p):sha(p) for p in frozen};(O/'before_fitting_manifest.json').write_text(json.dumps(dict(stage='before_any_new_fit',input_hashes=hashes),indent=2),encoding='utf-8');(O/'checkpoints').mkdir(exist_ok=True)
logs=[];pred=[];start=time.time()
for seed in P['seeds']:
 for f in range(5):
  tr=np.where(fold!=f)[0];te=np.where(fold==f)[0];assert not set(groups[tr])&set(groups[te]);torch.manual_seed(seed+100*f);m=Model();initial=copy.deepcopy(m.state_dict());ih=whash(m);del m;assert ih==prior[(seed,f)]['initial_weights_hash']
  for arm in P['new_arms']:
   m=Model();m.load_state_dict(initial);sc=StandardScaler().fit(raw[arm][tr]);tx=tensor(sc.transform(raw[arm][tr]).astype(np.float32));vx=tensor(sc.transform(raw[arm][te]).astype(np.float32));ty=tensor(y[tr]);pw=float((y[tr]==0).sum()/(y[tr]==1).sum());opt=torch.optim.AdamW(m.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay']);rng=np.random.default_rng(seed+100*f);bh=hashlib.sha256();steps=0;begin=time.time();m.train()
   for epoch in range(P['epochs']):
    order=rng.permutation(len(tr))
    for k in range(0,len(tr),P['batch_size']):
     b=order[k:k+P['batch_size']];bh.update(b.tobytes());bt=torch.tensor(b.tolist(),dtype=torch.long);yy=ty[bt];loss=(F.binary_cross_entropy_with_logits(m(tx[bt]),yy,reduction='none')*torch.where(yy>.5,pw,1.)).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();steps+=1
   assert steps==prior[(seed,f)]['steps'] and bh.hexdigest()==prior[(seed,f)]['batch_hash'];m.eval()
   with torch.inference_mode():ll=m(vx).tolist()
   pp=expit(np.array(ll,dtype=np.float64))
   for i,l,p in zip(te,ll,pp):pred.append(dict(seed=seed,fold=f,arm=arm,sample_id=ids[i],label=int(y[i]),logit=float(l),probability=float(p)))
   ck=O/'checkpoints'/f'{seed}_fold{f}_{arm}.pt';torch.save(dict(state_dict=m.state_dict(),seed=seed,fold=f,arm=arm,scaler_mean=sc.mean_.tolist(),scaler_scale=sc.scale_.tolist(),train_ids=[ids[i] for i in tr],test_ids=[ids[i] for i in te]),ck)
   log=dict(seed=seed,fold=f,arm=arm,MCC=float(matthews_corrcoef(y[te],pp>=.5)),steps=steps,initial_weights_hash=ih,batch_hash=bh.hexdigest(),parameters=sum(p.numel() for p in m.parameters()),seconds=time.time()-begin,checkpoint_sha256=sha(ck),matched_prior_baseline=True);logs.append(log);print(f'{seed} fold{f} {arm}: MCC={log["MCC"]:.4f}, {log["seconds"]:.1f}s',flush=True)
   (O/'oof_partial.json').write_text(json.dumps(pred),encoding='utf-8');(O/'fit_logs_partial.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
assert len(pred)==6387*4 and len(logs)==20 and all(sha(Path(p))==h for p,h in hashes.items())
for n,obj in [('oof_predictions.json',pred),('fit_logs.json',logs),('training_verification.json',dict(new_fits=20,prior_models_refit=0,prior_encoders_refit=0,seconds=time.time()-start,all_initialization_batches_steps_match_prior_baseline=True,all_original_group_splits_disjoint=True,all_frozen_source_inputs_unchanged=True,noProtT5_or_physchem_input_to_new_heads=True))]: (O/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
print('All20 SaProt-only controls completed.',flush=True)
