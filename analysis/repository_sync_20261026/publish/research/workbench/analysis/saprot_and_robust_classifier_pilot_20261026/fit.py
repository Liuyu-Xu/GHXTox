"""Four fixed heads, separating structure representation from robust-loss effects."""
from pathlib import Path
import json,hashlib,time,copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef,average_precision_score
O=Path(__file__).resolve().parent;P=json.loads((O/'protocol.json').read_text(encoding='utf-8-sig'));z=np.load(O/'frozen_features.npz')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];groups=z['groups'];base=z['base'];zero=np.zeros((len(y),480),dtype=np.float32)
raw={'baseline_BCE':np.concatenate([base,zero],axis=1),'plus_SaProt_masked_BCE':np.concatenate([base,z['saprot_masked']],axis=1),'plus_SaProt_3Di_BCE':np.concatenate([base,z['saprot_struct']],axis=1)};raw['baseline_GCE_q07']=raw['baseline_BCE']
assert len(y)==6387 and len(set(groups))==5258
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
class Model(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def tensor(x):return torch.tensor(x.tolist(),dtype=torch.float32)
def whash(model):
 h=hashlib.sha256()
 for x in model.state_dict().values():h.update(np.array(x.tolist(),dtype=np.float32).tobytes())
 return h.hexdigest()
inputs={str(p):sha(p) for p in [O/'protocol.json',O/'frozen_features.npz',O/'embedding_manifest.json']}
(O/'before_fitting_manifest.json').write_text(json.dumps(dict(stage='before_any_head_fit',input_hashes=inputs),indent=2),encoding='utf-8');(O/'checkpoints').mkdir(exist_ok=True)
pred=[];logs=[];start=time.time()
for seed in P['seeds']:
 for f in range(5):
  tr=np.where(fold!=f)[0];te=np.where(fold==f)[0];assert not set(groups[tr])&set(groups[te])
  torch.manual_seed(seed+100*f);m=Model();initial=copy.deepcopy(m.state_dict());ih=whash(m);del m
  for arm in P['arms']:
   model=Model();model.load_state_dict(initial);assert whash(model)==ih
   scaler=StandardScaler().fit(raw[arm][tr]);tx=tensor(scaler.transform(raw[arm][tr]).astype(np.float32));vx=tensor(scaler.transform(raw[arm][te]).astype(np.float32));ty=tensor(y[tr]);pw=float((y[tr]==0).sum()/(y[tr]==1).sum())
   opt=torch.optim.AdamW(model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay']);rng=np.random.default_rng(seed+f*100);bh=hashlib.sha256();steps=0;begin=time.time();model.train()
   for epoch in range(P['epochs']):
    order=rng.permutation(len(tr))
    for k in range(0,len(tr),P['batch_size']):
     b=order[k:k+P['batch_size']];bh.update(b.tobytes());bt=torch.tensor(b.tolist(),dtype=torch.long);yy=ty[bt];logit=model(tx[bt]);ce=F.binary_cross_entropy_with_logits(logit,yy,reduction='none');weights=torch.where(yy>.5,pw,1.)
     per=-torch.expm1(-.7*ce)/.7 if arm=='baseline_GCE_q07' else ce
     loss=(weights*per).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();steps+=1
   model.eval()
   with torch.inference_mode():pp=torch.sigmoid(model(vx)).tolist()
   for i,p in zip(te,pp):pred.append(dict(seed=seed,fold=f,arm=arm,sample_id=ids[i],label=int(y[i]),probability=float(p)))
   ck=O/'checkpoints'/f'{seed}_fold{f}_{arm}.pt';torch.save(dict(state_dict=model.state_dict(),seed=seed,fold=f,arm=arm,scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),train_ids=[ids[i] for i in tr],test_ids=[ids[i] for i in te]),ck)
   log=dict(seed=seed,fold=f,arm=arm,MCC=float(matthews_corrcoef(y[te],np.array(pp)>=.5)),AUPRC=float(average_precision_score(y[te],pp)),train_n=len(tr),test_n=len(te),steps=steps,initial_weights_hash=ih,batch_hash=bh.hexdigest(),parameters=sum(v.numel() for v in model.parameters()),seconds=time.time()-begin,checkpoint_sha256=sha(ck));logs.append(log)
   print(f'{seed} fold{f} {arm}: MCC={log["MCC"]:.4f}, {log["seconds"]:.1f}s',flush=True)
   (O/'oof_partial.json').write_text(json.dumps(pred),encoding='utf-8');(O/'fit_logs_partial.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
for seed in P['seeds']:
 for f in range(5):
  a=[x for x in logs if x['seed']==seed and x['fold']==f]
  assert len(a)==4
  for k in ['steps','initial_weights_hash','batch_hash','parameters']:assert len({x[k] for x in a})==1
assert len(pred)==6387*8 and all(sha(Path(p))==h for p,h in inputs.items())
for n,obj in [('oof_predictions.json',pred),('fit_logs.json',logs),('training_verification.json',dict(fits=40,n_predictions=len(pred),seconds=time.time()-start,all_initialization_steps_batches_capacity_matched=True,source_and_frozen_inputs_unchanged=True,labels_unchanged=True,pretrained_encoders_frozen=True,distillation_performed=False,GHXTox_weights_modified=False))]: (O/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
print('All40 prespecified heads complete.',flush=True)
