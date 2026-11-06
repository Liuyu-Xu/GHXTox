"""Only two additional baseline seeds for fixed equal ensemble-size validation."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
O=Path(__file__).resolve().parent;R=O.parents[1];S=R/'analysis/saprot_and_robust_classifier_pilot_20261026';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=read(O/'protocol.json');z=np.load(S/'frozen_features.npz');ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];groups=z['groups'];X=np.concatenate([z['base'],np.zeros((6387,480),dtype=np.float32)],axis=1)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
class Model(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(3561,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
 def forward(self,x):return self.net(x).squeeze(-1)
tensor=lambda a:torch.tensor(a.tolist(),dtype=torch.float32)
frozen={str(p):sha(p) for p in [O/'protocol.json',S/'frozen_features.npz']};(O/'before_fitting_manifest.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');(O/'checkpoints').mkdir(exist_ok=True);out=[];logs=[];start=time.time()
for seed in P['new_seeds']:
 for f in range(5):
  tr=np.where(fold!=f)[0];te=np.where(fold==f)[0];assert not set(groups[tr])&set(groups[te]);sc=StandardScaler().fit(X[tr]);tx=tensor(sc.transform(X[tr]).astype(np.float32));vx=tensor(sc.transform(X[te]).astype(np.float32));yy=tensor(y[tr]);pw=float((y[tr]==0).sum()/(y[tr]==1).sum());torch.manual_seed(seed+100*f);m=Model();opt=torch.optim.AdamW(m.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay']);rng=np.random.default_rng(seed+100*f);bh=hashlib.sha256();steps=0;begin=time.time();m.train()
  for epoch in range(P['epochs']):
   order=rng.permutation(len(tr))
   for k in range(0,len(tr),P['batch_size']):
    b=order[k:k+P['batch_size']];bh.update(b.tobytes());bt=torch.tensor(b.tolist(),dtype=torch.long);target=yy[bt];loss=(F.binary_cross_entropy_with_logits(m(tx[bt]),target,reduction='none')*torch.where(target>.5,pw,1.)).mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();steps+=1
  m.eval()
  with torch.inference_mode():ll=m(vx).tolist()
  for i,l in zip(te,ll):out.append(dict(seed=seed,fold=f,sample_id=ids[i],label=int(y[i]),logit=l))
  ck=O/'checkpoints'/f'{seed}_fold{f}.pt';torch.save(dict(state_dict=m.state_dict(),seed=seed,fold=f,scaler_mean=sc.mean_.tolist(),scaler_scale=sc.scale_.tolist(),train_ids=[ids[i] for i in tr],test_ids=[ids[i] for i in te]),ck);log=dict(seed=seed,fold=f,MCC=float(matthews_corrcoef(y[te],np.array(ll)>=0)),seconds=time.time()-begin,steps=steps,batch_hash=bh.hexdigest(),checkpoint_sha256=sha(ck));logs.append(log);print(f'Baseline extra seed{seed} fold{f}: MCC{log["MCC"]:.4f}',flush=True)
assert len(out)==6387*2 and len(logs)==10 and all(sha(Path(p))==h for p,h in frozen.items())
for n,obj in [('oof_logits.json',out),('fit_logs.json',logs),('training_verification.json',dict(new_fits=10,seconds=time.time()-start,prior_models_refit=0,input_hashes_unchanged=True,original_labels_and_group_splits_unchanged=True))]: (O/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
