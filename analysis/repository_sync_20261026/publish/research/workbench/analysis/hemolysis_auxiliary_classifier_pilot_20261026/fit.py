"""Matched three-arm toxicity classifier pilot; no encoder/GHXTox updates."""
from pathlib import Path
import json,hashlib,time,copy,math
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score,confusion_matrix
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;A=R/'analysis/hemolysis_auxiliary_feasibility_20261026'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=read(O/'protocol.json');main=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');aux=read(A/'auxiliary_source_records.json');folds=read(A/'fold_manifests.json')
z=np.load(O/'frozen_mean_features.npz');ids=z['main_ids'].tolist();ai=z['aux_ids'].tolist();mi={s:i for i,s in enumerate(ids)};aid={s:i for i,s in enumerate(ai)};amain={r['aux_id']:r for r in aux}
assert ids==[r['sample_id'] for r in main]
y=np.array([int(r['label']) for r in main]);ay=np.array([amain[s]['label'] for s in ai]);X=z['main_x'];AX=z['aux_x'];assert X.shape==(6387,1024)
freeze=[O/'protocol.json',O/'frozen_mean_features.npz',A/'fold_manifests.json',A/'joint_sequence_components.json']
hashes={str(p):sha(p) for p in freeze};(O/'before_fitting_manifest.json').write_text(json.dumps(dict(stage='before_any_fit',input_hashes=hashes),indent=2),encoding='utf-8')
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.trunk=nn.Sequential(nn.Linear(1024,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU());self.tox=nn.Linear(64,1);self.aux=nn.Linear(64,1)
 def forward(self,x,task='tox'):
  h=self.trunk(x);return (self.tox if task=='tox' else self.aux)(h).squeeze(-1)
def tensor(a):return torch.tensor(a.tolist(),dtype=torch.float32)
def metrics(yy,pp):
 pred=pp>=.5;tn,fp,fn,tp=confusion_matrix(yy,pred,labels=[0,1]).ravel()
 return dict(n=len(yy),MCC=float(matthews_corrcoef(yy,pred)),AUPRC=float(average_precision_score(yy,pp)),AUROC=float(roc_auc_score(yy,pp)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))
def weight_hash(model):
 h=hashlib.sha256()
 for v in model.state_dict().values():h.update(np.array(v.tolist(),dtype=np.float32).tobytes())
 return h.hexdigest()
predictions=[];logs=[];start=time.time();(O/'checkpoints').mkdir(exist_ok=True)
for seed in P['seeds']:
 for f in folds:
  fi=f['fold'];tr=np.array([mi[s] for s in f['train_main_ids']]);te=np.array([mi[s] for s in f['test_main_ids']]);atr=np.array([aid[s] for s in f['aux_train_ids']])
  scale=StandardScaler().fit(X[tr]);tx=tensor(scale.transform(X[tr]).astype(np.float32));vx=tensor(scale.transform(X[te]).astype(np.float32));ax=tensor(scale.transform(AX[atr]).astype(np.float32));ty=tensor(y[tr]);trueay=ay[atr]
  torch.manual_seed(seed+fi*100);initial=Model();state=copy.deepcopy(initial.state_dict());init_hash=weight_hash(initial);del initial
  perm=np.random.default_rng(seed+10000+fi).permutation(trueay)
  for arm in P['arms']:
   model=Model();model.load_state_dict(state);assert weight_hash(model)==init_hash
   opt=torch.optim.AdamW(model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
   labels=perm if arm=='permuted_hemolysis_aux' else trueay;aly=tensor(labels);pw=torch.tensor(float((y[tr]==0).sum()/(y[tr]==1).sum()));apw=torch.tensor(float((labels==0).sum()/(labels==1).sum()))
   mainloss=nn.BCEWithLogitsLoss(pos_weight=pw);auxloss=nn.BCEWithLogitsLoss(pos_weight=apw);lam=0. if arm=='toxicity_only' else P['aux_loss_weight']
   rng=np.random.default_rng(seed+fi*100);batchhash=hashlib.sha256();steps=0;begin=time.time();model.train()
   for epoch in range(P['epochs']):
    order=rng.permutation(len(tr))
    for k in range(0,len(tr),P['batch_size']):
     b=order[k:k+P['batch_size']];ab=rng.integers(0,len(atr),P['aux_batch_size']);batchhash.update(b.tobytes());batchhash.update(ab.tobytes())
     bt=torch.tensor(b.tolist(),dtype=torch.long);abt=torch.tensor(ab.tolist(),dtype=torch.long)
     opt.zero_grad(set_to_none=True);loss=mainloss(model(tx[bt]),ty[bt])+lam*auxloss(model(ax[abt],'aux'),aly[abt]);assert torch.isfinite(loss);loss.backward();opt.step();steps+=1
   model.eval()
   with torch.inference_mode():pp=np.array(torch.sigmoid(model(vx)).tolist())
   assert np.isfinite(pp).all() and len(pp)==len(te)
   for i,p in zip(te,pp):predictions.append(dict(seed=seed,arm=arm,sample_id=ids[i],label=int(y[i]),fold=fi,probability=float(p)))
   ck=O/'checkpoints'/f'{seed}_fold{fi}_{arm}.pt'
   torch.save(dict(state_dict=model.state_dict(),scaler_mean=scale.mean_.tolist(),scaler_scale=scale.scale_.tolist(),seed=seed,fold=fi,arm=arm,train_main_ids=f['train_main_ids'],train_aux_ids=f['aux_train_ids'],test_ids=f['test_main_ids']),ck)
   log=dict(seed=seed,fold=fi,arm=arm,metrics=metrics(y[te],pp),optimizer_steps=steps,batch_hash=batchhash.hexdigest(),initial_weights_hash=init_hash,checkpoint_sha256=sha(ck),elapsed_seconds=time.time()-begin,parameters=sum(p.numel() for p in model.parameters()),device='cpu',threads=2);logs.append(log)
   print(f'{seed} fold{fi} {arm}: MCC={log["metrics"]["MCC"]:.4f}, {log["elapsed_seconds"]:.1f}s',flush=True)
   (O/'oof_predictions_partial.json').write_text(json.dumps(predictions,ensure_ascii=False),encoding='utf-8')
   (O/'fit_logs_partial.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2),encoding='utf-8')
for seed in P['seeds']:
 for f in range(5):
  g=[x for x in logs if x['seed']==seed and x['fold']==f]
  assert len({x['batch_hash'] for x in g})==1 and len({x['optimizer_steps'] for x in g})==1 and len({x['initial_weights_hash'] for x in g})==1
assert len(predictions)==6387*6 and all(sha(p)==hashes[str(p)] for p in freeze)
for n,obj in [('oof_predictions.json',predictions),('fit_logs.json',logs),('training_verification.json',dict(fits=len(logs),total_seconds=time.time()-start,all_inputs_unchanged=True,all_arms_initialization_batches_and_steps_matched=True,threshold=.5,head_training_only=True,encoder_training=False,GHXTox_weights_modified=False))]: (O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
print('All30 fixed classifier fits complete.',flush=True)
