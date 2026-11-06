import sys,json
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
P=Path(__file__).resolve().parent;R=P.parents[1];Q=P.parent/'local_residue_interaction_20261030';torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tr=torch.load(Q/'pilot_features.pt',map_location='cpu',weights_only=False)['train'];va=torch.load(Q/'penalty_validation_features.pt',map_location='cpu',weights_only=False);meta=json.loads((R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json').read_text(encoding='utf8'))
aa='ACDEFGHIKLMNPQRSTVWY'
def comp(indices):
 return np.array([[len(meta[i]['sequence']),meta[i]['cys_count'],*[meta[i]['sequence'].count(a)/len(meta[i]['sequence']) for a in aa]] for i in indices])
X=comp(tr['indices']);V=comp(va['indices']);y=tr['labels'].numpy();yv=va['labels'].numpy();groups=np.array([meta[i]['group_id'] for i in tr['indices']]);assert not set(groups)&{meta[i]['group_id'] for i in va['indices']};shortcut=np.zeros(len(y))
def make():return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
for fit,ev in GroupKFold(3).split(X,y,groups):
 m=make();m.fit(X[fit],y[fit]);shortcut[ev]=m.decision_function(X[ev])
m=make();m.fit(X,y);vshort=m.decision_function(V);bias=torch.tensor(shortcut,device=device,dtype=torch.float32)
# Cross-fitted shortcut logit is used only in fitting loss; inference uses the main head alone.
f=tr['features'];mean=f.mean(0);std=f.std(0).clamp_min(1e-4);f=((f-mean)/std).to(device);v=((va['features']-mean)/std).to(device);yt=tr['labels'].float().to(device);dim=f.shape[1];allpred={};result={}
weights=torch.where(yt>0,1-torch.sigmoid(bias),torch.sigmoid(bias)).clamp(.1,1.);weights/=weights.mean()
for mode in ['BCE','difficulty_weight','shortcut_loss']:
 pred=[]
 for seed in [42,123,2025]:
  torch.manual_seed(seed);h=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,1)).to(device);opt=torch.optim.AdamW(h.parameters(),lr=.0003,weight_decay=.01);g=torch.Generator().manual_seed(seed)
  for epoch in range(10):
   h.train()
   for ix in torch.randperm(len(y),generator=g).split(64):
    ix=ix.to(device);opt.zero_grad();l=h(f[ix]).squeeze(-1);l=l+bias[ix] if mode=='shortcut_loss' else l;loss=nn.functional.binary_cross_entropy_with_logits(l,yt[ix],reduction='none');loss=(loss*weights[ix]).mean() if mode=='difficulty_weight' else loss.mean();loss.backward();opt.step()
  h.eval()
  with torch.inference_mode():pred.append(torch.sigmoid(h(v).squeeze(-1)).cpu().numpy())
  torch.save(h.cpu().state_dict(),P/f'{mode}_seed{seed}.pt')
 p=np.mean(pred,axis=0);allpred[mode]=p;c=np.array([meta[i]['cys_count'] for i in va['indices']]);q=p>=.5;result[mode]={'mcc':float(matthews_corrcoef(yv,q)),'auroc':float(roc_auc_score(yv,p)),'auprc':float(average_precision_score(yv,p)),'errors':int((q!=yv).sum()),'low_pos_fn':int(((c<=1)&(yv==1)&~q).sum()),'low_neg_fp':int(((c<=1)&(yv==0)&q).sum()),'high_neg_fp':int(((c>=6)&(yv==0)&q).sum()),'high_pos_fn':int(((c>=6)&(yv==1)&~q).sum())};print(mode,result[mode],flush=True)
(P/'results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(P/'predictions.npz',**allpred,labels=yv,indices=np.array(va['indices']),shortcut_validation=vshort)
