import csv,json,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.metrics import matthews_corrcoef,log_loss
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'src'))
meta=json.loads((R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json').read_text(encoding='utf8'))
train=torch.load(P/'pilot_features.pt',map_location='cpu',weights_only=False)['train'];val=torch.load(P/'penalty_validation_features.pt',map_location='cpu',weights_only=False);dim=train['features'].shape[1]//2
class Head(nn.Module):
 def __init__(self,kind):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1));self.cys=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1)) if kind=='C' else None
 def forward(self,x):return self.net(x[:,:dim]).squeeze(-1)+(self.cys(x[:,dim:]).squeeze(-1) if self.cys is not None else 0)
out=[];cases=[]
for split,d in [('train',train),('validation',val)]:
 y=d['labels'].numpy();base=d['logits'].numpy();c=d['cys'].numpy();bpred=base>=0
 for kind in ('B','C'):
  for penalty in (0.01,0.001,0.0):
   head=Head(kind);head.load_state_dict(torch.load(P/f'penalty_{kind}_{penalty}.pt',map_location='cpu',weights_only=True));head.eval()
   with torch.inference_mode():delta=head(d['features']).numpy()
   prob=torch.sigmoid(torch.tensor(base+delta)).numpy();pred=prob>=.5;fix=(bpred!=y)&(pred==y);harm=(bpred==y)&(pred!=y);flag=np.array([bool(meta[i]['uniform_flag']) for i in d['indices']]);uncertain=np.abs(base)<1
   for i in np.flatnonzero(fix|harm):
    m=meta[d['indices'][i]];side=m.get('current_family_sidecar') or {};cases.append({'split':split,'method':kind,'penalty':penalty,'sample_id':m['sample_id'],'group_id':m['group_id'],'label':int(y[i]),'cys_count':int(c[i]),'base_logit':float(base[i]),'delta_logit':float(delta[i]),'change':'corrected' if fix[i] else 'harmed','flag':int(flag[i]),'family':' | '.join(side.get('parent_families') or []),'name':m.get('name','')})
   out.append({'split':split,'method':kind,'penalty':penalty,'n':len(y),'base_error':int((bpred!=y).sum()),'new_error':int((pred!=y).sum()),'base_bce':float(log_loss(y,torch.sigmoid(torch.tensor(base)).numpy(),labels=[0,1])),'new_bce':float(log_loss(y,prob,labels=[0,1])),'corrected':int(fix.sum()),'harmed':int(harm.sum()),'harmed_flag':int((harm&flag).sum()),'corrected_flag':int((fix&flag).sum()),'harmed_near_logit_1':int((harm&uncertain).sum()),'harmed_low_pos':int((harm&(c<=1)&(y==1)).sum()),'harmed_high_neg':int((harm&(c>=6)&(y==0)).sum()),'mean_abs_delta':float(np.abs(delta).mean()),'mean_delta_on_base_error':float(np.mean(np.abs(delta[bpred!=y]))) if (bpred!=y).any() else None})
with (P/'penalty_train_validation_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=cases[0].keys());w.writeheader();w.writerows(cases)
(P/'penalty_train_validation_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
