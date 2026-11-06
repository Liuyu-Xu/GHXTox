import json,sys
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.metrics import matthews_corrcoef,average_precision_score,log_loss
P=Path(__file__).resolve().parent;torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d=torch.load(P/'pilot_features.pt',map_location='cpu',weights_only=False)['train'];u=torch.load(P/'base_unseen_training_features.pt',map_location='cpu',weights_only=False);v=torch.load(P/'penalty_validation_features.pt',map_location='cpu',weights_only=False);assert not set(u['indices'])&set(v['indices']) and not set(u['indices'])&set(d['indices']);n=len(u['labels']);choice=np.random.default_rng(42).choice(len(d['labels']),n,replace=False);seen={k:(t[choice] if torch.is_tensor(t) else t) for k,t in d.items()};dim=u['features'].shape[1]//2
class Head(nn.Module):
 def __init__(self,kind):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1));self.cys=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1)) if kind=='C' else None
  for z in (self.net,self.cys):
   if z is not None:nn.init.zeros_(z[-1].weight);nn.init.zeros_(z[-1].bias)
 def forward(self,x):return self.net(x[:,:dim]).squeeze(-1)+(self.cys(x[:,dim:]).squeeze(-1) if self.cys is not None else 0)
y=v['labels'].numpy();c=v['cys'].numpy();base=v['logits'].numpy();a=base>=0;results=[]
for mode,data in [('seen_matched',seen),('base_unseen',u)]:
 for kind in ('B','C'):
  for pen in (.01,.001,0.):
   torch.manual_seed(42);head=Head(kind).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01);x=data['features'].to(device);yy=data['labels'].float().to(device);l=data['logits'].to(device);g=torch.Generator().manual_seed(42)
   for epoch in range(5):
    for ix in torch.randperm(n,generator=g).split(64):
     ix=ix.to(device);opt.zero_grad();delta=head(x[ix]);loss=nn.functional.binary_cross_entropy_with_logits(l[ix]+delta,yy[ix])+pen*delta.square().mean();loss.backward();opt.step()
   head.eval()
   with torch.inference_mode():delta=head(v['features'].to(device)).cpu().numpy()
   p=torch.sigmoid(torch.tensor(base+delta)).numpy();q=p>=.5;results.append({'train_features':mode,'train_n':n,'base_errors_in_training':int(((data['logits'].numpy()>=0)!=data['labels'].numpy()).sum()),'method':kind,'penalty':pen,'mcc':float(matthews_corrcoef(y,q)),'auprc':float(average_precision_score(y,p)),'bce':float(log_loss(y,p,labels=[0,1])),'errors':int((q!=y).sum()),'corrected':int(((a!=y)&(q==y)).sum()),'harmed':int(((a==y)&(q!=y)).sum()),'low_pos_fn':int(((c<=1)&(y==1)&~q).sum()),'high_neg_fp':int(((c>=6)&(y==0)&q).sum()),'mean_abs_delta_logit':float(np.abs(delta).mean())});print(results[-1],flush=True)
(P/'unseen_feature_comparison.json').write_text(json.dumps(results,indent=2),encoding='utf8')
