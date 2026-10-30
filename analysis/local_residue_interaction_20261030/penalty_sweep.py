import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
root=Path(r'D:\dachuang2026\GHXTox');out=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.constants import AA_TO_IDX
from ghxtox.utils import move_batch_to_device
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); folder=out/'runs/topology_strict_oof_v1/base/fold0';prov=json.loads((folder/'provenance.json').read_text());d=torch.load(out/'pilot_features.pt',map_location='cpu',weights_only=False);val=prov['diagnostic_validation_indices'];assert not set(val)&set(d['train']['indices']) and not set(val)&set(d['test']['indices'])
vcache=out/'penalty_validation_features.pt'
if not vcache.exists():
 rec=torch.load(r'D:\dachuang2026\GHXTox-ziliao\train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',map_location='cpu',weights_only=False)['records'];ck=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=False);model=GHXToxModel(ck['config']).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval();vs=[];ys=[];ls=[];cs=[]
 with torch.inference_mode():
  for b in DataLoader([rec[i] for i in val],batch_size=16,shuffle=False,collate_fn=lambda x:collate_peptides(x,include_structure=True,include_atom=False)):
   b=move_batch_to_device(b,device);o=model(b);h=o['spatial_nodes'];mask=b['mask'].bool();pair=mask[:,1:]&mask[:,:-1];adj=((h[:,1:]*h[:,:-1])*pair.unsqueeze(-1)).sum(1)/pair.sum(1).clamp_min(1).unsqueeze(-1);cm=(b['aa_ids']==AA_TO_IDX['C'])&mask;ch=(h*cm.unsqueeze(-1)).sum(1)/cm.sum(1).clamp_min(1).unsqueeze(-1);vs.append(torch.cat([adj,ch],-1).cpu());ys.append(b['labels'].cpu());ls.append(o['logits'].cpu());cs.append(cm.sum(1).cpu())
 torch.save({'features':torch.cat(vs),'labels':torch.cat(ys),'logits':torch.cat(ls),'cys':torch.cat(cs),'indices':val},vcache);del model,rec
v=torch.load(vcache,map_location='cpu',weights_only=False);dim=v['features'].shape[1]//2
class Head(nn.Module):
 def __init__(self,kind):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1));self.cys=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1)) if kind=='C' else None
  for x in [self.net,self.cys]:
   if x is not None:nn.init.zeros_(x[-1].weight);nn.init.zeros_(x[-1].bias)
 def forward(self,x):return self.net(x[:,:dim]).squeeze(-1)+(self.cys(x[:,dim:]).squeeze(-1) if self.cys is not None else 0)
y=v['labels'].numpy();c=v['cys'].numpy();baseline=(v['logits'].numpy()>=0);x=d['train']['features'].to(device);yt=d['train']['labels'].float().to(device);l=d['train']['logits'].to(device);results=[];preds={}
def evaluate(kind,penalty,delta):
 p=torch.sigmoid(v['logits']+torch.tensor(delta)).numpy();q=p>=.5
 return {'method':kind,'penalty':penalty,'mcc':float(matthews_corrcoef(y,q)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'errors':int((q!=y).sum()),'low_positive_fn':int(((c<=1)&(y==1)&~q).sum()),'low_positive_n':int(((c<=1)&(y==1)).sum()),'high_negative_fp':int(((c>=6)&(y==0)&q).sum()),'high_negative_n':int(((c>=6)&(y==0)).sum()),'corrected':int(((baseline!=y)&(q==y)).sum()),'harmed':int(((baseline==y)&(q!=y)).sum()),'mean_abs_delta_logit':float(np.mean(np.abs(delta))),'p90_abs_delta_logit':float(np.quantile(np.abs(delta),.9)),'mean_abs_delta_probability':float(np.mean(abs(p-torch.sigmoid(v['logits']).numpy())))}
results.append(evaluate('A',None,np.zeros(len(y),dtype='float32')))
for kind in ('B','C'):
 for pen in (.01,.001,0.):
  torch.manual_seed(42);head=Head(kind).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.01);g=torch.Generator().manual_seed(42)
  for epoch in range(5):
   for ix in torch.randperm(len(yt),generator=g).split(64):
    ix=ix.to(device);opt.zero_grad();delta=head(x[ix]);loss=nn.functional.binary_cross_entropy_with_logits(l[ix]+delta,yt[ix])+pen*delta.square().mean();loss.backward();opt.step()
  head.eval()
  with torch.inference_mode():delta=head(v['features'].to(device)).cpu().numpy()
  tag=f'{kind}_{pen}';preds[tag]=delta;results.append(evaluate(kind,pen,delta));torch.save(head.state_dict(),out/f'penalty_{tag}.pt');print(tag,results[-1],flush=True)
(out/'penalty_sweep_results.json').write_text(json.dumps(results,indent=2),encoding='utf8');np.savez_compressed(out/'penalty_sweep_deltas.npz',**preds,labels=y,cys=c,indices=np.asarray(val))
