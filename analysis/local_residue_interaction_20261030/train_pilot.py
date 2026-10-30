import sys,json,hashlib,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
root=Path(r'D:\dachuang2026\GHXTox');out=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.constants import AA_TO_IDX
from ghxtox.utils import move_batch_to_device
torch.set_num_threads(2); torch.manual_seed(42);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir=out/'runs/topology_strict_oof_v1/base/fold0';prov=json.loads((dir/'provenance.json').read_text());ckpath=dir/'best_model.pt';data=Path(r'D:\dachuang2026\GHXTox-ziliao\train_chemical_sites_final_esm2_multisource_1_30_full2795.pt')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
assert sha(ckpath)==prov['checkpoint_sha256'] and sha(data)==prov['training_data_sha256']
train=prov['fitting_indices'];test=prov['outer_test_indices'];assert not set(train)&set(test)
cache=out/'pilot_features.pt'
if not cache.exists():
 records=torch.load(data,map_location='cpu',weights_only=False)['records'];ck=torch.load(ckpath,map_location='cpu',weights_only=False);model=GHXToxModel(ck['config']).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval();dest={}
 for mode,indices in [('train',train),('test',test)]:
  batches=DataLoader([records[i] for i in indices],batch_size=16,shuffle=False,collate_fn=lambda x:collate_peptides(x,include_structure=True,include_atom=False));vs=[];ys=[];ls=[];cs=[]
  with torch.inference_mode():
   for b in batches:
    b=move_batch_to_device(b,device);o=model(b);h=o['spatial_nodes'];mask=b['mask'].bool();pairs=mask[:,1:]&mask[:,:-1];adj=h[:,1:]*h[:,:-1];adj=(adj*pairs.unsqueeze(-1)).sum(1)/pairs.sum(1).clamp_min(1).unsqueeze(-1);cm=(b['aa_ids']==AA_TO_IDX['C'])&mask;ch=(h*cm.unsqueeze(-1)).sum(1)/cm.sum(1).clamp_min(1).unsqueeze(-1);vs.append(torch.cat([adj,ch],-1).cpu());ys.append(b['labels'].cpu());ls.append(o['logits'].cpu());cs.append(cm.sum(1).cpu())
  dest[mode]={'features':torch.cat(vs),'labels':torch.cat(ys),'logits':torch.cat(ls),'cys':torch.cat(cs),'indices':indices};print('features',mode,len(indices),flush=True)
 torch.save(dest,cache);del model,records
d=torch.load(cache,map_location='cpu',weights_only=False);dim=d['train']['features'].shape[1]//2
class Head(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1));self.cys=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1)) if kind=='C' else None
  nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
  if self.cys is not None:nn.init.zeros_(self.cys[-1].weight);nn.init.zeros_(self.cys[-1].bias)
 def forward(self,x):
  v=self.net(x[:,:dim]).squeeze(-1)
  return v if self.cys is None else v+self.cys(x[:,dim:]).squeeze(-1)
outputs={'A':d['test']['logits'].numpy().tolist()}
for kind in ('B','C'):
 torch.manual_seed(42);head=Head(kind).to(device);opt=torch.optim.AdamW(head.parameters(),lr=0.0003,weight_decay=.01);x=d['train']['features'].to(device);y=d['train']['labels'].to(device).float();l=d['train']['logits'].to(device);g=torch.Generator().manual_seed(42)
 for epoch in range(1,6):
  order=torch.randperm(len(y),generator=g);total=0
  for ix in order.split(64):
   ix=ix.to(device);opt.zero_grad();delta=head(x[ix]);loss=nn.functional.binary_cross_entropy_with_logits(l[ix]+delta,y[ix])+.01*delta.square().mean();loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  print(kind,'epoch',epoch,'loss',total/len(y),flush=True)
 head.eval()
 with torch.inference_mode():outputs[kind]=(d['test']['logits']+head(d['test']['features'].to(device)).cpu()).numpy().tolist()
 torch.save(head.state_dict(),out/f'head_{kind}.pt')
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
y=d['test']['labels'].numpy();c=d['test']['cys'].numpy();p={k:1/(1+np.exp(-np.clip(np.array(v),-50,50))) for k,v in outputs.items()};results={}
for k,v in p.items():
 q=v>=.5;results[k]={'mcc':float(matthews_corrcoef(y,q)),'auroc':float(roc_auc_score(y,v)),'auprc':float(average_precision_score(y,v)),'errors':int((q!=y).sum()),'low_pos_fn':int(((c<=1)&(y==1)&~q).sum()),'low_pos_n':int(((c<=1)&(y==1)).sum()),'high_neg_fp':int(((c>=6)&(y==0)&q).sum()),'high_neg_n':int(((c>=6)&(y==0)).sum())}
(out/'pilot_results.json').write_text(json.dumps(results,indent=2));np.savez_compressed(out/'pilot_predictions.npz',**p,labels=y,cys=c,indices=np.array(test));print(json.dumps(results,indent=2),flush=True)
