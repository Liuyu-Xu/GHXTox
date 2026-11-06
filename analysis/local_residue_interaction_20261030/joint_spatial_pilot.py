import json,sys,hashlib,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
root=Path(r'D:\dachuang2026\GHXTox');P=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.constants import AA_TO_IDX
from ghxtox.utils import move_batch_to_device
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');D=P/'runs/topology_strict_oof_v1/base/fold3';prov=json.loads((D/'provenance.json').read_text());data=Path(r'D:\dachuang2026\GHXTox-ziliao\train_chemical_sites_final_esm2_multisource_1_30_full2795.pt')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
assert sha(D/'best_model.pt')==prov['checkpoint_sha256'] and sha(data)==prov['training_data_sha256']
train=prov['fitting_indices'];val=prov['diagnostic_validation_indices'];assert not set(train)&set(val) and not set(val)&set(prov['fitting_indices'])
records=torch.load(data,map_location='cpu',weights_only=False)['records'];ck=torch.load(D/'best_model.pt',map_location='cpu',weights_only=False);dim=ck['config']['model']['hidden_dim'];print('device',device,'ntrain',len(train),'nval',len(val),'dim',dim,flush=True)
class Adapter(nn.Module):
 def __init__(self,conditional):
  super().__init__();self.conditional=conditional;self.message=nn.Sequential(nn.Linear(dim*3,64),nn.GELU(),nn.Linear(64,dim));nn.init.zeros_(self.message[-1].weight);nn.init.zeros_(self.message[-1].bias)
 def forward(self,h,b):
  mask=b['mask'].bool();coords=b['coords'];dist=torch.cdist(coords,coords);L=h.shape[1];valid=mask[:,:,None]&mask[:,None,:]&~torch.eye(L,device=h.device,dtype=torch.bool)[None];w=torch.exp(-dist.square()/128)*valid;normal=w/w.sum(-1,keepdim=True).clamp_min(1e-8);allmsg=torch.bmm(normal,h)
  if self.conditional:
   c=(b['aa_ids']==AA_TO_IDX['C'])&mask; wc=w*c[:,None,:]; context=torch.bmm(wc/wc.sum(-1,keepdim=True).clamp_min(1e-8),h)
  else:
   w2=torch.exp(-dist.square()/32)*valid;context=torch.bmm(w2/w2.sum(-1,keepdim=True).clamp_min(1e-8),h)
  return h+self.message(torch.cat([h,allmsg,context],-1))*mask.unsqueeze(-1)
def loadbase():
 m=GHXToxModel(ck['config']).to(device);m.load_state_dict(ck['model_state'],strict=True);m.eval()
 for p in m.parameters():p.requires_grad_(False)
 return m
def loader(indices,shuffle):
 gen=torch.Generator().manual_seed(42);return DataLoader([records[i] for i in indices],batch_size=8,shuffle=shuffle,generator=gen,collate_fn=lambda x:collate_peptides(x,include_structure=True,include_atom=False))
def evaluate(model,adapter):
 ys=[];ps=[];cs=[];current=[None];hook=None
 if adapter is not None:
  adapter.eval();hook=model.spatial_branch.register_forward_hook(lambda mod,args,o:adapter(o,current[0]))
 with torch.inference_mode():
  for b in loader(val,False):
   b=move_batch_to_device(b,device);current[0]=b;o=model(b);ys.extend(b['labels'].cpu().tolist());ps.extend(torch.sigmoid(o['logits']).cpu().tolist());cs.extend(((b['aa_ids']==AA_TO_IDX['C'])&b['mask'].bool()).sum(1).cpu().tolist())
 if hook:hook.remove()
 y=np.array(ys);p=np.array(ps);c=np.array(cs);q=p>=.5
 return {'mcc':float(matthews_corrcoef(y,q)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'errors':int((q!=y).sum()),'low_pos_fn':int(((c<=1)&(y==1)&~q).sum()),'low_pos_n':int(((c<=1)&(y==1)).sum()),'high_neg_fp':int(((c>=6)&(y==0)&q).sum()),'high_neg_n':int(((c>=6)&(y==0)).sum())},p,y

base=loadbase();results={};results['frozen'],pa,y=evaluate(base,None);pred={'frozen':pa};del base
for name,cond in [('A',None),('B',False),('C',True)]:
 torch.manual_seed(42);base=loadbase();last=len(base.spatial_branch.layers)-1
 for n,p in base.named_parameters():
  if n.startswith(f'spatial_branch.layers.{last}.') or n.startswith('classifier.'):p.requires_grad_(True)
 adapter=None if cond is None else Adapter(cond).to(device);current=[None];hook=None
 if adapter is not None:hook=base.spatial_branch.register_forward_hook(lambda mod,args,o:adapter(o,current[0]))
 params=[p for p in base.parameters() if p.requires_grad]+([] if adapter is None else list(adapter.parameters()));opt=torch.optim.AdamW(params,lr=.0001,weight_decay=.01);hist=[]
 print(name,'params',sum(p.numel() for p in params),flush=True)
 for epoch in range(3):
  base.eval()
  if adapter is not None:adapter.train()
  total=0
  for b in loader(train,True):
   b=move_batch_to_device(b,device);current[0]=b;opt.zero_grad();z=base(b)['logits'];loss=nn.functional.binary_cross_entropy_with_logits(z,b['labels'].float());loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step();total+=float(loss.detach())*len(z)
  hist.append(total/len(train));print(name,'epoch',epoch+1,'loss',hist[-1],flush=True)
 if hook:hook.remove()
 results[name],pred[name],_=evaluate(base,adapter);results[name]['history']=hist;results[name]['trainable_params']=sum(p.numel() for p in params)
 torch.save({'model_state':base.cpu().state_dict(),'adapter':None if adapter is None else adapter.cpu().state_dict(),'fold':3,'train_indices':train,'validation_indices':val},P/f'joint_fold3_{name}.pt')
 a=pred['frozen']>=.5;q=pred[name]>=.5;results[name]['corrected']=int(((a!=y)&(q==y)).sum());results[name]['harmed']=int(((a==y)&(q!=y)).sum());del base,adapter,opt;torch.cuda.empty_cache()
(P/'joint_fold3_results.json').write_text(json.dumps(results,indent=2),encoding='utf8');np.savez_compressed(P/'joint_fold3_predictions.npz',**pred,labels=y,indices=np.array(val));print(json.dumps(results,indent=2),flush=True)
