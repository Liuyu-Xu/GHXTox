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
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); folder=out/'runs/topology_strict_oof_v1/base/fold0';prov=json.loads((folder/'provenance.json').read_text());d=torch.load(out/'pilot_features.pt',map_location='cpu',weights_only=False);val=prov['unused_calibration_indices'];assert not set(val)&set(d['train']['indices']) and not set(val)&set(d['test']['indices'])
vcache=out/'base_unseen_training_features.pt'
if not vcache.exists():
 rec=torch.load(r'D:\dachuang2026\GHXTox-ziliao\train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',map_location='cpu',weights_only=False)['records'];ck=torch.load(folder/'best_model.pt',map_location='cpu',weights_only=False);model=GHXToxModel(ck['config']).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval();vs=[];ys=[];ls=[];cs=[]
 with torch.inference_mode():
  for b in DataLoader([rec[i] for i in val],batch_size=16,shuffle=False,collate_fn=lambda x:collate_peptides(x,include_structure=True,include_atom=False)):
   b=move_batch_to_device(b,device);o=model(b);h=o['spatial_nodes'];mask=b['mask'].bool();pair=mask[:,1:]&mask[:,:-1];adj=((h[:,1:]*h[:,:-1])*pair.unsqueeze(-1)).sum(1)/pair.sum(1).clamp_min(1).unsqueeze(-1);cm=(b['aa_ids']==AA_TO_IDX['C'])&mask;ch=(h*cm.unsqueeze(-1)).sum(1)/cm.sum(1).clamp_min(1).unsqueeze(-1);vs.append(torch.cat([adj,ch],-1).cpu());ys.append(b['labels'].cpu());ls.append(o['logits'].cpu());cs.append(cm.sum(1).cpu())
 torch.save({'features':torch.cat(vs),'labels':torch.cat(ys),'logits':torch.cat(ls),'cys':torch.cat(cs),'indices':val},vcache);del model,rec
v=torch.load(vcache,map_location='cpu',weights_only=False);dim=v['features'].shape[1]//2
