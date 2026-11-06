import csv,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.optimize import linear_sum_assignment
root=Path(r'D:\dachuang2026\GHXTox');base=Path(r'D:\dachuang2026\GHXTox-ziliao');out=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
records={str(r['sample_id']):r for r in torch.load(base/'train_chemical_sites_final_esm2_multisource_1_30_full2795.pt',map_location='cpu',weights_only=False)['records']}
with (root/'data/folds/train_cdhit080_fallback_folds.csv').open(encoding='utf8',newline='') as f:folds=list(csv.DictReader(f))
idx=json.loads((root/'configs/provenance/historical_selected_base_v1/index.json').read_text());results=[];coverage=[]
for e in idx['entries']:
 fold=e['fold'];cp=e['checkpoint'].replace('dachuang2026\\ziliao','dachuang2026\\GHXTox-ziliao');ck=torch.load(cp,map_location='cpu',weights_only=False);model=GHXToxModel(ck['config']).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval();eligible=[];n_cys=0
 for row in folds:
  if int(row['fold'])!=fold:continue
  rec=records[row['sample_id']];mask=rec['chemical_site_mask'];seq=row['sequence'];c=[i for i,a in enumerate(seq) if a=='C' and bool(mask[i].any())];n=[i for i,a in enumerate(seq) if a!='C' and bool(mask[i].any())]
  if not c:continue
  n_cys+=1
  if len(n)<len(c):continue
  costs=np.array([[abs(i-j)/max(len(seq)-1,1) if int(mask[i].sum())==int(mask[j].sum()) and abs(i-j)/max(len(seq)-1,1)<=.2 else 1000 for j in n] for i in c]); ii,jj=linear_sum_assignment(costs)
  if len(ii)!=len(c) or np.any(costs[ii,jj]>=1000):continue
  eligible.append((row,c,[n[j] for j in jj],float(costs[ii,jj].mean())))
 coverage.append({'fold':fold,'with_cys_sites':n_cys,'matched':len(eligible)})
 with torch.inference_mode():
  for start in range(0,len(eligible),16):
   part=eligible[start:start+16];batch=move_batch_to_device(collate_peptides([records[r['sample_id']] for r,c,n,d in part],include_structure=True,include_atom=False),device);orig=batch['chemical_site_mask'].clone();p0=torch.sigmoid(model(batch)['logits']).cpu().numpy();values=[]
   for col in (1,2):
    mask=orig.clone()
    for i,item in enumerate(part):mask[i,item[col],:]=False
    batch['chemical_site_mask']=mask;values.append(torch.sigmoid(model(batch)['logits']).cpu().numpy())
   for i,(r,c,n,d) in enumerate(part):results.append({'sample_id':r['sample_id'],'fold':fold,'group_id':r['group_id'],'label':int(r['label']),'cys_count':r['sequence'].count('C'),'matched_residues':len(c),'position_gap_mean':d,'base':float(p0[i]),'cys_mask':float(values[0][i]),'control_mask':float(values[1][i])})
 print('fold',fold,'coverage',coverage[-1],flush=True)
with (out/'cys_matched_mask_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
(out/'cys_matched_mask_coverage.json').write_text(json.dumps(coverage,indent=2))
