import csv,json,sys,hashlib
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score,log_loss
root=Path(r'D:\dachuang2026\GHXTox');P=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device
torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');datapath=Path(r'D:\dachuang2026\GHXTox-ziliao\train_chemical_sites_final_esm2_multisource_1_30_full2795.pt')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
records=torch.load(datapath,map_location='cpu',weights_only=False)['records'];out={}
for fold in (1,2):
 D=P/f'runs/topology_strict_oof_v1/base/fold{fold}';prov=json.loads((D/'provenance.json').read_text());assert sha(D/'best_model.pt')==prov['checkpoint_sha256'] and sha(datapath)==prov['training_data_sha256'];train=prov['unused_calibration_indices'];val=prov['diagnostic_validation_indices'];assert not set(train)&set(val) and not set(train)&set(prov['fitting_indices']);z=np.load(P/f'spatial_message_fold{fold}_predictions.npz');assert np.array_equal(z['indices'],val)
 ck=torch.load(D/'best_model.pt',map_location='cpu',weights_only=False);model=GHXToxModel(ck['config']).to(device);model.load_state_dict(ck['model_state'],strict=True);model.eval();logits=[];labels=[]
 with torch.inference_mode():
  for b in DataLoader([records[i] for i in train],batch_size=8,shuffle=False,collate_fn=lambda x:collate_peptides(x,include_structure=True,include_atom=False)):
   b=move_batch_to_device(b,device);logits.extend(model(b)['logits'].cpu().tolist());labels.extend(b['labels'].cpu().tolist())
 cal=LogisticRegression(C=1e6,max_iter=1000).fit(np.array(logits)[:,None],labels);val_logits=[]
 with torch.inference_mode():
  for b in DataLoader([records[i] for i in val],batch_size=8,shuffle=False,collate_fn=lambda x:collate_peptides(x,include_structure=True,include_atom=False)):
   b=move_batch_to_device(b,device);val_logits.extend(model(b)['logits'].cpu().tolist())
 val_logits=np.array(val_logits); reconstructed=torch.sigmoid(torch.tensor(val_logits,dtype=torch.float32)).numpy();assert np.max(abs(reconstructed-z['A']))<1e-6
 pcal=cal.predict_proba(val_logits[:,None])[:,1];y=z['labels'];result={}
 for name,p in [('A',z['A']),('Platt',pcal),('B',z['B']),('C',z['C'])]:
  p=np.asarray(p,dtype=float);q=p>=.5;v=np.clip(p,1e-7,1-1e-7);result[name]={'mcc':float(matthews_corrcoef(y,q)),'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p)),'errors':int((q!=y).sum()),'FN':int(((y==1)&~q).sum()),'FP':int(((y==0)&q).sum()),'brier':float(np.mean((p-y)**2)),'logloss_clipped':float(log_loss(y,v,labels=[0,1])),'positive_below_0.001':int(((y==1)&(p<.001)).sum())}
 out[str(fold)]={'n_fit':len(train),'n_val':len(val),'platt_slope':float(cal.coef_[0,0]),'platt_intercept':float(cal.intercept_[0]),'raw_logit_auroc':float(roc_auc_score(y,val_logits)),'raw_logit_auprc':float(average_precision_score(y,val_logits)),'base_reproduction_max_abs':float(np.max(abs(reconstructed-z['A']))),'metrics':result};np.savez_compressed(P/f'platt_fold{fold}_predictions.npz',platt=pcal,indices=z['indices']);print(fold,json.dumps(out[str(fold)]),flush=True)
(P/'platt_vs_message_results.json').write_text(json.dumps(out,indent=2),encoding='utf8')
