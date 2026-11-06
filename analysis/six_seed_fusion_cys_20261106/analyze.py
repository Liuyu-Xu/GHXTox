"""Retrospective fixed six-seed+structure vs six-seed analysis; no gate fitting."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
OOF=R/'analysis/sequence_structure_cys_20261106/branch_predictions.csv'; TEST=R/'analysis/reviewer_response_20261102/three_seed_full_received_20261106'; TH=.47022256531284556
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for split,path in [('training_oof',OOF),('test1',TEST/'test1_predictions.csv'),('test2',TEST/'test2_predictions.csv')]:
 data=read(path);assert len({x['sample_id'] for x in data})==len(data)==({'training_oof':6387,'test1':1126,'test2':582}[split])
 for r in data:
  seq=r['sequence']; p=float(r['p_sequence'] if split=='training_oof' else r['six_sequence']);s=float(r['p_structure'] if split=='training_oof' else r['p3d']);fusion=.72*p+.28*s
  if split!='training_oof':assert abs(fusion-float(r['six_plus_3d']))<1e-12
  y=int(r['label']);a=int(p>=TH);b=int(fusion>=TH);sp=int(s>=TH);cat='both_correct' if a==y and b==y else 'sequence_only_correct' if a==y else 'fusion_only_correct' if b==y else 'both_wrong'
  rows.append({'split':split,'sample_id':r['sample_id'],'sequence':seq,'group_id':r.get('group_id',''),'fold':r.get('fold',''),'label':y,'cys':seq.count('C'),'length':len(seq),'p_seq6':p,'p_3d':s,'p_fused6':fusion,'seq_pred':a,'structure_pred_diagnostic':sp,'fusion_pred':b,'category':cat})
groups=[('zero',lambda c:c==0),('zero_or_one',lambda c:c<=1),('seven_plus',lambda c:c>=7),('all',lambda c:True)];tables=[];metrics={}
for split in ('training_oof','test1','test2'):
 ss=[x for x in rows if x['split']==split];yy=np.array([x['label'] for x in ss]);metrics[split]={}
 for key in ('p_seq6','p_fused6'):
  p=np.array([x[key] for x in ss]);metrics[split][key]={'mcc':float(matthews_corrcoef(yy,p>=TH)),'auroc':float(roc_auc_score(yy,p)),'auprc':float(average_precision_score(yy,p))}
 for name,fn in groups:
  for label in (0,1):
   sub=[x for x in ss if fn(x['cys']) and x['label']==label];n=len(sub); counts={k:sum(x['category']==k for x in sub) for k in ('both_correct','sequence_only_correct','fusion_only_correct','both_wrong')};assert sum(counts.values())==n
   folds=[]
   if split=='training_oof':
    for fold in range(5):
     ff=[x for x in sub if int(x['fold'])==fold];folds.append({'fold':fold,'n':len(ff),'fusion_corrects':sum(x['category']=='fusion_only_correct' for x in ff),'fusion_harms':sum(x['category']=='sequence_only_correct' for x in ff)})
   tables.append({'split':split,'cys_group':name,'label':label,'n':n,**counts,'seq_errors':counts['fusion_only_correct']+counts['both_wrong'],'fusion_errors':counts['sequence_only_correct']+counts['both_wrong'],'net_corrected':counts['fusion_only_correct']-counts['sequence_only_correct'],'folds':folds})
with (O/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (O/'group_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 keys=[x for x in tables[0] if x!='folds'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in tables])
res={'inputs_sha256':{'training_oof':sha(OOF),'test1':sha(TEST/'test1_predictions.csv'),'test2':sha(TEST/'test2_predictions.csv')},'protocol':{'weight':'0.72 seq6 + 0.28 structure','threshold':TH,'training_oof':'six calibrated 1D + per-fold calibrated 3D retrained backbone; new post-hoc calculation, not earlier three-seed fusion','tests':'six full models + cached 3D deployed model, with mixed-OOF calibration; retrospective, Test2 subset of Test1','limitations':'comparison with fixed fusion only; no adaptive model has been fitted or evaluated; labels never routing inputs'},'metrics':metrics,'tables':tables}
(O/'summary.json').write_text(json.dumps(res,indent=2),encoding='utf8');print(json.dumps(res,indent=2))
