"""Descriptive branch comparison from pre-existing held-out predictions; no training."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent; B=R/'analysis/reviewer_response_20261102/six_seed_received_20261106'; MAN=R/'analysis/reviewer_response_20261102/six_seed_nested_autodl_20261106/data/nested_folds.csv'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cal(p,c):
 p=np.clip(p,1e-6,1-1e-6);return expit(c[0]*np.log(p/(1-p))+c[1])
summary=json.loads((B/'comparison/summary.json').read_text(encoding='utf8'));params={x['fold']:x for x in summary['parameters']};raw=read(B/'comparison/predictions.csv'); man=read(MAN); held={(x['sample_id'],int(x['outer_fold'])):x for x in man if x['role']=='test'}
assert len(raw)==len(held)==6387 and len({r['sample_id'] for r in raw})==6387
structure={}
for fold in range(5):
 p=B/f'structure/fold{fold}/test.csv'; expected=next(v for k,v in summary['structure_input_sha256'].items() if k.replace('\\','/').endswith(f'/fold{fold}/test.csv')); assert sha(p)==expected
 r=read(p); d={x['sample_id']:x for x in r};assert len(d)==len(r);structure[fold]=d
rows=[]
for r in raw:
 fold=int(r['fold']);sid=r['sample_id'];m=held[(sid,fold)];s=structure[fold][sid];assert r['label']==m['label']==s['label'] and r['group_id']==m['group_id'] and m['sequence']==s['sequence']
 seq=m['sequence'];n=len(seq);count=lambda letters:sum(seq.count(c) for c in letters);p1=float(r['six_fixed']);p3=float(cal(float(s['toxicity_probability']),params[fold]['platt_structure']));assert 0<=p1<=1 and 0<=p3<=1
 x={'sample_id':sid,'group_id':r['group_id'],'fold':fold,'label':int(r['label']),'sequence':seq,'length':n,'cys_count':seq.count('C'),'basic_fraction':count('KRH')/n,'acidic_fraction':count('DE')/n,'hydrophobic_fraction':count('AVILMFWY')/n,'charge_approx':count('KR')-count('DE'),'p_sequence':p1,'p_structure':p3,'p_structure_raw':float(s['toxicity_probability'])}
 for t,name in ((0.5,'t05'),(.47022256531284556,'historical')):
  a=p1>=t;b=p3>=t;y=x['label'];x[name+'_sequence_pred']=int(a);x[name+'_structure_pred']=int(b);x[name+'_category']='both_correct' if a==y and b==y else 'sequence_only_correct' if a==y else 'structure_only_correct' if b==y else 'both_wrong'
 rows.append(x)
def cy(n):return '0-1' if n<=1 else '2-3' if n<=3 else '4-6' if n<=6 else '7+'
def lg(n):return '11-20' if n<=20 else '21-35' if n<=35 else '36-50'
tables=[]
for threshold in ('t05','historical'):
 for dimension,func in [('cys_4bin',lambda r:cy(r['cys_count'])),('cys_2bin',lambda r:'<4' if r['cys_count']<4 else '>=4'),('length',lambda r:lg(r['length'])),('all',lambda r:'all')]:
  keys=sorted({func(r) for r in rows})
  for key in keys:
   for label in (0,1):
    part=[r for r in rows if func(r)==key and r['label']==label];n=len(part)
    counts={cat:sum(r[threshold+'_category']==cat for r in part) for cat in ('both_correct','sequence_only_correct','structure_only_correct','both_wrong')};assert sum(counts.values())==n
    folds=[]
    for fold in range(5):
     rr=[r for r in part if r['fold']==fold];folds.append({'fold':fold,'n':len(rr),'seq_correct':sum(r[threshold+'_sequence_pred']==label for r in rr),'struct_correct':sum(r[threshold+'_structure_pred']==label for r in rr)})
    tables.append({'threshold':threshold,'dimension':dimension,'group':key,'label':label,'n':n,**counts,'sequence_errors':counts['structure_only_correct']+counts['both_wrong'],'structure_errors':counts['sequence_only_correct']+counts['both_wrong'],'structure_correct_net_vs_sequence':counts['structure_only_correct']-counts['sequence_only_correct'],'folds':folds})
for r in rows:r['cys_group']=cy(r['cys_count']);r['length_group']=lg(r['length'])
with (O/'branch_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (O/'group_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 keys=[k for k in tables[0] if k!='folds'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in tables])
y=np.array([r['label'] for r in rows]);seq=np.array([r['p_sequence'] for r in rows]);s3=np.array([r['p_structure'] for r in rows]);output={'input_sha256':{'predictions':sha(B/'comparison/predictions.csv'),'summary':sha(B/'comparison/summary.json'),'manifest':sha(MAN)},'n':len(rows),'warnings':['exploratory on previously examined training OOF, not blind; six 1D models vs one 3D seed','probabilities calibrated per fold; thresholds 0.5 and 0.47022 are diagnostics, not separately optimized per model','pLDDT not available in current prediction CSV; sequence-derived cys is not proven mechanism','held-out 3D backbone is retrained proxy, not deployment checkpoint'], 'overall':{'sequence_auroc':roc_auc_score(y,seq),'structure_auroc':roc_auc_score(y,s3),'sequence_auprc':average_precision_score(y,seq),'structure_auprc':average_precision_score(y,s3)},'tables':tables}
(O/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf8');print(json.dumps({'overall':output['overall'],'cys_2bin_05':[r for r in tables if r['threshold']=='t05' and r['dimension']=='cys_2bin'],'all_05':[r for r in tables if r['threshold']=='t05' and r['dimension']=='all']},indent=2))
