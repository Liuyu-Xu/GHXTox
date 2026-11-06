"""Frozen per-fold disagreement probe; outer fold labels never enter training."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;P=R/'analysis/reviewer_response_20261102'; B=P/'six_seed_received_20261106'; ORIGINAL=P/'six_seed_nested_autodl_20261106'; OLD=P/'nested_1d_received_20261106/nested_1d_results/runs'; NEW=B/'six_seed_new_results/runs';TH=.47022256531284556; SEEDS=(42,123,2025,314,2718,2026)
spec=importlib.util.spec_from_file_location('validated_comparison',ORIGINAL/'compare.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prob(p,c):
 p=np.clip(p,1e-6,1-1e-6);return expit(c[0]*np.log(p/(1-p))+c[1])
manifest=read(ORIGINAL/'data/nested_folds.csv'); param={x['fold']:x for x in json.loads((B/'comparison/summary.json').read_text())['parameters']};expected={'manifest_sha256':sha(ORIGINAL/'data/nested_folds.csv'),'config_sha256':sha(ORIGINAL/'config/model.json'),'source_sha256':sha(ORIGINAL/'src/ghxtox/lora_mamba_1d.py'),'prott5_weight_sha256':'7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157','prott5_aux_sha256':{'config.json':'c93cc8c31433fb062cad1825a9c16cde962a2015c8a3b10cfdc5f76a85128db3','spiece.model':'74da7b4afcde53faa570114b530c726135bdfcdb813dec3abfb27f9d44db7324'}}
def collect(fold,role):
 rows=[r for r in manifest if int(r['outer_fold'])==fold and r['role']==role]; ids=[r['sample_id'] for r in rows];arr=[]
 for seed in SEEDS:
  m=prior.load_run(OLD if seed in SEEDS[:3] else NEW,seed,fold,role,expected)
  if set(m)!=set(ids):raise ValueError('sample mismatch')
  for x in rows:
   q=m[x['sample_id']]
   if any(q[k]!=x[k] for k in ('label','sequence','group_id')) or int(q['seed'])!=seed or int(q['fold'])!=fold:raise ValueError('row mismatch')
  arr.append([float(m[i]['probability']) for i in ids])
 sfile=B/f'structure/fold{fold}/{role}.csv';assert sha(sfile)==prior.STRUCT_SHA[(fold,role)];s=read(sfile);by={x['sample_id']:x for x in s};assert len(by)==len(s)==len(rows) and set(by)==set(ids)
 assert all(by[x['sample_id']]['sequence']==x['sequence'] and by[x['sample_id']]['label']==x['label'] for x in rows)
 arr=np.array(arr);q=prob(arr.mean(axis=0),param[fold]['platt_6']);p3=prob(np.array([float(by[i]['toxicity_probability']) for i in ids]),param[fold]['platt_structure']);std=arr.std(axis=0,ddof=1)
 if not (np.isfinite(arr).all() and ((arr>=0)&(arr<=1)).all() and np.isfinite(p3).all()):raise ValueError('invalid probabilities')
 y=np.array([int(r['label']) for r in rows]);cys=np.array([r['sequence'].count('C') for r in rows]);length=np.array([len(r['sequence']) for r in rows]);groups=np.array([r['group_id'] for r in rows]);a=q>=TH;b=p3>=TH;dis=a!=b;target=(b==y).astype(int);fixed=.72*q+.28*p3
 return {'rows':rows,'y':y,'q':q,'p3':p3,'std':std,'cys':cys,'length':length,'groups':groups,'seq':a,'structure':b,'dis':dis,'target':target,'fixed':fixed,'arr':arr,'raw3':np.array([float(by[i]['toxicity_probability']) for i in ids])}
def xfeatures(d,extra):
 base=np.column_stack((d['q'],d['p3'],d['std']))
 return np.column_stack((base,d['length'],(d['cys']==0).astype(float),(d['cys']>=7).astype(float))) if extra else base
def score(y,p):return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
def counts(target,choose):return {'n':int(len(target)),'corrects':int(((target==1)&choose).sum()),'harms':int(((target==0)&choose).sum()),'net':int(((target==1)&choose).sum()-((target==0)&choose).sum())}
def main():
 evaluation=[];foldreports=[]
 for fold in range(5):
  cal=collect(fold,'calibration'); held=collect(fold,'test');fi,sel=next(StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20261106+fold).split(np.zeros(len(cal['y'])),cal['y'],cal['groups']))
  assert set(cal['groups'][fi]).isdisjoint(set(cal['groups'][sel])) and set(cal['groups']).isdisjoint(set(held['groups']))
  train=sel[cal['dis'][sel]];test=np.flatnonzero(held['dis']);assert len(test)>0
  targets=cal['target'][train];z=held['target'][test];group_counts={str(k):int(len(set(cal['groups'][train][targets==k]))) for k in (0,1)};eligible=min(group_counts.values())>=10
  row={'fold':fold,'calibration_fit_n':len(fi),'probe_train_n':len(train),'probe_train_target_counts':{str(k):int((targets==k).sum()) for k in (0,1)},'probe_train_groups_by_target':group_counts,'held_disagreement_n':len(test),'held_structure_correct':int(z.sum()),'held_sequence_correct':int(len(z)-z.sum()),'fit_allowed':eligible,'methods':{}}
  fixed_delta=(held['fixed']>=TH)!=held['seq']; row['held_fixed_fusion_changed']={'n':int(fixed_delta.sum()),'corrects':int(((held['fixed']>=TH)==held['y'])[fixed_delta].sum()),'harms':int(((held['fixed']>=TH)!=held['y'])[fixed_delta].sum())}
  confidence=np.abs(held['p3'][test]-TH)-np.abs(held['q'][test]-TH); row['methods']['margin']={'ranking_auroc':score(z,confidence),**counts(z,confidence>0)}
  preds={}
  if eligible:
   for key,extra in [('probability',False),('probability_cys_length',True)]:
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=20261106));model.fit(xfeatures(cal,extra)[train],targets);predict=model.predict_proba(xfeatures(held,extra)[test])[:,1];preds[key]=predict;row['methods'][key]={'ranking_auroc':score(z,predict),**counts(z,predict>.5)}
  for j,i in enumerate(test):
   x=held['rows'][i];evaluation.append({'fold':fold,'sample_id':x['sample_id'],'group_id':x['group_id'],'label':int(held['y'][i]),'sequence':x['sequence'],'cys':int(held['cys'][i]),'length':int(held['length'][i]),'p_seq':float(held['q'][i]),'p_3d':float(held['p3'][i]),'seed_std_raw':float(held['std'][i]),'structure_correct':int(z[j]),'margin_choose_3d':int(confidence[j]>0),'margin_score':float(confidence[j]),'probability_gate':float(preds['probability'][j]) if eligible else '', 'cys_gate':float(preds['probability_cys_length'][j]) if eligible else ''})
  foldreports.append(row)
 assert len({x['sample_id'] for x in evaluation})==len(evaluation)
 with (O/'held_disagreements.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(evaluation[0]));w.writeheader();w.writerows(evaluation)
 report={'protocol':'PROTOCOL.md','source_sha256':{'manifest':sha(ORIGINAL/'data/nested_folds.csv'),'previous_results':sha(B/'comparison/summary.json'),'previous_comparator':sha(ORIGINAL/'compare.py')},'folds':foldreports,'note':'fold training labels used for model fitting only; held labels for evaluation only; not tested on Test1/Test2'}
 (O/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(foldreports,indent=2))

if __name__=="__main__":main()
