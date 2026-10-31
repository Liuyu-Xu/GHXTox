import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.metrics import roc_auc_score
read=lambda q:json.loads(q.read_text(encoding='utf-8'))
a=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');b=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');z=read(R/'analysis/cys_probability_adjustment_20261028/predictions.json');assert len(a)==len(b)==len(z)==6387
assert all(x['sample_id']==v['sample_id']==w['sample_id'] and x['sequence']==v['sequence'] for x,v,w in zip(a,b,z))
y=np.array([x['label'] for x in a]);c=np.array([x['cys_count'] for x in a]);fold=np.array([int(x['fold']) for x in a]);grp=np.array([x['group_id'] for x in a]);err=np.array([v['A_prediction']!=v['label'] for v in z]);flag=np.array([bool(x['uniform_flag']) for x in a])
def num(x):
 try:
  y=float(x);return y if np.isfinite(y) else np.nan
 except (ValueError,TypeError):return np.nan
features=['sequence_length','net_charge_proxy','charge_density','mean_hydropathy','aromatic_fraction','proline_fraction','glycine_fraction','mean_plddt','low_plddt_fraction','hydrophobic_contact_excess','charge_hydrophobic_separation','nonlocal_contact_density','ca_radius_gyration','max_otherfold_kmer_cosine','p_seq','p_struct','sequence_seed_std']
values={k:np.array([num(v.get(k)) for v in b]) for k in features}
values['branch_abs_gap']=np.abs(values['p_seq']-values['p_struct'])
values['branch_disagreement']=((values['p_seq']>=.5)!=(values['p_struct']>=.5)).astype(float)
sets={'low_positive':(c<=1)&(y==1),'high_negative':(c>=6)&(y==0)}
rng=np.random.default_rng(20261028)
def metric(mask,x):
 ix=mask&np.isfinite(x); e=err[ix]; xx=x[ix]
 if len(np.unique(e))<2:return None
 return {'n_wrong':int(e.sum()),'n_correct':int((~e).sum()),'median_wrong':float(np.median(xx[e])),'median_correct':float(np.median(xx[~e])),'auc_error':float(roc_auc_score(e,xx))}
out={}
for name,mask in sets.items():
 featuresout={}
 for k,x in values.items():
  total=metric(mask,x); byfold=[metric(mask&(fold==q),x) for q in range(5)]; auc=[s['auc_error'] if s else None for s in byfold]
  featuresout[k]={'overall':total,'fold_auc':auc,'positive_folds':sum(v is not None and v>.5 for v in auc),'negative_folds':sum(v is not None and v<.5 for v in auc)}
 out[name]={'n':int(mask.sum()),'wrong':int((mask&err).sum()),'correct':int((mask&~err).sum()),'flag_wrong':int((mask&err&flag).sum()),'flag_correct':int((mask&~err&flag).sum()),'fold_counts':[{'fold':q,'wrong':int((mask&err&(fold==q)).sum()),'correct':int((mask&~err&(fold==q)).sum())} for q in range(5)],'features':featuresout}
(P/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
for g,v in out.items():
 print('\n',g,'n/wrong/correct/flags',v['n'],v['wrong'],v['correct'],v['flag_wrong'],v['flag_correct'],'folds',v['fold_counts'])
 for k,t in v['features'].items():
  x=t['overall'];print(k,round(x['median_wrong'],3),round(x['median_correct'],3),round(x['auc_error'],3),[round(u,2) if u is not None else None for u in t['fold_auc']])
