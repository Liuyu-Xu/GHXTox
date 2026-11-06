import json,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent; R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
read=lambda x:json.loads(x.read_text(encoding='utf-8'))
def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
bp=R/'analysis/error_evidence_linkage_20261025/all_training_linked.json';rp=R/'analysis/cys_probability_adjustment_20261028/predictions.json';b=read(bp);r=read(rp)
assert len(b)==len(r)==6387 and all(a['sample_id']==z['sample_id'] for a,z in zip(b,r))
def num(v):
 try:
  v=float(v);return v if np.isfinite(v) else np.nan
 except (ValueError,TypeError):return np.nan
features=['p_seq','p_struct','sequence_length','cys_count','charge_density','mean_hydropathy','aromatic_fraction','proline_fraction','glycine_fraction','mean_plddt','low_plddt_fraction','ca_radius_gyration','nonlocal_contact_density','sequence_seed_std']
X=np.array([[num(v.get(k)) for k in features] for v in b]); p=np.array([z['A_probability'] for z in r]); labels=np.array([z['label'] for z in r]); pred=np.array([z['A_prediction'] for z in r]); e=(pred!=labels).astype(int); folds=np.array([z['fold'] for z in r]); groups=np.array([z['group_id'] for z in r]);c=np.array([z['cys_count'] for z in r]);t=0.47022256531284556
# Margin and predicted class are available at inference; never use target label or annotation as features.
base=np.column_stack([p, np.abs(p-t),pred]); Xfull=np.column_stack([base,X]); risk={'distance':-np.abs(p-t),'probability_logistic':np.empty(len(p)),'feature_logistic':np.empty(len(p))}; chosen={k:np.zeros(len(p),dtype=bool) for k in risk}; details=[]
for f in range(5):
 train=folds!=f; valid=~train; assert not set(groups[train])&set(groups[valid]); d={'fold':f,'train_n':int(train.sum()),'valid_n':int(valid.sum()),'valid_errors':int(e[valid].sum())}
 for name,xx in [('probability_logistic',base),('feature_logistic',Xfull)]:
  model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261029))
  model.fit(xx[train],e[train]);risk[name][valid]=model.predict_proba(xx[valid])[:,1]
 # Top 10% each held-out fold, rank-only; budget fixed before evaluating.
 ids=np.flatnonzero(valid); n=int(np.ceil(len(ids)*.1))
 for name in risk:
  order=np.lexsort((ids,-risk[name][ids]));chosen[name][ids[order[:n]]]=True
 details.append(d)
def calc(mask):
 o={}
 for k,s in risk.items():
  yy=e[mask]; selected=chosen[k]&mask; caught=int((selected&(e==1)).sum()); count=int(selected.sum()); wrong=int(yy.sum());
  o[k]={'n':int(mask.sum()),'error_count':wrong,'AUROC':float(roc_auc_score(yy,s[mask])) if len(np.unique(yy))>1 else None,'AUPRC':float(average_precision_score(yy,s[mask])) if len(np.unique(yy))>1 else None,'brier':float(brier_score_loss(yy,s[mask])) if k!='distance' else None,'selected':count,'selected_errors':caught,'precision_selected':caught/count if count else None,'error_recall_selected':caught/wrong if wrong else None,'false_alarms':count-caught}
 return o
masks={'all':np.ones(len(p),dtype=bool),'low_pos_diagnostic_only':(c<=1)&(labels==1),'high_neg_diagnostic_only':(c>=6)&(labels==0),'low_all':c<=1,'high_all':c>=6}
metrics={k:calc(v) for k,v in masks.items()}; byfold={str(k):calc(folds==k) for k in range(5)}
save('metrics.json',metrics);save('fold_metrics.json',byfold);save('folds.json',details)
save('predictions.json',[{'sample_id':r[i]['sample_id'],'fold':int(folds[i]),'group_id':groups[i],'label':int(labels[i]),'original_prediction':int(pred[i]),'error':int(e[i]),**{k+'_risk':float(v[i]) for k,v in risk.items()},**{k+'_selected':bool(v[i]) for k,v in chosen.items()}} for i in range(len(p))])
save('protocol.json',{'goal':'out-of-fold error-risk detection, no change of GHXTox predictions','base_features':['p_fusion','distance_to_fixed_threshold','predicted_class'],'additional_features':features,'selection':'fixed top 10% within each held-out fold','training':'one logistic regression per fold, median imputation and standardization fitted on four development folds, fixed C=1','reference_similarity':'excluded: historical similarity uses other folds and is not appropriate for independent validation','test1_test2_used':False,'upstream_fully_nested':False,'inputs_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (bp,rp)}})
print(json.dumps({'metrics':metrics,'folds':byfold},indent=2))
