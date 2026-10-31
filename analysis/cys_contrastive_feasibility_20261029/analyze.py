import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
read=lambda q:json.loads(q.read_text(encoding='utf8'))
b=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json');rows=read(R/'analysis/cys_probability_adjustment_20261028/predictions.json');meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');roles=read(R/'analysis/group_weight_trial_20261027/role_weight_contract.json');assert len(rows)==len(meta)==len(b)==6387
assert all(u['sample_id']==v['sample_id']==w['sample_id'] and u['sequence']==w['sequence'] for u,v,w in zip(b,rows,meta))
y=np.array([v['label'] for v in rows]);fold=np.array([v['fold'] for v in rows]);grp=np.array([v['group_id'] for v in rows]);p=np.array([v['A_probability'] for v in rows]); c=np.array([v['cys_count'] for v in rows]);seq=[v['sequence'] for v in b];th=.47022256531284556
X1=np.array([[len(s),s.count('C')] for s in seq]);X2=np.array([[len(s)]+[s.count(a)/len(s) for a in 'ACDEFGHIKLMNPQRSTVWY'] for s in seq]);scores={'ghxtox':p}
for name,X in [('cys_length',X1),('composition',X2)]:
 out=np.zeros(len(y))
 for f in range(5):
  tr=fold!=f;ev=~tr;assert not set(grp[tr])&set(grp[ev]);model=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,max_features=1.,n_jobs=2,random_state=20261029);model.fit(X[tr],p[tr]);out[ev]=model.predict(X[ev])
 scores[name]=out
panels={'all':np.ones(len(y),dtype=bool),'low_positive':(c<=1)&(y==1),'low_negative':(c<=1)&(y==0),'high_positive':(c>=6)&(y==1),'high_negative':(c>=6)&(y==0)};out={}
for key,mask in panels.items():
 out[key]={}
 for name,s in scores.items():
  yy=y[mask];d=(s[mask]>=th);out[key][name]={'n':int(mask.sum()),'errors':int((d!=yy).sum()),'FN':int(((yy==1)&~d).sum()),'FP':int(((yy==0)&d).sum()),'agreement_with_ghxtox':float(np.mean(d==(p[mask]>=th))),'MCC':float(matthews_corrcoef(yy,d)) if len(set(yy))==2 else None,'AUPRC':float(average_precision_score(yy,s[mask])) if len(set(yy))==2 else None}
def count_pairs(items):
 pos=[x for x in items if x['label']==1];neg=[x for x in items if x['label']==0]; gp=Counter(x['group_id'] for x in pos);gn=Counter(x['group_id'] for x in neg);cross=len(pos)*len(neg)-sum(n*gn[g] for g,n in gp.items())
 def same(a):
  gc=Counter(x['group_id'] for x in a);return len(a)*(len(a)-1)//2-sum(n*(n-1)//2 for n in gc.values())
 return {'positive_samples':len(pos),'negative_samples':len(neg),'positive_groups':len(gp),'negative_groups':len(gn),'cross_label_different_group_pairs':cross,'positive_same_label_different_group_pairs':same(pos),'negative_same_label_different_group_pairs':same(neg)}
counts=[]
for f in range(5):
 train=[x for x in roles if x['outer_fold']==f and x['role']=='train'];ids=[x['source_index'] for x in train];assert all(meta[i]['sample_id']==x['sample_id'] for x,i in zip(train,ids));item=[]
 for x,i in zip(train,ids):item.append({'label':x['label'],'group_id':x['group_id'],'cys':meta[i]['cys_count'],'flag':bool(meta[i]['uniform_flag'])})
 rec={'fold':f,'train_n':len(item),'groups':{}}
 for g,fn in [('low',lambda x:x['cys']<=1),('mid',lambda x:2<=x['cys']<=5),('high',lambda x:x['cys']>=6)]:
  group=[x for x in item if fn(x)];rec['groups'][g]={'all':count_pairs(group),'unflagged':count_pairs([x for x in group if not x['flag']])}
 counts.append(rec)
(P/'metrics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf8');(P/'pairs.json').write_text(json.dumps(counts,indent=2,ensure_ascii=False),encoding='utf8');(P/'predictions.json').write_text(json.dumps([{'sample_id':rows[i]['sample_id'],'fold':int(fold[i]),'label':int(y[i]),'cys_count':int(c[i]),**{n:float(s[i]) for n,s in scores.items()}} for i in range(len(y))],indent=2),encoding='utf8')
print(json.dumps({'metrics':out,'pairs':counts},indent=2))
