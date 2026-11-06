from pathlib import Path
import json
from collections import Counter
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
P=read(O/'family_followup_protocol.json');z=np.load(O/'crossfit_scores.npz');ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];half=z['halves'];groups=z['groups'];n=len(y);rows={r['sample_id']:r for r in read(R/'analysis/saprot_contribution_20261026/sample_contributions.json')};rows=[rows[k] for k in ids];base=z['baseline4_raw'];length=np.array([r['features']['sequence_length'] for r in rows]);cys=np.array([r['features']['cys_count'] for r in rows]);span=np.array([r['features']['cys_span_fraction'] for r in rows]);seq=np.column_stack([base,length,cys,cys/length,span,base*(cys>=4)]);missing=np.array([[not r['curated_family'],not r['name_class']] for r in rows],dtype=float);tokens=[set(['curated:'+t for t in r['curated_family']]+['name:'+t for t in r['name_class']]) for r in rows];scores={name:np.full(n,np.nan) for name in P['new_arms']};models=[];vocabs=[]
for ff in range(5):
 for hh in range(2):
  tr=(fold==ff)&(half!=hh);te=(fold==ff)&(half==hh);assert not set(groups[tr])&set(groups[te]);counts=Counter(t for i in np.where(tr)[0] for t in tokens[i]);vocab=sorted(t for t,c in counts.items() if c>=5);vocabs.append(dict(fold=ff,half=hh,vocab=vocab));names=np.column_stack([np.array([t in ts for ts in tokens],dtype=float) for t in vocab])
  for arm,X in [('baseline4_CysLengthMissingOnly',np.column_stack([seq,missing])),('baseline4_CysLengthNamesOnly',np.column_stack([seq,names]))]:
   sc=StandardScaler().fit(X[tr]);m=LogisticRegression(C=1,solver='lbfgs',max_iter=3000).fit(sc.transform(X[tr]),y[tr]);assert m.n_iter_.max()<3000;scores[arm][te]=m.decision_function(sc.transform(X[te]));models.append(dict(arm=arm,fold=ff,half=hh,mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=m.coef_.tolist(),intercept=m.intercept_.tolist(),vocab=vocab if arm.endswith('NamesOnly') else []))
scores.update({k:z[k] for k in ['baseline4_CysLength','baseline4_CysLengthFamily']});_,gi=np.unique(groups,return_inverse=True);ng=gi.max()+1;rng=np.random.default_rng(20261028);W=np.array([np.bincount(rng.integers(0,ng,ng),minlength=ng) for _ in range(2000)],dtype=float);boot={};metrics=[]
for arm,score in scores.items():
 assert np.isfinite(score).all();p=score>=0;cat=np.where(y==0,np.where(p,1,0),np.where(p,3,2));cm=np.zeros((ng,4));np.add.at(cm,(gi,cat),1);cc=W@cm;tn,fp,fn,tp=cc.T;den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));boot[arm]=(tp*tn-fp*fn)/den;metrics.append(dict(arm=arm,MCC=float(matthews_corrcoef(y,p)),AUPRC=float(average_precision_score(y,score)),Brier=float(np.mean((expit(score)-y)**2)),FN=int(((y==1)&~p).sum()),FP=int(((y==0)&p).sum())))
look={r['arm']:r for r in metrics};comparisons=[]
for a,b in [('baseline4_CysLengthMissingOnly','baseline4_CysLength'),('baseline4_CysLengthFamily','baseline4_CysLengthMissingOnly'),('baseline4_CysLengthNamesOnly','baseline4_CysLength')]:
 d=boot[a]-boot[b];comparisons.append(dict(candidate=a,reference=b,delta_MCC=look[a]['MCC']-look[b]['MCC'],CI95=np.quantile(d,[.025,.975]).tolist(),CI98_3333=np.quantile(d,[.05/6,1-.05/6]).tolist()))
out=dict(metrics=metrics,comparisons=comparisons,vocabularies=vocabs,eligible_curated_token_count=sum(t.startswith('curated:') for r in vocabs for t in r['vocab']),scope='posthoc exploratory metadata check, not biological attribution')
(O/'family_followup_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(O/'family_followup_models.json').write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(O/'family_followup_scores.npz',sample_ids=np.array(ids),**scores);print(json.dumps(out,ensure_ascii=False,indent=2))
