import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
P=Path(__file__).resolve().parent; R=P.parents[1]
sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(P/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
rows=read(R/'analysis/cys_probability_adjustment_20261028/predictions.json')
y=np.array([r['label'] for r in rows]); c=np.array([r['cys_count'] for r in rows]); f=np.array([r['fold'] for r in rows]); p=np.array([r['A_probability'] for r in rows]); groups=np.array([r['group_id'] for r in rows]); t=0.47022256531284556
fixed=np.array([r['C_fixed_probability'] for r in rows]); fixedpred=np.array([r['C_fixed_prediction'] for r in rows]); out=p.copy(); choices=[]
# A group has one shared additive offset. Fit log loss plus L2 penalty on training folds only.
def fit(mask,bounds,dev_n):
 x=p[mask]; z=y[mask]
 def loss(d):
  q=np.clip(x+d,1e-7,1-1e-7)
  return float(np.sum(-z*np.log(q)-(1-z)*np.log1p(-q))/dev_n + 0.01*d*d)
 v=minimize_scalar(loss,bounds=bounds,method='bounded',options={'xatol':1e-10})
 candidates=[bounds[0],bounds[1],float(v.x),0.0];return min(candidates,key=lambda a:(loss(a),abs(a)))
for k in range(5):
 dev=f!=k; ev=~dev
 assert not set(groups[dev])&set(groups[ev])
 lo=fit(dev&(c<=1),(0.,0.15),int(dev.sum())); hi=fit(dev&(c>=6),(-0.15,0.),int(dev.sum()))
 out[ev]=np.clip(p[ev]+np.where(c[ev]<=1,lo,np.where(c[ev]>=6,hi,0)),0,1)
 choices.append({'fold':k,'development_n':int(dev.sum()),'evaluation_n':int(ev.sum()),'low_add':lo,'high_add':hi})
methods={'original':(p,(p>=t).astype(int)),'fixed_previous':(fixed,fixedpred),'learned':(out,(out>=t).astype(int))}
def calc(mask):
 yy=y[mask]; results={}
 for name,(prob,pred) in methods.items():
  z=prob[mask]; d=pred[mask]; correct=methods['original'][1][mask]==yy; new=d==yy
  results[name]={'n':len(yy),'errors':int((~new).sum()),'FN':int(((yy==1)&(d==0)).sum()),'FP':int(((yy==0)&(d==1)).sum()),'MCC':float(matthews_corrcoef(yy,d)) if len(np.unique(yy))==2 else None,'AUPRC':float(average_precision_score(yy,z)) if len(np.unique(yy))==2 else None,'AUROC':float(roc_auc_score(yy,z)) if len(np.unique(yy))==2 else None,'Brier':float(np.mean((z-yy)**2)),'corrected':int((~correct&new).sum()),'harmed':int((correct&~new).sum())}
 return results
panels={'all':np.ones(len(y),dtype=bool),'low_pos':(c<=1)&(y==1),'low_neg':(c<=1)&(y==0),'high_pos':(c>=6)&(y==1),'high_neg':(c>=6)&(y==0)}
metrics={k:calc(v) for k,v in panels.items()}; folds={str(k):calc(f==k) for k in range(5)}
save('selection.json',choices); save('metrics.json',metrics); save('fold_metrics.json',folds)
save('predictions.json',[{'sample_id':r['sample_id'],'fold':int(f[i]),'cys_count':int(c[i]),'label':int(y[i]),'original_probability':float(p[i]),'learned_probability':float(out[i]),'original_prediction':int(methods['original'][1][i]),'learned_prediction':int(methods['learned'][1][i])} for i,r in enumerate(rows)])
save('protocol.json',{'method':'per-group additive offsets learned using BCE plus 0.01 * sum(offset^2), bounded low [0,0.15] and high [-0.15,0]','threshold':t,'test1_test2_used':False,'training':'4-fold development, 1-fold evaluation; historical upstream predictions are not nested','n':len(y)})
print(json.dumps({'selection':choices,'metrics':metrics},indent=2))
