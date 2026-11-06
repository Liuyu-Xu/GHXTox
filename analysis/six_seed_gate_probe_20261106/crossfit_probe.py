"""Post-amendment cross-calibration and discrete gate feasibility; never fits outer labels."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
from run_probe import R,O,B,ORIGINAL,TH,collect,prob,prior,sha,score,counts

folds=[];out=[]
for fold in range(5):
 dev=collect(fold,'calibration');test=collect(fold,'test');yy=dev['y'];g=dev['groups'];a,b=next(StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20261106+fold).split(np.zeros(len(yy)),yy,g));assert set(g[a]).isdisjoint(set(g[b])) and set(g).isdisjoint(set(test['groups']))
 q=np.empty(len(yy));s=np.empty(len(yy))
 for train,predict in ((a,b),(b,a)):
  c1=prior.fit(dev['arr'].mean(axis=0)[train],yy[train]);c3=prior.fit(dev['raw3'][train],yy[train]);q[predict]=prob(dev['arr'].mean(axis=0)[predict],c1);s[predict]=prob(dev['raw3'][predict],c3)
 c1=prior.fit(dev['arr'].mean(axis=0),yy);c3=prior.fit(dev['raw3'],yy);tq=prob(test['arr'].mean(axis=0),c1);ts=prob(test['raw3'],c3)
 dq=q>=TH;ds=s>=TH;dis=dq!=ds;target=(ds==yy).astype(int);train=np.flatnonzero(dis);t1=tq>=TH;t3=ts>=TH;tdis=t1!=t3;ti=np.flatnonzero(tdis);tz=(t3[ti]==test['y'][ti]).astype(int)
 tg={str(k):len(set(g[train][target[train]==k])) for k in (0,1)};eligible=min(tg.values())>=10
 grid=sorted(set(np.round(np.arange(0,1.01,.1),10))|{.28});weight=max((matthews_corrcoef(yy,((1-w)*q+w*s)>=TH),-abs(w-.28),-w,w) for w in grid)[3]
 def features(q,s,std,length,cys,extra):
  x=np.column_stack((q,s,std));return np.column_stack((x,length,(cys==0).astype(float),(cys>=7).astype(float))) if extra else x
 predictions={}; entry={'fold':fold,'calibration_n':len(yy),'calibration_crossfit_group_a':len(a),'calibration_crossfit_group_b':len(b),'disagreement_train_n':len(train),'train_target_counts':{str(k):int((target[train]==k).sum()) for k in (0,1)},'train_target_unique_groups':tg,'fit_allowed':eligible,'outer_n':len(test['y']),'outer_disagreement_n':len(ti),'outer_3d_correct':int(tz.sum()),'outer_1d_correct':int(len(tz)-tz.sum()),'calibration_full':{'1d':c1,'3d':c3},'methods':{}}
 entry['selected_fixed_3d_weight']=weight
 margin=np.abs(ts[ti]-TH)-np.abs(tq[ti]-TH);entry['methods']['margin']={'auc':score(tz,margin),**counts(tz,margin>0)}
 if eligible:
  for name,extra in [('probability',False),('probability_cys_length',True)]:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,random_state=20261106));model.fit(features(q,s,dev['std'],dev['length'],dev['cys'],extra)[train],target[train]);vv=model.predict_proba(features(tq,ts,test['std'],test['length'],test['cys'],extra)[ti])[:,1];predictions[name]=vv;entry['methods'][name]={'auc':score(tz,vv),**counts(tz,vv>.5)}
 for i in range(len(test['y'])):
  pos=np.searchsorted(ti,i);is_dis=bool(tdis[i]);assert not is_dis or ti[pos]==i
  selects={name:bool(predictions[name][pos]>.5) if is_dis and eligible else False for name in ('probability','probability_cys_length')};selects['margin']=bool(abs(ts[i]-TH)>abs(tq[i]-TH)) if is_dis else False
  row=test['rows'][i];vals={'fold':fold,'sample_id':row['sample_id'],'group_id':row['group_id'],'sequence':row['sequence'],'label':int(test['y'][i]),'cys':int(test['cys'][i]),'length':int(test['length'][i]),'p_sequence':float(tq[i]),'p_structure':float(ts[i]),'seed_std_raw':float(test['std'][i]),'disagreement':int(is_dis),'structure_correct_target':int(t3[i]==test['y'][i]) if is_dis else '', 'seq_pred':int(t1[i]),'fixed_probability':float(.72*tq[i]+.28*ts[i]),'fixed_pred':int(.72*tq[i]+.28*ts[i]>=TH),'selected_fixed_3d_weight':weight,'selected_fixed_pred':int((1-weight)*tq[i]+weight*ts[i]>=TH)}
  for name in ('probability','probability_cys_length','margin'):
   vals[name+'_choose_3d']=int(selects[name]);vals[name+'_pred']=int(t3[i] if selects[name] else t1[i]);vals[name+'_score']=float(predictions[name][pos]) if name in predictions and is_dis else ''
  out.append(vals)
 folds.append(entry)
assert len(out)==6387 and len({r['sample_id'] for r in out})==6387
with (O/'crossfit_outer_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
y=np.array([r['label'] for r in out]);groups=np.array([r['group_id'] for r in out]);assert all(len({r['fold'] for r in out if r['group_id']==g})==1 for g in set(groups));p=np.array([r['p_sequence'] for r in out]);struct=np.array([r['p_structure'] for r in out]);seq_pred=np.array([r['seq_pred'] for r in out]);fixed_pred=np.array([r['fixed_pred'] for r in out]);rng=np.random.default_rng(20261106);uniq=np.unique(groups);chunks=[np.flatnonzero(groups==g) for g in uniq];inds=[np.concatenate([chunks[k] for k in rng.integers(len(chunks),size=len(chunks))]) for _ in range(2000)]
methods={}
for name, col in [('sequence','seq_pred'),('fixed_028','fixed_pred'),('selected_fixed','selected_fixed_pred'),('margin','margin_pred'),('probability','probability_pred'),('probability_cys_length','probability_cys_length_pred')]:
 d=np.array([r[col] for r in out]);correct=(d==y);base=(seq_pred==y);changes=d!=seq_pred;ds=[]
 if name=='sequence':ranking=p
 elif name=='fixed_028':ranking=.72*p+.28*struct
 elif name=='selected_fixed':
  weight=np.array([r['selected_fixed_3d_weight'] for r in out]);ranking=(1-weight)*p+weight*struct
 else:ranking=np.where(np.array([r[name+'_choose_3d'] for r in out],dtype=bool),struct,p)
 for ix in inds:ds.append(matthews_corrcoef(y[ix],d[ix])-matthews_corrcoef(y[ix],seq_pred[ix]))
 subgroups={}
 for cat,cond in [('zero',lambda r:r['cys']==0),('seven_plus',lambda r:r['cys']>=7)]:
  for label in (0,1):
   ix=np.array([i for i,r in enumerate(out) if cond(r) and r['label']==label]);subgroups[f'{cat}_label{label}']={'n':len(ix),'seq_errors':int((seq_pred[ix]!=y[ix]).sum()),'errors':int((d[ix]!=y[ix]).sum()),'corrects':int((correct[ix]&~base[ix]).sum()),'harms':int((~correct[ix]&base[ix]).sum())}
 methods[name]={'mcc':float(matthews_corrcoef(y,d)),'auroc':float(roc_auc_score(y,ranking)),'auprc':float(average_precision_score(y,ranking)),'sensitivity':float(d[y==1].mean()),'specificity':float((1-d[y==0]).mean()),'errors':int((~correct).sum()),'corrects_vs_sequence':int((correct&~base).sum()),'harms_vs_sequence':int((~correct&base).sum()),'mcc_delta_vs_sequence_ci95':np.quantile(ds,[.025,.975]).tolist(),'subgroups':subgroups}
comparisons={}
for col in ('margin_pred','probability_pred','probability_cys_length_pred'):
 d=np.array([r[col] for r in out]);comparisons[col]={}
 for baseline in ('fixed_pred','selected_fixed_pred'):
  base=np.array([r[baseline] for r in out]);delta=np.array([matthews_corrcoef(y[ix],d[ix])-matthews_corrcoef(y[ix],base[ix]) for ix in inds]);comparisons[col][baseline]={'delta_mcc':float(matthews_corrcoef(y,d)-matthews_corrcoef(y,base)),'group_bootstrap_ci95':np.quantile(delta,[.025,.975]).tolist()}
report={'protocol':'CROSSFIT_PROTOCOL.md','seed':20261106,'n_outer':len(out),'folds':folds,'methods':methods,'comparisons':comparisons,'notes':['post-hoc calibration cross-fitting; no Test1/Test2; neither deployed nor continuous adaptive fusion','gate ranking uses chosen branch score; it is not verified as calibrated probability','pooled fixed predictions group bootstrap ignores training, model selection and previous exploration uncertainty','old 3D upstream lineage is not certified by this exercise']}
(O/'crossfit_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({'folds':folds,'methods':methods},indent=2))
