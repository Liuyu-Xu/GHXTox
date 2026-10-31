from pathlib import Path
import json,collections
import numpy as np
from sklearn.metrics import matthews_corrcoef,roc_auc_score,average_precision_score
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')
preds={};runs={}
for arm in ['A','B','C']:
 rows=[];records=[]
 for fold in range(5):
  f=P/'runs/seed42'/f'fold{fold}'/arm/'result.json'
  if not f.exists():raise SystemExit('Not complete: '+str(f))
  r=read(f);records.append(r);rows+=r['predictions']
 rows.sort(key=lambda x:x['source_index'])
 assert len(rows)==6387 and [r['source_index'] for r in rows]==list(range(6387))
 for r,m in zip(rows,meta):assert r['sample_id']==m['sample_id'] and r['label']==m['label'] and r['fold']==m['fold']
 preds[arm]=rows;runs[arm]=records
save('all_oof_predictions.json',preds)
y=np.array([r['label'] for r in meta])
def metrics(idx,prob,decision):
 yy=y[idx];pp=prob[idx];dd=decision[idx]
 d={'n':len(idx),'positive_n':int(yy.sum()),'TP':int(((dd==1)&(yy==1)).sum()),'TN':int(((dd==0)&(yy==0)).sum()),'FP':int(((dd==1)&(yy==0)).sum()),'FN':int(((dd==0)&(yy==1)).sum()),'errors':int((dd!=yy).sum())}
 d['MCC']=float(matthews_corrcoef(yy,dd)) if len(set(yy.tolist()))==2 else None
 d['AUROC']=float(roc_auc_score(yy,pp)) if len(set(yy.tolist()))==2 else None
 d['AUPRC']=float(average_precision_score(yy,pp)) if len(set(yy.tolist()))==2 else None
 return d
panels={'all':list(range(len(meta))),'primary_flagged':[i for i,m in enumerate(meta) if m['uniform_flag']],'unflagged_not_gold':[i for i,m in enumerate(meta) if not m['uniform_flag']]}
for k in range(4):panels['physical_'+str(k)]=[i for i,m in enumerate(meta) if 'physical_'+str(k) in m['category_keys']]
for family,title in [('belongs to the mcd family','MCD'),('belongs to the non-disulfide-bridged peptide (ndbp) superfamily','NDBP')]:
 panels[title]=[i for i,m in enumerate(meta) if m['label']==1 and m['cys_count']<4 and m['current_family_sidecar']['parent_families']==[family]]
panels['MCD_NDBP']=sorted(panels['MCD']+panels['NDBP'])
for fold in range(5):panels['fold_'+str(fold)]=[i for i,m in enumerate(meta) if m['fold']==fold]
for family in sorted({f for m in meta for f in m['current_family_sidecar']['parent_families']}):
 for label in [0,1]:
  idx=[i for i,m in enumerate(meta) if m['label']==label and m['current_family_sidecar']['parent_families']==[family]]
  if idx:panels[f'family_label{label}:'+family]=idx
scores={};decisions={};results={}
for mode in ['structure','fusion']:
 scores[mode]={};decisions[mode]={};results[mode]={}
 for arm,rows in preds.items():
  prob=np.array([r[mode+'_probability'] for r in rows]);decision=np.array([r[mode+'_prediction'] for r in rows]);scores[mode][arm]=prob;decisions[mode][arm]=decision
  results[mode][arm]={key:metrics(np.array(idx,dtype=int),prob,decision) for key,idx in panels.items()}
save('metrics.json',results)
groupnames=sorted({m['group_id'] for m in meta});gid={g:i for i,g in enumerate(groupnames)};groupidx=np.array([gid[m['group_id']] for m in meta])
def mcc_cm(v):
 tn,fp,fn,tp=[v[...,k] for k in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,dtype=float),where=den>0)
comparisons={}
for mode in ['structure','fusion']:
 cm=[]
 for arm in ['A','B','C']:
  d=decisions[mode][arm];arr=np.zeros((len(groupnames),4),dtype=float)
  np.add.at(arr,(groupidx,2*y+d),1);cm.append(arr)
 flat=np.concatenate(cm,axis=1);rng=np.random.default_rng(20261027);boot=[]
 for start in range(0,2000,100):
  w=rng.multinomial(len(groupnames),np.full(len(groupnames),1/len(groupnames)),size=min(100,2000-start))
  count=(w@flat).reshape(-1,3,4);boot.append(mcc_cm(count))
 boot=np.concatenate(boot)
 comp=[]
 for control,treatment in [('A','B'),('B','C'),('A','C')]:
  c=decisions[mode][control];t=decisions[mode][treatment]
  corr=np.flatnonzero((c!=y)&(t==y));harm=np.flatnonzero((c==y)&(t!=y))
  ai=['A','B','C'].index(control);bi=['A','B','C'].index(treatment);delta=boot[:,bi]-boot[:,ai]
  comp.append({'control':control,'treatment':treatment,'delta_MCC':results[mode][treatment]['all']['MCC']-results[mode][control]['all']['MCC'],'paired_sequence_group_bootstrap_CI95':np.quantile(delta,[.025,.975]).tolist(),'CI97_5_for_two_primary_fusion_contrasts':np.quantile(delta,[.0125,.9875]).tolist() if mode=='fusion' and (control,treatment) in [('A','B'),('B','C')] else None,'corrected':len(corr),'harmed':len(harm),'net_fewer_errors':len(corr)-len(harm),'corrected_ids':[meta[i]['sample_id'] for i in corr],'harmed_ids':[meta[i]['sample_id'] for i in harm],'fold_nets':{str(f):sum(meta[i]['fold']==f for i in corr)-sum(meta[i]['fold']==f for i in harm) for f in range(5)},'MCD_NDBP_corrected':sum(i in panels['MCD_NDBP'] for i in corr),'MCD_NDBP_harmed':sum(i in panels['MCD_NDBP'] for i in harm),'scope':'Conditional historical OOF prediction uncertainty only; does not include training-seed variability or family selection bias.'})
 comparisons[mode]=comp
save('comparisons.json',comparisons)
target=[]
for i in panels['MCD_NDBP']:
 m=meta[i];target.append({'sample_id':m['sample_id'],'name':m['current_family_sidecar']['exact_entries'][0]['protein_name'],'label':m['label'],'fold':m['fold'],'group_id':m['group_id'],'uniform_flag':m['uniform_flag'],'strict_original_label_verified':m['strict_original_label_verified'],'parent_families':m['current_family_sidecar']['parent_families'],'arms':{a:preds[a][i] for a in preds}})
save('all11_target_records.json',target)
save('runtime_summary.json',{'arms':{a:{'seconds':sum(r['seconds'] for r in rr),'selected_epochs':[r['selected_epoch'] for r in rr],'peak_cuda_bytes':max(r['peak_cuda_bytes'] for r in rr)} for a,rr in runs.items()},'nn_fits':15,'seeds':[42],'epochs_each':6,'primary_contrasts':['B-A','C-B'],'server_training_cost':0,'claim':'exploratory actual structural retraining + conditional whole-fusion, no deployment'})
print('FUSION',{a:results['fusion'][a]['all'] for a in preds},flush=True)
print('TARGET',{a:results['fusion'][a]['MCD_NDBP'] for a in preds},flush=True)
print('COMPARISONS',comparisons['fusion'],flush=True)
