"""Cross-fit a postprocessor on historical original-model OOF; base model is NOT nested."""
import json,sys,hashlib
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'analysis/repository_sync_20261026/validation_numpy1_clean'))
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score,log_loss
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(P/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
protocol=read(P/'protocol.json');mp=R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json';bp=R/'analysis/error_evidence_linkage_20261025/all_training_linked.json';meta=read(mp);base=read(bp);assert len(meta)==len(base)==6387
for m,b in zip(meta,base):assert m['sample_id']==b['sample_id'] and m['sequence']==b['sequence'] and m['label']==int(b['label']) and int(m['fold'])==int(b['fold'])
y=np.array([m['label'] for m in meta],dtype=int);c=np.array([m['cys_count'] for m in meta]);fold=np.array([int(m['fold']) for m in meta]);p=np.array([float(b['p_fusion']) for b in base]);assert np.all((p>=0)&(p<=1));assert all(abs(float(b['p_fusion'])-(.72*float(b['p_seq'])+.28*float(b['p_struct'])))<2e-7 for b in base)
flags=np.array([bool(m['uniform_flag']) for m in meta]);groups=np.array([m['group_id'] for m in meta]);assert len({m['sample_id'] for m in meta})==6387
for f in range(5):assert not set(groups[fold==f])&set(groups[fold!=f])
thresholds=np.array(sorted(set(protocol['candidates']['thresholds'])));original=float(protocol['threshold']);pred={k:np.zeros(len(y),dtype=int) for k in protocol['arms']};prob={k:p.copy() for k in protocol['arms']};selected=[]
def mcc_vec(tn,fp,fn,tp):
 den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));return np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,dtype=float),where=den>0)
def scores(y,z,th):
 idx=np.argsort(z);z=z[idx];ys=y[idx];i=np.searchsorted(z,th,side='left');cum=np.concatenate([[0],np.cumsum(ys)]);fn=cum[i].astype(float);tn=i-fn;tp=float(cum[-1])-fn;fp=(len(y)-i)-tp
 return mcc_vec(tn,fp,fn,tp)
def adjust(x,cc,lo,hi):return np.clip(x+np.where(cc<=1,lo,np.where(cc>=6,hi,0)),0,1)
def choose(dev,adjusted,ths,with_shift):
 best=None
 for lo in ([0] if not with_shift else protocol['candidates']['low']):
  for hi in ([0] if not with_shift else protocol['candidates']['high']):
   z=adjust(p[dev],c[dev],lo,hi);s=scores(y[dev],z,ths)
   for t,v in zip(ths,s):
    key=(round(float(v),12),-(abs(lo)+abs(hi)),-abs(float(t)-original),-float(t))
    if best is None or key>best[0]:best=(key,float(lo),float(hi),float(t),float(v))
 return {'low_offset':best[1],'high_offset':best[2],'threshold':best[3],'dev_MCC':best[4]}
for f in range(5):
 dev=fold!=f;ev=fold==f;assert len(set(groups[dev])&set(groups[ev]))==0
 choices={'A':{'low_offset':0.,'high_offset':0.,'threshold':original},'B':choose(dev,False,thresholds,False),'C_fixed':choose(dev,True,np.array([original]),True),'C_joint':choose(dev,True,thresholds,True)}
 for a,v in choices.items():
  prob[a][ev]=adjust(p[ev],c[ev],v['low_offset'],v['high_offset']);pred[a][ev]=(prob[a][ev]>=v['threshold']).astype(int)
 selected.append({'fold':f,'dev_n':int(dev.sum()),'test_n':int(ev.sum()),'choices':choices})
print('selected',json.dumps(selected),flush=True)
def panel(ix):
 o={};yy=y[ix];both=len(np.unique(yy))==2
 for a in protocol['arms']:
  z=prob[a][ix];d=pred[a][ix];tp=int(((yy==1)&(d==1)).sum());tn=int(((yy==0)&(d==0)).sum());fp=int(((yy==0)&(d==1)).sum());fn=int(((yy==1)&(d==0)).sum())
  o[a]={'n':len(yy),'TP':tp,'TN':tn,'FP':fp,'FN':fn,'errors':fp+fn,'sensitivity':tp/(tp+fn) if tp+fn else None,'specificity':tn/(tn+fp) if tn+fp else None,'MCC':float(matthews_corrcoef(yy,d)) if both else None,'AUPRC':float(average_precision_score(yy,z)) if both else None,'AUROC':float(roc_auc_score(yy,z)) if both else None,'Brier':float(np.mean((z-yy)**2)) if len(yy) else None,'log_loss':float(log_loss(yy,np.clip(z,1e-7,1-1e-7),labels=[0,1])) if len(yy) else None}
 return o
panels={'all':np.ones(len(y),dtype=bool),'low_cys_positive':(c<=1)&(y==1),'high_cys_negative':(c>=6)&(y==0),'middle_cys':(c>=2)&(c<=5),'flagged':flags,'unflagged_not_gold':~flags,'low_pos_flagged':(c<=1)&(y==1)&flags,'low_pos_unflagged':(c<=1)&(y==1)&~flags,'high_neg_flagged':(c>=6)&(y==0)&flags,'high_neg_unflagged':(c>=6)&(y==0)&~flags}
metrics={name:panel(ix) for name,ix in panels.items()};byfold={str(f):panel(fold==f) for f in range(5)};comparison={}
for name,ix in panels.items():
 comparison[name]={}
 for a in ['B','C_fixed','C_joint']:
  correctA=pred['A'][ix]==y[ix];correctB=pred[a][ix]==y[ix]
  comparison[name][a]={'corrected':int((~correctA&correctB).sum()),'harmed':int((correctA&~correctB).sum())}
# Fold-stratified resampling of sequence clusters; conditional on fitted postprocessors.
buckets=defaultdict(dict)
for i,(f,g) in enumerate(zip(fold,groups)):buckets[int(f)].setdefault(g,[]).append(i)
arrs=[]
for f,gs in buckets.items():
 counts=np.zeros((len(gs),len(pred),4),dtype=np.float64)
 for j,inds in enumerate(gs.values()):
  for k,a in enumerate(pred):np.add.at(counts[j,k],2*y[inds]+pred[a][inds],1)
 arrs.append(counts)
rng=np.random.default_rng(20261028);boot=np.zeros((2000,len(pred),4))
for q in arrs:boot+=(rng.multinomial(len(q),np.full(len(q),1/len(q)),size=2000)@q.reshape(len(q),-1)).reshape(2000,len(pred),4)
tn,fp,fn,tp=[boot[...,i] for i in range(4)];mb=mcc_vec(tn,fp,fn,tp);names=list(pred);ci={a:np.quantile(mb[:,names.index(a)]-mb[:,names.index('A')],[.025,.975]).tolist() for a in ['B','C_fixed','C_joint']}
rows=[{'sample_id':m['sample_id'],'fold':int(fold[i]),'group_id':m['group_id'],'label':int(y[i]),'cys_count':int(c[i]),'flag':bool(flags[i]),**{a+'_probability':float(prob[a][i]) for a in prob},**{a+'_prediction':int(pred[a][i]) for a in pred}} for i,m in enumerate(meta)]
save('predictions.json',rows);save('selection.json',selected);save('metrics.json',metrics);save('fold_metrics.json',byfold);save('comparisons.json',comparison);save('bootstrap.json',ci)
save('verification.json',{'passed':True,'n':len(y),'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [mp,bp,P/'protocol.json']},'original_fusion_reconstructed':True,'same_sample_label_fold_and_group':True,'outer_group_isolation':True,'A_errors':metrics['all']['A']['errors'],'A_errors_expected_327':metrics['all']['A']['errors']==327,'threshold_only_AUPRC_unchanged':metrics['all']['A']['AUPRC']==metrics['all']['B']['AUPRC'],'Test1_Test2_unused':True,'upstream_full_pipeline_independence_not_established':True})
print('ALL',json.dumps(metrics['all'],ensure_ascii=False),flush=True);print('GROUPS',json.dumps({k:metrics[k] for k in ['low_cys_positive','high_cys_negative']},ensure_ascii=False),flush=True);print('CI',json.dumps(ci),flush=True)
