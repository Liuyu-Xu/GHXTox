"""Frozen, retrospective full GHXTox + SaProt OOF comparisons; no NN fit."""
from pathlib import Path
import json, csv, hashlib, os, sys
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis'
sys.path.insert(0,str(A/'repository_sync_20261026/validation_numpy1_clean'))
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.special import expit,logit,log_expit,logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
H=A/'upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def csvread(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return x.item()
    return x
def save(n,d):(O/n).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

P=read(O/'protocol.json')
paths=[O/'protocol.json',A/'error_evidence_linkage_20261025/all_training_linked.json',
 H/'runs/structure_candidates_20261022/report/S3_plus_no_inner_transformer_oof.csv',
 H/'runs/structure_candidates_20261022/protocol/nested_roles.csv',
 A/'saprot_standalone_20261026/ensemble_scores.npz',A/'saprot_standalone_20261026/oof_predictions.json',
 A/'saprot_explanation_20261026/crossfit_scores.npz',
 A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz']
hashes={str(p):sha(p) for p in paths};save('input_manifest.json',hashes)
z=np.load(paths[-1]);ids=z['sample_ids'].tolist();seq=z['sequences'].tolist();y=z['labels'].astype(int);fold=z['folds'].astype(int);groups=z['groups'].astype(str);N=len(ids);ix={s:i for i,s in enumerate(ids)}
assert N==6387 and len(set(ids))==N
meta=read(paths[1]);oldmap={x['sample_id']:x for x in meta};newmap={x['sample_id']:x for x in csvread(paths[2])};assert set(oldmap)==set(newmap)==set(ids)
for k in ids:
    j=ix[k];a,b=oldmap[k],newmap[k]
    assert a['sequence']==b['sequence']==seq[j]
    assert int(a['label'])==int(b['label'])==y[j]
    assert int(a['fold'])==int(b['outer_fold'])==fold[j]
    assert a['group_id']==groups[j]
c=np.load(paths[-2]);assert c['sample_ids'].tolist()==ids and np.array_equal(c['labels'],y) and np.array_equal(c['folds'],fold) and np.array_equal(c['groups'].astype(str),groups)
halves=c['halves'].astype(int)
s=np.load(paths[4]);assert s['sample_ids'].tolist()==ids and np.array_equal(s['labels'],y)
raw_probs={'old':np.array([float(oldmap[k]['p_fusion']) for k in ids]),'new':np.array([float(newmap[k]['probability']) for k in ids])}
thresholds={'old':np.full(N,.47022256531284556),'new':np.array([float(newmap[k]['threshold']) for k in ids])}
scores={v:logit(np.clip(p,1e-12,1-1e-12))-logit(thresholds[v]) for v,p in raw_probs.items()}
for mode in ['masked','3Di']:scores['teacher_'+mode]=s['SaProt_only_'+mode].astype(float)
teacher_rows=read(paths[5]);seed_logits={}
for mode in ['masked','3Di']:
    for seed in [20261026,20261027]:
        rr=[x for x in teacher_rows if x['arm']=='SaProt_only_'+mode and x['seed']==seed]
        assert len(rr)==N and len({x['sample_id'] for x in rr})==N
        arr=np.zeros(N)
        for x in rr:
            j=ix[x['sample_id']];assert x['label']==y[j] and x['fold']==fold[j];arr[j]=x['logit']
        name=f'teacher_{mode}_seed{seed}';scores[name]=arr;seed_logits[mode,seed]=arr
    stack=np.stack([seed_logits[mode,k] for k in [20261026,20261027]])
    stable=logsumexp(log_expit(stack),axis=0)-logsumexp(log_expit(-stack),axis=0)
    assert np.max(np.abs(stable-scores['teacher_'+mode]))<1e-10

# Audit available model training membership without loading a network.
import torch
audit_models=[]
for mode in ['masked','3Di']:
    for seed in [20261026,20261027]:
        for f in range(5):
            p=A/f'saprot_standalone_20261026/checkpoints/{seed}_fold{f}_SaProt_only_{mode}.pt'
            d=torch.load(p,map_location='cpu',weights_only=False)
            train=set(d['train_ids']);test=set(d['test_ids'])
            assert train==set(np.array(ids)[fold!=f]) and test==set(np.array(ids)[fold==f]) and not train&test
            assert not {groups[ix[k]] for k in train}&{groups[ix[k]] for k in test}
            audit_models.append({'mode':mode,'seed':seed,'fold':f,'train_n':len(train),'test_n':len(test),'train_test_group_disjoint':True,'checkpoint':str(p)})
            del d
roles=csvread(paths[3]);role_summary={}
for f in range(5):
    rr=[x for x in roles if int(x['outer_fold'])==f];assert len(rr)==N
    byrole={k:{x['sample_id'] for x in rr if x['role']==k} for k in ['train','validation','calibration','test']}
    assert byrole['test']==set(np.array(ids)[fold==f])
    assert set.union(*byrole.values())==set(ids)
    rg={k:{groups[ix[j]] for j in v} for k,v in byrole.items()}
    for a in rg:
        for b in rg:
            if a!=b:assert not rg[a]&rg[b]
    role_summary[f]={k:len(v) for k,v in byrole.items()}
audit={'sample_sequence_label_outerfold_group_alignment':True,'n':N,'groups':len(set(groups)),
       'saved_SaProt_models_membership_audited':audit_models,'new_structure_roles':role_summary,
       'source_caveats':['Old full predictions used global OOF calibration/selection and repeated development.',
       'New structure has inner train/validation/calibration roles; SaProt used all other outer folds for fixed20epoch fitting: not same training budget.',
       'New fusion reused original sequence OOF. Original inner-calibration-row sequence predictors may have trained on current outer fold; original training memberships unavailable for full forensic verification. Do not call new fusion fully independent nested validation.',
       'New global study candidates already chosen using development data; new withinfold calibration does not remove historical use.',
       'No equivalent extra full-GHXTox-member or six-sequence-seed saved control supplied; cannot rule out ensemble size.']}
save('source_audit.json',audit)
print('Alignment and twenty SaProt checkpoint membership audits passed',flush=True)

cal={k:np.zeros(N) for k in scores};models=[]
for f in range(5):
    for h in [0,1]:
        tr=(fold==f)&(halves!=h);te=(fold==f)&(halves==h)
        assert not set(groups[tr])&set(groups[te]);assert set(np.unique(y[tr]))=={0,1}
        for arm,x in scores.items():
            sc=StandardScaler().fit(x[tr,None]);m=LogisticRegression(C=1,solver='lbfgs',max_iter=3000).fit(sc.transform(x[tr,None]),y[tr])
            assert m.coef_[0,0]>0
            cal[arm][te]=expit(m.decision_function(sc.transform(x[te,None])))
            models.append({'arm':arm,'fold':f,'evaluation_half':h,'n_fit':int(tr.sum()),'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'coefficient':m.coef_.tolist(),'intercept':m.intercept_.tolist()})
        print(f'fold{f} evalhalf{h}: {tr.sum()} calibration / {te.sum()} evaluation',flush=True)

probs={};preds={};comparisons=[];seed_results={}
for v in ['old','new']:
    probs[v+'_raw']=raw_probs[v];preds[v+'_raw']=(raw_probs[v]>=thresholds[v]).astype(int)
    probs[v+'_cal']=cal[v];preds[v+'_cal']=(cal[v]>=.5).astype(int)
    for mode in ['masked','3Di']:
        for w in [.2,.5]:
            arm=f'{v}_{mode}_w{int(w*100)}';probs[arm]=(1-w)*cal[v]+w*cal['teacher_'+mode];preds[arm]=(probs[arm]>=.5).astype(int)
            comparisons.append({'candidate':arm,'baseline':v+'_cal','weight_SaProt':w,'version':v,'mode':mode})
        for seed in [20261026,20261027]:
            name=f'{v}_{mode}_w20_seed{seed}';probs[name]=.8*cal[v]+.2*cal[f'teacher_{mode}_seed{seed}'];preds[name]=(probs[name]>=.5).astype(int)

mem=read(A/'classification_mechanism_map_20261026/membrane76_comparison_pool.json')['all76']
memids={x['sample_id'] if isinstance(x,dict) else x for x in mem};assert len(memids)==76 and memids<=set(ids)
cys=np.array([s.count('C') for s in seq]);strata={'lowCys_positive':(cys<4)&(y==1),'membrane76_annotation_only':np.array([s in memids for s in ids])}
def metric(p,pred,mask=None):
    mask=np.ones(N,bool) if mask is None else mask;yy=y[mask];pp=p[mask];dd=pred[mask];both=len(set(yy))==2
    return {'n':int(mask.sum()),'MCC':float(matthews_corrcoef(yy,dd)) if both else None,'AUPRC':float(average_precision_score(yy,pp)) if both else None,'AUROC':float(roc_auc_score(yy,pp)) if both else None,'Brier':float(np.mean((pp-yy)**2)),
            'TP':int(((yy==1)&(dd==1)).sum()),'TN':int(((yy==0)&(dd==0)).sum()),'FP':int(((yy==0)&(dd==1)).sum()),'FN':int(((yy==1)&(dd==0)).sum())}
metrics={k:metric(p,preds[k]) for k,p in probs.items()}
foldmetrics={k:{f:metric(p,preds[k],fold==f) for f in range(5)} for k,p in probs.items()}
submetrics={k:{s:metric(p,preds[k],m) for s,m in strata.items()} for k,p in probs.items()}
for comp in comparisons:
    ca,ba=comp['candidate'],comp['baseline'];d=preds[ca];b=preds[ba]
    comp['corrected']=int(((d==y)&(b!=y)).sum());comp['harmed']=int(((d!=y)&(b==y)).sum());comp['net_fewer_errors']=comp['corrected']-comp['harmed']
    comp['delta_MCC']=metrics[ca]['MCC']-metrics[ba]['MCC'];comp['delta_AUPRC']=metrics[ca]['AUPRC']-metrics[ba]['AUPRC']
    comp['fold_net_fewer_errors']=[int(((preds[ba]!=y)&(fold==f)).sum()-((preds[ca]!=y)&(fold==f)).sum()) for f in range(5)]
    comp['delta_to_raw_MCC']=metrics[ca]['MCC']-metrics[comp['version']+'_raw']['MCC']

# Paired original-group bootstrap on saved conditional predictions.
ug,gi=np.unique(groups,return_inverse=True);G=len(ug);armnames=list(probs);counts=[]
for arm in armnames:
    d=preds[arm];counts.append(np.stack([np.bincount(gi,weights=cond,minlength=G) for cond in [(y==1)&(d==1),(y==0)&(d==0),(y==0)&(d==1),(y==1)&(d==0)]],axis=1))
counts=np.stack(counts,axis=1).reshape(G,-1);rng=np.random.default_rng(20261027);boot=[]
for start in range(0,2000,100):
    mult=rng.multinomial(G,np.full(G,1/G),size=min(100,2000-start));a=(mult@counts).reshape(-1,len(armnames),4)
    tp,tn,fp,fn=[a[:,:,j] for j in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));boot.append(np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0))
boot=np.concatenate(boot);bd={}
for comp in comparisons:
    delta=boot[:,armnames.index(comp['candidate'])]-boot[:,armnames.index(comp['baseline'])];bd[comp['candidate']]=delta
    comp['delta_MCC_95CI']=np.quantile(delta,[.025,.975]).tolist();comp['delta_MCC_99_375CI']=np.quantile(delta,[.003125,.996875]).tolist()
    v,mode=comp['version'],comp['mode']
    seeds=[metrics[f'{v}_{mode}_w20_seed{k}']['MCC']-metrics[v+'_cal']['MCC'] for k in [20261026,20261027]]
    comp['single_teacher_seed_w20_delta_MCC']=seeds
    comp['primary_masked20_gate']=bool(mode=='masked' and comp['weight_SaProt']==.2 and comp['delta_MCC']>=.005 and comp['delta_MCC_99_375CI'][0]>0 and sum(x>=0 for x in comp['fold_net_fewer_errors'])>=4 and comp['delta_AUPRC']>=-.002 and min(seeds)>=0)

rows=[]
for j,k in enumerate(ids):
    row={'sample_id':k,'sequence':seq[j],'label':int(y[j]),'fold':int(fold[j]),'group_id':groups[j],'calibration_half':int(halves[j]),'name':oldmap[k].get('name','')}
    for v in ['old','new']:row[v+'_raw_threshold']=float(thresholds[v][j])
    for arm in probs:
        row[arm+'_probability']=float(probs[arm][j]);row[arm+'_prediction']=int(preds[arm][j])
    for comp in comparisons:
        ca,ba=comp['candidate'],comp['baseline'];b=preds[ba][j];d=preds[ca][j]
        row[ca+'_role']='corrected' if b!=y[j] and d==y[j] else 'harmed' if b==y[j] and d!=y[j] else 'both_wrong' if d!=y[j] else 'both_correct'
    rows.append(row)
save('all_sample_predictions.json',rows);save('calibration_models.json',models);save('metrics.json',metrics);save('fold_metrics.json',foldmetrics);save('subgroup_metrics.json',submetrics);save('comparisons.json',comparisons)
np.savez_compressed(O/'crossfit_predictions.npz',sample_ids=np.array(ids),labels=y,folds=fold,groups=groups,halves=halves,**{k+'_prob':v for k,v in probs.items()},**{k+'_pred':v for k,v in preds.items()},**{k+'_calibrated_component':v for k,v in cal.items()})
np.savez_compressed(O/'bootstrap_deltas.npz',**bd)
assert hashes=={str(p):sha(p) for p in paths}
save('execution_summary.json',{'n':N,'calibration_fits':len(models),'neural_training':False,'external_tests_used':False,'source_hashes_unchanged':True,'comparisons':comparisons,'all_saved':True})
print(json.dumps(clean({'metrics':{k:v for k,v in metrics.items() if '_seed' not in k},'comparisons':comparisons}),ensure_ascii=False,indent=2),flush=True)
