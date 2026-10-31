"""Frozen selective-value atlas and honest meta-holdout probes; no neural fit."""
from pathlib import Path
import json,csv,hashlib,os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from scipy.special import expit,logit,log_expit,logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis'
H=A/'upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def cr(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(n,d):(O/n).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def center(p,t):return expit(logit(np.clip(p,1e-12,1-1e-12))-logit(np.clip(t,1e-12,1-1e-12)))
files=[O/'protocol.json',A/'saprot_fullmodel_fusion_20261027/crossfit_predictions.npz',
 A/'error_evidence_linkage_20261025/all_training_linked.json',
 H/'runs/structure_candidates_20261022/report/S3_plus_no_inner_transformer_oof.csv',
 A/'saprot_standalone_20261026/ensemble_scores.npz',A/'saprot_standalone_20261026/oof_predictions.json',
 A/'saprot_contribution_20261026/sample_contributions.json',A/'saprot_contribution_20261026/matched_controls.json']
manifest={str(p):sha(p) for p in files};save('input_manifest.json',manifest)
z=np.load(files[1]);ids=z['sample_ids'].tolist();y=z['labels'].astype(int);fold=z['folds'].astype(int);groups=z['groups'].astype(str);halves=z['halves'].astype(int);N=len(y);assert N==6387
old={x['sample_id']:x for x in read(files[2])};new={x['sample_id']:x for x in cr(files[3])};meta={x['sample_id']:x for x in read(files[6])};seq=[meta[k]['sequence'] for k in ids]
assert set(old)==set(new)==set(meta)==set(ids)
for j,k in enumerate(ids):
    assert old[k]['sequence']==new[k]['sequence']==seq[j]
    assert int(old[k]['label'])==int(new[k]['label'])==y[j]
    assert int(old[k]['fold'])==int(new[k]['outer_fold'])==fold[j]
    assert old[k]['group_id']==groups[j]
gp={'old':np.array([float(old[k]['p_fusion']) for k in ids]),'new':np.array([float(new[k]['probability']) for k in ids])}
gt={'old':np.full(N,.47022256531284556),'new':np.array([float(new[k]['threshold']) for k in ids])}
gx={v:logit(np.clip(p,1e-12,1-1e-12))-logit(gt[v]) for v,p in gp.items()}
s=np.load(files[4]);assert s['sample_ids'].tolist()==ids and np.array_equal(s['labels'],y)
sx={m:s['SaProt_only_'+m] for m in ['masked','3Di']}
seedrows=read(files[5]);sxseed={}
for mode in sx:
    for seed in [20261026,20261027]:
        mm={x['sample_id']:x['logit'] for x in seedrows if x['seed']==seed and x['arm']=='SaProt_only_'+mode};assert len(mm)==N
        sxseed[mode,seed]=np.array([mm[k] for k in ids])
    st=np.stack([sxseed[mode,k] for k in [20261026,20261027]])
    assert np.max(np.abs(sx[mode]-(logsumexp(log_expit(st),axis=0)-logsumexp(log_expit(-st),axis=0))))<1e-10
ix={k:j for j,k in enumerate(ids)}

# Natural expert decisions: disjoint four-way atlas, not fused outcomes.
atlas={};class_vectors={}
for v in gx:
    dg=(gp[v]>=gt[v]).astype(int)
    for mode in sx:
        ds=(sx[mode]>=0).astype(int);gc=dg==y;sc=ds==y
        cls=np.where(gc,np.where(sc,'both_right','G_right_S_wrong'),np.where(sc,'G_wrong_S_right','both_wrong'))
        class_vectors[v,mode]=cls
        seedsc=np.stack([sxseed[mode,k]>=0 for k in [20261026,20261027]])==y
        result={}
        for name in ['G_wrong_S_right','G_right_S_wrong','both_wrong','both_right']:
            mask=cls==name
            result[name]={'n':int(mask.sum()),'positive':int((mask&(y==1)).sum()),'negative':int((mask&(y==0)).sum()),
                          'fold_counts':[int((mask&(fold==f)).sum()) for f in range(5)],
                          'both_SaProt_seeds_correct':int((mask&seedsc.all(axis=0)).sum()),
                          'both_SaProt_seeds_wrong':int((mask&(~seedsc).all(axis=0)).sum()),
                          'sample_ids':np.array(ids)[mask].tolist()}
        oracle=dg.copy();oracle[(~gc)&sc]=y[(~gc)&sc]
        atlas[f'{v}_{mode}']={'classes':result,'GHXTox_errors':int((~gc).sum()),'SaProt_errors':int((~sc).sum()),
                              'label_aware_oracle_MCC_not_deployable':float(matthews_corrcoef(y,oracle))}
save('source_fourway_atlas.json',atlas)

extra_names=['log_length','cys_fraction','charge_density','hydropathy','aromatic_fraction','proline_fraction','mean_plddt','unmasked_fraction','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation','nonlocal_contact_density']
feat=lambda n:np.array([meta[k]['features'][n] for k in ids],float)
length=feat('sequence_length')
extra=np.column_stack([np.log1p(length),feat('cys_count')/length,feat('net_charge_proxy')/length,
                       feat('mean_hydropathy'),feat('aromatic_fraction'),feat('proline_fraction'),
                       feat('mean_plddt'),feat('unmasked_fraction'),feat('hydrophobic_contact_excess'),
                       feat('positive_hydrophobic_contact_excess'),feat('charge_hydrophobic_separation'),feat('nonlocal_contact_density')])
extra[~np.isfinite(extra)]=np.nan
score_names=['G_probability','S_probability','G_margin','S_margin','absolute_probability_disagreement']
models=[];selections=[];gate_models=[];prob={};decision={};aux={};inner_membership=[]

def scalar_fit(x,tr,te,arm,outer_f,eval_h,inner):
    sc=StandardScaler().fit(x[tr,None]);m=LogisticRegression(C=1,solver='lbfgs',max_iter=3000).fit(sc.transform(x[tr,None]),y[tr])
    assert m.coef_[0,0]>0
    models.append({'arm':arm,'outer_fold':outer_f,'evaluation_half':eval_h,'inner_eval_fold':inner,
                   'mean':sc.mean_.tolist(),'scale':sc.scale_.tolist(),'coefficient':m.coef_.tolist(),'intercept':m.intercept_.tolist(),
                   'fit_ids':np.array(ids)[tr].tolist(),'predict_ids':np.array(ids)[te].tolist()})
    return expit(m.decision_function(sc.transform(x[te,None])))

def best_threshold(p,yy):
    order=np.argsort(p,kind='stable');ps=p[order];ys=yy[order];un,first=np.unique(ps,return_index=True)
    cum=np.r_[0,np.cumsum(ys)];total=yy.sum();neg=len(yy)-total
    tp=total-cum[first];fp=neg-(first-cum[first]);fn=total-tp;tn=neg-fp
    den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mm=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den,float),where=den>0)
    th=np.r_[un,.5,np.nextafter(ps[-1],np.inf)]
    d=p>=.5;t=float(matthews_corrcoef(yy,d)) if len(set(yy))==2 else 0.
    mm=np.r_[mm,t,0.];best=mm.max();ii=np.where(mm>=best-1e-12)[0];j=ii[np.argmin(np.abs(th[ii]-.5))]
    return float(th[j]),float(mm[j])

def gate_fit_predict(Xtr,Xte,target,fitrows,pair,f,h,kind):
    counts=np.bincount(target,minlength=2)
    fallback=len(target)<20 or counts.min()<3
    median=np.nanmedian(Xtr,axis=0) if len(Xtr) else np.zeros(Xte.shape[1]);median=np.where(np.isfinite(median),median,0)
    Xtr=np.where(np.isfinite(Xtr),Xtr,median);Xte=np.where(np.isfinite(Xte),Xte,median)
    rec={'pair':pair,'outer_fold':f,'evaluation_half':h,'kind':kind,'feature_names':score_names+(extra_names if kind=='feature_gate' else []),
         'fit_ids':np.array(ids)[fitrows].tolist(),'target':target.tolist(),'helpful_n':int(target.sum()),'harmful_n':int(len(target)-target.sum()),'fallback':fallback,'median':median.tolist()}
    if fallback:
        q=(float(target.sum())+1)/(len(target)+2);out=np.full(len(Xte),q);rec['constant']=q
    else:
        sc=StandardScaler().fit(Xtr);m=LogisticRegression(C=.1,solver='lbfgs',max_iter=3000).fit(sc.transform(Xtr),target)
        out=m.predict_proba(sc.transform(Xte))[:,1]
        rec.update(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coefficient=m.coef_.tolist(),intercept=m.intercept_.tolist())
    gate_models.append(rec);return out

for v in gx:
    for mode in sx:
        pair=f'{v}_{mode}'
        for method in ['baseline','weight_selected','score_gate','feature_gate']:
            prob[f'{pair}_{method}']=np.zeros(N);decision[f'{pair}_{method}']=np.zeros(N,int)
        pge=np.zeros(N);pse=np.zeros(N);tge=np.zeros(N);tse=np.zeros(N);qmap={k:np.zeros(N) for k in ['score_gate','feature_gate']};choosemap={k:np.zeros(N,bool) for k in qmap}
        for f in range(5):
            for h in [0,1]:
                dev=(fold==f)&(halves!=h);ev=(fold==f)&(halves==h);assert not set(groups[dev])&set(groups[ev])
                devgroups=sorted(set(groups[dev]),key=lambda g:hashlib.sha256((g+'|saprot_selective_inner_v1').encode()).hexdigest());gm={g:j%3 for j,g in enumerate(devgroups)}
                inner=np.array([gm.get(g,-1) for g in groups]);pgtr=np.full(N,np.nan);pstr=np.full(N,np.nan)
                for j in range(3):
                    tr=dev&(inner!=j);va=dev&(inner==j);assert not set(groups[tr])&set(groups[va])
                    pgtr[va]=scalar_fit(gx[v],tr,va,pair+'_G',f,h,j);pstr[va]=scalar_fit(sx[mode],tr,va,pair+'_S',f,h,j)
                pgev=scalar_fit(gx[v],dev,ev,pair+'_G',f,h,-1);psev=scalar_fit(sx[mode],dev,ev,pair+'_S',f,h,-1)
                tg,_=best_threshold(pgtr[dev],y[dev]);ts,_=best_threshold(pstr[dev],y[dev])
                candidates=[]
                for w in np.linspace(0,1,11):
                    t,mc=best_threshold((1-w)*pgtr[dev]+w*pstr[dev],y[dev]);candidates.append({'weight_SaProt':float(w),'threshold':t,'development_inner_MCC':mc})
                best=max(x['development_inner_MCC'] for x in candidates);choice=next(x for x in candidates if x['development_inner_MCC']>=best-1e-12)
                w,t=choice['weight_SaProt'],choice['threshold']
                selections.append({'pair':pair,'outer_fold':f,'evaluation_half':h,'development_ids':np.array(ids)[dev].tolist(),'evaluation_ids':np.array(ids)[ev].tolist(),
                                   'G_threshold':tg,'S_threshold':ts,'selected':choice,'all_candidates':candidates})
                dgt=pgtr>=tg;dst=pstr>=ts;gatefit=dev&(dgt!=dst);target=(dst[gatefit]==y[gatefit]).astype(int)
                dge=pgev>=tg;dse=psev>=ts
                Xtr=np.column_stack([pgtr[gatefit],pstr[gatefit],pgtr[gatefit]-tg,pstr[gatefit]-ts,np.abs(pgtr[gatefit]-pstr[gatefit])])
                Xev=np.column_stack([pgev,psev,pgev-tg,psev-ts,np.abs(pgev-psev)])
                gb=center(pgev,tg);sb=center(psev,ts);pb=(1-w)*pgev+w*psev
                prob[pair+'_baseline'][ev]=gb;decision[pair+'_baseline'][ev]=dge
                prob[pair+'_weight_selected'][ev]=center(pb,t);decision[pair+'_weight_selected'][ev]=pb>=t
                pge[ev]=pgev;pse[ev]=psev;tge[ev]=tg;tse[ev]=ts
                for kind in qmap:
                    xx,xe=(Xtr,Xev) if kind=='score_gate' else (np.column_stack([Xtr,extra[gatefit]]),np.column_stack([Xev,extra[ev]]))
                    q=gate_fit_predict(xx,xe,target,gatefit,pair,f,h,kind);use=(dge!=dse)&(q>.5)
                    prob[pair+'_'+kind][ev]=np.where(use,sb,gb);decision[pair+'_'+kind][ev]=np.where(use,dse,dge)
                    qmap[kind][ev]=q;choosemap[kind][ev]=use
                print(f'{pair} fold{f} evalhalf{h}: dev{dev.sum()} eval{ev.sum()} gatefit{gatefit.sum()} selectedw{w:.1f}',flush=True)
        aux[pair]={'G_probability':pge,'S_probability':pse,'G_threshold':tge,'S_threshold':tse,**{k+'_q':a for k,a in qmap.items()},**{k+'_choose_S':a for k,a in choosemap.items()}}

def metrics(score,d,mask=None):
    mask=np.ones(N,bool) if mask is None else mask;yy=y[mask];s=score[mask];dd=d[mask];both=len(set(yy))==2
    return {'n':int(mask.sum()),'MCC':float(matthews_corrcoef(yy,dd)) if both else None,'AUPRC':float(average_precision_score(yy,s)) if both else None,'AUROC':float(roc_auc_score(yy,s)) if both else None,
            'TP':int(((yy==1)&(dd==1)).sum()),'TN':int(((yy==0)&(dd==0)).sum()),'FP':int(((yy==0)&(dd==1)).sum()),'FN':int(((yy==1)&(dd==0)).sum())}
met={k:metrics(a,decision[k]) for k,a in prob.items()};fm={k:{f:metrics(a,decision[k],fold==f) for f in range(5)} for k,a in prob.items()}
contrasts=[]
for v in gx:
    for mode in sx:
        pair=f'{v}_{mode}'
        for method,baseline in [('weight_selected','baseline'),('score_gate','baseline'),('feature_gate','baseline'),('feature_gate','weight_selected'),('feature_gate','score_gate')]:
            a,b=pair+'_'+method,pair+'_'+baseline;da,db=decision[a],decision[b]
            contrasts.append({'candidate':a,'baseline':b,'delta_MCC':met[a]['MCC']-met[b]['MCC'],'delta_AUPRC':met[a]['AUPRC']-met[b]['AUPRC'],
                              'corrected':int(((da==y)&(db!=y)).sum()),'harmed':int(((da!=y)&(db==y)).sum()),
                              'fold_net_fewer_errors':[int(((db!=y)&(fold==f)).sum()-((da!=y)&(fold==f)).sum()) for f in range(5)]})
ug,gi=np.unique(groups,return_inverse=True);G=len(ug);names=list(prob);counts=[]
for k in names:
    d=decision[k];counts.append(np.stack([np.bincount(gi,weights=a,minlength=G) for a in [(y==1)&(d==1),(y==0)&(d==0),(y==0)&(d==1),(y==1)&(d==0)]],axis=1))
counts=np.stack(counts,axis=1).reshape(G,-1);rng=np.random.default_rng(20261028);boots=[]
for _ in range(20):
    a=(rng.multinomial(G,np.full(G,1/G),100)@counts).reshape(100,len(names),4);tp,tn,fp,fn=[a[:,:,j] for j in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));boots.append(np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0))
boots=np.concatenate(boots);bd={}
for j,c in enumerate(contrasts):
    d=boots[:,names.index(c['candidate'])]-boots[:,names.index(c['baseline'])];bd[f'contrast{j}']=d;c['delta_MCC_95CI']=np.quantile(d,[.025,.975]).tolist();c['delta_MCC_99_75CI']=np.quantile(d,[.00125,.99875]).tolist()

gate_metrics={}
for pair,data in aux.items():
    dg=data['G_probability']>=data['G_threshold'];ds=data['S_probability']>=data['S_threshold'];dis=dg!=ds;target=ds[dis]==y[dis];gate_metrics[pair]={'evaluation_disagreements':int(dis.sum()),'helpful_n':int(target.sum()),'harmful_n':int((~target).sum()),'helpful_prevalence':float(target.mean())}
    for kind in ['score_gate','feature_gate']:
        q=data[kind+'_q'][dis];sel=data[kind+'_choose_S'];both=len(set(target))==2
        gate_metrics[pair][kind]={'AUROC_teacher_helpful':float(roc_auc_score(target,q)) if both else None,'AUPRC_teacher_helpful':float(average_precision_score(target,q)) if both else None,
                                 'selected_n':int(sel.sum()),'selected_helpful':int((sel&(ds==y)).sum()),'selected_harmful':int((sel&(ds!=y)).sum())}

# Reuse old controls only; no rematching or new sequence search.
oldpairs=read(files[7]);oldpairs=oldpairs if isinstance(oldpairs,list) else oldpairs.get('pairs',[])
retained=[]
for v in gx:
    for mode in sx:
        cls=class_vectors[v,mode]
        for p in oldpairs:
            ca,co=p['case_id'],p['control_id'];j,k=ix[ca],ix[co]
            if cls[j] in ['G_wrong_S_right','G_right_S_wrong'] and cls[k]=='both_right':
                assert y[j]==y[k] and fold[j]==fold[k] and groups[j]!=groups[k]
                retained.append({'pair':v+'_'+mode,'case_class':str(cls[j]),**p})
feature_summary={}
for pair,classes in [(f'{v}_{mode}',class_vectors[v,mode]) for v in gx for mode in sx]:
    result={}
    for label in [0,1]:
        good=(classes=='G_wrong_S_right')&(y==label);bad=(classes=='G_right_S_wrong')&(y==label)
        rr={'helpful_n':int(good.sum()),'harmful_n':int(bad.sum()),'features':{}}
        for j,name in enumerate(extra_names):
            dd=[]
            for f in range(5):
                a=extra[good&(fold==f),j];b=extra[bad&(fold==f),j];dd.append(float(np.nanmedian(a)-np.nanmedian(b)) if len(a) and len(b) else None)
            rr['features'][name]={'helpful_median':float(np.nanmedian(extra[good,j])) if good.any() else None,'harmful_median':float(np.nanmedian(extra[bad,j])) if bad.any() else None,'fold_median_differences':dd}
        for clsname,mask in [('helpful',good),('harmful',bad)]:
            rr[clsname+'_curated_family_records']=int(sum(bool(meta[k]['curated_family']) for k in np.array(ids)[mask]));rr[clsname+'_name_proxy_records']=int(sum(bool(meta[k]['name_class']) for k in np.array(ids)[mask]))
        result[label]=rr
    feature_summary[pair]=result
coef_summary={}
for pair in aux:
    rr=[x for x in gate_models if x['pair']==pair and x['kind']=='feature_gate' and not x['fallback']]
    coef_summary[pair]={name:{'fits':len(rr),'positive_coefficients':int(sum(x['coefficient'][0][j]>0 for x in rr)),'negative_coefficients':int(sum(x['coefficient'][0][j]<0 for x in rr)),'median_standardized_coefficient':float(np.median([x['coefficient'][0][j] for x in rr])) if rr else None} for j,name in enumerate(score_names+extra_names)}

rows=[]
for j,k in enumerate(ids):
    rr={'sample_id':k,'sequence':seq[j],'label':int(y[j]),'fold':int(fold[j]),'group_id':str(groups[j]),'name':meta[k]['name'],'curated_family':meta[k]['curated_family'],'name_class':meta[k]['name_class'],'source_urls':meta[k]['source_urls']}
    for (v,mode),cls in class_vectors.items():rr[v+'_'+mode+'_source_class']=str(cls[j])
    for pair,a in aux.items():
        for key,value in a.items():rr[pair+'_'+key]=clean(value[j])
    for k2 in prob:rr[k2+'_score']=float(prob[k2][j]);rr[k2+'_prediction']=int(decision[k2][j])
    rows.append(rr)
save('all_sample_records.json',rows);save('metrics.json',met);save('fold_metrics.json',fm);save('comparisons.json',contrasts);save('gate_prediction_metrics.json',gate_metrics)
save('scalar_models.json',models);save('selection_records.json',selections);save('gate_models.json',gate_models)
save('retained_old_matched_controls.json',retained);save('class_feature_summary.json',feature_summary);save('gate_coefficient_summary.json',coef_summary)
np.savez_compressed(O/'evaluation_predictions.npz',sample_ids=np.array(ids),labels=y,folds=fold,groups=groups,halves=halves,features_extra=extra,
 **{k+'_score':v for k,v in prob.items()},**{k+'_pred':v for k,v in decision.items()},**{pair+'_'+key:value for pair,a in aux.items() for key,value in a.items()})
np.savez_compressed(O/'bootstrap_deltas.npz',**bd)
assert manifest=={str(p):sha(p) for p in files}
save('execution_summary.json',{'n':N,'scalar_calibration_fits':len(models),'gate_fits':len(gate_models),'gate_constant_fallbacks':sum(x['fallback'] for x in gate_models),'selection_splits':len(selections),'old_pairs_retained':len(retained),'no_neural_training':True,'no_test_data_used':True,'all_input_hashes_unchanged':True})
print(json.dumps(clean({'atlas':{k:{a:b['n'] for a,b in v['classes'].items()} for k,v in atlas.items()},'gate_metrics':gate_metrics,'metrics':met}),ensure_ascii=False,indent=2))
