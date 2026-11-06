"""Frozen score/threshold factorial diagnostic, no fitting or parameter search."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
from sklearn.metrics import matthews_corrcoef,average_precision_score,roc_auc_score
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';S=A/'saprot_selective_value_20261027'
rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(n,d):(O/n).write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
files=[O/'protocol.json',S/'evaluation_predictions.npz',S/'selection_records.json',S/'comparisons.json',A/'saprot_contribution_20261026/sample_contributions.json',A/'saprot_contribution_20261026/matched_controls.json']
hashes={str(p):sha(p) for p in files};save('input_manifest.json',hashes)
z=np.load(files[1]);ids=z['sample_ids'].tolist();N=len(ids);assert N==6387
y=z['labels'].astype(int);fold=z['folds'];half=z['halves'];groups=z['groups'];ix={k:i for i,k in enumerate(ids)}
selections=rd(files[2]);oldcontrast=rd(files[3]);meta={x['sample_id']:x for x in rd(files[4])};assert set(meta)==set(ids)
pairs=['old_masked','old_3Di','new_masked','new_3Di'];parts={};scores={};preds={};factorial={};rows=[]
def metric(p,d,mask=None):
    mask=np.ones(N,bool) if mask is None else mask;yy=y[mask];pp=p[mask];dd=d[mask];both=len(set(yy))==2
    return {'n':int(mask.sum()),'MCC':float(matthews_corrcoef(yy,dd)) if both else None,
            'AUPRC_uncentered_score':float(average_precision_score(yy,pp)) if both else None,'AUROC_uncentered_score':float(roc_auc_score(yy,pp)) if both else None,
            'FN':int(((yy==1)&(dd==0)).sum()),'FP':int(((yy==0)&(dd==1)).sum()),'correct':int((yy==dd).sum())}
for pair in pairs:
    pg=z[pair+'_G_probability'];ps=z[pair+'_S_probability'];tg=z[pair+'_G_threshold'];ts=z[pair+'_S_threshold'];w=np.full(N,np.nan);tf=np.full(N,np.nan)
    for sel in selections:
        if sel['pair']!=pair:continue
        mask=(fold==sel['outer_fold'])&(half==sel['evaluation_half'])
        assert set(np.array(ids)[mask])==set(sel['evaluation_ids'])
        w[mask]=sel['selected']['weight_SaProt'];tf[mask]=sel['selected']['threshold']
    assert np.isfinite(w).all() and np.isfinite(tf).all()
    pm=(1-w)*pg+w*ps
    pr={'A_G_tG':pg>=tg,'B_G_tF':pg>=tf,'C_mix_tG':pm>=tg,'D_mix_tF':pm>=tf}
    assert np.array_equal(pr['A_G_tG'],z[pair+'_baseline_pred']) and np.array_equal(pr['D_mix_tF'],z[pair+'_weight_selected_pred'])
    mm={k:metric(pg if k.startswith(('A','B')) else pm,v) for k,v in pr.items()}
    score_first=mm['C_mix_tG']['MCC']-mm['A_G_tG']['MCC'];score_afterthreshold=mm['D_mix_tF']['MCC']-mm['B_G_tF']['MCC']
    threshold_first=mm['B_G_tF']['MCC']-mm['A_G_tG']['MCC'];threshold_afterscore=mm['D_mix_tF']['MCC']-mm['C_mix_tG']['MCC']
    decomposition={'score_first_MCC':score_first,'threshold_after_score_MCC':threshold_afterscore,
                   'threshold_first_MCC':threshold_first,'score_after_threshold_MCC':score_afterthreshold,
                   'score_ShAP_allocation_MCC':.5*(score_first+score_afterthreshold),'threshold_ShAP_allocation_MCC':.5*(threshold_first+threshold_afterscore),
                   'total_delta_MCC':mm['D_mix_tF']['MCC']-mm['A_G_tG']['MCC']}
    ca={k:mm[k]['correct'] for k in mm}
    decomposition.update(score_allocation_netcorrect=.5*(ca['C_mix_tG']-ca['A_G_tG']+ca['D_mix_tF']-ca['B_G_tF']),threshold_allocation_netcorrect=.5*(ca['B_G_tF']-ca['A_G_tG']+ca['D_mix_tF']-ca['C_mix_tG']),total_netcorrect=ca['D_mix_tF']-ca['A_G_tG'])
    assert abs(decomposition['score_ShAP_allocation_MCC']+decomposition['threshold_ShAP_allocation_MCC']-decomposition['total_delta_MCC'])<1e-12
    a,d=pr['A_G_tG'],pr['D_mix_tF'];role=np.where(a!=y,np.where(d==y,'corrected','both_wrong'),np.where(d!=y,'harmed','both_correct'))
    bchange=pr['B_G_tF']!=a;cchange=pr['C_mix_tG']!=a
    subtype=np.where(bchange,np.where(cchange,'either_alone_sufficient','threshold_alone_only'),np.where(cchange,'score_alone_only','both_changes_required'))
    attrib={}
    for rr in ['corrected','harmed']:
        mask=role==rr;attrib[rr]={'n':int(mask.sum()),'positive':int((mask&(y==1)).sum()),'negative':int((mask&(y==0)).sum()),'subtypes':dict(Counter(subtype[mask])),'per_fold_counts':[int((mask&(fold==f)).sum()) for f in range(5)],
                                'SaProt_alone_at_its_own_selected_threshold_correct':int((mask&((ps>=ts)==y)).sum())}
    for rr in ['corrected','harmed','both_wrong','both_correct']:
        mask=role==rr
        for j in np.where(mask)[0]:
            record={'pair':pair,'sample_id':ids[j],'sequence':meta[ids[j]]['sequence'],'label':int(y[j]),'fold':int(fold[j]),'half':int(half[j]),'group_id':str(groups[j]),'role':rr,
                    'G_probability':float(pg[j]),'S_probability':float(ps[j]),'weight_SaProt':float(w[j]),'threshold_G':float(tg[j]),'threshold_fusion':float(tf[j]),
                    'A_prediction':int(a[j]),'B_prediction':int(pr['B_G_tF'][j]),'C_prediction':int(pr['C_mix_tG'][j]),'D_prediction':int(d[j]),
                    'subtype':str(subtype[j]) if rr in ['corrected','harmed'] else None,
                    'original_margin':float(pg[j]-tg[j]),'fusion_margin':float(pm[j]-tf[j]),
                    'SaProt_score_push':float(w[j]*(ps[j]-pg[j])),'threshold_push':float(tg[j]-tf[j]),
                    'name':meta[ids[j]]['name'],'curated_family':meta[ids[j]]['curated_family'],'name_class':meta[ids[j]]['name_class'],
                    'disulfide_status':meta[ids[j]]['disulfide_status'],'modification_status':meta[ids[j]]['modification_status'],
                    'annotation_negation_warning':meta[ids[j]]['annotation_negation_warning'],'mechanism_mentions':meta[ids[j]]['mechanism_mentions'],
                    'source_urls':meta[ids[j]]['source_urls'],'cached_papers':meta[ids[j]]['cached_papers'],'features':meta[ids[j]]['features']}
            assert abs(record['fusion_margin']-record['original_margin']-record['SaProt_score_push']-record['threshold_push'])<1e-12
            rows.append(record)
    factorial[pair]={'metrics':mm,'decomposition':decomposition,'changed_case_attribution':attrib,'fold_metrics':{k:{f:metric(pg if k.startswith(('A','B')) else pm,v,fold==f) for f in range(5)} for k,v in pr.items()},
                     'rank_only':{'G_AUPRC':mm['A_G_tG']['AUPRC_uncentered_score'],'mixture_AUPRC':mm['D_mix_tF']['AUPRC_uncentered_score'],'G_AUROC':mm['A_G_tG']['AUROC_uncentered_score'],'mixture_AUROC':mm['D_mix_tF']['AUROC_uncentered_score']}}
    parts[pair]={'pg':pg,'ps':ps,'tg':tg,'tf':tf,'w':w,'pm':pm,'role':role}
    for k,v in pr.items():preds[pair+'_'+k]=v;scores[pair+'_'+k]=pg if k.startswith(('A','B')) else pm
save('factorial_results.json',factorial);save('all_sample_factorial_records.json',rows)

contrasts=[]
for pair in pairs:
    for a,b in [('D_mix_tF','A_G_tG'),('C_mix_tG','A_G_tG'),('B_G_tF','A_G_tG'),('D_mix_tF','B_G_tF'),('D_mix_tF','C_mix_tG')]:
        da,db=preds[pair+'_'+a],preds[pair+'_'+b];mm=factorial[pair]['metrics'];contrasts.append({'pair':pair,'candidate':a,'baseline':b,'delta_MCC':mm[a]['MCC']-mm[b]['MCC'],'corrected':int(((da==y)&(db!=y)).sum()),'harmed':int(((da!=y)&(db==y)).sum())})
ug,gi=np.unique(groups,return_inverse=True);G=len(ug);names=list(preds);counts=[]
for name in names:
    d=preds[name];counts.append(np.stack([np.bincount(gi,weights=cc,minlength=G) for cc in [(y==1)&d,(y==0)&~d,(y==0)&d,(y==1)&~d]],axis=1))
counts=np.stack(counts,axis=1).reshape(G,-1);rng=np.random.default_rng(20261030);boots=[]
for _ in range(20):
    bb=(rng.multinomial(G,np.full(G,1/G),100)@counts).reshape(100,len(names),4);tp,tn,fp,fn=[bb[:,:,j] for j in range(4)];den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));boots.append(np.divide(tp*tn-fp*fn,den,out=np.zeros_like(den),where=den>0))
boots=np.concatenate(boots);bootdiff={}
for j,c in enumerate(contrasts):
    dd=boots[:,names.index(c['pair']+'_'+c['candidate'])]-boots[:,names.index(c['pair']+'_'+c['baseline'])];bootdiff[f'contrast{j}']=dd;c['delta_MCC_95CI']=np.quantile(dd,[.025,.975]).tolist();c['delta_MCC_99_75CI']=np.quantile(dd,[.00125,.99875]).tolist()
save('factorial_comparisons.json',contrasts);np.savez_compressed(O/'bootstrap_deltas.npz',**bootdiff)

# Conditional shuffling is a diagnostic null, not an evaluated new model.
nulls={};nullarray={};rng=np.random.default_rng(20261029)
for pair,data in parts.items():
    pg,ps,w,tf=data['pg'],data['ps'],data['w'],data['tf'];bins=[]
    for f in range(5):
        for h in [0,1]:
            ii=np.where((fold==f)&(half==h))[0];ii=sorted(ii,key=lambda j:(pg[j],ids[j]));bins.extend(np.array_split(np.array(ii),10))
    mm=[];ap=[];fewer=[]
    for _ in range(200):
        sp=ps.copy()
        for ii in bins:sp[ii]=ps[rng.permutation(ii)]
        p=(1-w)*pg+w*sp;d=p>=tf
        mm.append(float(matthews_corrcoef(y,d)));ap.append(float(average_precision_score(y,p)));fewer.append(int((d==y).sum()-(preds[pair+'_A_G_tG']==y).sum()))
    actual=factorial[pair]['metrics']['D_mix_tF']['MCC'];nulls[pair]={'actual_MCC':actual,'conditional_shuffle_MCC_mean':float(np.mean(mm)),'conditional_shuffle_MCC_95_range':np.quantile(mm,[.025,.975]).tolist(),'shuffles_with_MCC_at_least_actual':int((np.array(mm)>=actual).sum()),'n_shuffles':200,'shuffle_net_fewer_errors_mean':float(np.mean(fewer)),'actual_AUPRC':factorial[pair]['rank_only']['mixture_AUPRC'],'shuffle_AUPRC_mean':float(np.mean(ap)),'scope':'Descriptive coarse-confidence conditional permutation; no pvalue, model-selection, group-exchangeability, or biological causality claim.'};nullarray[pair+'_MCC']=np.array(mm);nullarray[pair+'_AUPRC']=np.array(ap)
save('conditional_shuffle_diagnostic.json',nulls);np.savez_compressed(O/'conditional_shuffle_values.npz',**nullarray)

primary=[r for r in rows if r['pair']=='old_3Di'];changed=[r for r in primary if r['role'] in ['corrected','harmed']];assert Counter(r['role'] for r in changed)=={'corrected':31,'harmed':20}
save('old3Di_corrected31_harmed20.json',changed)
byid={r['sample_id']:r for r in primary};rawpairs=rd(files[5]);rawpairs=rawpairs if isinstance(rawpairs,list) else rawpairs.get('pairs',[]);matches={}
for p in rawpairs:
    ca,co=p['case_id'],p['control_id']
    if byid[ca]['role'] in ['corrected','harmed'] and byid[co]['role']=='both_correct':
        assert byid[ca]['label']==byid[co]['label'] and byid[ca]['fold']==byid[co]['fold'] and byid[ca]['group_id']!=byid[co]['group_id']
        matches[ca,co]={'candidate_role':byid[ca]['role'],**p}
save('reused_matched_controls.json',list(matches.values()))
features=['sequence_length','cys_count','cys_span_fraction','net_charge_proxy','mean_hydropathy','mean_plddt','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation','nonlocal_contact_density']
biology={}
for lab in [0,1]:
    br={}
    for role in ['corrected','harmed','both_wrong','both_correct']:
        rr=[x for x in primary if x['label']==lab and x['role']==role];means={}
        for feature in features:
            val=[x['features'][feature] for x in rr if x['features'][feature] is not None and np.isfinite(x['features'][feature])];means[feature]={'n_available':len(val),'n_missing':len(rr)-len(val),'median':float(np.median(val)) if val else None}
        br[role]={'n':len(rr),'features':means,'subtypes':dict(Counter(x['subtype'] for x in rr if x['subtype'])),
                  'curated_family_records':sum(bool(x['curated_family']) for x in rr),'curated_family_tokens':dict(Counter(k for x in rr for k in x['curated_family'])),
                  'name_proxy_tokens':dict(Counter(k for x in rr for k in x['name_class'])),
                  'explicit_disulfide':sum(x['disulfide_status']=='explicit_annotation' for x in rr),'explicit_modification':sum(x['modification_status']=='explicit_annotation' for x in rr),
                  'full_feature_record_unavailable':sum(x['disulfide_status']=='full_feature_record_unavailable' for x in rr),
                  'mechanism_candidate_mentions_excluding_negation_warning':dict(Counter(k for x in rr if not x['annotation_negation_warning'] for k,v in x['mechanism_mentions'].items() if v)),
                  'fold_counts':[sum(x['fold']==f for x in rr) for f in range(5)]}
    biology[lab]=br
save('whole_class_biology.json',biology)
save('reused_control_coverage.json',{'unique_case_control_pairs':len(matches),'covered_corrected_cases':len({k[0] for k,p in matches.items() if p['candidate_role']=='corrected'}),'covered_harmed_cases':len({k[0] for k,p in matches.items() if p['candidate_role']=='harmed'}),'family_support_counts':dict(Counter(p['family_support'] for p in matches.values())),'scope':'Only existing cache pairs; sparse, posthoc outcome-selected coverage, not a representative sample or new family-level validation.'})
np.savez_compressed(O/'factorial_predictions.npz',sample_ids=np.array(ids),labels=y,folds=fold,halves=half,groups=groups,**{k+'_pred':v for k,v in preds.items()},**{pair+'_'+key:value for pair,data in parts.items() for key,value in data.items() if key!='role'})
assert hashes=={str(p):sha(p) for p in files}
save('execution_summary.json',{'n':N,'arms':4,'factorial_contrasts':len(contrasts),'new_model_fits':0,'neural_training':False,'test_data_used':False,'source_hashes_unchanged':True,'primary_cases':{'corrected':31,'harmed':20}})
print(json.dumps(clean({'factorial':factorial,'conditional_shuffle':nulls,'control_coverage':rd(O/'reused_control_coverage.json')}),ensure_ascii=False,indent=2),flush=True)
