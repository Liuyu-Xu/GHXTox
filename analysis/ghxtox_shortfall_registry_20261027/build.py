"""Consolidate existing error evidence; add only missing version/category joins."""
from pathlib import Path
import json,csv,hashlib
from collections import Counter
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';H=A/'upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027'
rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def cr(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig',newline='')))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    return v
def save(n,v):(O/n).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2),encoding='utf-8')
files=[O/'protocol.json',A/'error_evidence_linkage_20261025/all_training_linked.json',
 H/'runs/structure_candidates_20261022/report/S3_plus_no_inner_transformer_oof.csv',
 A/'classification_mechanism_map_20261026/all_training_category_index.json',
 A/'classification_mechanism_map_20261026/category_matrix.json',
 A/'membrane_class_feature_diagnosis_20261026/semantic_and_feature_rows.json',
 A/'ghxtox_error_atlas_20261025/matched_pairs.csv',A/'family_mechanism_comparison_20261026/class_pair_evidence.json',
 A/'saprot_contribution_20261026/sample_contributions.json',A/'saprot_selective_value_20261027/source_fourway_atlas.json',
 H/'data/folds/train_cdhit080_fallback_folds.csv']
hashes={str(p):sha(p) for p in files};save('input_manifest.json',hashes)
oldrows=rd(files[1]);ids=[x['sample_id'] for x in oldrows];N=len(ids);assert N==6387 and len(set(ids))==N
old={x['sample_id']:x for x in oldrows};new={x['sample_id']:x for x in cr(files[2])};cat={x['sample_id']:x for x in rd(files[3])};meta={x['sample_id']:x for x in rd(files[8])};semantic={x['sample_id']:x for x in rd(files[5])};newmanifest={x['sample_id']:x for x in cr(files[10])}
assert set(old)==set(new)==set(cat)==set(meta)
y=np.array([int(old[k]['label']) for k in ids]);fold=np.array([int(old[k]['fold']) for k in ids]);groups=np.array([old[k]['group_id'] for k in ids]);seq=[old[k]['sequence'] for k in ids];cys=np.array([s.count('C') for s in seq])
for j,k in enumerate(ids):
    assert new[k]['sequence']==cat[k]['sequence']==meta[k]['sequence']==seq[j]
    assert int(new[k]['label'])==cat[k]['label']==meta[k]['label']==y[j]
    assert int(new[k]['outer_fold'])==cat[k]['fold']==meta[k]['fold']==fold[j]
    assert cat[k]['group_id']==meta[k]['group_id']==newmanifest[k]['group_id']==groups[j]
    assert newmanifest[k]['sequence']==seq[j] and int(newmanifest[k]['label'])==y[j] and int(newmanifest[k]['fold'])==fold[j]
    assert int(old[k]['cys_count'])==cys[j]
oldp=np.array([float(old[k]['p_fusion']) for k in ids]);newp=np.array([float(new[k]['probability']) for k in ids]);newt=np.array([float(new[k]['threshold']) for k in ids])
op=(oldp>=.47022256531284556).astype(int);npred=(newp>=newt).astype(int);oe=op!=y;ne=npred!=y
assert oe.sum()==327 and ne.sum()==357
assert all((old[k]['fusion_error']=='True')==bool(oe[j]) for j,k in enumerate(ids))
joint=np.where(oe,np.where(ne,'both_prediction_streams_wrong','old_wrong_new_right'),np.where(ne,'old_right_new_wrong','both_right'))
flags={'fragment_reference_flag':np.array([bool(meta[k]['fragment_flag']) for k in ids]),
       'cross_source_label_conflict':np.array([bool(meta[k]['cross_source_label_conflict']) for k in ids]),
       'identity_unrecovered':np.array([not bool(old[k].get('exact_match_accessions')) for k in ids])}
direct=np.array([any(a.get('evidence_status')=='有直接测量记录' and a.get('result') for a in old[k].get('assay_records',[])) for k in ids],bool)
source_modes={}
for j,k in enumerate(ids):
    if not oe[j]:continue
    se=old[k]['sequence_error']=='True';te=old[k]['structure_error']=='True'
    mode='both_branches_wrong' if se and te else 'sequence_right_structure_wrong' if not se and te else 'sequence_wrong_structure_right' if se and not te else 'both_branches_right_fusion_wrong'
    source_modes[k]=mode
assert Counter(source_modes.values())=={'both_branches_wrong':229,'sequence_right_structure_wrong':48,'sequence_wrong_structure_right':50}

titles={'membrane_or_lysis':'膜作用/裂解注释线索','ion_channel':'离子通道注释线索','receptor_or_signaling':'受体/信号注释线索','hemostasis_or_proteolysis':'凝血/蛋白水解注释线索','antimicrobial':'抗微生物注释线索','hemolysis_text':'溶血文字线索（含否定/弱活性）','disulfide_annotation':'详细二硫键注释缓存','chemical_annotation':'化学修饰特征缓存','public_hemolysis_record':'公开溶血测量关联缓存','manual_direct_hemolysis':'人工直接溶血证据缓存'}
spec=[]
for l,condition,title in [(1,cys<4,'低Cys原有毒类（Cys<4）'),(0,cys>=4,'富Cys原无毒类（Cys≥4）'),(1,cys>=4,'富Cys原有毒背景'),(0,cys<4,'低Cys原无毒背景')]:
    # Explicit ordinal keys preserve the original four-cell order.
    spec.append({'key':f'physical_{len(spec)}','title':title,'label':l,'mask':condition&(y==l),'basis':'complete_population_physical_partition','grade':'A_historical_statistical_shortfall' if len(spec)<2 else 'background_reference','priority':1 if len(spec)<2 else 4})
for token in sorted(titles):
    for l in [0,1]:
        mask=np.array([token in cat[k]['categories'] for k in ids])&(y==l)
        spec.append({'key':token+f'_label{l}','title':titles[token],'label':l,'mask':mask,'basis':'existing_annotation_or_evidence_snapshot_not_gold','grade':'C_annotation_or_evidence_coverage_not_confirmed_mechanism','priority':2 if token in ['membrane_or_lysis','ion_channel','receptor_or_signaling','hemostasis_or_proteolysis'] and l==1 else 3})
for state in sorted({x['semantic_state'] for x in semantic.values()}):
    mask=np.array([k in semantic and semantic[k]['semantic_state']==state for k in ids])
    spec.append({'key':'membrane_semantic_'+state,'title':'膜候选语义子层：'+state,'label':1,'mask':mask,'basis':'existing_76_semantic_review_not_gold','grade':'B_or_C_candidate_evidence_not_complete_endpoint_class','priority':2 if state=='affirmative_annotation_not_gold' else 3})
for name,mask in flags.items():spec.append({'key':'dataflag_'+name,'title':'资料疑点：'+name,'label':None,'mask':mask,'basis':'data_quality_flag_not_confirmed_error','grade':'C_label_or_object_audit','priority':3})

# Attach old control edges without new matching, preserving matching strength.
edges=[]
for p in cr(files[6]):
    edges.append({'case_id':p['case_id'],'control_id':p['control_id'],'source_archive':'initial_composition151','family_support':'not_matched_for_family','source_same_fold':True,'scope':'composition_matched_diagnostic_not_endpoint_or_family_matched'})
for r in rd(files[7]):
    p=r['pair'];edges.append({'case_id':p['case_id'],'control_id':p['control_id'],'source_archive':'class_family27','family_support':p['family_support'],'source_same_fold':p['case_fold']==p['control_fold'],'identity':p.get('identity'),'length_ratio':p.get('length_ratio'),'scope':'family_or_name_proxy_exploratory_many_crossfold'})
ix={k:j for j,k in enumerate(ids)}
for e in edges:
    j,k=ix[e['case_id']],ix[e['control_id']];assert y[j]==y[k] and groups[j]!=groups[k]
    assert oe[j] and not oe[k]
teacher=rd(files[9]);registry=[];control_records=[]
for s in spec:
    mask=s['mask'];members=set(np.array(ids)[mask]);n=int(mask.sum());stat={}
    for v,err in [('old_archived',oe),('new_noTransformer',ne)]:
        wrong=mask&err;right=mask&~err
        stat[v]={'n':n,'wrong':int(wrong.sum()),'right':int(right.sum()),'error_rate':float(wrong.sum()/n) if n else None,'wrong_ids':np.array(ids)[wrong].tolist(),'right_pool_ids':np.array(ids)[right].tolist(),
                 'per_fold':[{'fold':f,'n':int((mask&(fold==f)).sum()),'wrong':int((wrong&(fold==f)).sum())} for f in range(5)]}
    modes={name:int((mask&(joint==name)).sum()) for name in ['both_prediction_streams_wrong','old_wrong_new_right','old_right_new_wrong','both_right']}
    controls={};background_controls={}
    for v,err in [('old_archived',oe),('new_noTransformer',ne)]:
        retained=[e for e in edges if e['case_id'] in members and e['control_id'] in members and err[ix[e['case_id']]] and not err[ix[e['control_id']]]]
        controls[v]={'composition_edges':sum(e['source_archive']=='initial_composition151' for e in retained),'family_or_name_edges':sum(e['source_archive']=='class_family27' for e in retained),
                     'curated_family_edges':sum(e['family_support']=='curated_family' for e in retained),'name_proxy_edges':sum(e['family_support']=='name_class_proxy' for e in retained),
                     'same_fold_edges':sum(e['source_same_fold'] for e in retained),'unique_cases_with_any_old_control':len({e['case_id'] for e in retained}),
                     'scope':'Retained original pairs only, may overlap source archives; not representative or endpoint-matched; no rematching.'}
        broader=[e for e in edges if e['case_id'] in members and err[ix[e['case_id']]] and not err[ix[e['control_id']]]]
        background_controls[v]={'all_same_label_archived_edges':len(broader),'controls_outside_category':sum(e['control_id'] not in members for e in broader),
                                'curated_family_edges':sum(e['family_support']=='curated_family' for e in broader),'name_proxy_edges':sum(e['family_support']=='name_class_proxy' for e in broader),
                                'unique_error_cases_covered':len({e['case_id'] for e in broader}),
                                'scope':'Same label archived controls may lie outside physical/annotation category; retain for shortcut/background analysis, do not confuse with within-category comparison pool.'}
        for e in broader:control_records.append({'category_key':s['key'],'prediction_stream':v,'control_inside_category':e['control_id'] in members,**e})
    td={}
    for v,prefix in [('old_archived','old'),('new_noTransformer','new')]:
        td[v]={}
        for mode in ['masked','3Di']:
            cl=teacher[prefix+'_'+mode]['classes'];helps=members&set(cl['G_wrong_S_right']['sample_ids']);harms=members&set(cl['G_right_S_wrong']['sample_ids'])
            td[v][mode]={'Gwrong_Sright_potential':len(helps),'Gright_Swrong_potential':len(harms),'scope':'Independent natural decisions, not actual fused correction or validated gate; reused existing fourway atlas.'}
    errunion=mask&(oe|ne);bothmask=mask&oe&ne
    defs={k:v for k,v in s.items() if k!='mask'}
    reg={**defs,'population':n,'n_groups':len(set(groups[mask])),'stats':stat,'joint_modes':modes,'joint_error_ids':np.array(ids)[bothmask].tolist(),'both_right_control_pool_ids':np.array(ids)[mask&~oe&~ne].tolist(),
         'old_error_branch_modes':dict(Counter(source_modes[k] for k in np.array(ids)[mask&oe])),
         'evidence':{'fragment_flags':int((mask&flags['fragment_reference_flag']).sum()),'label_conflicts':int((mask&flags['cross_source_label_conflict']).sum()),'has_cached_direct_measurement':int((mask&direct).sum()),'cached_measurements_in_both_stream_error_subset':int((bothmask&direct).sum()),
                     'explicit_disulfide_cache':sum(meta[k]['disulfide_status']=='explicit_annotation' for k in members),'explicit_modification_cache':sum(meta[k]['modification_status']=='explicit_annotation' for k in members),'unknown_disulfide_record':sum(meta[k]['disulfide_status']=='full_feature_record_unavailable' for k in members),
                     'annotation_negation_warning':sum(bool(cat[k]['annotation_negation_warning']) for k in members),'full_snapshot_record_available':sum(bool(cat[k]['full_record_available']) for k in members),'endpoint_scope':'Measurement presence is not complete molecule/endpoint match or goldlabel; unknown not negative.'},
         'existing_control_coverage':controls,'existing_background_control_coverage':background_controls,'existing_SaProt_diagnostic':td}
    registry.append(reg)

# Consistency checks against archived category matrix; not a new estimate.
for oldmatrix in rd(files[4]):
    found=next(x for x in registry if x['key']==oldmatrix['category']+f"_label{oldmatrix['original_label']}")
    assert found['population']==oldmatrix['all_cached']['n'] and found['stats']['old_archived']['wrong']==oldmatrix['all_cached']['fusion_errors']
assert [(x['population'],x['stats']['old_archived']['wrong']) for x in registry[:4]]==[(421,175),(445,76),(1397,17),(4124,59)]

membership={k:[] for k in ids}
for r in registry:
    for k in r['stats']['old_archived']['wrong_ids']+r['stats']['old_archived']['right_pool_ids']:membership[k].append(r['key'])
rows=[]
for j,k in enumerate(ids):
    rr={'sample_id':k,'sequence':seq[j],'label':int(y[j]),'fold':int(fold[j]),'group_id':str(groups[j]),'cys_count':int(cys[j]),'name':meta[k]['name'],
        'old_archived_probability':float(oldp[j]),'old_archived_threshold':.47022256531284556,'old_archived_wrong':bool(oe[j]),'new_probability':float(newp[j]),'new_threshold':float(newt[j]),'new_wrong':bool(ne[j]),'joint_mode':str(joint[j]),
        'old_error_branch_mode':source_modes.get(k),'category_keys':membership[k],'curated_family':meta[k]['curated_family'],'name_class':meta[k]['name_class'],
        'mechanism_mentions':meta[k]['mechanism_mentions'],'annotation_negation_warning':meta[k]['annotation_negation_warning'],'fragment_flag_not_confirmed_label_error':bool(flags['fragment_reference_flag'][j]),'cross_source_conflict_not_confirmed_label_error':bool(flags['cross_source_label_conflict'][j]),
        'disulfide_status':meta[k]['disulfide_status'],'modification_status':meta[k]['modification_status'],'manual_review':old[k].get('manual_review'),
        'assay_records':old[k].get('assay_records',[]),'has_cached_direct_measurement':bool(direct[j]),'entity_input_audit':old[k].get('entity_input_audit'),
        'source_urls':sorted(set(meta[k]['source_urls']+old[k].get('source_urls',[]))),'features':meta[k]['features'],'original_label_unchanged':True}
    rows.append(rr)
save('all_sample_shortfall_index.json',rows);save('category_registry.json',registry);save('existing_control_edges.json',control_records)
save('shared_error_samples.json',[x for x in rows if x['joint_mode']=='both_prediction_streams_wrong'])
version={'n':N,'original_labels_unchanged':True,'source_alignment':'ID/sequence/label/fold/group checked; old counts consistency-checked only',
 'old':{'source':str(files[1]),'identity':'archived operatingpoint, not arbitrary full2795 deployment lineage','threshold':.47022256531284556,'wrong':int(oe.sum()),'FN':int((oe&(y==1)).sum()),'FP':int((oe&(y==0)).sum()),'same_fullmodel_seed_replications_available':False},
 'new':{'source':str(files[2]),'identity':'handoff seed42 no-innerTransformer structure+originalS3 sequence OOF; differing inner protocol','threshold':'each original row threshold','wrong':int(ne.sum()),'FN':int((ne&(y==1)).sum()),'FP':int((ne&(y==0)).sum()),'same_fullmodel_seed_replications_available':False},
 'joint_modes':dict(Counter(joint)),'groups':len(set(groups)),
 'scope':'Descriptive intersection, not architecturecausality/independentconfirmation/crossseedstability; no Test1/Test2 used.'}
save('version_manifest.json',version)
save('execution_summary.json',{'n':N,'categories':len(registry),'old_20_category_rows_consistency_checked':True,'physical4_old_cells_consistency_checked':True,'new_version_perclass_and_overlap_joined':True,'new_matching':False,'new_training':False,'new_significance_tests':False,'all_input_hashes_unchanged':hashes=={str(p):sha(p) for p in files}})
print(json.dumps(clean({'version':version,'primary':[{k:v for k,v in x.items() if k in ['key','title','population','joint_modes','old_error_branch_modes']}|{'oldwrong':x['stats']['old_archived']['wrong'],'newwrong':x['stats']['new_noTransformer']['wrong']} for x in registry[:4]]}),ensure_ascii=False,indent=2))
