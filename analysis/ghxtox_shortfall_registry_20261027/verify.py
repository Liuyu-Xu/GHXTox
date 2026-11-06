"""Independent registry replay; no modeling, matching, or significance testing."""
from pathlib import Path
import json,csv,hashlib
from collections import Counter
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';rd=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
rows=rd(O/'all_sample_shortfall_index.json');registry=rd(O/'category_registry.json');byid={r['sample_id']:r for r in rows};assert len(rows)==len(byid)==6387
old=rd(A/'error_evidence_linkage_20261025/all_training_linked.json');old={r['sample_id']:r for r in old}
newfile=A/'upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/runs/structure_candidates_20261022/report/S3_plus_no_inner_transformer_oof.csv'
new={r['sample_id']:r for r in csv.DictReader(newfile.open(encoding='utf-8-sig',newline=''))}
newmanifest={r['sample_id']:r for r in csv.DictReader((newfile.parents[3]/'data/folds/train_cdhit080_fallback_folds.csv').open(encoding='utf-8-sig',newline=''))}
cat={r['sample_id']:r for r in rd(A/'classification_mechanism_map_20261026/all_training_category_index.json')}
sem={r['sample_id']:r for r in rd(A/'membrane_class_feature_diagnosis_20261026/semantic_and_feature_rows.json')}
checked=0
for k,row in byid.items():
    assert row['sequence']==old[k]['sequence']==new[k]['sequence']==cat[k]['sequence']
    assert row['label']==int(old[k]['label'])==int(new[k]['label'])==cat[k]['label']
    assert row['fold']==int(old[k]['fold'])==int(new[k]['outer_fold'])==cat[k]['fold']==int(newmanifest[k]['fold']) and row['group_id']==cat[k]['group_id']==newmanifest[k]['group_id']
    assert row['cys_count']==row['sequence'].count('C')
    oe=int(float(old[k]['p_fusion'])>=.47022256531284556)!=row['label'];ne=int(float(new[k]['probability'])>=float(new[k]['threshold']))!=row['label']
    assert row['old_archived_wrong']==oe and row['new_wrong']==ne
    mode='both_prediction_streams_wrong' if oe and ne else 'old_wrong_new_right' if oe else 'old_right_new_wrong' if ne else 'both_right';assert row['joint_mode']==mode;checked+=1

def members(reg):
    key=reg['key']
    if key.startswith('physical_'):
        i=int(key[-1]);return {k for k,r in byid.items() if (r['label']==1 and r['cys_count']<4 if i==0 else r['label']==0 and r['cys_count']>=4 if i==1 else r['label']==1 and r['cys_count']>=4 if i==2 else r['label']==0 and r['cys_count']<4)}
    if key.startswith('membrane_semantic_'):return {k for k in sem if sem[k]['semantic_state']==key.removeprefix('membrane_semantic_')}
    if key.startswith('dataflag_'):
        name=key.removeprefix('dataflag_')
        if name=='fragment_reference_flag':return {k for k,r in byid.items() if r['fragment_flag_not_confirmed_label_error']}
        if name=='cross_source_label_conflict':return {k for k,r in byid.items() if r['cross_source_conflict_not_confirmed_label_error']}
        return {k for k,r in old.items() if not r.get('exact_match_accessions')}
    token,label=key.rsplit('_label',1);return {k for k in cat if token in cat[k]['categories'] and cat[k]['label']==int(label)}

category_checks=0;lookup={}
for r in registry:
    mm=members(r);lookup[r['key']]=mm;assert len(mm)==r['population'] and len({byid[k]['group_id'] for k in mm})==r['n_groups']
    for v,err in [('old_archived','old_archived_wrong'),('new_noTransformer','new_wrong')]:
        wrong={k for k in mm if byid[k][err]};right=mm-wrong;s=r['stats'][v]
        assert wrong==set(s['wrong_ids']) and right==set(s['right_pool_ids']) and len(wrong)==s['wrong'] and len(right)==s['right'] and len(mm)==s['n']
        for f in s['per_fold']:assert sum(byid[k]['fold']==f['fold'] for k in wrong)==f['wrong'] and sum(byid[k]['fold']==f['fold'] for k in mm)==f['n']
    joint=Counter(byid[k]['joint_mode'] for k in mm);assert all(joint[key]==value for key,value in r['joint_modes'].items())
    assert {k for k in mm if byid[k]['joint_mode']=='both_prediction_streams_wrong'}==set(r['joint_error_ids'])
    assert {k for k in mm if byid[k]['joint_mode']=='both_right'}==set(r['both_right_control_pool_ids'])
    category_checks+=1
physical=[lookup['physical_'+str(i)] for i in range(4)];assert set.union(*physical)==set(byid) and sum(map(len,physical))==6387
edgechecks=0
for e in rd(O/'existing_control_edges.json'):
    mm=lookup[e['category_key']];case,control=byid[e['case_id']],byid[e['control_id']];assert case['sample_id'] in mm and case['label']==control['label'] and case['group_id']!=control['group_id']
    err='old_archived_wrong' if e['prediction_stream']=='old_archived' else 'new_wrong';assert case[err] and not control[err]
    assert e['control_inside_category']==(control['sample_id'] in mm);assert e['source_same_fold']==(case['fold']==control['fold']);edgechecks+=1
shared=rd(O/'shared_error_samples.json');assert {r['sample_id'] for r in shared}=={k for k,r in byid.items() if r['old_archived_wrong'] and r['new_wrong']}
for p,h in rd(O/'input_manifest.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
out={'passed':True,'all_sample_id_sequence_label_fold_group_checks':checked,'categories_replayed':category_checks,'physical_partition_is_exhaustive_disjoint':True,'all_error_correct_pools_and_joint_memberships_replayed':True,'existing_control_edges_rechecked':edgechecks,'shared_error_ids_verified':len(shared),'input_hashes_unchanged':True,'neural_training':False,'new_matching':False,'test_data_used':False}
(O/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
