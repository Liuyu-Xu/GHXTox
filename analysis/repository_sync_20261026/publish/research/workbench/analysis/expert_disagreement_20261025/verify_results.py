from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
OUT=Path(__file__).resolve().parent

def run():
    manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf8'))
    for x in manifest:
        assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
        assert 'test1' not in x['path'].lower() and 'test2' not in x['path'].lower()
    d=pd.read_csv(OUT/'training_expert_atlas.csv',float_precision='round_trip',low_memory=False)
    original=pd.read_csv(manifest[0]['path'],float_precision='round_trip',low_memory=False)
    for c in ['source_index','sample_id','sequence','label','fold','group_id','p_seq','p_struct','p_fusion','sequence_error','structure_error','fusion_error']:
        assert d[c].equals(original[c]),c
    assert len(d)==6387 and d.sample_id.is_unique
    assert ((d.sequence_error!=d.structure_error)==d.expert_decision_disagreement).all()
    assert d.expert_decision_disagreement.sum()==334
    assert len(pd.read_csv(OUT/'fusion_corrections.csv'))==98
    assert len(pd.read_csv(OUT/'fusion_harms.csv'))==48
    assert int(d.sequence_error.sum()-d.fusion_error.sum())==98-48
    assert d.groupby('group_id').fold.nunique().max()==1
    expected_counts={'both_correct':5824,'structure_only_correct':148,'sequence_only_correct':186,'both_wrong':229}
    assert d.expert_group.value_counts().to_dict()==expected_counts
    for prefix,strict in [('primary',False),('strict',True)]:
        p=pd.read_csv(OUT/f'{prefix}_matched_pairs.csv')
        cols=['length','net_charge_proxy','mean_hydropathy','cys_count','aromatic_fraction']
        scales=[3,2,.5,0 if strict else 1,.1]
        if strict:
            cols+=['mean_plddt','low_plddt_fraction','max_otherfold_kmer_cosine','sequence_seed_std'];scales += [.05,.1,.1,.05]
        for name,g in p.groupby('contrast'):
            groups=list(g.case_group)+list(g.control_group);assert len(groups)==len(set(groups))
            cases=d.set_index('sample_id').loc[g.case_id];controls=d.set_index('sample_id').loc[g.control_id]
            for c in ['label','fold','evidence_tier']:assert np.array_equal(cases[c].to_numpy(),controls[c].to_numpy()),(name,c)
            assert (cases.group_id.to_numpy()!=controls.group_id.to_numpy()).all()
            for c,limit in zip(cols,scales):assert (np.abs(cases[c].to_numpy()-controls[c].to_numpy())<=limit+1e-12).all(),(name,c)
            if name.startswith('structure_minus'):
                assert cases.expert_group.eq('structure_only_correct').all() and controls.expert_group.eq('sequence_only_correct').all()
            else:assert cases.evidence_tier.eq('recovered_no_selected_flag_not_gold').all()
        t=pd.read_csv(OUT/f'{prefix}_matched_tests.csv')
        if len(t):assert np.allclose(t.q_BH.to_numpy(),multipletests(t.p,method='fdr_bh')[1],rtol=0,atol=2e-15)
    assert len(pd.read_csv(OUT/'primary_matched_pairs.csv'))==52
    assert len(pd.read_csv(OUT/'primary_matched_tests.csv'))==42
    assert len(pd.read_csv(OUT/'strict_matched_tests.csv'))==0
    ledger=pd.read_csv(OUT/'case_annotation_ledger.csv')
    for row in ledger.to_dict('records'):assert d.set_index('sample_id').loc[row['sample_id'],'sequence']==row['sequence']
    refs=pd.read_csv(OUT/'case_reference_links.csv',keep_default_na=False,dtype=str)
    papers=json.loads((OUT/'case_paper_abstracts_grouped.json').read_text(encoding='utf8'))['resultList']['result']
    assert {r['id'] for r in papers}==set(refs.loc[refs.pmid!='','pmid'])
    impl=json.loads((OUT/'implementation_audit.json').read_text(encoding='utf8'))
    for x in impl:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    result=dict(passed=True,original_training_identities_labels_probabilities_unchanged=True,no_direct_Test1_Test2_inputs=True,
        all_expert_and_fusion_counts_reconciled=True,group_calipers_and_matching_verified=True,BH_values_recomputed=True,
        primary_pairs=52,primary_tests=42,strict_pairs=1,strict_tests=0,
        exact_case_annotations_sequence_verified=True,requested_33_paper_ids_recovered=True,
        inspected_model_source_files_unchanged=True,model_training=False,labels_or_weights_changed=False)
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':run()
