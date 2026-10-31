"""Whole-class cached evidence coverage, no new matching or literature search."""
from pathlib import Path
import json
from collections import Counter,defaultdict
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
meta={x['sample_id']:x for x in read(A/'saprot_contribution_20261026/sample_contributions.json')}
atlas=read(O/'source_fourway_atlas.json');out={}
for pair,record in atlas.items():
    out[pair]={}
    for label in [0,1]:
        dd={}
        for cls in ['G_wrong_S_right','G_right_S_wrong','both_wrong','both_right']:
            rr=[meta[k] for k in record['classes'][cls]['sample_ids'] if meta[k]['label']==label]
            families=Counter(t for r in rr for t in r['curated_family']);names=Counter(t for r in rr for t in r['name_class'])
            mechanisms=Counter(t for r in rr if not r.get('annotation_negation_warning') for t,a in r['mechanism_mentions'].items() if a)
            dd[cls]={'n':len(rr),'explicit_disulfide_records':sum(r['disulfide_status']=='explicit_annotation' for r in rr),
                     'explicit_modification_records':sum(r['modification_status']=='explicit_annotation' for r in rr),
                     'full_feature_record_unavailable':sum(r['disulfide_status']=='full_feature_record_unavailable' for r in rr),
                     'curated_family_records':sum(bool(r['curated_family']) for r in rr),'curated_family_tokens':dict(families),
                     'name_proxy_tokens':dict(names),'mechanism_candidate_mentions_excluding_negation_flags':dict(mechanisms),
                     'fragment_flags_not_proven_label_errors':sum(r['fragment_flag'] for r in rr),
                     'label_conflicts_not_proven_label_errors':sum(r['cross_source_label_conflict'] for r in rr)}
        out[pair][label]=dd
(O/'annotation_class_coverage.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
pairs=read(O/'retained_old_matched_controls.json');unique={}
for p in pairs:unique[p['pair'],p['case_id'],p['control_id']]=p
summaries={}
features=['cys_count','mean_plddt','mean_hydropathy','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation']
for pair in atlas:
    d={}
    for cls in ['G_wrong_S_right','G_right_S_wrong']:
        pp=[v for (p,_,_),v in unique.items() if p==pair and v['case_class']==cls]
        d[cls]={'unique_pairs':len(pp),'curated_family_pairs':sum(v['family_support']=='curated_family' for v in pp),
                'name_proxy_pairs':sum(v['family_support']=='name_class_proxy' for v in pp),
                'sequence_only_pairs':sum(v['family_support']=='sequence_only' for v in pp),'feature_case_minus_control':{}}
        for feature in features:
            values=[(meta[v['case_id']]['features'][feature],meta[v['control_id']]['features'][feature]) for v in pp]
            delta=[a-b for a,b in values if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
            d[cls]['feature_case_minus_control'][feature]={'available_pairs':len(delta),'missing_pairs':len(pp)-len(delta),'mean':float(np.mean(delta)) if delta else None,'positive_pairs':sum(x>0 for x in delta),'negative_pairs':sum(x<0 for x in delta)}
    summaries[pair]=d
record={'retained_records_before_dedup':len(pairs),'within_pair_unique_control_pairs':len(unique),'globally_unique_case_control_pairs':len(set((p['case_id'],p['control_id']) for p in pairs)),
        'scope':'Existing lightweight-study matches reused only when source full/SaProt classes fit. Sparse outcome-selected coverage, not a fresh matched sample for entire population. No experimental endpoint equality established.',
        'by_pair_and_class':summaries}
(O/'matched_control_summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
contrasts=read(O/'comparisons.json');metrics=read(O/'metrics.json');decisions={}
for pair in atlas:
    b=next(c for c in contrasts if c['candidate']==pair+'_feature_gate' and c['baseline']==pair+'_baseline')
    w=next(c for c in contrasts if c['candidate']==pair+'_feature_gate' and c['baseline']==pair+'_weight_selected')
    passed=b['delta_MCC']>=.005 and b['delta_MCC_99_75CI'][0]>0 and w['delta_MCC']>=.005 and w['delta_MCC_99_75CI'][0]>0 and b['delta_AUPRC']>=-.002 and sum(x>=0 for x in b['fold_net_fewer_errors'])>=4
    decisions[pair]={'prespecified_feature_gate_continue_gate_passed':passed,'delta_MCC_vs_baseline':b['delta_MCC'],'delta_MCC_vs_controlled_weight':w['delta_MCC'],'no_promotion_or_distillation':True}
(O/'decisions.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'unique_old_control_pairs':record['globally_unique_case_control_pairs'],'decisions':decisions},ensure_ascii=False))
