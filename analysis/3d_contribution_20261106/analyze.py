"""Describe fixed six-seed + 3D changes; never fit or select on evaluation labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CASES = ROOT / 'analysis/six_seed_fusion_cys_20261106/cases.csv'
QUALITY = OUT / 'quality.csv'
TH = 0.47022256531284556

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(y, p):
    pred = p >= TH
    return {'mcc': float(matthews_corrcoef(y, pred)),
            'auroc': float(roc_auc_score(y, p)),
            'auprc': float(average_precision_score(y, p)),
            'brier': float(np.mean((p-y)**2)),
            'sensitivity': float(pred[y == 1].mean()),
            'specificity': float((~pred[y == 0]).mean())}

def main():
    d = pd.read_csv(CASES, dtype={'sample_id': str, 'group_id': str})
    q = pd.read_csv(QUALITY, dtype={'sample_id': str})
    assert not d.duplicated(['split','sample_id']).any() and not q.duplicated(['split','sample_id']).any()
    d = d.merge(q, on=['split','sample_id'], validate='one_to_one', how='left', indicator=True)
    assert (d['_merge'] == 'both').all() and len(d) == len(q) == 8095
    d.drop(columns='_merge', inplace=True)
    assert d.mean_plddt.between(0,1).all() and d.low_plddt_fraction.between(0,1).all()
    assert np.allclose(d.p_fused6, .72*d.p_seq6 + .28*d.p_3d, atol=1e-12)
    assert (d.seq_pred == (d.p_seq6 >= TH)).all() and (d.fusion_pred == (d.p_fused6 >= TH)).all()
    d['cys_bin'] = np.select([d.cys.eq(0),d.cys.ge(7)], ['0','>=7'], default='1-6')
    d['length_bin'] = np.select([d.length.le(20),d.length.le(35)], ['<=20','21-35'], default='>=36')
    d['quality_bin'] = np.select([d.mean_plddt.lt(.5),d.mean_plddt.lt(.7),d.mean_plddt.lt(.9)],
                                  ['<0.5','0.5-<0.7','0.7-<0.9'], default='>=0.9')
    d['disagreement_bin'] = np.select([abs(d.p_seq6-d.p_3d).lt(.1),abs(d.p_seq6-d.p_3d).lt(.3)],
                                      ['<0.1','0.1-<0.3'], default='>=0.3')
    d['brier_change'] = (d.p_fused6-d.label)**2-(d.p_seq6-d.label)**2
    d['structure_pred'] = (d.p_3d >= TH).astype(int)
    assert (d.structure_pred == d.structure_pred_diagnostic).all()
    d['seq_correct'] = d.seq_pred.eq(d.label)
    d['structure_correct'] = d.structure_pred.eq(d.label)
    d['fusion_correct'] = d.fusion_pred.eq(d.label)
    d['corrected'] = ~d.seq_correct & d.fusion_correct
    d['harmed'] = d.seq_correct & ~d.fusion_correct
    d['seq_only_branch_right'] = d.seq_correct & ~d.structure_correct
    d['struct_only_branch_right'] = ~d.seq_correct & d.structure_correct
    d['both_branch_wrong'] = ~d.seq_correct & ~d.structure_correct
    rows = []
    for split, sub in d.groupby('split', sort=False):
        for dimension in ['all','cys_bin','length_bin','quality_bin','disagreement_bin']:
            for group, part in ([('all',sub)] if dimension == 'all' else sub.groupby(dimension, sort=True)):
                for label in ['all',0,1]:
                    p = part if label == 'all' else part[part.label == label]
                    if not len(p): continue
                    rows.append({'split':split,'dimension':dimension,'group':group,'label':label,'n':len(p),
                         'independent_groups':int(p.group_id.nunique()) if split=='training_oof' else None,
                         'seq_errors':int((~p.seq_correct).sum()),'fusion_errors':int((~p.fusion_correct).sum()),
                         'corrected':int(p.corrected.sum()),'harmed':int(p.harmed.sum()),
                         'net_corrected':int(p.corrected.sum()-p.harmed.sum()),
                         'only_seq_branch_right':int(p.seq_only_branch_right.sum()),
                         'only_struct_branch_right':int(p.struct_only_branch_right.sum()),
                         'both_branch_wrong':int(p.both_branch_wrong.sum()),
                         'mean_brier_delta_fusion_minus_seq':float(p.brier_change.mean()),
                         'fold_corrected_harmed': ' | '.join(f'{int(f)}:{int(x.corrected.sum())}/{int(x.harmed.sum())}' for f,x in p.groupby('fold')) if split=='training_oof' else ''})
    table = pd.DataFrame(rows)
    for split, sub in d.groupby('split'):
        allrow=table[(table.split==split)&(table.dimension=='all')&(table.label=='all')].iloc[0]
        for dimension in ['cys_bin','length_bin','quality_bin','disagreement_bin']:
            part=table[(table.split==split)&(table.dimension==dimension)&(table.label=='all')]
            assert part.n.sum()==len(sub)
            assert part.corrected.sum()==allrow.corrected and part.harmed.sum()==allrow.harmed
    overlap = d[d.split=='test2'].merge(d[d.split=='test1'], on='sequence', suffixes=('_2','_1'), validate='one_to_one')
    assert len(overlap)==582 and (overlap.label_2 == overlap.label_1).all()
    quality_diff = (overlap.mean_plddt_2-overlap.mean_plddt_1).abs()
    d.to_csv(OUT/'sample_analysis.csv',index=False,encoding='utf-8-sig')
    table.to_csv(OUT/'strata.csv',index=False,encoding='utf-8-sig')
    d[d.corrected | d.harmed].to_csv(OUT/'changed_cases.csv',index=False,encoding='utf-8-sig')
    res={'sha256':{'cases':sha(CASES),'quality':sha(QUALITY)}, 'threshold':TH, 'fusion':'0.72 * six_1d + 0.28 * 3d',
         'descriptive_only':True,'test2_subset_test1_overlap_n':len(overlap),
         'test1_test2_same_sequence_max_abs_quality_diff':float(quality_diff.max()),
         'overall':{s:{'n':len(p),'sequence':metrics(p.label.to_numpy(),p.p_seq6.to_numpy()),
                       'fusion':metrics(p.label.to_numpy(),p.p_fused6.to_numpy())} for s,p in d.groupby('split')}}
    (OUT/'summary.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2))
    print(table[(table.label=='all')][['split','dimension','group','n','seq_errors','fusion_errors','corrected','harmed','mean_brier_delta_fusion_minus_seq']].to_string(index=False))

if __name__ == '__main__': main()
