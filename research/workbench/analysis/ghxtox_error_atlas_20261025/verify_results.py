"""Independent output-integrity and geometric-invariance checks."""
import json
import numpy as np
import pandas as pd
from build_atlas import OUT, P, structure_features

d=pd.read_csv(OUT/'all_training_atlas.csv',float_precision='round_trip')
e=pd.read_csv(OUT/'errors_only.csv',float_precision='round_trip')
p=pd.read_csv(OUT/'matched_pairs.csv',float_precision='round_trip')
assert len(d)==6387 and d.sample_id.is_unique
pred=d.p_fusion.to_numpy()>=P['fusion_threshold']
assert np.array_equal(pred!=d.label.to_numpy(),d.fusion_error.to_numpy())
assert set(e.sample_id)==set(d.loc[d.fusion_error,'sample_id'])
assert len(e)==327 and (e.fusion_category=='FP').sum()==135 and (e.fusion_category=='FN').sum()==192
assert np.max(abs(.72*d.p_seq+.28*d.p_struct-d.p_fusion))<1e-15
assert d.groupby('group_id').fold.nunique().max()==1
for kind,g in p.groupby('error_type'):
    groups=list(g.case_group)+list(g.control_group);assert len(groups)==len(set(groups))
    for _,r in g.iterrows():
        a=d.iloc[r.case_index];b=d.iloc[r.control_index]
        assert a.label==b.label and a.fold==b.fold and a.group_id!=b.group_id
        assert a.fusion_error and not b.fusion_error
        assert abs(a.length-b.length)<=3 and abs(a.net_charge_proxy-b.net_charge_proxy)<=2
        assert abs(a.mean_hydropathy-b.mean_hydropathy)<=.5+1e-12
        assert abs(a.cys_count-b.cys_count)<=1
        assert abs(a.aromatic_fraction-b.aromatic_fraction)<=.1+1e-12
# Geometry descriptors must not change under rigid rotation and translation.
rng=np.random.default_rng(5);xyz=rng.normal(size=(12,3))*4
rot=np.linalg.qr(rng.normal(size=(3,3)))[0]
first=structure_features('AKLCWREDVIYF',xyz)
second=structure_features('AKLCWREDVIYF',xyz@rot+np.array([11,4,-7]))
for k in first:assert abs(first[k]-second[k])<1e-12,k
for c in json.loads((OUT/'evidence_findings.json').read_text(encoding='utf-8'))['cases']:
    assert d.set_index('sample_id').loc[c['sample_id'],'sequence']==c['sequence']
print('PASS: probabilities, boundary decisions, full/error sets, pair calipers, group independence, geometric invariance, and evidence identities.')
