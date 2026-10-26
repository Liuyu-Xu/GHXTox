"""Verify saved diagnosis against original stable OOF scores, without fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=read(O/'sample_contributions.json');s=np.load(R/'analysis/saprot_ensemble_size_control_20261026/ensemble_scores.npz');idx={r['sample_id']:r for r in rows};assert len(idx)==6387
arms=['fusion_3Di4','fusion_masked4'];summary=read(O/'contribution_summary.json')
for a in arms:
 for i,k in enumerate(s['sample_ids']):
  r=idx[str(k)];assert r['label']==int(s['labels'][i]);b=(s['baseline4'][i]>=0)==r['label'];f=(s[a][i]>=0)==r['label'];role='corrected' if not b and f else 'harmed' if b and not f else 'persistent_correct' if b and f else 'persistent_wrong'
  assert r[a]['role']==role and abs(r[a]['probability']-expit(s[a][i]))<1e-15
 for label in ['all',0,1]:
  subset=[r for r in rows if label=='all' or r['label']==label];t=next(t for t in summary if t['arm']==a and t['label']==label)
  for role in ['corrected','harmed','persistent_wrong','persistent_correct']:assert t[role]==sum(r[a]['role']==role for r in subset)
for p in read(O/'matched_controls.json'):
 c,d=idx[p['case_id']],idx[p['control_id']];a=p['arm'];assert c[a]['role']=='corrected' and d[a]['role']==p['control_role'];assert c['label']==d['label'] and c['fold']==d['fold'] and c['group_id']!=d['group_id'];assert p['identity']>=.4 and min(p['coverage_case'],p['coverage_control'])>=.7
 for key,v in p['feature_differences'].items():
  x,y=c['features'][key],d['features'][key]
  if x is not None and y is not None:assert abs(v-(x-y))<1e-12
manifest=read(O/'input_manifest.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.items())
old=read(O/'verification.json');old.update(independently_checked_role_and_probability_rows=6387*2,independently_checked_summary_counts=True,independently_checked_matching_constraints=True,source_hashes_rechecked=True)
(O/'verification.json').write_text(json.dumps(old,indent=2),encoding='utf-8');print(json.dumps(old,indent=2))
