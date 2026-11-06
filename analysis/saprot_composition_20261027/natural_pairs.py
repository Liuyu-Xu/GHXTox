from pathlib import Path
from collections import defaultdict
from itertools import combinations
import json
import numpy as np
from scipy.special import expit
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';z=np.load(A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');p=np.load(A/'saprot_standalone_20261026/ensemble_scores.npz');ids=z['sample_ids'].tolist();seq=z['sequences'].tolist();fold=z['folds'];groups=z['groups'];labels=z['labels'];bins=defaultdict(list);aa='ACDEFGHIKLMNPQRSTVWY'
for i,s in enumerate(seq):bins[(int(fold[i]),tuple(s.count(c) for c in aa))].append(i)
out=[]
for (f,counts),ii in bins.items():
 for i,j in combinations(ii,2):
  if seq[i]==seq[j]:continue
  out.append(dict(left_id=ids[i],right_id=ids[j],left_sequence=seq[i],right_sequence=seq[j],original_fold=f,same_group=bool(groups[i]==groups[j]),left_label_metadata_only=int(labels[i]),right_label_metadata_only=int(labels[j]),same_original_label=bool(labels[i]==labels[j]),teacher_left_probability=float(expit(p['SaProt_only_masked'][i])),teacher_right_probability=float(expit(p['SaProt_only_masked'][j])),teacher_probability_gap=float(abs(expit(p['SaProt_only_masked'][i])-expit(p['SaProt_only_masked'][j]))),baseline_probability_gap=float(abs(expit(p['baseline'][i])-expit(p['baseline'][j])))))
summary=dict(n_pairs=len(out),n_distinct_samples=len({r[k] for r in out for k in ['left_id','right_id']}),same_group_pairs=sum(r['same_group'] for r in out),same_label_pairs=sum(r['same_original_label'] for r in out),fold_counts={str(f):sum(r['original_fold']==f for r in out) for f in range(5)},teacher_mean_gap=float(np.mean([r['teacher_probability_gap'] for r in out])) if out else None,teacher_max_gap=max((r['teacher_probability_gap'] for r in out),default=None),teacher_gap_ge025=sum(r['teacher_probability_gap']>=.25 for r in out),teacher_opposite_decisions=sum((r['teacher_left_probability']>=.5)!=(r['teacher_right_probability']>=.5) for r in out),scope='all exactcomposition/samefold natural pairs, descriptive; not mechanism intervention')
(O/'natural_composition_pairs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');(O/'natural_pair_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
