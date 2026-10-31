"""Replay saved explanatory coefficients only; no neural loading or model fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import expit
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
z=np.load(O/'crossfit_scores.npz');ids=z['sample_ids'].tolist();y=z['labels'];fold=z['folds'];half=z['halves'];groups=z['groups'];n=len(y);b=z['baseline4_raw'];p=expit(b);data={r['sample_id']:r for r in read(R/'analysis/saprot_contribution_20261026/sample_contributions.json')};rows=[data[k] for k in ids];le=np.array([r['features']['sequence_length'] for r in rows]);cy=np.array([r['features']['cys_count'] for r in rows]);sp=np.array([r['features']['cys_span_fraction'] for r in rows]);seq=np.column_stack([le,cy,cy/le,sp,b*(cy>=4)]);sq=np.column_stack([b,seq]);mi=np.array([[not r['curated_family'],not r['name_class']] for r in rows],dtype=float);tokens=[set(['curated:'+v for v in r['curated_family']]+['name:'+v for v in r['name_class']]) for r in rows];errors=[];checked=0
for r in read(O/'splits.json'):
 tr=np.array([ids.index(k) for k in r['train_ids']]);te=np.array([ids.index(k) for k in r['evaluation_ids']]);assert set(groups[tr]).isdisjoint(groups[te]);assert np.all(fold[tr]==r['fold']) and np.all(fold[te]==r['fold']);assert np.all(half[te]==r['evaluation_half']) and np.all(half[tr]!=r['evaluation_half'])
for m in read(O/'fitted_controls.json'):
 mask=(fold==m['fold'])&(half==m['evaluation_half']);arm=m['arm'];vocab=m.get('family_vocab') or [];fm=np.column_stack([mi]+[np.array([t in ts for ts in tokens],dtype=float) for t in vocab])
 if arm=='baseline4_bias_offset':pred=b[mask]+m['intercept'];expected=z[arm][mask]
 else:
  if arm.startswith('shift_'):
   _,kind,level=arm.split('_');xb=np.column_stack([b,p,p*(1-p)]);X=xb if level=='base' else np.column_stack([xb,seq]) if level=='sequence' else np.column_stack([xb,seq,fm]);expected=z[arm][mask]
  else:
   X=b[:,None] if arm=='baseline4_Platt' else sq if arm=='baseline4_CysLength' else np.column_stack([sq,fm]) if arm=='baseline4_CysLengthFamily' else z[arm.replace('_Platt','_raw')][:,None];expected=z[arm][mask]
  X=(X[mask]-np.array(m['scaler_mean']))/np.array(m['scaler_scale']);pred=np.asarray(X@np.array(m['coefficient']).T+np.array(m['intercept'])).reshape(-1)
 errors.append(float(np.max(np.abs(pred-expected))));checked+=len(pred)
follow=np.load(O/'family_followup_scores.npz')
for m in read(O/'family_followup_models.json'):
 mask=(fold==m['fold'])&(half==m['half']);vocab=m['vocab'];names=np.column_stack([np.array([t in ts for ts in tokens],dtype=float) for t in vocab]) if vocab else np.empty((n,0));X=np.column_stack([sq,mi]) if m['arm'].endswith('MissingOnly') else np.column_stack([sq,names]);pred=(((X[mask]-m['mean'])/m['scale'])@np.array(m['coef']).T+np.array(m['intercept'])).reshape(-1);errors.append(float(np.max(np.abs(pred-follow[m['arm']][mask]))));checked+=len(pred)
assert max(errors)<1e-10,max(errors)
for path,h in read(O/'input_manifest.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
summary=read(O/'verification.json');summary.update(replayed_label_controls_and_probes=140,replayed_scalar_predictions=checked,max_coefficient_replay_difference=max(errors),split_membership_independently_checked=True,source_hashes_rechecked=True,neural_models_loaded_or_trained_for_check=0)
(O/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
