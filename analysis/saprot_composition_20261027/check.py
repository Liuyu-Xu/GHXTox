"""Replay saved scalar probe coefficients, without model refitting or neural loads."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.pairwise import rbf_kernel
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));z=np.load(R/'analysis/saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');saved=np.load(O/'probe_predictions.npz');seqs=z['sequences'].tolist();n=len(seqs);X=np.zeros((n,21));D=np.zeros((n,400));aa='ACDEFGHIKLMNPQRSTVWY';ai={a:i for i,a in enumerate(aa)}
for i,s in enumerate(seqs):
 c=np.array([s.count(a) for a in aa],float);L=len(s);X[i]=np.r_[c/L,L];obs=np.zeros((20,20))
 for a,b in zip(s,s[1:]):obs[ai[a],ai[b]]+=1
 D[i]=(obs/(L-1)-(np.outer(c,c)-np.diag(c))/(L*(L-1))).ravel()
assert np.max(abs(D.sum(axis=1)))<1e-12;Q=PolynomialFeatures(2,include_bias=False).fit_transform(X);fold=saved['folds'];half=saved['halves'];groups=saved['groups'];maxdiff=0.;count=0
for model in read(O/'teacher_probability_models.json'):
 arm=model['arm'];tr=(fold==model['fold'])&(half!=model['evaluation_half']);te=(fold==model['fold'])&(half==model['evaluation_half']);assert set(groups[tr]).isdisjoint(groups[te]);mu=model['teacher_target_mean'];assert abs(mu-float(saved['true_teacher_probability'][tr].mean()))<1e-12
 if model['kind']=='constant':p=np.full(te.sum(),mu)
 elif model['kind']=='linear':
  design=X if arm=='composition_linear' else Q if arm=='composition_quadratic' else np.column_stack([X,D]);normalized=(design[te]-model['design_mean'])/model['design_scale'];p=normalized@np.array(model['teacher_coefficient'])+mu
 else:
  assert np.array_equal(np.where(tr)[0],model['train_indices']);ct=(X[tr]-model['composition_mean'])/model['composition_scale'];cv=(X[te]-model['composition_mean'])/model['composition_scale'];K=rbf_kernel(cv,ct,gamma=1/21)
  if arm=='composition_order_kernel':dt=(D[tr]-model['order_mean'])/model['order_scale'];dv=(D[te]-model['order_mean'])/model['order_scale'];K=(K+dv@dt.T/model['order_normalizer'])/2
  p=K@np.array(model['teacher_coefficient'])+mu
 diff=float(np.max(abs(p-saved['teacher_'+arm][te])));maxdiff=max(maxdiff,diff);count+=int(te.sum())
assert maxdiff<1e-10,maxdiff
for path,h in read(O/'input_manifest.json').items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
verification=read(O/'verification.json');verification.update(scalar_probe_models_replayed=60,scalar_predictions_replayed=count,max_scalar_replay_difference=maxdiff,residual_dipeptide_frequency_sums_zero=True,all_replayed_meta_groups_disjoint=True,source_hashes_rechecked=True)
(O/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print(json.dumps(verification,indent=2))
