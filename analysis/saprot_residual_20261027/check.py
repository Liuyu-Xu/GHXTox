"""Check stored split targets, strict definitions and seed-probe coefficient replay."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import expit
from sklearn.metrics.pairwise import rbf_kernel
O=Path(__file__).resolve().parent;R=O.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));z=np.load(O/'seed_residuals.npz');old=np.load(R/'analysis/saprot_composition_20261027/probe_predictions.npz');fold=z['folds'];half=z['halves'];groups=z['groups'];labels=z['labels'];ids=z['sample_ids'].tolist();records={r['sample_id']:r for r in read(O/'all_residual_records.json')};aa='ACDEFGHIKLMNPQRSTVWY';X=np.array([[records[k]['sequence'].count(a)/len(records[k]['sequence']) for a in aa]+[len(records[k]['sequence'])] for k in ids]);maxdiff=0.
for m in read(O/'seed_probe_models.json'):
 tr=(fold==m['fold'])&(half!=m['half']);te=(fold==m['fold'])&(half==m['half']);assert set(groups[tr]).isdisjoint(groups[te]);assert np.array_equal(np.where(tr)[0],m['train_indices']);a=(X[tr]-m['composition_mean'])/m['composition_scale'];b=(X[te]-m['composition_mean'])/m['composition_scale'];prediction=rbf_kernel(b,a,gamma=1/21)@np.array(m['coefficients'])+m['teacher_target_means'];maxdiff=max(maxdiff,float(np.max(abs(prediction-z['composition_predictions'][te]))))
assert maxdiff<1e-10;tp=expit(z['teacher_logits']);res=tp-np.clip(z['composition_predictions'],0,1);assert np.allclose(res,z['residual_clipped']);bok=(z['baseline_logits']>=0)==labels[:,None];aok=(z['actual_seed_fusion_scores']>=0)==labels[:,None];pok=(z['composition_seed_fusion_scores']>=0)==labels[:,None];ori=res*(2*labels[:,None]-1);strict=np.all(~bok&aok&~pok,axis=1)&np.all(ori>0,axis=1);bad=np.all(bok&~aok&pok,axis=1)&np.all(ori<0,axis=1);assert np.array_equal(strict,z['strict_useful']) and np.array_equal(bad,z['strict_misleading']);assert np.max(abs(z['composition_predictions'].mean(axis=1)-old['teacher_composition_RBF']))<1e-10
for p,h in read(O/'input_manifest.json').items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
v=read(O/'verification.json');v.update(seed_predictions_replayed=6387*2,max_replay_difference=maxdiff,strict_and_mirror_rules_independently_checked=True,average_probe_reconstruction_rechecked=True,source_hashes_rechecked=True);(O/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8');print(json.dumps(v,indent=2))
