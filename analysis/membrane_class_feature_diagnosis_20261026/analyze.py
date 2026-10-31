"""Fixed-cohort exploratory feature diagnosis. No toxicity model or correction."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
paths=[R/'analysis/classification_mechanism_map_20261026/membrane76_comparison_pool.json',R/'analysis/error_evidence_linkage_20261025/all_training_linked.json',O/'protocol.json']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in paths};pool=read(paths[0]);old={r['sample_id']:r for r in read(paths[1])};protocol=read(paths[2])
rows=pool['all76'];assert len(rows)==76
def ids(s):return {'peptide_'+x for x in s.split()}
negative=ids('174 431 444 463 510 572 631 1199')
protective=ids('477');unrelated=ids('324');low=ids('44 102 733');mixed=ids('107 1434')
tentative=ids('18 203 291 314 337 619 781 821 999 1047 1444')
indirect=ids('422 544 562 655 699 800 851');conditional=ids('58 59 110 870')
modeltext=ids('741')
known_entity=ids('74 201 160 288 652 1551 733 251 424')
notes={
 'peptide_324':'血淋巴凝固，非红细胞溶血/膜破坏；旧关键词命中不相干。',
 'peptide_477':'抑制补体途径溶血，不能当肽自身诱发溶血。',
 'peptide_444':'注释否定溶血；旧原文记录包含阈值/检测范围，不能把有测量当有阳性效应。',
 'peptide_463':'注释否定溶血但提及细胞毒性；旧原文弱溶血/半数阈值证据保留，非同一终点。',
 'peptide_74':'完整双链毒素作用不能直接分配给孤立B链。',
 'peptide_201':'完整双链毒素作用不能直接分配给孤立A链。',
 'peptide_160':'此前逐肽原文核查未找到本条直接溶血，不借Lycotoxin I实验。',
 'peptide_288':'2a直接溶血正确对照已撤回，不借2b曲线。',
 'peptide_652':'2c同家族注释不能替代逐肽直接实验。',
 'peptide_1551':'共享功能陈述不能证明本条实验与输入分子对应。',
 'peptide_733':'原句标明Cryptide TyPep-8；需区分其与本输入的完整对象。',
 'peptide_251':'H1化学形式冲突未解决。',
 'peptide_424':'H2化学形式冲突未解决。',
 'peptide_741':'该命中句描述膜作用位点排布/相似性推断，不是独立直接溶血实验。'}
for r in rows:
 sid=r['sample_id'];state='affirmative_annotation_not_gold'
 for group,label in [(negative,'negative_endpoint_text'),(protective,'protective_inhibits_hemolysis'),(unrelated,'unrelated_hemolymph'),(low,'low_activity_text'),(mixed,'dose_or_endpoint_mixed_text'),(tentative,'tentative_or_similarity_only'),(indirect,'indirect_hemolysis_text'),(conditional,'condition_or_cofactor_dependent_text'),(modeltext,'structural_explanation_not_endpoint')]:
  if sid in group:state=label
 r['semantic_state']=state;r['semantic_note']=notes.get(sid,'仅复核已有命中原句的含义，不升级为指定完整肽的机制实验证据。')
 r['known_entity_or_evidence_caution']=sid in known_entity
 r['no_simple_flag']=not r['fragment_flag'] and not r['cross_source_label_disagreement']
 r['evidence_caution_subset']=r['no_simple_flag'] and sid not in (negative|protective|unrelated|known_entity|modeltext)
 r['fusion_error']=r['fusion_category']=='FN'
features=protocol['primary_features'];background=protocol['background']
def val(r,k):
 try:return float(old[r['sample_id']][k])
 except (ValueError,TypeError,KeyError):return np.nan
X=np.array([[val(r,k) for k in background] for r in rows]);fold=np.array([r['fold'] for r in rows]);groups=np.array([r['group_id'] for r in rows]);err=np.array([r['fusion_error'] for r in rows]);assert np.isfinite(X).all()
def diff(z,mask):
 mask=mask & np.isfinite(z)
 vals=[]
 for outcome in [True,False]:
  g=[float(np.mean(z[mask & (groups==k) & (err==outcome)])) for k in sorted(set(groups[mask & (err==outcome)]))]
  if not g:return None
  vals.append(float(np.mean(g)))
 return vals[0]-vals[1]
def boot(z):
 # Same cluster draw supplies both outcomes if a group contains both.
 unique=sorted(set(groups[np.isfinite(z)]));matrix=np.full((len(unique),2),np.nan)
 for i,k in enumerate(unique):
  for j,v in enumerate([True,False]):
   m=(groups==k)&(err==v)&np.isfinite(z)
   if m.any():matrix[i,j]=z[m].mean()
 rng=np.random.default_rng(protocol['fixed_seed']);a=[]
 for _ in range(5000):
  picked=matrix[rng.integers(0,len(unique),len(unique))]
  if np.isfinite(picked).sum(axis=0).min()>0:a.append(float(np.nanmean(picked[:,0])-np.nanmean(picked[:,1])))
 return dict(ci95=np.quantile(a,[.025,.975]).tolist(),ci9875=np.quantile(a,[.00625,.99375]).tolist(),valid_draws=len(a))
results=[];fit_checks=[]
for feature in features:
 y=np.array([val(r,feature) for r in rows]);valid=np.isfinite(y);res=np.full(76,np.nan)
 for f in range(5):
  train=(fold!=f)&valid;test=(fold==f)&valid
  assert not set(groups[train])&set(groups[test])
  mu=y[train].mean();sd=y[train].std();assert sd>0
  model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10.0))
  model.fit(X[train],(y[train]-mu)/sd)
  res[test]=(y[test]-mu)/sd-model.predict(X[test])
  fit_checks.append(dict(feature=feature,fold=f,n_train=int(train.sum()),n_test=int(test.sum()),training_y_mean=float(mu),training_y_sd=float(sd),groups_disjoint=True,task='predict spatial feature, not toxicity or error'))
 for i,r in enumerate(rows):r.setdefault('spatial_residuals',{})[feature]=float(res[i]) if np.isfinite(res[i]) else None
 allmask=np.ones(76,dtype=bool);estimate=diff(res,allmask);ci=boot(res)
 byfold={str(f):diff(res,fold==f) for f in range(5)}
 simple=np.array([r['no_simple_flag'] for r in rows]);evid=np.array([r['evidence_caution_subset'] for r in rows])
 s1=diff(res,simple);s2=diff(res,evid);sign=lambda a,b:a is not None and b is not None and a*b>0
 consistent=sum(sign(v,estimate) for v in byfold.values());passes=(ci['ci9875'][0]>0 or ci['ci9875'][1]<0) and consistent>=4 and sign(s1,estimate) and sign(s2,estimate)
 results.append(dict(feature=feature,n_valid=int(valid.sum()),missing_sample_ids=[rows[i]['sample_id'] for i in range(76) if not valid[i]],raw_group_difference=diff(y,allmask),adjusted_residual_group_difference=estimate,**ci,fold_differences=byfold,folds_same_sign=consistent,no_simple_flag_difference=s1,evidence_caution_difference=s2,continue_screen_passed=bool(passes),scope='residual association with old error; not causal effect, classifier gain, family-matched or heldout study validation'))
# Background differences are descriptive only, no repeat significance scan.
bg=[dict(feature=k,raw_group_difference=diff(X[:,j],np.ones(76,dtype=bool))) for j,k in enumerate(background)]
counts=dict(n=76,semantics=dict(collections.Counter(r['semantic_state'] for r in rows)),no_simple_flag_n=sum(r['no_simple_flag'] for r in rows),no_simple_flag_errors=sum(r['no_simple_flag'] and r['fusion_error'] for r in rows),evidence_caution_n=sum(r['evidence_caution_subset'] for r in rows),evidence_caution_errors=sum(r['evidence_caution_subset'] and r['fusion_error'] for r in rows),new_spatial_feature_regressions=20,new_toxicity_model_fits=0,spatial_extraction_runs=0)
for n,obj in [('semantic_and_feature_rows.json',rows),('spatial_comparisons.json',results),('background_descriptive.json',bg),('adjustment_fit_audit.json',fit_checks),('summary.json',counts)]: (O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
assert before=={str(p):sha(p) for p in paths}
(O/'verification.json').write_text(json.dumps(dict(source_hashes=before,source_unchanged=True,n_rows=76,n_features=4,background_all_finite=True,missing_targets_not_imputed=True,all_adjustment_train_test_groups_disjoint=True,new_toxicity_model_fits=0,test_data_read=False,probabilities_labels_and_original_patterns_unchanged=True),indent=2),encoding='utf-8')
print(json.dumps(dict(counts=counts,results=results),ensure_ascii=False,indent=2))
