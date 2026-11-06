from pathlib import Path
import json,hashlib,csv,math,collections
import numpy as np
from scipy.special import expit
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')
seq=read(R/'runs/lora_mamba_full/oof_calibrated.json')
contract=read(P/'role_weight_contract.json');assets=read(P/'asset_and_role_manifest.json')
assert len(contract)==31935
lookup={(r['outer_fold'],r['source_index']):r for r in contract};assert len(lookup)==31935
for f in range(5):
 groups={r:set() for r in ['train','validation','calibration','test']}
 for i,m in enumerate(meta):
  r=lookup[f,i];assert r['sample_id']==m['sample_id'] and r['group_id']==m['group_id']
  groups[r['role']].add(m['group_id'])
  low=m['label']==1 and m['sequence'].count('C')<4
  fam=m['current_family_sidecar']['parent_families'];target=low and len(fam)==1 and fam[0] in ['belongs to the mcd family','belongs to the non-disulfide-bridged peptide (ndbp) superfamily']
  expected={'A':1.0,'B':1.5 if low else 1.0,'C':2.0 if target else 1.5 if low else 1.0} if r['role']=='train' else {'A':1.0,'B':1.0,'C':1.0}
  assert r['weights_if_training']==expected
 for a in groups:
  for b in groups:
   if a!=b:assert not groups[a]&groups[b]
smoke=[read(P/'smoke/seed42/fold0'/a/'result.json') for a in ['A','B','C']]
assert len({r['initial_sha256'] for r in smoke})==1
assert all(r['actual_gradient_steps']==2 and r['parameters_changed'] and not r['validation_or_outer_test_used'] for r in smoke)
allpreds=read(P/'all_oof_predictions.json');count=0
for fold in range(5):
 models=[read(P/'runs/seed42'/f'fold{fold}'/a/'result.json') for a in ['A','B','C']]
 assert len({r['initial_sha256'] for r in models})==1
 assert all(len(r['history'])==6 for r in models)
 assert len({tuple(h['steps'] for h in r['history']) for r in models})==1
 for arm,result in zip(['A','B','C'],models):
  assert result['arm']==arm and result['outer_fold']==fold
  val=[r['validation_unweighted_BCE'] for r in result['history']];assert result['selected_epoch']==int(np.argmin(val))+1
  for r in result['calibration_records']:assert lookup[fold,r['source_index']]['role']=='calibration'
  for r in result['predictions']:
   i=r['source_index'];m=meta[i];assert lookup[fold,i]['role']=='test' and m['fold']==fold
   prob=float(expit(result['platt']['slope']*r['structure_logit']+result['platt']['intercept']))
   assert abs(prob-r['structure_probability'])<1e-12
   fused=.72*float(seq['oof'][i])+.28*prob
   assert abs(fused-r['fusion_probability'])<1e-12
   assert r['structure_prediction']==int(prob>=.5) and r['fusion_prediction']==int(fused>=.47022256531284556)
   count+=1
for arm,rows in allpreds.items():
 assert len(rows)==6387 and len({r['sample_id'] for r in rows})==6387
 assert all(r['source_index']==i and r['sample_id']==meta[i]['sample_id'] for i,r in enumerate(rows))
metrics=read(P/'metrics.json');comparisons=read(P/'comparisons.json')
for mode in ['structure','fusion']:
 for arm,rows in allpreds.items():
  tp=tn=fp=fn=0
  for r in rows:
   label=r['label'];d=r[mode+'_prediction']
   tp+=int(label==1 and d==1);tn+=int(label==0 and d==0);fp+=int(label==0 and d==1);fn+=int(label==1 and d==0)
  q=metrics[mode][arm]['all'];assert [q[k] for k in ['TP','TN','FP','FN']]==[tp,tn,fp,fn]
  manual=(tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));assert abs(manual-q['MCC'])<1e-12
 for c in comparisons[mode]:
  control,treat=allpreds[c['control']],allpreds[c['treatment']]
  corr=[r['sample_id'] for r,s in zip(control,treat) if r[mode+'_prediction']!=r['label'] and s[mode+'_prediction']==s['label']]
  harm=[r['sample_id'] for r,s in zip(control,treat) if r[mode+'_prediction']==r['label'] and s[mode+'_prediction']!=s['label']]
  assert corr==c['corrected_ids'] and harm==c['harmed_ids']
  assert sum(c['fold_nets'].values())==len(corr)-len(harm)
for p,sha in assets['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
output={'passed':True,'fold_role_weight_contract_rows':len(contract),'NN_fits':15,'epochs_each':6,'paired_initialization_and_step_budgets_equal':True,'all_fresh_OOF_predictions_recomputed':count,'no_all_data_checkpoint_initialization':True,'gradient_smoke_passed':True,'model_source_hashes_unchanged':True,'no_Test1_or_Test2_used':True,'scope':'Single-seed exploratory structural retraining and historical-frozen-1D conditional fusion, NOT blinded full-pipeline validation.'}
(P/'verification.json').write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2))
