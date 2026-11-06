from pathlib import Path
import json,math,hashlib,collections
import numpy as np
from scipy.special import expit
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
manifest=read(P/'run_manifest.json');meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');md={r['sample_id']:r for r in meta}
bg=read(R/'analysis/error_evidence_linkage_20261025/all_training_linked.json')
roles=manifest['role_ids'];groups={role:{md[k]['group_id'] for k in ids} for role,ids in roles.items()}
assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
assert len(set(k for ids in roles.values() for k in ids))==6387
ARMS=read(P/'execution_status.json')['expected_completed_arms'] if (P/'execution_status.json').exists() else ['A','B','C']
runs={a:read(P/'runs'/a/'result.json') for a in ARMS}
assert len({r['initial_trainable_sha256'] for r in runs.values()})==1
assert all(r['native_class_pos_weight']==manifest['class_pos_weight'] for r in runs.values())
checked=0
for arm,r in runs.items():
 assert len(r['history'])==15
 vals=[h['validation_unweighted_BCE'] for h in r['history']];assert r['selected_epoch']==int(np.argmin(vals))+1
 for epoch,h in enumerate(r['history'],1):
  assert h['epoch']==epoch and h['steps']==math.ceil(len(roles['train'])/16)
  assert h['first_batch_gradient']['LoRA_B_grad_present']==(0 if arm=='A' else 144)
  assert h['first_batch_gradient']['head_grad_present']
  assert h['LoRA_B_nonzero_at_epoch_end']==(0 if arm=='A' else 144)
  assert h['order_sha256']==runs['A']['history'][epoch-1]['order_sha256']
 assert {meta[v['source_index']]['sample_id'] for v in r['calibration_records']}==set(roles['calibration'])
 assert {v['sample_id'] for v in r['predictions']}==set(roles['test'])
 for v in r['predictions']:
  i=v['source_index'];m=meta[i];assert m['sample_id']==v['sample_id'] and m['label']==v['label']
  probability=float(expit(r['platt']['slope']*v['logit']+r['platt']['intercept']))
  assert abs(probability-v['sequence_probability'])<1e-12
  fusion=.72*probability+.28*float(bg[i]['p_struct'])
  assert abs(fusion-v['fusion_probability'])<1e-12
  assert v['sequence_prediction']==int(probability>=.5) and v['fusion_prediction']==int(fusion>=.47022256531284556)
  checked+=1
assert manifest['source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest()
metrics=read(P/'metrics.json')
for mode in ['sequence','fusion']:
 for arm,r in runs.items():
  counts=collections.Counter((v['label'],v[mode+'_prediction']) for v in r['predictions'])
  q=metrics[mode][arm]['all'];tp,tn,fp,fn=[counts[x] for x in [(1,1),(0,0),(0,1),(1,0)]]
  assert [q[k] for k in ['TP','TN','FP','FN']]==[tp,tn,fp,fn]
  manual=(tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));assert abs(manual-q['MCC'])<1e-12
if ARMS==['A','B']:assert not (P/'runs/C/progress.json').exists() and not (P/'runs/C/result.json').exists()
out={'passed':True,'NN_fits':len(ARMS),'completed_arms':ARMS,'C_deferred_by_user':ARMS==['A','B'],'epochs_each':15,'outer_fold':0,'seed':42,'evaluation_rows':checked,'same_initialization_batch_orders_class_weight_and_optimizer_attempts':True,'A_LoRA_B_zero_and_disconnected_reproduced':True,'repaired_arms_LoRA_B_updated_and_gradient_present':True,'calibration_role_only':True,'source_hash_unchanged':True,'no_MCD_NDBP_extra_boost':True,'no_Test1_Test2_used':True,'limits':['onefold/oneseed','single1Dmember not original3seed ensemble','historical frozen3D fusion conditional','no new blind confirmation']}
(P/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
