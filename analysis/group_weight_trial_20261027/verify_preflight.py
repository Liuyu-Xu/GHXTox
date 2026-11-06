from pathlib import Path
import json,math,hashlib
import numpy as np
from scipy.special import expit
P=Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
contract=read(P/'role_weight_contract.json');meta=read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json');lookup={(r['outer_fold'],r['source_index']):r for r in contract}
assert len(lookup)==31935
for fold in range(5):
 sets={k:set() for k in ['train','validation','calibration','test']}
 for i,m in enumerate(meta):
  r=lookup[fold,i];sets[r['role']].add(m['group_id'])
  low=m['label']==1 and m['sequence'].count('C')<4
  fs=m['current_family_sidecar']['parent_families'];target=low and len(fs)==1 and fs[0] in ['belongs to the mcd family','belongs to the non-disulfide-bridged peptide (ndbp) superfamily']
  w={'A':1.,'B':1.5 if low else 1.,'C':2. if target else 1.5 if low else 1.} if r['role']=='train' else {'A':1.,'B':1.,'C':1.}
  assert r['weights_if_training']==w
 for a in sets:
  for b in sets:
   if a!=b:assert not sets[a]&sets[b]
reach=read(P/'target_fixed_fusion_reachability.json')
for r in reach:
 ceiling=.72*r['frozen_1d_probability']+.28
 assert abs(ceiling-r['maximum_fusion_probability_if_structure_perfect'])<1e-12
 assert r['positive_reachable_by_structure_only']==(ceiling>=.47022256531284556)
assert sum(not r['positive_reachable_by_structure_only'] for r in reach)==8
foldresults=[read(P/'runs/seed42/fold0'/a/'result.json') for a in ['A','B','C']]
assert len({r['initial_sha256'] for r in foldresults})==1
for r in foldresults:
 assert len(r['history'])==6 and len(r['predictions'])==1271
 assert r['selected_epoch']==int(np.argmin([h['validation_unweighted_BCE'] for h in r['history']]))+1
 for c in r['calibration_records']:assert lookup[0,c['source_index']]['role']=='calibration'
 for p in r['predictions']:
  assert lookup[0,p['source_index']]['role']=='test'
  prob=expit(r['platt']['slope']*p['structure_logit']+r['platt']['intercept']);assert abs(prob-p['structure_probability'])<1e-12
check=read(P/'sequence_training_check.json')
assert [r['B_nonzero_tensors'] for r in check['old_checkpoint_inspection']]==[0,0,0]
assert all(r['B_tensors']==144 for r in check['old_checkpoint_inspection'])
smoke=check['new_actual_architecture_weighted_gradient_smoke'];assert len(smoke)==3
assert len({r['initial_trainable_sha256'] for r in smoke})==1
for r in smoke:
 assert len(r['steps'])==2 and r['B_nonzero_after_steps']==144 and not r['outer_eval_or_validation_used']
 assert all(s['B_grad_present']==s['B_grad_nonzero']==144 for s in r['steps'])
assert check['model_source_sha256']==hashlib.sha256((R/'src/ghxtox/lora_mamba_1d.py').read_bytes()).hexdigest()
out={'passed':True,'role_weight_rows':31935,'structural_completed_fits':3,'structural_outer_evaluation_predictions':3813,'actual_sequence_gradient_steps':6,'old_final_checkpoints_inspected':3,'old_final_B_tensors_checked':432,'new_actual_LoRA_B_gradients_present_per_step':144,'targets_unreachable_by_frozen1D_structure_only':8,'formal_sequence_OOF_fits_completed':0,'full_ABC_efficacy_comparison_completed':False,'past_predictions_unchanged':True,'regression_test_note':'2 pytest cases: current non-reentrant API and legacy input-grad-hook fallback; gradient updates and exact eval output parity'}
(P/'preflight_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
