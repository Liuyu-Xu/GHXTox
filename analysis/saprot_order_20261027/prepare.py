from pathlib import Path
import json,hashlib,random
from collections import Counter,defaultdict
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];A=R/'analysis';read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(A/'saprot_contribution_20261026/sample_contributions.json');by={r['sample_id']:r for r in rows};pairs=read(A/'saprot_contribution_20261026/matched_controls.json');chosen=set();reasons=defaultdict(set)
def add(k,why):chosen.add(k);reasons[k].add(why)
for r in rows:
 if r['fusion_masked4']['role'] in ['corrected','persistent_wrong']:add(r['sample_id'],'all_baseline4_errors')
 if any(r[a]['role']=='harmed' for a in ['fusion_masked4','fusion_3Di4']):add(r['sample_id'],'all_fusion_harms')
buckets=defaultdict(list)
for r in rows:
 if all(r[a]['role']=='persistent_correct' for a in ['fusion_masked4','fusion_3Di4']):buckets[(r['label'],r['features']['cys_count']>=4,len(r['sequence'])>25,r['fold'])].append(r['sample_id'])
for keys,bb in sorted(buckets.items()):
 for k in sorted(bb,key=lambda k:hashlib.sha256((k+'|background_v1').encode()).hexdigest())[:5]:add(k,'stratified_persistent_correct')
for p in pairs:add(p['control_id'],'prior_matched_control')
cohort=[dict(r,selection_reasons=sorted(reasons[r['sample_id']])) for r in rows if r['sample_id'] in chosen];out=[];modes=['local_nonCys_shuffle','global_nonCys_shuffle','Cys_relocation','global_all_shuffle']
def mutate(s,mode,rng):
 a=list(s);positions=[i for i,c in enumerate(s) if c!='C']
 if mode=='local_nonCys_shuffle':
  for b in range(0,len(s),4):
   pp=[i for i in positions if b<=i<b+4];vv=[a[i] for i in pp];rng.shuffle(vv)
   for i,c in zip(pp,vv):a[i]=c
 elif mode=='global_nonCys_shuffle':
  vv=[a[i] for i in positions];rng.shuffle(vv)
  for i,c in zip(positions,vv):a[i]=c
 elif mode=='Cys_relocation':
  cp=set(rng.sample(range(len(s)),s.count('C')));it=iter(c for c in s if c!='C');a=['C' if i in cp else next(it) for i in range(len(s))]
 else:rng.shuffle(a)
 return ''.join(a)
z=np.load(A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz');seqfold=defaultdict(set)
for s,f in zip(z['sequences'],z['folds']):seqfold[str(s)].add(int(f))
for r in cohort:
 k,s=r['sample_id'],r['sequence'];out.append(dict(key=k+'|original',sample_id=k,mode='original',replicate=-1,sequence=s,original_sequence=s,fold=r['fold'],changed_fraction=0.,eligible=True,exact_otherfold_training=False))
 for mode in modes:
  for rep in range(2):
   seed=int(hashlib.sha256(f'{k}|{mode}|{rep}|saprot_order_v1'.encode()).hexdigest()[:16],16);rng=random.Random(seed);ss=s
   for _ in range(20):
    ss=mutate(s,mode,rng)
    if ss!=s:break
   assert len(ss)==len(s) and Counter(ss)==Counter(s)
   if 'nonCys' in mode:assert [i for i,c in enumerate(ss) if c=='C']==[i for i,c in enumerate(s) if c=='C']
   collision=any(f!=r['fold'] for f in seqfold.get(ss,set()));out.append(dict(key=f'{k}|{mode}|{rep}',sample_id=k,mode=mode,replicate=rep,sequence=ss,original_sequence=s,fold=r['fold'],changed_fraction=sum(a!=b for a,b in zip(s,ss))/len(s),eligible=ss!=s and not collision,unchanged=ss==s,exact_otherfold_training=collision,random_seed=seed))
for name,data in [('cohort.json',cohort),('sequence_manifest.json',out)]: (O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
summary=dict(n_cohort=len(cohort),role_counts={a:dict(Counter(r[a]['role'] for r in cohort)) for a in ['fusion_masked4','fusion_3Di4']},n_sequences=len(out),unique_sequences=len({r['sequence'] for r in out}),ineligible_by_mode={m:sum(r['mode']==m and not r['eligible'] for r in out) for m in modes},matched_pairs_retained=len(pairs))
(O/'cohort_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(O/'source_hashes.json').write_text(json.dumps({str(p):sha(p) for p in [O/'protocol.json',A/'saprot_contribution_20261026/sample_contributions.json',A/'saprot_contribution_20261026/matched_controls.json',A/'saprot_and_robust_classifier_pilot_20261026/frozen_features.npz']},indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
