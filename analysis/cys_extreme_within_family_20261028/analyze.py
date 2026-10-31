"""Descriptive new extreme-Cys family matching on two archived OOF outputs."""
import json,csv,hashlib,statistics,math
from pathlib import Path
from collections import Counter,defaultdict
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
paths={'meta':R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json','old':R/'analysis/error_evidence_linkage_20261025/all_training_linked.json','new':R/'analysis/lora_multifold_screen_20261028/all_fold_predictions.json'};data={k:read(p) for k,p in paths.items()};meta=data['meta'];old={x['sample_id']:x for x in data['old']};new={x['sample_id']:x for x in data['new']};assert len(meta)==len(old)==len(new)==6387
threshold=.47022256531284556
features=['sequence_length','cys_count','cys_span_fraction','net_charge_proxy','mean_hydropathy','aromatic_fraction','local_positive_excess','local_hydrophobic_excess','ideal_alpha_hydrophobic_moment','mean_plddt','hydrophobic_contact_excess','positive_hydrophobic_contact_excess','charge_hydrophobic_separation','nonlocal_contact_density','ca_radius_gyration']
rows=[]
for m in meta:
 o=old[m['sample_id']];n=new[m['sample_id']];y=m['label'];assert int(o['label'])==n['label']==y and int(o['fold'])==n['fold']==int(m['fold']) and n['group_id']==m['group_id'] and o['sequence']==m['sequence']
 oldp=float(o['p_fusion']);newp=float(n['fusion_base']);assert bool(m['old_archived_wrong'])==((oldp>=threshold)!=bool(y))
 fs=m['current_family_sidecar']['full_families'];rows.append({'sample_id':m['sample_id'],'sequence':m['sequence'],'group_id':m['group_id'],'fold':n['fold'],'label':y,'cys':m['cys_count'],'length':m['length'],'flag':bool(m['uniform_flag']),'family':fs[0] if len(fs)==1 else None,'family_parents':m['current_family_sidecar']['parent_families'],'old_error':(oldp>=threshold)!=bool(y),'new_error':(newp>=threshold)!=bool(y),'old_p':oldp,'new_p':newp,'seq_p':float(o['p_seq']),'struct_p':float(o['p_struct']),'features':m['features'],'identity_status':m['identity_status'],'original_endpoint':m['original_endpoint'],'disulfide_status':m['disulfide_status'],'modification_status':m['modification_status']})
cohorts={'low_0_1_positive':[x for x in rows if x['label']==1 and x['cys']<=1],'high_6plus_negative':[x for x in rows if x['label']==0 and x['cys']>=6]}
summary={};matches={};families={}
for name,subset in cohorts.items():
 summary[name]={};families[name]={};matches[name]={}
 for version in ['old','new']:
  err=[x for x in subset if x[version+'_error']];good=[x for x in subset if not x[version+'_error']]
  summary[name][version]={'n':len(subset),'errors':len(err),'correct':len(good),'error_flags':sum(x['flag'] for x in err),'correct_flags':sum(x['flag'] for x in good),'error_unique_groups':len({x['group_id'] for x in err}),'correct_unique_groups':len({x['group_id'] for x in good}),'unique_family_error':sum(x['family'] is not None for x in err),'unique_family_correct':sum(x['family'] is not None for x in good)}
  fam=defaultdict(lambda:{'n':0,'errors':0,'flagged':0,'groups':set()})
  for x in subset:
   if x['family'] is not None:
    z=fam[x['family']];z['n']+=1;z['errors']+=x[version+'_error'];z['flagged']+=x['flag'];z['groups'].add(x['group_id'])
  families[name][version]=[{'family':k,'n':v['n'],'errors':v['errors'],'correct':v['n']-v['errors'],'flagged':v['flagged'],'groups':len(v['groups'])} for k,v in fam.items() if v['errors']>=1 and v['n']-v['errors']>=1]
  families[name][version].sort(key=lambda x:(-x['n'],x['family']))
  # Greedy outcome-stratified matching; unique group globally, exact family/fold/flag, cys within 1, lengths within 20%.
  used=set();paired=[]
  for e in sorted(err,key=lambda x:x['sample_id']):
   if not e['family'] or e['group_id'] in used:continue
   candidates=[c for c in good if c['group_id'] not in used and c['group_id']!=e['group_id'] and c['fold']==e['fold'] and c['flag']==e['flag'] and c['family']==e['family'] and abs(c['cys']-e['cys'])<=1 and abs(c['length']-e['length'])<=.2*max(c['length'],e['length'])]
   if not candidates:continue
   c=min(candidates,key=lambda c:(abs(c['cys']-e['cys']),abs(c['length']-e['length']),c['sample_id']));used.update([e['group_id'],c['group_id']])
   diffs={}
   for f in features:
    a=e['features'].get(f);b=c['features'].get(f)
    if isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b):diffs[f]=float(a-b)
   paired.append({'error_id':e['sample_id'],'correct_id':c['sample_id'],'error_group':e['group_id'],'correct_group':c['group_id'],'family':e['family'],'fold':e['fold'],'flag':e['flag'],'error_length':e['length'],'correct_length':c['length'],'error_cys':e['cys'],'correct_cys':c['cys'],'feature_difference_error_minus_correct':diffs})
  matches[name][version]=paired
save('summary.json',summary);save('mixed_family_panels.json',families);save('matched_pairs.json',matches)
with (P/'extreme_cohorts.csv').open('w',encoding='utf-8-sig',newline='') as h:
 flat=[{k:v for k,v in x.items() if k not in ['features','family_parents']} for group in cohorts.values() for x in group];w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
manifest={'inputs':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},'sample_count':len(rows),'cross_version_alignment_verified':True,'old_full_3seed_with_historical_3D':True,'new_one_seed_conditional_3D':True,'no_Test1_Test2':True,'matching':'same label by cohort, same fold, exact unique full family and flag, different unreused sequence groups, Cys diff<=1, length diff<=20% longer; no identity or endpoint equivalence required','limitations':'opportunistic posthoc, greedy matching not optimal, selected outcome, no causal or statistical inference; residual quality/sequence unknown; no blank treated negative'}
save('verification.json',manifest)
for n in cohorts:
 print(n,summary[n]);print('matches',{k:len(v) for k,v in matches[n].items()});print('families',{k:v[:8] for k,v in families[n].items()})
