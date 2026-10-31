"""Descriptive audit of saved predictions only. No fitting or label changes."""
from pathlib import Path
import json, csv, hashlib, collections, itertools
import numpy as np
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
P=R/'analysis/dose_monotonic_pilot_20261026'
inputs=[P/'prepared_data.json',P/'oof_predictions.csv']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in inputs}
data=json.loads(inputs[0].read_text(encoding='utf-8-sig'))
pred=list(csv.DictReader(inputs[1].open(encoding='utf-8-sig')))
rows=[]; contrasts=[]
for dataset,d in data.items():
 points={str(x['id']):x for x in d['points']}
 for model in ['unconstrained','monotonic_dose']:
  grouped=collections.defaultdict(list)
  for x in pred:
   if x['dataset']==dataset and x['model']==model: grouped[x['panel_id']].append(x)
  assert set(grouped)=={x['panel_id'] for x in d['panels']}
  for panel in d['panels']:
   g=grouped[panel['panel_id']]; rec=[points[x['record_id']] for x in g]
   y=np.array([float(x['observed_percent']) for x in g]); z=np.array([float(x['predicted_percent']) for x in g])
   assert all(float(x['observed_percent'])==r['hemolysis_percent'] for x,r in zip(g,rec))
   r=rec[0]; offset=float(np.mean(z-y))
   rows.append(dict(dataset=dataset,model=model,**panel,name=r['name'],c_terminal=r['c_terminal'],n_terminal=r['n_terminal'],observed_mean=float(y.mean()),predicted_mean=float(z.mean()),signed_offset_pp=offset,abs_offset_pp=abs(offset),mae_pp=float(np.mean(abs(z-y))),dose_min_uM=min(x['dose_uM'] for x in rec),dose_max_uM=max(x['dose_uM'] for x in rec),has_literal50=any(x==50 for x in y),record_ids=[x['id'] for x in rec]))
  # Only compare contexts for exactly the same backbone within the same paper.
  by=collections.defaultdict(list)
  for r in rows:
   if r['dataset']==dataset and r['model']==model: by[(r['pmid'],r['sequence'])].append(r)
  for pair in by.values():
   for a,b in itertools.combinations(pair,2):
    changed=[k for k in ['species','n_terminal','c_terminal','pH'] if a[k]!=b[k]]
    if len(changed)!=1: continue
    ga={float(x['dose_uM']):x for x in grouped[a['panel_id']]}; gb={float(x['dose_uM']):x for x in grouped[b['panel_id']]}
    shared=sorted(ga.keys()&gb.keys())
    contrasts.append(dict(dataset=dataset,model=model,pmid=a['pmid'],sequence=a['sequence'],panel_a=a['panel_id'],panel_b=b['panel_id'],changed_field=changed[0],value_a=a[changed[0]],value_b=b[changed[0]],shared_doses_uM=shared,n_shared_doses=len(shared),scope='metadata matched, missing assay conditions not certified controlled',mean_observed_b_minus_a_on_shared=float(np.mean([float(gb[t]['observed_percent'])-float(ga[t]['observed_percent']) for t in shared])) if shared else None))
def summarize(g):
 return dict(n_panels=len(g),n_pmid=len({x['pmid'] for x in g}),mean_offset_pp=float(np.mean([x['signed_offset_pp'] for x in g])),mean_abs_offset_pp=float(np.mean([x['abs_offset_pp'] for x in g])),macro_mae_pp=float(np.mean([x['mae_pp'] for x in g])),overestimated=sum(x['signed_offset_pp']>0 for x in g),underestimated=sum(x['signed_offset_pp']<0 for x in g))
summary=[]
for (ds,model) in itertools.product(data,['unconstrained','monotonic_dose']):
 g=[x for x in rows if x['dataset']==ds and x['model']==model]
 for field in ['species','c_terminal','n_terminal','has_literal50','pmid']:
  by=collections.defaultdict(list)
  for r in g: by[str(r[field])].append(r)
  for v,gg in sorted(by.items()): summary.append(dict(dataset=ds,model=model,field=field,value=v,**summarize(gg)))
cache=[]
import xml.etree.ElementTree as ET
for f in (R/'analysis').rglob('*.xml'):
 try:
  root=ET.parse(f).getroot()
  pmids={''.join(e.itertext()).strip() for e in root.findall('.//article-id') if e.get('pub-id-type')=='pmid'}
  hit=pmids&{r['pmid'] for r in rows}
  if hit: cache.append(dict(path=str(f),pmids=sorted(hit),sha256=sha(f)))
 except (ET.ParseError,OSError): pass
stats=dict(n_fits=0,group_summaries_descriptive_only=True,missingness={ds:{'panels':len(d['panels']),'pH_known_panels':sum(x['pH'] is not None for x in d['panels']),'point_fields':sorted(d['points'][0])} for ds,d in data.items()},cache_matches=cache,matched_context_pairs=len(contrasts),interpretation='No causal attribution; groups confounded by peptide/background/source. Literal50 is not a confirmed error flag. Unknown time/temp/buffer/density not imputed.')
for name,obj in [('panel_offsets.json',rows),('group_summaries.json',summary),('matched_context_pairs.json',contrasts),('summary.json',stats)]:
 (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
assert before=={str(p):sha(p) for p in inputs}
(O/'verification.json').write_text(json.dumps(dict(input_hashes=before,input_unchanged=True,panels_joined=len(rows),all_observed_values_match_source=True,new_fits=0),indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
print(json.dumps([r for r in summary if r['dataset']=='confirmed_overlay_primary' and r['model']=='unconstrained' and r['field']!='pmid'],ensure_ascii=False,indent=2))
