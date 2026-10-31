import csv,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
mp=R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json';pp=R/'analysis/lora_multifold_screen_20261028/all_fold_predictions.json';meta=read(mp);pred=read(pp)
md={m['sample_id']:m for m in meta};assert len(md)==len(meta)==len(pred)==6387 and len({x['sample_id'] for x in pred})==6387
rows=[]
for x in pred:
 m=md[x['sample_id']];assert x['label']==m['label'] and x['group_id']==m['group_id'] and int(m['fold'])==x['fold']
 c=m['cys_count'];b='0–1' if c<=1 else '2–3' if c<=3 else '4–6' if c<=6 else '≥7';q=x['fusion_base'];correct=(q>=.47022256531284556)==bool(m['label'])
 rows.append({'sample_id':m['sample_id'],'fold':x['fold'],'group_id':m['group_id'],'label':m['label'],'cys_count':c,'cys_bin':b,'flag':int(bool(m['uniform_flag'])),'family':'; '.join(m['current_family_sidecar']['full_families']),'fusion_probability':q,'error':int(not correct)})
results=[]
for b in ['0–1','2–3','4–6','≥7']:
 for label in [1,0]:
  for flag in ['all','flagged','unflagged_not_gold']:
   rr=[x for x in rows if x['cys_bin']==b and x['label']==label and (flag=='all' or (x['flag']==1)==(flag=='flagged'))]
   results.append({'bin':b,'label':label,'flag':flag,'n':len(rr),'errors':sum(x['error'] for x in rr),'error_rate':sum(x['error'] for x in rr)/len(rr) if rr else None,'independent_sequence_groups':len({x['group_id'] for x in rr}),'annotated_family_samples':sum(bool(x['family']) for x in rr),'annotated_families':len({z for x in rr for z in x['family'].split('; ') if z})})
assert sum(z['n'] for z in results if z['flag']=='all')==6387
assert sum(z['errors'] for z in results if z['flag']=='all')==408
for b in ['0–1','2–3','4–6','≥7']:
 for label in [1,0]:
  q=[z for z in results if z['bin']==b and z['label']==label];assert q[0]['n']==q[1]['n']+q[2]['n'] and q[0]['errors']==q[1]['errors']+q[2]['errors']
(P/'statistics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
with (P/'samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
manifest={'input_files':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [mp,pp]},'n':len(rows),'errors':408,'crossvalidated':'fivefold single-seed non-LoRA 1D plus historical 3D OOF','threshold':.47022256531284556,'no_new_training':True,'not_new_blind_test':True,'flags':'fragment-match or broad cross-source conflict, not verified label correctness'}
(P/'verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
for z in results:
 if z['flag']=='all':print(z)
