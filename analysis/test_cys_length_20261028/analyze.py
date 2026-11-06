import csv,json,hashlib,statistics
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def summarize(rows):
 n=len(rows);wrong=sum(z['error'] for z in rows)
 return {'n':n,'errors':wrong,'error_rate':wrong/n if n else None,'distinct_sequences':len({z['sequence'] for z in rows}),'mean_cys':statistics.mean(z['cys_count'] for z in rows) if n else None,'median_cys':statistics.median(z['cys_count'] for z in rows) if n else None,'mean_length':statistics.mean(z['length'] for z in rows) if n else None,'median_length':statistics.median(z['length'] for z in rows) if n else None}
results={};allrows={};inputs=[]
for split in ['test1','test2']:
 predpath=R/f'runs/default_multidimensional_fusion_single_structure/{split}_predictions.csv';seqpath=R/f'runs/single_structure_test_components/{split}_structure_predictions.csv';flagpath=R/f'analysis/test_ambiguity_association_20261025/{split}_sample_audit.csv'
 inputs += [predpath,seqpath,flagpath];pred=read(predpath);seq=read(seqpath);flag=read(flagpath);assert len(pred)==len(seq)==len(flag)
 print('FLAG_KEYS',split,list(flag[0]))
 fdict={z['sample_id']:z for z in flag};assert len(fdict)==len(flag)
 rows=[]
 for a,b in zip(pred,seq):
  assert all(a[k]==b[k] for k in ['source_index','sample_id','label']);sequence=b['sequence'].strip().upper();assert sequence and set(sequence)<=set('ACDEFGHIKLMNPQRSTVWY')
  q=float(a['toxicity_probability']);label=int(a['label']);decision=int(a['prediction']);threshold=float(a['decision_threshold']);assert decision==int(q>=threshold)
  c=sequence.count('C');n=len(sequence);flagrow=fdict[a['sample_id']];assert int(flagrow['label'])==label and flagrow['sequence']==sequence;assert abs(q-(float(a['p_1d'])*float(a['weight_1d'])+float(a['p_3d'])*(1-float(a['weight_1d']))))<2e-9
  rows.append({'sample_id':a['sample_id'],'sequence':sequence,'label':label,'probability':q,'prediction':decision,'error':int(label!=decision),'cys_count':c,'length':n,'review_flag':int(flagrow['primary_review_flag'].lower() in ('1','true')),'cys_bin':'0–1' if c<2 else '2–3' if c<4 else '4–6' if c<7 else '≥7','length_bin':'≤20' if n<=20 else '21–30' if n<=30 else '31–40' if n<=40 else '≥41'})
 assert len({x['sample_id'] for x in rows})==len(rows) and len({x['sequence'] for x in rows})==len(rows);assert len(rows)==(1126 if split=='test1' else 582)
 results[split]={'total':summarize(rows),'cys':{},'length':{},'class_correctness':{}}
 for group,ordered in [('cys',['0–1','2–3','4–6','≥7']),('length',['≤20','21–30','31–40','≥41'])]:
  for b in ordered:
   results[split][group][b]={}
   for label in [1,0]:results[split][group][b][str(label)]=summarize([x for x in rows if x[group+'_bin']==b and x['label']==label])
 for label in [1,0]:
  results[split]['class_correctness'][str(label)]={str(err):summarize([x for x in rows if x['label']==label and x['error']==err]) for err in [0,1]}
 allrows[split]=rows
 with (P/(split+'_samples.csv')).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert {r['sequence'] for r in allrows['test2']} <= {r['sequence'] for r in allrows['test1']}
(P/'statistics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'verification.json').write_text(json.dumps({'passed':True,'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'sample_alignment_verified':True,'prediction_threshold_verified':True,'test2_sequence_subset_of_test1':True,'no_model_training':True,'test_sets_previously_examined_not_blind':True,'no_multiple_testing_claim':True},indent=2),encoding='utf-8')
for split in results:
 print(split, 'total',results[split]['total'])
 for axis in ['cys','length']:
  print(axis,[(b,[(k,z['n'],z['errors']) for k,z in pair.items()]) for b,pair in results[split][axis].items()])
 print('class',results[split]['class_correctness'])
