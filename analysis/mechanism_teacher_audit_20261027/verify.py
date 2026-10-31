"""Independent key/sequence, threshold, exact overlap, and aggregate checks."""
import json,pathlib,csv,math,hashlib,collections
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
src=read(R/'analysis/ghxtox_shortfall_registry_20261027/all_sample_shortfall_index.json')
ov=read(P/'overlap_records.json');summary=read(P/'diagnostic_summary.json')
assert len(src)==len(ov)==6387
assert len({s['sample_id'] for s in src})==6387
exact_checks=0
for name,kind in [('hemopi2',''),('ntxpred2','peptides'),('ntxpred2','protein')]:
 for role,filename in [('train','cross_val_dataset.csv'),('independent','independent_dataset.csv')]:
  key='_'.join(k for k in [name,kind,role] if k)
  rows=list(csv.DictReader((P/name/'Dataset'/kind/filename).open(encoding='utf-8-sig')))
  sequences=collections.Counter(r.get('SEQUENCE',r.get('Sequence')).strip().upper() for r in rows)
  for s,o in zip(src,ov):
   assert s['sample_id']==o['sample_id'] and s['sequence']==o['sequence']
   assert len(o['sources'][key]['exact_records'])==sequences[s['sequence']]
   exact_checks+=1
checks={}
for name in ['hemopi2','hemopi2_hybrid','ntxpred2']:
 rows=read(P/(name+'_predictions.json'));pop=[r for r in rows if r['population']=='GHXTox']
 assert len(pop)==6387
 for s,r in zip(src,pop):
  assert s['sample_id']==r['sample_id'] and s['sequence']==r['sequence']
  assert 0<=r['probability']<=1 and math.isfinite(r['probability'])
  assert r['prediction']==int(r['raw_hybrid_score']>0.58) if name=='hemopi2_hybrid' else r['prediction']==int(r['probability']>=r['threshold'])
 joined=read(P/(name+'_joined_sample_records.json'))
 for c in ['shared_low_cys_FN','shared_high_cys_FP','physical_0_both_correct_pool','physical_1_both_correct_pool']:
  key='physical_0' if 'low' in c or c.startswith('physical_0') else 'physical_1'
  base=[(s,r) for s,r in zip(src,pop) if key in s['category_keys'] and ((s['old_archived_wrong'] and s['new_wrong']) if c.startswith('shared') else (not s['old_archived_wrong'] and not s['new_wrong']))]
  q=summary[name]['categories'][c]['all'];assert q['n']==len(base) and q['endpoint_positive']==sum(r['prediction'] for s,r in base)
  model='hemopi2' if name.startswith('hemopi2') else 'ntxpred2'
  clean=[(s,r) for s,r in base if next(o for o in ov if o['sample_id']==s['sample_id'])[model+'_public_max_similarity']<.7-1e-7]
  q=summary[name]['categories'][c]['below_0_7_public_edit_similarity'];assert q['n']==len(clean) and q['endpoint_positive']==sum(r['prediction'] for s,r in clean)
 checks[name]={'population':len(pop),'id_sequence_threshold_aggregate_checks':'passed'}
ntx=read(P/'ntxpred2_inference.json');hf=read(P/'ntx_hf_info.json')
hfhash=next(x['lfs']['sha256'] for x in hf['siblings'] if x['rfilename']=='model.safetensors')
assert ntx['weight_sha256']==hfhash
checks['ntx_weight_vs_HF']='exact SHA256 match; this verifies artifact identity, not training lineage'
git_checks=0
for name in ['hemopi2','ntxpred2']:
 tree=read(P/(name+'_tree.txt'))
 for entry in tree['tree']:
  if entry['type']!='blob':continue
  f=P/name/entry['path']
  if entry['path']==('hemopi2_classification.py' if name=='hemopi2' else 'ntxpred2.py'):f=P/(name+'_code.txt')
  if not f.exists():continue
  raw=f.read_bytes()
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==entry['sha'],str(f)
  git_checks+=1
checks['author_Git_blob_hashes']=git_checks
panel=read(P/'common_low_similarity_panel.json')
for cat,v in panel.items():
 assert v['n']==len(v['ids'])
 for sid in v['ids']:
  o=next(x for x in ov if x['sample_id']==sid)
  assert o['hemopi2_public_max_similarity']<.7-1e-7 and o['ntxpred2_public_max_similarity']<.7-1e-7
checks['common_low_similarity_panel']='all IDs and both exclusions passed'
out={'passed':True,'exact_overlap_pair_checks':exact_checks,'prediction_rows_checked':sum(len(read(P/(n+'_predictions.json'))) for n in ['hemopi2','hemopi2_hybrid','ntxpred2']),'checks':checks,'limits':'No independent biological goldset or teacher training-lineage certification; no fitting or promotion.'}
(P/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
