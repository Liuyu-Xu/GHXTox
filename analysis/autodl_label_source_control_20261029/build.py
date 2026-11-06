import json,csv,collections
from pathlib import Path
P=Path('analysis/autodl_label_source_control_20261029');R=P.parents[1];read=lambda p:json.loads(p.read_text(encoding='utf8'))
s=[x for x in read(R/'analysis/autodl_lowcys_fp_audit_20261029/samples.json') if x['fold']==2 and x['role']=='new_fp'];meta={x['sample_id']:x for x in read(R/'analysis/lowcys_background_20261027/all_samples_with_symmetric_family.json')};seqs={meta[x['sample_id']]['sequence'] for x in s}; root=R/'analysis/dataset_comparison_20261025'
def load(name):
 with (root/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
reviews={x['source_id']:x for x in load('ghxtox_label_disagreement_review.csv') if x['source_id'] in {v['sample_id'] for v in s}}
ref={x['sequence']:x for x in load('maomao_endpoint_cross_reference.csv') if x['sequence'] in seqs}
matched=collections.defaultdict(list)
with (root/'all_records.csv').open(encoding='utf-8-sig',newline='') as f:
 for x in csv.DictReader(f):
  if x['sequence'] in seqs:matched[x['sequence']].append(x)
rows=[]
for z in s:
 m=meta[z['sample_id']];q=reviews.get(z['sample_id'],{});same=matched[m['sequence']];u=ref.get(m['sequence'],{});rows.append({'sample_id':z['sample_id'],'sequence':m['sequence'],'flag':z['label_flag'],'conflict':z['conflict'],'fragment':z['fragment'],'uniprot':';'.join(m['source_urls']),'positive_sources':q.get('positive_sources',''),'positive_families':q.get('positive_sources_data_families',''),'negative_sources':q.get('negative_sources',''),'negative_families':q.get('negative_sources_data_families',''),'original_label':'0','original_endpoint':next((x['endpoint'] for x in same if x['dataset']=='GHXTox' and x['split']=='train'),'unknown'),'original_evidence':next((x['label_evidence'] for x in same if x['dataset']=='GHXTox' and x['split']=='train'),'unknown'),'maomao_toxic':u.get('toxic','unknown'),'maomao_hemolytic':u.get('hemolytic','unknown'),'maomao_neurotoxic':u.get('neurotoxic','unknown'),'cached_direct_assay':bool(m.get('has_cached_direct_measurement')),'family':' ; '.join(z['family']),'A_probability':z['A_fusion'],'B_probability':z['B_fusion'],'C_probability':z['C_fusion'],'D_probability':z['D_fusion']})
with (P/'sample_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sumry={'n':len(rows),'conflict':sum(r['conflict'] for r in rows),'fragment':sum(r['fragment'] for r in rows),'with_cached_direct_assay_flag':sum(r['cached_direct_assay'] for r in rows),'positive_sources':dict(collections.Counter(r['positive_families'] for r in rows if r['positive_families'])),'original_endpoints':dict(collections.Counter(r['original_endpoint'] for r in rows)),'original_evidence':dict(collections.Counter(r['original_evidence'] for r in rows)),'maomao_toxic':dict(collections.Counter(r['maomao_toxic'] for r in rows)),'maomao_hemolytic':dict(collections.Counter(r['maomao_hemolytic'] for r in rows))};(P/'summary.json').write_text(json.dumps(sumry,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(sumry,ensure_ascii=False,indent=2))
