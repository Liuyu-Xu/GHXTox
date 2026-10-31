import pathlib,json,collections,csv,hashlib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,matthews_corrcoef
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def load(path):return json.loads(path.read_text(encoding='utf-8'))
samples=load(R/'analysis/ghxtox_shortfall_registry_20261027/all_sample_shortfall_index.json')
overlap=load(P/'overlap_records.json');categories=load(R/'analysis/ghxtox_shortfall_registry_20261027/category_registry.json')
assert all(s['sample_id']==o['sample_id'] and s['sequence']==o['sequence'] for s,o in zip(samples,overlap))
cats={c['key']:[i for i,s in enumerate(samples) if c['key'] in s['category_keys']] for c in categories}
cats['all']=list(range(len(samples)))
for key,poolkey in [('physical_0','shared_low_cys_FN'),('physical_1','shared_high_cys_FP')]:
 cats[poolkey]=[i for i,s in enumerate(samples) if key in s['category_keys'] and s['old_archived_wrong'] and s['new_wrong']]
 cats[key+'_both_correct_pool']=[i for i,s in enumerate(samples) if key in s['category_keys'] and not s['old_archived_wrong'] and not s['new_wrong']]
idxbyid={r['sample_id']:i for i,r in enumerate(samples)}
baseedges=load(R/'analysis/ghxtox_shortfall_registry_20261027/existing_control_edges.json')
edges=list({(r['case_id'],r['control_id'],r['source_archive']):r for r in baseedges}.values())
result={};allrows=[]
for name in ['hemopi2','hemopi2_hybrid','ntxpred2']:
 model='hemopi2' if name.startswith('hemopi2') else 'ntxpred2'
 raw=load(P/(name+'_predictions.json'));d={r['sample_id']:r for r in raw if r['population']=='GHXTox'}
 assert set(d)==set(idxbyid) and all(d[s['sample_id']]['sequence']==s['sequence'] for s in samples)
 y=np.array([s['label'] for s in samples]);pred=np.array([d[s['sample_id']]['prediction'] for s in samples]);prob=np.array([d[s['sample_id']]['probability'] for s in samples])
 ind=[r for r in raw if r['population']=='author_independent'];iy=[r['author_endpoint_label'] for r in ind];ip=[r['prediction'] for r in ind];iq=[r['probability'] for r in ind]
 author={'n':len(ind),'MCC':float(matthews_corrcoef(iy,ip)),'AUROC':float(roc_auc_score(iy,iq)),'AUPRC':float(average_precision_score(iy,iq)),'errors':sum(a!=b for a,b in zip(iy,ip))}
 groupstats={}
 for cat,idx in cats.items():
  groupstats[cat]={}
  for stratum in ['all','no_exact_public','below_0_7_public_edit_similarity','training_length_range_and_below_0_7']:
   sel=idx
   if stratum=='no_exact_public':sel=[i for i in sel if not any(v['exact_records'] for k,v in overlap[i]['sources'].items() if k.startswith(model))]
   if stratum in ['below_0_7_public_edit_similarity','training_length_range_and_below_0_7']:sel=[i for i in sel if overlap[i][model+'_public_max_similarity']<0.7-1e-7]
   if stratum=='training_length_range_and_below_0_7':sel=[i for i in sel if 6<=len(samples[i]['sequence'])<=39] if model=='hemopi2' else [i for i in sel if 7<=len(samples[i]['sequence'])<=50]
   z={'n':len(sel),'endpoint_positive':int(sum(pred[i] for i in sel)),'original_label_agreement':int(sum(pred[i]==y[i] for i in sel)),'original_label_disagreement':int(sum(pred[i]!=y[i] for i in sel)),'sample_ids':[samples[i]['sample_id'] for i in sel],'mean_score':float(np.mean(prob[sel])) if sel else None}
   z['historical_comparison']={}
   for version,wrongkey in [('old_archived','old_archived_wrong'),('new_noTransformer','new_wrong')]:
    support=[i for i in sel if samples[i][wrongkey] and pred[i]==y[i]];harm=[i for i in sel if not samples[i][wrongkey] and pred[i]!=y[i]]
    z['historical_comparison'][version]={'potential_label_agreement_when_G_wrong':len(support),'endpoint_label_disagreement_when_G_right':len(harm),'potential_ids':[samples[i]['sample_id'] for i in support],'disagreement_ids':[samples[i]['sample_id'] for i in harm],'scope':'Descriptive cross-endpoint agreement, NOT validated corrections or a fusion policy'}
   z['by_heldout_fold']={str(f):{'n':sum(samples[i]['fold']==f for i in sel),'endpoint_positive':sum(int(pred[i]) for i in sel if samples[i]['fold']==f)} for f in sorted(set(s['fold'] for s in samples))}
   groupstats[cat][stratum]=z
 joined=[]
 for i,s in enumerate(samples):
  row={**s,'teacher_name':name,'teacher_endpoint_positive':int(pred[i]),'teacher_endpoint_score':float(prob[i]),'teacher_original_label_agreement':bool(pred[i]==y[i]),'teacher_public_max_edit_similarity':overlap[i][model+'_public_max_similarity'],'exact_public_sources':[k for k,v in overlap[i]['sources'].items() if k.startswith(model) and v['exact_records']],'conservative_below_0_7':overlap[i][model+'_public_max_similarity']<.7-1e-7,'within_observed_training_length':6<=len(s['sequence'])<=39 if model=='hemopi2' else 7<=len(s['sequence'])<=50}
  joined.append(row)
 (P/(name+'_joined_sample_records.json')).write_text(json.dumps(joined,ensure_ascii=False,indent=2),encoding='utf-8')
 pairrows=[]
 for edge in edges:
  a,b=idxbyid[edge['case_id']],idxbyid[edge['control_id']]
  pairrows.append({**edge,'teacher_name':name,'case_positive':int(pred[a]),'control_positive':int(pred[b]),'case_original_label_agreement':bool(pred[a]==y[a]),'control_original_label_agreement':bool(pred[b]==y[b]),'both_below_0_7_public_edit_similarity':overlap[a][model+'_public_max_similarity']<.7-1e-7 and overlap[b][model+'_public_max_similarity']<.7-1e-7})
 (P/(name+'_reused_control_edges.json')).write_text(json.dumps(pairrows,ensure_ascii=False,indent=2),encoding='utf-8')
 result[name]={'author_independent_endpoint_check':author,'original_total_toxicity_label_agreement_diagnostic_ONLY':{'MCC':float(matthews_corrcoef(y,pred)),'AUROC':float(roc_auc_score(y,prob)),'AUPRC':float(average_precision_score(y,prob)),'errors':int((pred!=y).sum())},'categories':groupstats,'reused_unique_control_edges':len(pairrows)}
 print(name,'AUTHOR',author,flush=True)
 for c in ['shared_low_cys_FN','physical_0_both_correct_pool','shared_high_cys_FP','physical_1_both_correct_pool']:
  print(c,[(st,x['n'],x['endpoint_positive']) for st,x in groupstats[c].items()],flush=True)
(P/'diagnostic_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
panel={}
for cat in ['shared_low_cys_FN','shared_high_cys_FP','physical_0_both_correct_pool','physical_1_both_correct_pool']:
 idx=[i for i in cats[cat] if overlap[i]['hemopi2_public_max_similarity']<.7-1e-7 and overlap[i]['ntxpred2_public_max_similarity']<.7-1e-7]
 ids=[samples[i]['sample_id'] for i in idx]
 panel[cat]={'n':len(ids),'ids':ids,'models':{}}
 for name in result:
  joined=json.loads((P/(name+'_joined_sample_records.json')).read_text(encoding='utf-8'))
  panel[cat]['models'][name]={'endpoint_positive':sum(joined[i]['teacher_endpoint_positive'] for i in idx)}
(P/'common_low_similarity_panel.json').write_text(json.dumps(panel,ensure_ascii=False,indent=2),encoding='utf-8')
