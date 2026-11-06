import sys,pathlib,json,csv,hashlib,collections,time
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'_vendor'))
import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
root=P.parents[1]
samples=json.loads((root/'analysis/ghxtox_shortfall_registry_20261027/all_sample_shortfall_index.json').read_text(encoding='utf-8'))
seqs=[s['sequence'] for s in samples]
source_rows={}; source_stats={};nearest={}
for model,kind in [('hemopi2',''),('ntxpred2','peptides'),('ntxpred2','protein')]:
    for role,filename in [('train','cross_val_dataset.csv'),('independent','independent_dataset.csv')]:
        path=P/model/'Dataset'/kind/filename
        key='_'.join(x for x in [model,kind,role] if x)
        rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
        seqkey='SEQUENCE' if model=='hemopi2' else 'Sequence'
        labelkey='label' if model=='hemopi2' else 'Label'
        records=[{'sequence':r[seqkey].strip().upper(),'label':int(r[labelkey]),'row':i+2, 'concentration_uM':r.get('μM')} for i,r in enumerate(rows)]
        assert all(set(r['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') for r in records)
        source_rows[key]=records
        lens=[len(r['sequence']) for r in records]
        source_stats[key]={'rows':len(rows),'unique_sequences':len(set(r['sequence'] for r in records)),'label_counts':dict(collections.Counter(r['label'] for r in records)),'min_length':min(lens),'max_length':max(lens),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print('distance',key,len(rows),flush=True)
        matrix=process.cdist(seqs,[r['sequence'] for r in records],scorer=Levenshtein.normalized_similarity,dtype=np.float32,workers=4)
        bestidx=matrix.argmax(axis=1);best=matrix[np.arange(len(seqs)),bestidx]
        exact=collections.defaultdict(list)
        for r in records:exact[r['sequence']].append({'row':r['row'],'label':r['label'],'concentration_uM':r['concentration_uM']})
        nearest[key]=[{'edit_similarity':float(best[i]),'nearest_row':records[int(bestidx[i])]['row'],'nearest_sequence':records[int(bestidx[i])]['sequence'],'nearest_endpoint_label':records[int(bestidx[i])]['label'],'exact_records':exact.get(s,[])} for i,s in enumerate(seqs)]
        del matrix

records=[]
for i,s in enumerate(samples):
    r={k:s[k] for k in ['sample_id','sequence','label','fold','group_id','joint_mode','category_keys']}
    r['length']=len(s['sequence']);r['sources']={k:v[i] for k,v in nearest.items()}
    # Union of public train and independent is deliberately conservative because not all checkpoint lineage is certified.
    r['hemopi2_public_max_similarity']=max(r['sources'][k]['edit_similarity'] for k in nearest if k.startswith('hemopi2'))
    r['ntxpred2_public_max_similarity']=max(r['sources'][k]['edit_similarity'] for k in nearest if k.startswith('ntxpred2'))
    r['ntxpred2_peptide_in_domain']=7<=r['length']<=50
    records.append(r)
categories={'all':list(range(len(samples))),'low_cys_positive':[i for i,s in enumerate(samples) if 'physical_0' in s['category_keys']],'high_cys_negative':[i for i,s in enumerate(samples) if 'physical_1' in s['category_keys']],'shared_low_cys_FN':[i for i,s in enumerate(samples) if 'physical_0' in s['category_keys'] and s['old_archived_wrong'] and s['new_wrong']],'shared_high_cys_FP':[i for i,s in enumerate(samples) if 'physical_1' in s['category_keys'] and s['old_archived_wrong'] and s['new_wrong']]}
summary={}
for cat,idx in categories.items():
    summary[cat]={'n':len(idx),'ntx_peptide_in_domain':sum(records[i]['ntxpred2_peptide_in_domain'] for i in idx),'sources':{k:{'exact':sum(bool(v[i]['exact_records']) for i in idx),'edit_sim_ge_0_9':sum(v[i]['edit_similarity']>=0.9-1e-7 for i in idx),'edit_sim_ge_0_8':sum(v[i]['edit_similarity']>=0.8-1e-7 for i in idx),'edit_sim_ge_0_7':sum(v[i]['edit_similarity']>=0.7-1e-7 for i in idx)} for k,v in nearest.items()}}
    for m in ['hemopi2','ntxpred2']:
        summary[cat][m+'_conservative_below_0_7']=sum(records[i][m+'_public_max_similarity']<0.7-1e-7 for i in idx)
assert len(categories['shared_low_cys_FN'])==151 and len(categories['shared_high_cys_FP'])==64
out={'metric':'global normalized Levenshtein edit similarity = 1 - edit_distance/max(lengths); substitutions/insertions/deletions cost 1; NOT alignment percent identity, not a phylogenetic/family test','thresholds_frozen':[0.7,0.8,0.9],'checkpoint_training_lineage':'train versus independent separately; all-public conservative union for candidate discovery only','source_statistics':source_stats,'categories':summary}
(P/'overlap_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(P/'overlap_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
