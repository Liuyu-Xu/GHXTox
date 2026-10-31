"""Cache primary-source identity records. No toxicity labels are created or edited."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json
import requests
import pandas as pd

OUT=Path(__file__).resolve().parent
ATLAS=OUT.parent/'all_training_atlas.csv'
FIELDS='accession,id,protein_name,organism_name,length,sequence,keyword,cc_function,ft_chain,ft_peptide'

def retrieve(url,params,path):
    if path.exists():return
    response=requests.get(url,params=params,timeout=55)
    response.raise_for_status();path.write_bytes(response.content)
    path.with_suffix(path.suffix+'.request.json').write_text(json.dumps({
        'retrieved_utc':datetime.now(timezone.utc).isoformat(),'url':response.url,
        'headers':dict(response.headers),'sha256':hashlib.sha256(response.content).hexdigest(),
        'purpose':'sequence identity and annotation recovery, not gold toxicity labeling'},indent=2),encoding='utf-8')

if __name__=='__main__':
    path=OUT/'uniprot_reviewed_length11_50_20261025.tsv'
    print('Fetch reviewed UniProt short-entry snapshot',flush=True)
    retrieve('https://rest.uniprot.org/uniprotkb/stream',{'query':'reviewed:true AND length:[11 TO 50]','format':'tsv','fields':FIELDS},path)
    u=pd.read_csv(path,sep='\t',keep_default_na=False);a=pd.read_csv(ATLAS,float_precision='round_trip')
    merged=a.merge(u,left_on='sequence',right_on='Sequence',how='left')
    merged['identity_relation']=merged.Entry.notna().map({True:'exact_full_entry_sequence',False:'not_recovered_from_short_entry_snapshot'})
    merged.to_csv(OUT/'training_uniprot_exact_matches.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    errors=merged[merged.fusion_error]
    errors.to_csv(OUT/'errors_uniprot_exact_matches.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    hits=errors.Entry.dropna().unique().tolist()
    print('Snapshot entries',len(u),'training sequences matched',merged[merged.Entry.notna()].sample_id.nunique(),
          'error sequences matched',errors[errors.Entry.notna()].sample_id.nunique(),'error accessions',len(hits),flush=True)
    for offset in range(0,len(hits),60):
        ids=hits[offset:offset+60]
        query=' OR '.join('accession:'+x for x in ids)
        retrieve('https://rest.uniprot.org/uniprotkb/stream',{'query':query,'format':'json'},OUT/f'uniprot_error_records_batch{offset//60}.json')
        print('Cached detailed annotations',min(offset+60,len(hits)),flush=True)
    summary={'snapshot_entries':len(u),'training_samples':len(a),'training_exact_identity_samples':int(merged[merged.Entry.notna()].sample_id.nunique()),
        'errors':int(a.fusion_error.sum()),'error_exact_identity_samples':int(errors[errors.Entry.notna()].sample_id.nunique()),
        'match_is_not_original_label_provenance':True,'no_labels_changed':True}
    (OUT/'identity_recovery_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
