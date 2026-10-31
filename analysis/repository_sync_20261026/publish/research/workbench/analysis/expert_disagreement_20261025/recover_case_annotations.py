"""Recover exact case identities and original annotation leads, not gold mechanisms."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import pandas as pd
import requests
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'ghxtox_error_atlas_20261025/evidence_phase2'

def run():
    queue=pd.read_csv(OUT/'mechanism_review_queue.csv',keep_default_na=False)
    records={};provenance={}
    for file in sorted(OLD.glob('uniprot_error_records_batch*.json')):
        if '.request.' in file.name:continue
        for r in json.loads(file.read_text(encoding='utf8'))['results']:
            records[r['primaryAccession']]=r;provenance[r['primaryAccession']]=str(file)
    accs=sorted({a for s in queue.exact_match_accessions for a in s.split(';') if a})
    missing=[a for a in accs if a not in records]
    if missing:
        path=OUT/'case_annotations_new.json'
        if not path.exists():
            params={'query':' OR '.join('accession:'+a for a in missing),'format':'json'}
            response=requests.get('https://rest.uniprot.org/uniprotkb/stream',params=params,timeout=55)
            response.raise_for_status();path.write_bytes(response.content)
            path.with_suffix('.request.json').write_text(json.dumps(dict(url=response.url,retrieved_utc=datetime.now(timezone.utc).isoformat(),
                sha256=hashlib.sha256(response.content).hexdigest(),headers=dict(response.headers)),indent=2),encoding='utf8')
        for r in json.loads(path.read_text(encoding='utf8'))['results']:
            records[r['primaryAccession']]=r;provenance[r['primaryAccession']]=str(path)
    ledger=[];citations=[]
    for row in queue.to_dict('records'):
        for acc in [a for a in row['exact_match_accessions'].split(';') if a]:
            if acc not in records:continue
            r=records[acc];assert r['sequence']['value']==row['sequence']
            name=r['proteinDescription'].get('recommendedName',{}).get('fullName',{}).get('value','')
            if not name:
                names=r['proteinDescription'].get('submissionNames',[]);name=names[0].get('fullName',{}).get('value','') if names else ''
            fn=' || '.join(t['value'] for c in r.get('comments',[]) if c['commentType']=='FUNCTION' for t in c.get('texts',[]))
            ev=sorted({e.get('evidenceCode','') for c in r.get('comments',[]) if c['commentType']=='FUNCTION' for t in c.get('texts',[]) for e in t.get('evidences',[])})
            feats=r.get('features',[])
            feature_types={t:sum(f['type']==t for f in feats) for t in ['Disulfide bond','Cross-link','Modified residue','Peptide','Chain']}
            detail=' || '.join(json.dumps(f,ensure_ascii=False) for f in feats if f['type'] in feature_types)
            mod_detail=' || '.join(f.get('description','') for f in feats if f['type']=='Modified residue')
            ids=[]
            for ref in r.get('references',[]):
                citation=ref.get('citation',{});pmid='';doi=''
                for x in citation.get('citationCrossReferences',[]):
                    if x['database']=='PubMed':pmid=x['id']
                    if x['database']=='DOI':doi=x['id']
                if pmid:ids.append(pmid)
                citations.append(dict(sample_id=row['sample_id'],accession=acc,pmid=pmid,doi=doi,title=citation.get('title',''),
                    citation_type=citation.get('citationType',''),link='https://doi.org/'+doi if doi else 'https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/' if pmid else '',
                    scope='record_reference_not_proof_of_exact_peptide_toxicity'))
            ledger.append(dict(sample_id=row['sample_id'],paired_id=row['paired_id'],review_reason=row['review_reason'],sequence=row['sequence'],
                label=row['label'],expert_group=row['expert_group'],fusion_change=row['fusion_change'],evidence_tier=row['evidence_tier'],accession=acc,
                protein_name=name,organism=r.get('organism',{}).get('scientificName',''),fragment=r['proteinDescription'].get('flag','')=='Fragment',
                function_text=fn,function_evidence_codes=';'.join(ev),pubmed_ids=';'.join(ids),
                disulfide_annotation_count=feature_types['Disulfide bond'],crosslink_annotation_count=feature_types['Cross-link'],
                modified_residue_count=feature_types['Modified residue'],modified_residue_descriptions=mod_detail,
                function_mentions_oligomer=any(t in fn.lower() for t in ['oligomer','dimer','polymer','multimer']),
                annotated_terminal_or_formyl_lead=any(t in mod_detail.lower() for t in ['amide','amidat','formyl']),
                molecular_feature_details=detail,annotation_cache=provenance[acc],source_url='https://www.uniprot.org/uniprotkb/'+acc+'/entry',
                interpretation='identity_and_annotation_leads; no_new_toxicity_or_mechanism_label'))
    pd.DataFrame(ledger).to_csv(OUT/'case_annotation_ledger.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(citations).to_csv(OUT/'case_reference_links.csv',index=False,encoding='utf-8-sig')
    # Cache primary literature abstracts without claiming full-method verification.
    paper_ids=sorted({x['pmid'] for x in citations if x['pmid']})
    path=OUT/'case_paper_abstracts_grouped.json'
    if paper_ids and not path.exists():
        query='('+' OR '.join('EXT_ID:'+p for p in paper_ids)+') AND SRC:MED'
        r=requests.get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',params={'query':query,'format':'json','resultType':'core','pageSize':100},timeout=55)
        r.raise_for_status();path.write_bytes(r.content)
        path.with_suffix('.request.json').write_text(json.dumps(dict(url=r.url,retrieved_utc=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(r.content).hexdigest()),indent=2),encoding='utf8')
    returned=json.loads(path.read_text(encoding='utf8')).get('resultList',{}).get('result',[]) if path.exists() else []
    print('Requested paper IDs',len(paper_ids),'returned',len(returned))
    print(json.dumps(dict(queue=len(queue),exact_identity_samples=len({r['sample_id'] for r in ledger}),accessions=len(ledger),new_accessions=len(missing),paper_ids=len(paper_ids)),ensure_ascii=True))

if __name__=='__main__':run()
