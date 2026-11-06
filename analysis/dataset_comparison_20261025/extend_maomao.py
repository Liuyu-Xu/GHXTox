"""Keep MAOMAO's 0/1/2/3/999 endpoint states separate from binary model labels."""
from pathlib import Path
import json
import pandas as pd
from align_candidates import init_worker,search_batch
OUT=Path(__file__).resolve().parent
AA=set('ACDEFGHIKLMNPQRSTVWY')

def run():
    src=OUT/'sources/kren-ai-lab__maomao'
    master=pd.read_csv(src/'maomao_sequence_pivot.csv')
    quant=pd.read_csv(src/'maomao_toxicity_measurements.csv',keep_default_na=False)
    data=pd.read_csv(OUT/'all_records.csv',low_memory=False)
    gh=data[data.dataset=='GHXTox'];train=set(gh.loc[gh.split=='train','sequence']);known=set(gh.sequence)
    unique=data[['sequence']].drop_duplicates().merge(master,on='sequence',how='left',validate='one_to_one')
    unique.to_csv(OUT/'maomao_endpoint_cross_reference.csv',index=False,encoding='utf-8-sig')
    quant['exact_overlap_ghxtox_train']=quant.sequence.isin(train)
    quant['exact_overlap_ghxtox_known']=quant.sequence.isin(known)
    quant['within_current_input_scope']=quant.sequence.apply(lambda s:set(s)<=AA and 11<=len(s)<=50)
    qs=sorted(set(quant.loc[quant.within_current_input_scope,'sequence']))
    for refname,mask in [('train',gh.split=='train'),('known',gh.split.notna())]:
        existing=pd.read_csv(OUT/f'alignment_to_{refname}_v2.csv',float_precision='round_trip')
        missing=sorted(set(qs)-set(existing.sequence))
        refs=gh[mask][['sequence','source_id','split','label']].drop_duplicates('sequence').to_dict('records')
        init_worker(refs);extra=search_batch(missing)
        allalign=pd.concat([existing,pd.DataFrame(extra)],ignore_index=True).drop_duplicates('sequence')
        allalign.to_csv(OUT/f'alignment_to_{refname}_v2.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
        cols=['sequence','max_identity','nearest_ref_id','nearest_ref_sequence']
        quant=quant.merge(allalign[cols].rename(columns={c:f'{c}_{refname}' for c in cols if c!='sequence'}),on='sequence',how='left',validate='many_to_one')
    quant['review_status']='source_reported_measurement; verify_primary_paper_assay_and_chemistry'
    quant.to_csv(OUT/'maomao_quantitative_candidates.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    endpoints=[c for c in master.columns if c not in ['id','sequence']]
    counts=[]
    for endpoint in endpoints:
        for state,n in master[endpoint].value_counts().items():counts.append(dict(endpoint=endpoint,evidence_code=int(state),n=int(n)))
    pd.DataFrame(counts).to_csv(OUT/'maomao_endpoint_state_counts.csv',index=False,encoding='utf-8-sig')
    summary=dict(master_records=len(master),unique_sequences=master.sequence.nunique(),exact_ghxtox_train=len(set(master.sequence)&train),
        exact_ghxtox_known=len(set(master.sequence)&known),quantitative_records=len(quant),quantitative_unique=quant.sequence.nunique(),
        quantitative_ghxtox_train_unique=quant.loc[quant.exact_overlap_ghxtox_train,'sequence'].nunique(),
        quantitative_ghxtox_known_unique=quant.loc[quant.exact_overlap_ghxtox_known,'sequence'].nunique(),
        quantitative_unseen_in_scope_unique=quant.loc[~quant.exact_overlap_ghxtox_known&quant.within_current_input_scope,'sequence'].nunique(),
        quantitative_unseen_in_scope_known_le40_unique=quant.loc[~quant.exact_overlap_ghxtox_known&quant.within_current_input_scope&(quant.max_identity_known<=.4),'sequence'].nunique(),
        measurement_types=quant.measurement_type.value_counts().to_dict(),relation_counts=quant.relation.value_counts().to_dict(),
        code_semantics={'0':'source_negative_evidence_not_universal_safety','1':'direct_or_hierarchy_supported_positive','2':'ambiguous','3':'unlabeled','999':'no_information'})
    (OUT/'maomao_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    protocol=json.loads((OUT/'alignment_protocol.json').read_text(encoding='utf8'))
    protocol['additional_MAOMAO_quantitative_sequences_in_scope']=len(qs)
    protocol['final_query_rows_each_reference_set']=len(allalign)
    protocol['extension_note']='Additional quantitative sequences searched with identical traceback algorithm; do not interpret original primary-run seconds as full extension runtime.'
    (OUT/'alignment_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=True))

if __name__=='__main__':run()
