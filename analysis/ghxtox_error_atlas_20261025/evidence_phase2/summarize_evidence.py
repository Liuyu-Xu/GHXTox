"""Consolidate exact identity, entity boundaries and evidence without relabeling."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
BASE=OUT.parent

NOTES={
 'peptide_4096':('复杂生物实体','Myrmexin 异二聚体中的 SS3 亚基；功能证据涉及异二聚体/复合物的抗炎活性，不是孤立 SS3 的毒性证明。'),
 'peptide_2165':('片段与功能外推','PLA2 CS24a 片段；酶活性功能注释含同源推断，不能视为该短序列的实测毒性。'),
 'peptide_3921':('片段与功能外推','疑似丝氨酸蛋白酶片段；已注释真菌细胞壁组分结合功能，不能据此证明孤立短肽无毒。'),
 'peptide_5033':('内源效应与实体边界','蛙利钠肽相关条目，血管活性/激素功能；记录还涉及前体和切割链，不能把生理活性等同于统一毒性。'),
 'peptide_4448':('片段与证据缺失','蛇毒 C 型凝集素蛋白的片段；没有恢复该短序列的独立毒性终点。'),
 'peptide_2653':('已确认的部分序列问题','Mytimycin 标记 Fragment；原始研究描述部分表征的约6.2 kDa抗真菌肽，而保留输入只有33 aa。抗真菌也不是宿主无毒证据。'),
 'peptide_3648':('无毒注释但实体不完整','条目注释对小鼠和昆虫无毒，但序列标记 Fragment，功能证据指向数据库提交。不能当作合成该34 aa片段已测得无毒。'),
 'peptide_2726':('缺少功能证据','pc16c 的序列和C端酰胺化有记录，未恢复对应毒性测定；没有 Toxin 关键词不等于已证实无毒。'),
 'peptide_5875':('片段与证据缺失','蜘蛛毒液蛋白 PKTx22C1 的片段；来源和家族相似性不能确定其宿主毒性。'),
 'peptide_5902':('无毒注释但实体不完整','注释称脑室注射小鼠未见毒性，但为 Fragment 且功能依据为提交记录；不能外推到任意实验条件或孤立片段。'),
 'peptide_1392':('原始阳性依据待追查','序列匹配 human xenin-25；内源激素身份既不证明安全，也不证明原标签有误。优先恢复原始毒性依据。'),
 'peptide_322':('效应与物种条件明确','Cono-NPY1，C端酰胺化；注释/原始论文支持小鼠脑室给药后的行为效应，不是溶血或某个已确证受体机制。'),
 'peptide_424':('可追查的真实效应','H2C/H2U 同序列，原始论文支持溶血和成纤维细胞毒性；需保留物种、剂量与N端甲酰化等分子信息。'),
 'peptide_806':('未恢复身份','本轮短条目精确匹配和精确网页搜索未恢复可靠身份；保持未知，不按模型意见补标签。'),
 'peptide_309':('已确认的部分序列问题','输入28 aa是38 kDa pleuturegin 的N端测序片段；原论文的翻译抑制实验对象为完整分离蛋白，未建立孤立28 aa的同等活性。'),
 'peptide_1623':('推定成熟体而非实测活性','ConoServer G9.1；由前体推定成熟序列，序列证据为核酸层面，不能当作成熟肽已实测毒性。'),
 'peptide_622':('片段与同源功能推断','Turripeptide Ici9.1；Fragment，离子通道毒性为同源推断，不能当作该精确短肽直接活性证据。'),
 'peptide_332':('标签与已测条件不一致风险','pc16a 原论文报告所测动物、Na/K通道及烟碱受体实验未见活性；不能据此证明普遍安全，但正标签不应被当作这些终点的实测阳性。'),
 'peptide_33':('物种选择性','Omega-HXTX-Hv2a：昆虫钙通道选择性证据，不能无条件解释为人类毒性；原记录还含C端修饰。'),
 'peptide_643':('片段与同源功能推断','Turripeptide OL11-like；Fragment，离子通道毒性为同源推断，优先检查实体和实验效应。')
}

def save(name,df):df.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    a=pd.read_csv(BASE/'all_training_atlas.csv',float_precision='round_trip')
    m=pd.read_csv(OUT/'training_uniprot_exact_matches.csv',float_precision='round_trip')
    fragments=set(pd.read_csv(OUT/'uniprot_fragment_accessions.tsv',sep='\t').Entry)
    m['fragment_hit']=m.Entry.isin(fragments)
    m['toxin_keyword_hit']=m.Keywords.fillna('').str.split(';').map(lambda xs:'Toxin' in xs)
    grouped=m.groupby('sample_id',sort=False)
    annotations=[]
    for sid,g in grouped:
        hits=g[g.Entry.notna()]
        annotations.append(dict(sample_id=sid,exact_match_accessions=';'.join(hits.Entry),
            exact_match_count=len(hits),any_fragment_match=bool(hits.fragment_hit.any()),
            all_matches_fragment=bool(len(hits)>0 and hits.fragment_hit.all()),
            mixed_fragment_status=bool(hits.fragment_hit.any() and not hits.fragment_hit.all()),
            any_current_toxin_keyword=bool(hits.toxin_keyword_hit.any()),
            identity_status='no_short_entry_match' if len(hits)==0 else 'fragment_match' if hits.fragment_hit.any() else 'matched_not_fragment_flagged'))
    ann=pd.DataFrame(annotations)
    data=a.merge(ann,on='sample_id',how='left',validate='one_to_one')
    assert len(data)==6387 and int(data.fusion_error.sum())==327
    data['audit_action']=np.where(data.any_fragment_match,'recover_tested_entity_boundary',
        np.where(data.exact_match_count==0,'recover_identity','recover_endpoint_and_original_label_source'))
    data['original_label_unchanged']=True
    save('training_evidence_flags.csv',data)
    save('error_evidence_flags.csv',data[data.fusion_error])
    save('negative_current_toxin_keyword_review.csv',m[(m.label==0)&m.toxin_keyword_hit])
    # Identity/status screening only: no sample is declared gold-standard from automatic flags.
    strata=[]
    for (y,cys,status),g in data.groupby(['label',data.cys_count>=4,'identity_status']):
        strata.append(dict(label=y,cys_ge4=bool(cys),identity_status=status,n=len(g),errors=int(g.fusion_error.sum()),error_rate=float(g.fusion_error.mean())))
    save('entity_stratified_error_rates.csv',pd.DataFrame(strata))
    records={}
    for f in OUT.glob('uniprot_error_records_batch*.json'):
        if '.request.' in f.name:continue
        for r in json.loads(f.read_text(encoding='utf-8'))['results']:records[r['primaryAccession']]=r
    ledger=[];refs=[]
    for _,row in m[m.fusion_error&m.Entry.notna()].iterrows():
        r=records[row.Entry]
        assert r['sequence']['value']==row.sequence
        desc=r.get('proteinDescription',{})
        funcs=[t for c in r.get('comments',[]) if c.get('commentType')=='FUNCTION' for t in c.get('texts',[])]
        codes=sorted({e.get('evidenceCode','') for t in funcs for e in t.get('evidences',[])})
        pmids=[];dois=[]
        for ref in r.get('references',[]):
            citation=ref['citation'];cross=citation.get('citationCrossReferences',[])
            ids=[x['id'] for x in cross if x['database']=='PubMed'];dd=[x['id'] for x in cross if x['database']=='DOI']
            pmids.extend(ids);dois.extend(dd)
            refs.append(dict(sample_id=row.sample_id,accession=row.Entry,reference_number=ref['referenceNumber'],
                citation_type=citation.get('citationType',''),title=citation.get('title',''),
                pmids=';'.join(ids),dois=';'.join(dd),positions=';'.join(ref.get('referencePositions',[]))))
        ptms=[f for f in r.get('features',[]) if f['type'] in ['Modified residue','Disulfide bond','Cross-link','Glycosylation']]
        ledger.append(dict(sample_id=row.sample_id,sequence=row.sequence,label=int(row.label),error_type=row.fusion_category,
            accession=row.Entry,entry_name=r.get('uniProtkbId',''),protein_name=desc.get('recommendedName',{}).get('fullName',{}).get('value',''),
            organism=r.get('organism',{}).get('scientificName',''),fragment_flag=desc.get('flag',''),
            function_evidence_codes=';'.join(codes),function_annotation=' | '.join(t['value'] for t in funcs),
            function_has_direct_experimental_code='ECO:0000269' in codes,
            function_has_inferred_code=bool(set(codes)&{'ECO:0000250','ECO:0000255'}),
            function_has_submission_evidence=any(e.get('source')=='Reference' for t in funcs for e in t.get('evidences',[])),
            chain_boundary_features=json.dumps([f for f in r.get('features',[]) if f['type'] in ['Chain','Peptide','Non-terminal residue']],ensure_ascii=False),
            documented_ptm_features=json.dumps(ptms,ensure_ascii=False),
            pmids=';'.join(sorted(set(pmids))),dois=';'.join(sorted(set(dois))),
            annotation_date=r['entryAudit']['lastAnnotationUpdateDate'],sequence_version=r['entryAudit']['sequenceVersion'],
            source_url='https://www.uniprot.org/uniprotkb/'+row.Entry+'/entry',
            original_ghxtox_source_accession_verified=False))
    led=pd.DataFrame(ledger);save('error_accession_evidence_ledger.csv',led);save('error_primary_reference_links.csv',pd.DataFrame(refs))
    q=pd.read_csv(BASE/'evidence_review_queue.csv',float_precision='round_trip');reviews=[]
    for _,row in q.iterrows():
        block=led[led.sample_id==row.sample_id];flags=data.set_index('sample_id').loc[row.sample_id]
        kind,note=NOTES[row.sample_id]
        extra_name='';extra_url=''
        if row.sample_id=='peptide_1392':extra_name='Human xenin-25';extra_url='https://gpcrdb.org/ligand/8890/info/'
        if row.sample_id=='peptide_1623':extra_name='ConoServer G9.1';extra_url='https://www.conoserver.org/?id=5758&page=card&table=protein'
        reviews.append(dict(sample_id=row.sample_id,sequence=row.sequence,label=int(row.label),error_type=row.error_type,
            cys_count=int(row.cys_count),p_fusion=row.p_fusion,
            recovered_identity=';'.join(block.protein_name.drop_duplicates()) or extra_name or 'unknown',
            accessions=';'.join(block.accession),any_fragment_match=bool(flags.any_fragment_match),
            evidence_issue=kind,review_note=note,
            source_links=';'.join(block.source_url.drop_duplicates()) or extra_url,
            primary_dois=';'.join(block.dois.drop_duplicates()),
            original_binary_label_provenance='not_recovered',label_action='unchanged',
            qualified_for_direct_human_toxicity_validation=False))
    reviews=pd.DataFrame(reviews);save('representative20_review.csv',reviews)
    pub=json.loads((OUT/'representative_paper_abstracts.json').read_text())
    neg=data[(data.label==0)&data.any_current_toxin_keyword]
    summary=dict(training_n=len(data),errors_n=int(data.fusion_error.sum()),
        current_uniprot_snapshot_entries=12032,training_exact_identity_n=int((data.exact_match_count>0).sum()),
        error_exact_identity_n=int(((data.exact_match_count>0)&data.fusion_error).sum()),
        error_matched_accessions=len(records),any_fragment_training_n=int(data.any_fragment_match.sum()),
        any_fragment_error_n=int((data.any_fragment_match&data.fusion_error).sum()),
        mixed_fragment_status_n=int(data.mixed_fragment_status.sum()),
        negative_current_toxin_keyword_n=len(neg),negative_current_toxin_keyword_fp_n=int(neg.fusion_error.sum()),
        representative_n=20,representative_identity_recovered=int((reviews.recovered_identity!='unknown').sum()),
        primary_article_abstracts_retrieved=len(pub['resultList']['result']),
        upstream_captp_sequence_label_agreement=True,
        labels_edited=0,records_removed=0,new_models_trained=0,external_test_labels_read=False,
        limitations=['Current exact identity is not proof of original database accession or original release annotation.',
            'Fragment status is an evidence audit flag, not proof that the isolated short sequence is nontoxic.',
            'Not fragment-flagged does not prove a complete tested molecular entity.',
            'Original dose, chemistry and organism-specific endpoints remain incomplete.',
            'No metric is presented as performance on a new clean independent dataset.'])
    dump('summary.json',summary)
    # Cache hashes make future changes in live annotations visible.
    manifest=[]
    for f in sorted(OUT.iterdir()):
        if f.is_file() and (f.name.startswith('uniprot_') or f.name in ['captp_training_source.fasta','representative_paper_abstracts.json']):
            manifest.append(dict(file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size))
    save('source_cache_manifest.csv',pd.DataFrame(manifest))
    make_report(summary,data,reviews)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def make_report(s,d,reviews):
    lines=['# GHXTox 错题证据追溯 第二轮','',
        '本轮发现，部分“模型错题”同时存在实验对象不一致、毒性终点不一致或标签证据不足的问题。应先区分这些问题，再训练纠错模型。没有修改标签、删除样本或训练新模型。','',
        '## 完成范围','',
        f"- 核对全部 {s['training_n']} 条训练记录，序列和原始标签与公开 CAPTP 训练文件逐条一致；公开文件使用通用 peptide|label 标题，没有保留原始 accession。",
        f"- 当前 UniProt reviewed 短条目快照共 12,032 条；精确匹配到 {s['training_exact_identity_n']} 条训练序列和 {s['error_exact_identity_n']}/{s['errors_n']} 条错误序列，错误对应 {s['error_matched_accessions']} 个 accession。",
        f"- 20 条重点样本中恢复了 {s['representative_identity_recovered']} 条身份线索，1 条仍未知；另缓存16篇关联原始论文摘要。摘要不是全文方法核查，不能据此声称完整恢复实验条件。",'',
        '## 发现一 片段序列可能继承了完整蛋白的效应标签','',
        f"全部训练记录中 {s['any_fragment_training_n']} 条至少精确匹配一个当前标记 Fragment 的条目；其中错误 {s['any_fragment_error_n']} 条。这是对象边界的审核提示，不是确认这些标签全部错误。",
        '',
        '最清楚的例子是 peptide_309：输入只有28个残基，精确匹配 UniProt P83245 的N端片段，链末端标为超出已知序列。原始论文测的是38 kDa pleuturegin 的翻译抑制活性，未证明孤立28 aa有相同活性。将其作为完整短肽预测结构，可能改变了真正研究的分子对象。',
        '[原始研究](https://doi.org/10.1006/bbrc.2001.5816) · [UniProt P83245](https://www.uniprot.org/uniprotkb/P83245/entry)。',
        '',
        'peptide_2653 的33 aa Mytimycin 同样标记 Fragment；原始研究描述的是部分表征的约6.2 kDa抗真菌肽。不能把抗真菌活性当成宿主无毒，也不能把部分测序片段当成同一完整实验对象。[原始研究](https://pubmed.ncbi.nlm.nih.gov/8702979/)。','',
        '## 发现二 毒液身份 功能活性 与统一毒性标签不等价','',
        'peptide_332 的 pc16a 被标作有毒，模型判断无毒。原始论文报告，在所测动物、Na/K通道和烟碱受体实验中没有观察到活性，靶点未知。它不能被视为这些终点的已测阳性，也不能据此改成任何条件下都安全。[原始研究](https://doi.org/10.1016/j.peptides.2011.10.026)。',
        '',
        'peptide_2726 的 pc16c 同属 Conus pictus，却被标作无毒；目前恢复到序列和酰胺化记录，没有对应的无毒实验证据。pc16a/pc16c 不适合仅依据当前二分类标签被当成一对机制明确的正负样本。[pc16c 记录](https://www.conoserver.org/?id=5862&page=card&table=protein)。',
        '',
        'peptide_1623 的 G9.1 由前体预测成熟序列，ConoServer注明核酸层面的序列证据；这不是成熟肽已合成并测得毒性的证据。[G9.1 记录](https://www.conoserver.org/?id=5758&page=card&table=protein)。',
        '',
        '富Cys负类 peptide_3648/P84000 和 peptide_5902/P85146 确有无毒注释，但二者均为 Fragment，功能依据来自数据库提交，实验对象和完整条件未恢复。不能把它们直接认作精确短肽的高质量无毒对照。',
        '[P84000](https://www.uniprot.org/uniprotkb/P84000/entry) · [P85146](https://www.uniprot.org/uniprotkb/P85146/entry)。','',
        '## 发现三 低Cys盲点仍然值得研究','',
        '只查看当前精确匹配且未被 Fragment 标记的记录，结果如下。这不是新清洗后的验证集，仅是原有 OOF 的描述性分层；未匹配者单列，不能当成完整序列。','',
        '|真实标签|Cys分组|片段审核状态|样本数|错误数|错误率|','|---|---|---|---:|---:|---:|']
    for y,cys in [(1,False),(1,True),(0,False),(0,True)]:
        g=d[(d.label==y)&((d.cys_count>=4)==cys)&(d.identity_status=='matched_not_fragment_flagged')]
        lines.append(f"|{'有毒' if y else '无毒'}|{'≥4' if cys else '<4'}|精确匹配且未标 Fragment|{len(g)}|{int(g.fusion_error.sum())}|{g.fusion_error.mean():.2%}|")
    lines+=['',
        '低Cys毒肽中，157条精确匹配且未标Fragment的记录仍错67条（42.7%）。这说明片段问题不能解释全部低Cys漏检；但这些记录仍可能存在终点、修饰或证据强度差异，不能自动视为金标准。',
        '',
        '一个具有明确效应线索的候选是 peptide_424/H2C：原始论文报告溶血和成纤维细胞毒性。另一个 peptide_322/Cono-NPY1 的证据是特定给药途径下的小鼠行为效应。应分别研究，不能都套用溶血机制。',
        '[H2C研究](https://pubmed.ncbi.nlm.nih.gov/18752624/) · [Cono-NPY研究](https://pubmed.ncbi.nlm.nih.gov/20705590/)。','',
        '## 发现四 当前注释和历史标签存在需复核的差异','',
        f"35条历史负类序列精确匹配至少一个当前带 Toxin 关键词的条目，其中5条被模型记为FP、30条记为TN。关键词和当前注释不是毒性实验金标准，也不能据此认定35条都标错；相同序列还可能对应不同物种或化学形式。已输出单独复核清单，未重标。",'',
        '## 下一项实验已经收敛','',
        '先建立“实验对象—具体终点—实验条件”一致的小规模证据面板，再检验模型能否识别组成相近但效应不同的肽。优先从低Cys膜活性肽的溶血终点开始，因为已有可追查的漏检案例；富Cys通道/毒液肽另设面板。不要为统一解释327条错误而训练一个泛化机制分类器。',
        '',
        '完整分子和片段、成熟体和前体、酰胺化和游离末端、二硫键连接方式必须分开记录。未测、未注释、所测条件无效、已测阳性分别编码；不得把unknown直接当负类。',
        '',
        '本次训练错误用于发现假设，不能再次作为新方法的独立测试集。新面板需要排除与原训练集的重叠和同源性，并在看新结果前固定规则；现有模型与简单组成基线、通用困难样本加权才有公平比较的基础。细化方案见 NEXT_EXPERIMENT.md。','',
        '## 重点样本复核表','',
        '|样本|相对原标签的错误|恢复身份|证据审核结论|','|---|---|---|---|']
    for _,r in reviews.iterrows():lines.append(f'|{r.sample_id}|{r.error_type}|{r.recovered_identity}|{r.review_note}|')
    lines+=['','## 文件与边界','',
        '- representative20_review.csv：20条人工整合的身份和证据问题。',
        '- error_accession_evidence_ledger.csv：精确匹配错误对应的逐accession注释、证据代码、片段边界、修饰和文献。',
        '- error_primary_reference_links.csv：原始参考文献链接；条目提到文献不等于已经核查其全文。',
        '- training_evidence_flags.csv / error_evidence_flags.csv：全训练及错误记录的身份/片段审核标志。',
        '- entity_stratified_error_rates.csv：原OOF分层统计。',
        '- negative_current_toxin_keyword_review.csv：当前注释差异复核队列。',
        '- source_cache_manifest.csv：缓存输入哈希。检索日期为2026-10-25，数据库注释会变化。',
        '',
        '原始数据集的具体来源accession、筛选理由与历史注释版本没有通过通用FASTA头恢复；因此这里报告的是身份与证据回查，尚不能逐条判定原始标签对错。旧OOF的校准/模型选择独立性限制依旧存在。']
    (OUT/'EVIDENCE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
