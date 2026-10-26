"""Produce reviewable Chinese findings and candidate tables from verified caches."""
from pathlib import Path
import json,hashlib
import pandas as pd
from audit_datasets import save,dump,OUT

FAMILY={'GHXTox':'CAPTP_lineage','CAPTP':'CAPTP_lineage','ToxMSRC':'CAPTP_lineage','ToxPLTC':'CAPTP_lineage','StrucToxNet':'CAPTP_lineage',
    'ToxiPep':'ToxiPep_lineage','ToxiPep_0.8':'ToxiPep_lineage','ToxiPep_0.9':'ToxiPep_lineage','ToxMamba':'ToxiPep_lineage',
    'HyPepTox-Fuse':'ToxinPred3_lineage','ToxinPred3':'ToxinPred3_lineage','PeptiTox':'PeptiTox_merged','HemoPI2':'HemoPI2_HC50','ToxiCompass':'ToxiCompass_operational'}

def table(df):
    columns=list(df.columns)
    lines=['|'+'|'.join(columns)+'|','|'+'|'.join(['---']*len(columns))+'|']
    for row in df.fillna('').itertuples(index=False,name=None):lines.append('|'+'|'.join(str(v).replace('|','/') for v in row)+'|')
    return '\n'.join(lines)

def finalize():
    d=pd.read_csv(OUT/'all_records.csv',low_memory=False,keep_default_na=False)
    conflicts=pd.read_csv(OUT/'same_sequence_label_review.csv',keep_default_na=False)
    align={name:pd.read_csv(OUT/f'alignment_to_{name}_v2.csv',float_precision='round_trip') for name in ['train','known']}
    candidates=pd.read_csv(OUT/'external_record_candidates.csv',low_memory=False)
    for name,a in align.items():
        candidates=candidates.merge(a[['sequence','max_identity','nearest_ref_id','nearest_ref_sequence']].rename(columns={c:f'{c}_{name}' for c in a.columns if c!='sequence'}),on='sequence',validate='many_to_one')
    known=sorted(set(d.loc[d.dataset=='GHXTox','sequence']))
    # A short exact fragment can have low full-length identity: flag containment separately.
    lowseq=set(candidates.loc[candidates.max_identity_known<=.4,'sequence'])
    contained={s:next((r for r in known if s in r or r in s),'') for s in sorted(lowseq)}
    candidates['exact_subsequence_known']=candidates.sequence.map(contained).fillna('not_checked_above40')
    candidates['same_endpoint_label_disagreement']=candidates.sequence.isin(set(conflicts.sequence))
    candidates['source_family']=candidates.dataset.map(FAMILY)
    candidates['screening_status']='pending_original_assay_entity_and_modification_review'
    save('external_candidates_aligned.csv',candidates)
    shortlist=candidates[(candidates.max_identity_known<=.4)&candidates.exact_subsequence_known.eq('')].copy()
    # Preserve conflicts on the review list instead of silently discarding/relabeling them.
    save('low_identity_review_candidates.csv',shortlist)
    panel=shortlist[(shortlist.dataset=='HemoPI2')&(shortlist.split=='independent')].copy()
    panel['role']='candidate_external_HC50_panel; no_model_outcomes_examined'
    panel['approved_gold_label']=False
    save('hemopi2_independent_priority_panel.csv',panel)
    ss=[]
    for (dataset,split),g in candidates.groupby(['dataset','split']):
        low=g.max_identity_known<=.4
        ss.append(dict(dataset=dataset,split=split,n_exact_unseen_in_scope=len(g),identity_to_train_le40=int((g.max_identity_train<=.4).sum()),
            identity_to_known_le40=int(low.sum()),identity_to_known_40_70=int(((g.max_identity_known>.4)&(g.max_identity_known<=.7)).sum()),
            identity_to_known_gt70=int((g.max_identity_known>.7).sum()),low40_exact_containment=int((low&g.exact_subsequence_known.ne('')).sum()),
            low40_review_candidates=int((low&g.exact_subsequence_known.eq('')).sum()),low40_pos=int((low&(g.label==1)).sum()),low40_neg=int((low&(g.label==0)).sum())))
    screen=pd.DataFrame(ss);save('external_similarity_screen.csv',screen)
    # Existing GHXTox labels that disagree elsewhere; retain the previous OOF/evidence flags.
    gh=d[d.dataset=='GHXTox']
    review=gh.merge(conflicts[conflicts.scope=='broad_annotation'][['sequence','positive_sources','negative_sources']],on='sequence')
    prior=pd.read_csv(OUT.parent/'ghxtox_error_atlas_20261025/evidence_phase2/training_evidence_flags.csv',keep_default_na=False)
    cols=['sequence','fusion_category','fusion_error','identity_status','any_fragment_match','exact_match_accessions']
    review=review.drop(columns=['identity_status','any_fragment_match','exact_match_accessions']).merge(prior[cols],on='sequence',how='left',validate='many_to_one')
    review['original_label_unchanged']=True
    for column in ['positive_sources','negative_sources']:
        review[column+'_data_families']=review[column].map(lambda s:';'.join(sorted({FAMILY.get(x.split('/')[0],x.split('/')[0]) for x in str(s).split(';') if x})))
    review['review_priority']=review.fusion_error.map({True:'1_existing_OOF_error',False:'2_existing_OOF_correct'}).fillna('3_historical_test_label_audit_only')
    review['conclusion']='source_label_disagreement; no_majority_vote_or_automatic_relabel'
    save('ghxtox_label_disagreement_review.csv',review.sort_values(['review_priority','split','source_id']))
    # Contrast broad and RBC labels separately; endpoint difference is not automatically a conflict.
    rbc=gh[['split','source_id','sequence','label']].merge(d[d.dataset=='HemoPI2'][['split','source_id','sequence','label','hc50_uM']],on='sequence',suffixes=('_GHXTox','_HemoPI2'))
    rbc['label_different']=rbc.label_GHXTox!=rbc.label_HemoPI2
    rbc['interpretation']='broad_toxin_label_vs_RBC_HC50; verify_endpoint_before_label_comparison'
    save('ghxtox_vs_hemopi2_endpoint_review.csv',rbc)
    # Actual global-identity strata for old tests, descriptive only; no re-selection of models/thresholds.
    hist=gh[gh.split!='train'].merge(align['train'],on='sequence',validate='many_to_one')
    hist['identity_stratum']=pd.cut(hist.max_identity,[-.01,.4,.7,1],labels=['<=40%','40-70%','>70%']).astype(str)
    save('historical_tests_identity_to_train.csv',hist[['split','source_id','sequence','label','identity_stratum','max_identity','nearest_ref_id','nearest_ref_sequence']])
    strata=hist.groupby(['split','identity_stratum'],observed=True).agg(n=('sequence','size'),positive=('label','sum')).reset_index()
    strata['negative']=strata.n-strata.positive;save('historical_identity_strata_counts.csv',strata)
    # Reuse/lineage summary; different method names do not provide independent label votes.
    relationships=[
      ('GHXTox / CAPTP / ToxMSRC / ToxPLTC','6387 train + 1126 test1 + 582 test2','all sequence/label pairs and row order identical','same benchmark; not four external datasets'),
      ('StrucToxNet','6387 train + 1126 test','identical to GHXTox train/test1','relevant same-data structural-model comparator'),
      ('ToxiPep0.9 / ToxMamba','7056 train + 1766 test + 958 independent','identical sequence/label pairs','single data family; paper ToxMamba table says1768 but public test1766'),
      ('HyPepTox-Fuse / ToxinPred3','8828 train + 2208 test','identical sequence/label pairs','single data family'),
      ('PeptiTox','3864 merged records','3144 overlap GHXTox train;544 historical tests;176 new','public merged file does not retain paper split'),
      ('ToxiCompass','6114 binary records','operational comparators;target/effect labels separate','assay prioritization; not universal human toxicity'),
      ('HemoPI2','1926 HC50 records','1540 cross-val +386 independent','specific RBC endpoint; source dose/species/chemistry still verify'),
      ('MAOMAO v1.1','71857 sequences;169 measurements/166 sequences','all7513 GHXTox known unique sequences present','evidence recovery; never entire resource as independent test')]
    save('dataset_relationships.csv',pd.DataFrame(relationships,columns=['data_family','composition','verified_relation','appropriate_use']))
    definitions=[
      ('CAPTP_lineage','broad mixed toxicity annotations','annotation/source-derived; not per-record experiment retained','annotation exclusion does not prove safety','original accession/endpoint/dose/chemistry largely absent','https://github.com/jiaoshihu/CAPTP'),
      ('ToxiPep_lineage_internal','broad mixed toxicity annotations','multi-source toxic annotations','reviewed UniProt/SwissProt toxin-annotation exclusion','per-record source accession and experiments not retained','https://pmc.ncbi.nlm.nih.gov/articles/PMC12171765/'),
      ('ToxiPep_lineage_independent','DBAASP-derived hemolytic positives vs annotation-exclusion negatives','paper reports479 experimentally measured hemolytic peptides','479 negatives are annotation exclusion, not paired measured nonhemolytic controls','public FASTA/CSV does not retain DBAASP IDs,assay cutoff,conditions,PTM','https://doi.org/10.1093/bib/bbag516'),
      ('ToxinPred3_lineage','broad peptide toxin labels','curated toxin datasets/annotations','SwissProt exclusion/synthetic-fragment source must verify','binary files contain sequences without original assays','https://github.com/raghavagps/toxinpred3'),
      ('PeptiTox_merged','broad peptide toxicity','public file sequence/label only','public file sequence/label only','original split and record evidence not recoverable from merged CSV alone','https://github.com/WYLqlit/PeptiTox'),
      ('HemoPI2_HC50','mammalian RBC HC50','benchmark label1 HC50<=100uM','benchmark label0 HC50>100uM; not universally nontoxic','RBC species,exposure,buffer,terminal PTM and original papers not in CSV','https://doi.org/10.1038/s42003-025-07615-w'),
      ('ToxiCompass_operational','toxin-like screening;positive target/effect assay priorities','BioTD/reviewed UniProt curated positive evidence','reviewed non-toxin/non-AMP operational comparator','target/effect evidence can be annotation supported;not proven causal mechanism','https://github.com/cihebiyql/ToxiCompass'),
      ('MAOMAO','eight distinct endpoints with 0/1/2/3/999 evidence states','direct or ontology-hierarchy supported;retain provenance','code0 is source negative evidence, not universal safety;2/3/999 are not negative','169 quantitative records mix HC50/LC50/LD50/MHC,units and censoring','https://doi.org/10.5281/zenodo.22312156')]
    save('label_evidence_definitions.csv',pd.DataFrame(definitions,columns=['data_family','endpoint','positive_meaning','negative_meaning','missing_or_limited_evidence','primary_source']))
    # All source snapshots, including raw archive and selective CSV provenance.
    hashes=[]
    for file in sorted((OUT/'sources').rglob('*')):
        if file.is_file():hashes.append(dict(path=str(file.relative_to(OUT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    save('source_cache_sha256.csv',pd.DataFrame(hashes))
    report_checks=json.loads((OUT/'verification.json').read_text())
    assert all(v if isinstance(v,bool) else all(v.values()) for v in report_checks.values())
    assert len(hist[hist.split=='test1'])==1126 and len(hist[hist.split=='test2'])==582
    assert len(review[review.split=='train'])==328
    assert candidates.max_identity_known.notna().all()
    assert all((a.alignment_columns>=a.sequence.str.len()).all() and (a.alignment_columns>=a.nearest_ref_sequence.str.len()).all() for a in align.values())
    assert all(not ((a.max_identity==1)&(a.sequence!=a.nearest_ref_sequence)).any() for a in align.values())
    ma=json.loads((OUT/'maomao_summary.json').read_text())
    s=json.loads((OUT/'summary.json').read_text())
    result=dict(**s,GHXTox_train_label_disagreements=328,GHXTox_test1_label_disagreements=54,GHXTox_test2_label_disagreements=47,
        GHXTox_unique_label_disagreement_sequences=review.sequence.nunique(),
        GHXTox_OOF_errors_with_label_disagreements=int(review.loc[review.split=='train','fusion_error'].eq(True).sum()),
        low_identity_review_records=len(shortlist),low_identity_review_unique=shortlist.sequence.nunique(),
        hemopi2_independent_priority_n=len(panel),hemopi2_independent_priority_pos=int(panel.label.sum()),
        hemopi2_independent_priority_neg=int((panel.label==0).sum()),maomao=ma,
        model_training_performed=False,external_model_predictions_performed=False,original_labels_modified=False)
    dump('final_summary.json',result)
    # Easy-to-read report with computed counts and explicit boundaries.
    display=screen[screen.dataset.isin(['ToxMamba','HemoPI2','ToxiCompass','PeptiTox'])][['dataset','split','n_exact_unseen_in_scope','identity_to_known_le40','low40_review_candidates']].rename(columns={
        'dataset':'来源','split':'原划分','n_exact_unseen_in_scope':'去重且长度合适','identity_to_known_le40':'与全部旧数据一致性≤40%','low40_review_candidates':'再排除精确片段包含'})
    text=f'''# GHXTox 数据关系与证据对照报告

完成日期：2026-10-25。结论：现在优先做标签与实验对象核查、建立具体终点的新验证面板。新网络应等待这些结果。

## 已经查明的数据关系

1. GHXTox、CAPTP、ToxMSRC、ToxPLTC 的训练集6387条、Test1 1126条、Test2 582条逐条序列、标签及顺序完全一致。StrucToxNet 的训练集和测试集也完全对应前两套。这有利于同数据比较模型，却不能当成几次独立的新数据验证。
2. GHXTox Test2 的582条全部在Test1中。训练集与两测试集没有完全重复序列，但同源性仍需分层考察。
3. ToxMamba 的训练、测试、独立集与 ToxiPep0.9 对应文件的序列/标签相同；HyPepTox-Fuse 与 ToxinPred3 的训练/测试相同。不能按方法名称重复计数或当成多份独立标签证据。
4. PeptiTox 公共文件3864条中，3144条在GHXTox训练集、544条在历史测试集，只有176条不在旧数据中；公开merged CSV未保留原始划分，本轮不声称恢复其论文划分。
5. MAOMAO新版71857条包含全部6387条训练序列及全部7513条已用过的独特序列。它能帮助查证据，整套并不构成新的独立验证。

## 同一序列，答案可能不一样

在本轮模型二分类/操作性比较标签中，共有433条独特序列出现正负标签并存。其中涉及GHXTox的382条：训练集328条、Test1 54条；Test2中的47条属于上述Test1记录。GHXTox侧这些记录均为原负类，其他来源至少一处为正类。

训练集这328条中，{result['GHXTox_OOF_errors_with_label_disagreements']}条属于先前327条OOF错题。这只是复核优先级，不能证明它们原标签错误，也不代表模型因此正确。

相反标签可能来自实验对象不同、溶血与广义毒素标签不同、浓度/物种不同、注释与实测不同，或资料更新。已输出逐序列双方来源；没有投票、修改或删除任何原始标签。HemoPI2的溶血标签与广义毒性标签另表比较，没有混入433条统计。

## 新验证数据筛选到哪一步

ToxiPep/ToxMamba独立集958条中，151条在GHXTox训练集、31条在历史测试集，剩776条无精确重叠；限制为当前模型的11–50个标准氨基酸后剩722条。两者是同一批样本，不能算两个独立验证集。

HemoPI2独立集386条中，14条在训练集、3条在历史测试集；剔除后369条，长度合适341条。其中69条与全部旧数据的最大一致性≤40%，再经精确片段包含检查后本轮优先面板为{len(panel)}条（原HC50高活性{int(panel.label.sum())}条、低活性{int((panel.label==0).sum())}条）。这是待核查的候选面板，尚未取得逐条实验条件与化学形式，也没有运行模型看结果。

{table(display)}

一致性由全序列全局比对计算，所有候选都对照全部训练序列和全部已用过的序列；匹配+2、不匹配−1、开缺口3、延伸1，末端缺口计入。分母包含所有对齐列。这里的40%与CD-HIT内部定义不同；它是明确记录的筛选口径，不等同于已排除所有家族/研究来源关联。长度、修饰、环化、二硫键及实验论文分组仍须核查。初次实现的stats长度忽略起始缺口，已通过片段对照检验发现并废弃；本报告只用traceback修正后的_v2表。

旧Test1中≤40%、40–70%、>70%分别288、339、499条；Test2分别173、171、238条。这是分层数量，不是新性能结果，没有依据它重新选择模型或阈值。

## 标签含义不同，不能直接合并

- ToxiPep/ToxMamba独立集：原文支持479条阳性具有DBAASP溶血实验来源；479条阴性使用排除毒素注释的筛选策略。整套并非479条已测溶血与479条已测不溶血的配对实验。公开文件没有保存逐条DBAASP ID和浓度条件。[ToxiPep原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12171765/)；[ToxMamba原文](https://doi.org/10.1093/bib/bbag516)。
- HemoPI2：公开CSV保留HC50，本轮核对其标签1均为≤100 μM、标签0均为>100 μM。HC50是造成50%溶血的浓度，越低表示越容易溶血；这个分类阈值不能直接解释为普遍安全/危险。CSV没有完整保留物种、时间、缓冲液与末端修饰。[原文](https://doi.org/10.1038/s42003-025-07615-w)。
- ToxiCompass：Stage1是毒素样序列与操作性对照的区分；Stage2为靶点/效应类的实验优先级，不是逐条已证实的因果机制。其负类不能直接当成实验证实无毒。[官方数据卡](https://github.com/cihebiyql/ToxiCompass/blob/87805bef0f652f89cae9dc2739f64c29d49f4f3f/docs/BENCHMARK_CARD.md)。
- MAOMAO：0负证据、1阳性或层级支持、2含混、3未标注、999无信息，保留这些状态。169条定量记录对应166条序列，混有HC50/LC50/LD50/MHC、μM/μg·mL⁻¹及大于/小于号。与旧数据无精确重复且长度合适的定量序列136条，其中≤40%为35条；不能把35条不同终点拼成一个统一毒性测试集。[官方版本](https://doi.org/10.5281/zenodo.22312156)。

本轮还复用了先前训练实体审核：1606条训练序列匹配当前Fragment条目。片段标记是审核线索，未自动视为错误。没有对新候选运行模型或按预测分数挑选样本。

## 接下来实际优先级

1. 先核查训练集328条标签相反序列，优先查看其中OOF错题、片段边界及有明确实测来源者。目标是恢复“到底测了哪个分子、什么效应、什么条件”，保留未知状态。
2. 以HemoPI2候选面板为入口，恢复原始论文、红细胞物种、HC50与浓度单位、暴露时间和肽修饰；MAOMAO/Hemolytik定量记录作为补充线索。同一序列/家族/论文不得跨开发与最终评价组。
3. 对确认后的具体终点面板，先冻结样本清单、阈值、模型文件哈希和评价规则，再做现有模型的外部验证。历史Test1/Test2保持回顾性定位；不能重新称为盲测。
4. 如果证据清楚、分子对象一致的低Cys膜活性肽仍系统漏判，再研究膜选择性或局部排列机制，并与简单组成模型、普通加权/集成基线比较。通道毒素另设终点面板，不强行用溶血解释。

## 可复核文件

- dataset_relationships.csv：数据复用关系；dataset_inventory.csv：逐划分数量。
- pairwise_exact_overlap.csv：32个划分两两精确重叠及标签差异。
- ghxtox_label_disagreement_review.csv：GHXTox待复核记录（保留重复Test2行以溯源）。
- same_sequence_label_review.csv：全来源433条广义标签差异；ghxtox_vs_hemopi2_endpoint_review.csv：另列溶血比较。
- external_similarity_screen.csv：候选筛选数量；hemopi2_independent_priority_panel.csv：优先逐条核查面板。
- maomao_endpoint_cross_reference.csv：已有序列与多终点证据状态；maomao_quantitative_candidates.csv：定量记录原关系符号/单位。
- alignment_to_train_v2.csv / alignment_to_known_v2.csv：最终全局比对明细；debug_invalid_stats_v1/中的两张初版表仅供错误排查，不得用于科学结论。
- source_manifest_extended.json、source_cache_sha256.csv及sources/中的固定commit树、压缩包/选择性下载记录：来源版本与哈希。

本轮完成的是数据审计与候选清单，不是新模型性能实验。还没有完成逐条原始实验全文与化学形式核实，因此没有把候选清单称为已验证的金标准外部测试集。原数据、模型、阈值和标签均未修改。
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf8')
    protocol=dict(task='prospective_metadata_review_then_frozen_endpoint_validation',created='2026-10-25',
        scope='linear standard20 length11-50; unrepresented modifications require explicit separate handling',
        primary_candidate_panel=str(OUT/'hemopi2_independent_priority_panel.csv'),panel_sha256=hashlib.sha256((OUT/'hemopi2_independent_priority_panel.csv').read_bytes()).hexdigest(),
        status='candidate_manifest_frozen_for_evidence_review; not_final_gold_test',
        label_policy='retain source HC50 and threshold100uM; validate assay comparability before binary scoring; unknown not negative',
        identity='final traceback global identity<=40% to allGHXToxknown, no exact fragment containment',
        additional_exclusions='same molecular entity/PTM/species/time/assay; family and study grouping',
        outcome_policy='no GHXTox predictions inspected on these candidate records; do not develop from final outcomes',
        required_before_evaluation=['original assay provenance','chemistry/terminal forms','study/family grouping','frozen model/checkpoint/calibration hashes','fixed thresholds/metrics/paired resampling'],
        caution='GHXTox broad toxin score applied to RBC endpoint is endpoint transfer, not a universal toxicity validation',
        metrics=['AUROC','AUPRC with prevalence','MCC at frozen threshold','sensitivity','specificity','paired confidence intervals; study/family clustering where possible'])
    dump('NEXT_VALIDATION_PROTOCOL.json',protocol)
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':finalize()
