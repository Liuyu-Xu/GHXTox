"""Five-study, evidence-level feasibility inventory. No training or simulations."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
O=Path(__file__).resolve().parent;R=O.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
studies=[
 dict(id='PMAP36',pmid='27251456',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC4890124/',series='PMAP-36截短系列',peptides_in_old_audit=5,structural_candidate_n=4,structural_evidence='four derivatives CD in PBS/TFE/SDS; helical-wheel and projection are not measured paired3D',environment_pair='aqueous buffer vs50%TFE/30mMSDS',host_endpoint='human RBC MHC5,2right-censored',host_variation_candidate=True,in_GHXTox_standard_scope=True,paired_atomic_coordinates_recovered=0,raw_per_peptide_structure_table_recovered=False,model_membrane_caution='TFE/SDS not host cell membrane; comparison is secondary structure',readiness='candidate_secondary_structure_comparison_not_training_ready'),
 dict(id='Pap12',pmid='30846839',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6405874/',series='Pap12系列',peptides_in_old_audit=6,structural_candidate_n=6,structural_evidence='CD in water/SDS/DPC; PEP-FOLD predictions are not paired experimental membrane conformations',environment_pair='water vs100mMSDS/50mMDPC',host_endpoint='100uM,1peptide3% and5undetectable; HC10tableimputed/inconsistent',host_variation_candidate=False,in_GHXTox_standard_scope=True,paired_atomic_coordinates_recovered=0,raw_per_peptide_structure_table_recovered=False,model_membrane_caution='micelles not RBC bilayer; no usable high/low host toxicity contrast at stated dose',readiness='negative_or_low_effect_context_only'),
 dict(id='L5K5W',pmid='23344198',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC6270530/',series='L5K5W位置异构系列',peptides_in_old_audit=11,structural_candidate_n=11,structural_evidence='CD,fluorescence,2D COSY; no recovered solved paired atomic structures',environment_pair='phosphate buffer vsTFE titration/SDS/DPC',host_endpoint='human RBC MHC5 and128ug/mLhemolysis Figure6A',host_variation_candidate=True,in_GHXTox_standard_scope=True,paired_atomic_coordinates_recovered=0,raw_per_peptide_structure_table_recovered=False,model_membrane_caution='reported correlation in TFE, not SDS/DPC; TFE not validated actual RBC-bound state',readiness='candidate_secondary_structure_comparison_not_training_ready'),
 dict(id='PG1',pmid='39085646',url='https://www.nature.com/articles/s41551-024-01243-1',series='PG-1变体系列',peptides_in_old_audit=None,structural_candidate_n=41,structural_evidence='same-study preprint describes parent+40variants CD with/withoutLPS; not41atomic structures',environment_pair='buffer vsLPS bacterial outer membrane proxy',host_endpoint='preprint assay human RBC128ug/mL3h; repository workbook experiment-input columns, precise row-to-cohort/form mapping pending',host_variation_candidate=True,in_GHXTox_standard_scope=True,paired_atomic_coordinates_recovered=0,raw_per_peptide_structure_table_recovered=False,model_membrane_caution='LPS is bacterial proxy, not mammalian membrane; published2024fulltext not recovered; repository links2023preprint',readiness='candidate_secondary_structure_comparison_not_training_ready'),
 dict(id='cyclic2025',pmid='39969852',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC11891913/',series='环状RW六肽对照',peptides_in_old_audit=None,structural_candidate_n=2,structural_evidence='lipid binding/packing/vesicle leakage, not peptide paired atomic coordinates',environment_pair='PC vsPG/PE model membranes',host_endpoint='relative host cytotoxicity/hemolysis cited earlier literature; not newly matched host assay recovered here',host_variation_candidate=False,in_GHXTox_standard_scope=False,paired_atomic_coordinates_recovered=0,raw_per_peptide_structure_table_recovered=False,model_membrane_caution='six-residue cyclic peptides outside current linear11–50scope; model vesicle effects not host toxicity gold labels',readiness='mechanistic_background_only')]
save('study_readiness.json',studies)
tables=read(O/'PG1_workbook_summary.json');tree=read(O/'sources/PG1_tree.json')['tree'];files=[x['path'] for x in tree if x.get('type')=='blob']
valid=[]
for name in ['cyclic2025.xml','PG1_preprint.xml','PG1_BioC.xml']:
 p=O/'sources'/name
 try:root=ET.parse(p).getroot();ok=root.tag=='article'
 except ET.ParseError:ok=False
 valid.append(dict(file=name,valid_article_xml=ok,sha256=sha(p)))
assert next(x for x in valid if x['file']=='PG1_BioC.xml')['valid_article_xml'] is False
decision=dict(studies_reviewed=5,new_studies=2,reused_studies=3,candidate_series_with_host_variation_and_environmental_CD=['PMAP36','L5K5W','PG1'],candidate_peptides_upper_bound_not_joined_training_rows=56,independence='operational series only, no claim3independent phylogenetic families',paired_water_membrane_atomic_structures_recovered=0,native_per_peptide_CD_plus_host_table_complete=False,ready_to_train_new_3D_network=False,ready_for_claimed_toxicity_improvement=False,limited_mechanistic_followup_possible=True,next_if_continuing='only recover existing source figure numerics and cohort/form mappings, then independent-series state-vs-static diagnostic; no further paper expansion or large simulations in this stage',stop='stop this five-study search; no neural training,MD,or peptide design',model_fits=0,simulations=0)
save('decision.json',decision)
sourcepaths=[R/'analysis/hemolysis_selectivity_audit_20261026/sources'/n for n in ['PMC4890124.xml','PMC6405874.xml','PMC6270530.xml']]+[O/'sources'/n for n in ['cyclic2025.xml','PG1_preprint.xml','PG1_tree.json','README.md','Cumulative_Peptide_data_PG1.xlsx','Cumulative_Peptide_data_PG1_v01.2.xlsx']]
save('verification.json',dict(source_hashes={str(p):sha(p) for p in sourcepaths},retrieval_content_validation=valid,n_studies=5,workbook_source_versions_not_concatenated=True,repository_atomic_coordinate_files=[f for f in files if f.lower().endswith(('.pdb','.cif','.xtc','.dcd','.gro'))],repository_named_spectrum_files=[f for f in files if any(k in f.lower() for k in ['dichro','spectr','lps','circular'])],no_model_fitting=True,no_simulations=True,no_original_labels_changed=True))
lines=['# 膜环境结构变化路线：有界数据可行性检查','',
'2026-10-26，仅主Agent。范围固定为5项研究：复用旧3篇，新增PG-1及2025环肽论文。**有生物学依据，存在3个候选系列，但当前恢复的数据不够直接训练“膜前后状态”的新三维网络。停止扩展论文搜索，不启动大规模模拟或训练。**',
'','## 我们这次实际确认了什么','',
'需要区分三个层次：CD说明二级结构随环境改变；人工膜结合/泄漏说明与某种脂质体系有相互作用；成对三维坐标才直接描述水相与膜结合状态的具体残基空间变化。这些不能当成同一种标签，尤其不能把CD曲线写成实验3D结构。',
'','|系列|结构相关证据|宿主毒性差异|本轮用途|','|---|---|---|---|',
'|PMAP-36|4个衍生肽有PBS/TFE/SDS的CD图|原5肽MHC5不同，部分删失|候选整类二级结构对照，尚未恢复逐肽数值表|',
'|Pap12|6肽水/SDS/DPC的CD，另有PEP-FOLD预测|100μM均很低，不能形成所需高/低毒性对照|低效应背景；不把代填HC10当标签|',
'|L5K5W|11个异构肽CD/荧光，图3/6给出结构参数|同研究MHC5和固定浓度溶血不同|候选；TFE关联不能自动代表真实宿主膜|',
'|PG-1|同研究预印本描述母体+40变体有/无LPS的CD|有逐变体溶血实验说明，作者实验输入表已恢复|候选；41肽CD身份映射及数值未闭合|',
'|2025环状RW六肽|两肽PC/PG-PE膜结合、泄漏等|宿主毒性差异引用早文，本轮未恢复同批新测|仅机制背景；6残基环肽不在当前11–50线性范围|',
'',
'PMAP4+L5K5W11+PG1母体及40变体为最多56个候选，不是56条已完成结构/毒性同分子关联的训练数据，也不是56个独立家族。Pap12不因有结构变化就自动列入毒性差异验证。',
'','## 两项需要修正的先前直觉','',
'第一，环境变化的影响并非“总是有/总是同一方向”。L5K5W原文报告TFE中的结构参数与溶血相关，但SDS/DPC中未出现同样关联；这能支持提出环境依赖假设，却不足以把某个模拟环境等同于红细胞膜。',
'',
'第二，PG-1的LPS模拟的是细菌外膜，不是哺乳动物红细胞膜。它的二级结构与宿主溶血相关，仍不等于测量到真正宿主膜结合态。不能据此直接建立“人膜结构”的监督标签。',
'','## PG-1新增恢复：有实验输入表，仍缺结构数值','',
'作者仓库[ziul-bio/DMS_ML_AMP](https://github.com/ziul-bio/DMS_ML_AMP)已取得目录快照、README和两份小型xlsx，Git blob SHA逐项核验。没有执行作者代码、下载大规模预测或重跑SLAY。',
'',
'原工作簿96行/96ID/93不同序列，v01.2为93行/93ID/93序列。它们是同一来源的不同视图，不能相加成189肽。列仅为ID、Sequence、MBC、%Hemolysis（旧表多一索引列），没有CD或坐标列；MBC不能静默改名MIC，工作簿未给出的测量单位/化学形式也不能自行补齐。96条表行不能全部当作具有CD的40个变体。',
'',
'当前仓库快照无命名的光谱数据文件或PDB/CIF/轨迹文件；这只说明本次恢复范围，不声称原作者没有任何其他数据。另有作者同研究2023预印本开放XML，明确说明40变体+母体有/无LPS CD，以及128μg/mL、3小时的人红细胞溶血测定。正式2024出版全文本轮未成功取回；README链接预印本，表版本与正式发表版本的逐项对应仍未证明。所有版本边界保留。',
'',
'HTTP200也不一定是有效全文：PG1_BioC.xml实际为“No result”网页，已标为失败；既有PG1的151字节文件也不当正文。没有把失败下载当资料恢复成功。',
'','## 判断：哪一级现在能做','',
'|目标|现状|决定|','|---|---|---|',
'|判断该生物假设有无原始研究依据|5研究范围内可核对|有依据，但不等于解释GHXTox错误|',
'|跨系列小型二级结构—宿主效应复核|3个候选系列；逐肽原始结构数值/身份未齐|有界补齐现有图表后才可进行；暂未计算新的相关或分类提升|',
'|训练水相/膜相三维状态编码器|本轮恢复的成对原子坐标为0|不启动|',
'|把新表征接入GHXTox并声称提升|没有新分类试验或独立验证|不启动|',
'',
'停止门槛按数据闭合判断，不以“读了多少篇”代替。当前缺的是可直接核对的输入—环境—结构—宿主测量关联，而非多一个神经网络模块。',
'','## 下一项任务应怎样限制','',
'如果继续这条路线，任务只能是对PMAP、L5K5W、PG-1这3个已有候选做一次图表数值/身份恢复；优先作者原始数值，无原始值时图像数字化必须报告不确定性，不能伪装成原始实验精度。尤其先确认PG-1作者表中哪些行对应CD图谱，以及N/C端形式是否一致。',
'',
'只有关联闭合后，才比较“结构变化”是否比“静态结构/序列理化”更能解释同研究内的宿主效应差异，并按整个系列留出检验。MHC5、固定浓度百分比、不同物种与时间分开，不能混成统一HC50或毒性标签。即使机制复核通过，最终仍必须在原毒性分类MCC/AUPRC及其余样本上证明收益。',
'',
'如果现有图表无法可靠恢复，就结束这条数据路线、降为机制背景，不扩大文献搜索、不用大规模MD替代缺失实验证据；研究资源优先回到既有分类模型的独立验证和必要消融。不能为了创新而把缺少监督的状态编码器强行接入。',
'','## 复用与限制','',
'未重算旧22肽/222菌株关联、旧错误分组、空间特征检验或30次辅助分类小试验。没有重新训练模型、修改原数据标签、读取测试预测、模拟肽/膜或提出提高毒性的序列设计。此轮只有数据恢复和证据层级核查。',
'',
'study_readiness.json与decision.json给出机器可读结论；cached_structural_sections.json、PG1_preprint_sections.json、cyclic2025_sections.json保留段落定位；PG1_workbook_extraction.json为只读原表转换；retrieval_manifest.json、PG1_download_manifest.json及verification.json记录来源和哈希。',
'','## 原始来源','',
'- [PMAP-36研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC4890124/)：CD结果/Fig2、溶血方法与表。',
'- [Pap12研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405874/)：Fig3及CD方法；沿用旧HC10修正。',
'- [L5K5W研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6270530/)：Fig3/Fig6、§2.5、CD/NMR/溶血方法。',
'- [PG-1正式论文](https://www.nature.com/articles/s41551-024-01243-1)、[同研究预印本](https://pmc.ncbi.nlm.nih.gov/articles/PMC10543523/)、[作者数据仓库](https://github.com/ziul-bio/DMS_ML_AMP)：保持版本区分。',
'- [2025环肽膜选择性研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC11891913/)：PC/PG-PE相互作用；宿主效应引用之前研究。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(decision,ensure_ascii=False,indent=2))
