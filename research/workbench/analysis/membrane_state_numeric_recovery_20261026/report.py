from pathlib import Path
import json,hashlib
O=Path(__file__).resolve().parent;R=O.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
l5=read(O/'L5_digitized_all11.json');pmap=read(O/'PMAP_digitized_all4.json');quality=read(O/'digitization_quality.json');assoc=read(O/'L5_descriptive_state_associations.json')
assert len(l5)==11 and len(pmap)==4
lines=['# 膜环境状态：限定三来源数值恢复结果','',
'2026-10-26，仅主Agent。**恢复了2个系列、15条肽的部分结构—宿主效应关联，但并未得到可直接训练新三维网络的跨系列状态数据。停止本轮补资料，不新增论文、不模拟或训练。**',
'','## 恢复结果','',
'|来源|本轮实际恢复|仍缺什么|','|---|---|---|',
'|L5K5W，11肽|图3D的3种环境×2波长共66个CD柱值；图6的11个溶血百分比、11个螺旋含量和11个CD乘积，共99个图读数；与旧序列/分子形式/MHC5关联|图3A水相谱线叠加且未恢复逐肽身份，因此水相数值留空；没有原子坐标|',
'|PMAP，4个衍生肽|从原始PDF图2提取原生图片；在222nm读水相和模拟环境包络区间，与旧MHC5及删失标记对应|TFE/SDS在该波长重叠，不硬拆为两个点值；没有成对3D坐标|',
'|PG-1|沿用已恢复的作者实验输入表和预印本条件|本轮未取得补充材料/CD原始数值或CD子集完整身份映射；不把93个序列全当CD数据|',
'',
'15是两组关联的肽数，不是15条统一目标、统一精度的训练数据。99是重复条件/波长/终点的读数，不是99个独立样本。主任务标签未改。',
'','## 图像读数怎样控制误差','',
'L5原图来自出版社公开PNG。按原图W1–W11顺序识别全部柱体，保存像素边界、轴刻度、原图哈希与读数脚本。图6螺旋含量纵轴从60%起，已正确扣除截断轴，未把柱高直接当0起点。',
'',
'每个L5读数带保守的4个检查视图像素误差包络，涵盖边缘和刻度读数；这是图像数字化容差，不是实验95%置信区间。直接实验原始表值仍来自旧核查，图读数不冒充作者原始数值。',
'',
f"质量检查：螺旋含量与溶血的图读R²={quality['helicity_r2_digitized']:.4f}，原文报告0.7321；CD乘积的图读R²={quality['theta_product_r2_digitized']:.4f}，原文0.9022。这是复现原图以核验读数，没有发现新的机制规律，也没有提高分类指标。",
'',
'进一步把图3D两波长相乘，与图6B独立读取的CD乘积比较，11条的读数容差范围均重叠。这是跨图的一致性检查，不能代替原实验误差。',
'',
'PMAP使用990×703的原生PDF内嵌图，无图片放大后伪造精度。读取222±约1nm窗口中的标记/线宽，另加3原生像素容差。原图标记：圆点为PBS，三角为TFE，方框为SDS；后两条在222nm部分重叠，统一保留覆盖两条线的包络区间。所有点值字段保持null，禁止后续程序默默取中点当实测值。',
'','## 能否证明“环境变化”比单状态结构更有用','',
'目前不能。PMAP只有4条，其中2条MHC5右删失；没有对删失值强填128。L5只有1个系列，且水相逐肽值未恢复。真正想检验的“水相→宿主膜”变化，在这批数据里仍不完整。',
'',
'仅对L5恢复值做事后描述，比较单环境CD强度与TFE−SDS差值和固定浓度溶血的关系：','',
'|可恢复指标|同研究内Pearson R²|','|---|---:|']
for x in assoc['results']:lines.append(f"|{x['feature']}|{x['pearson_r2']:.4f}|")
lines+=['',
'TFE−SDS是两种人工环境的差，不是水→膜；表中结果没有证明这种差值优于单环境数值。这个表只有11条、事后选择可恢复量、带数字化误差，没有显著性或跨样本预测保证；不把它当作否定所有构象动态或膜状态建模的证据，也不因TFE列较好就用本组结果调选模型环境。',
'',
'L5全部肽的长度和氨基酸组成相同，有助于控制全局理化差别；但序列顺序不同，不能宣称CD提供的信息必然超出序列模型。MHC5、128μg/mL百分比和不同环境CD没有混成HC50或广义毒性标签。',
'','## 停止条件与下一步优先级','',
'本轮三来源补齐到此结束。仍缺第三个数值闭合系列、可比较水/膜状态，以及宿主膜相关的结构证据，因此**不启动状态三维分支、MD或新的辅助分类配置**。不是“这条生物学机制没有价值”，而是当前数据尚不足以支持所拟议的具体改法。',
'',
'已恢复的数据保留为机制复核资源，不继续围绕单个系列搜索或调参。研究主线应回到已有毒性分类模型的严格独立评估、必要消融和同协议基线，先确定现有性能与稳定短板；未来只有新的可用状态数据或明确独立假设出现，才重启该方向。分类收益仍必须看整体MCC/AUPRC和其余类别，不能把本报告的相关系数当作毒性分类提升。',
'','## 文件与取回限制','',
'L5_digitized_all11.json与PMAP_digitized_all4.json为逐肽关联；digitization_calibration.json、digitization_bar_boxes.json和PMAP_digitization_provenance.json保存像素依据；cross_figure_consistency.json和digitization_quality.json为质量复核。',
'',
'PMAP正文公开PDF成功取得。旧NCBI OA地址404，现站OA入口403；Nature图页跳转授权不可读；EuropePMC两项补充附件超时。保留失败状态，没有把HTML/超时文件算作数据。只使用公开正文、出版社开放原图和之前缓存，未发送邮件或联系作者。',
'',
'本轮使用PDF技能读取原图；没有编辑PDF。所有新产物保存本目录，未重算旧基础错误分组、旧52/27配对或30次分类辅助小试验，未训练模型或修改GHXTox权重。',
'','## 原始来源','',
'[L5研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC6270530/)，图3/6、方法；[PMAP原文](https://www.nature.com/articles/srep27258)，图2/Table4；[PG-1作者仓库](https://github.com/ziul-bio/DMS_ML_AMP)及[同研究预印本](https://pmc.ncbi.nlm.nih.gov/articles/PMC10543523/)，保留与正式2024版本的差别。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
save=lambda n,o:(O/n).write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
save('decision.json',dict(sources_targeted=3,numeric_series_recovered=2,matched_peptides=15,L5_peptides=11,PMAP_peptides=4,PG1_CD_numeric_recovered=0,paired_atomic_structure_records=0,cross_series_state_supervision_ready=False,new_3D_training=False,new_model_fits=0,MD_runs=0,stop_bounded_recovery_now=True,scope='figure-derived source-specific evidence, not raw spectra or new classification gains'))
table=['# 两组全样本数值关联','',
'所有图读值均为近似值，详细容差见JSON；PMAP保持区间不填中点。','',
'|系列/样本|固定剂量溶血%（图读）|TFE螺旋含量%（图读）|水→模拟环境θ222差值区间|直接MHC5（旧原表）|','|---|---:|---:|---|---|']
for r in l5:
 d=r['digitized_values'];h=r['existing_direct_MHC5'];table.append(f"|L5/{r['name']}|{d['hemolysis_percent']['value']:.1f}|{d['helicity_TFE_percent']['value']:.1f}|缺水相身份，不计算|{h['operator']}{h['value']} {h['unit']}|")
for r in pmap:
 d=r['state_contrast_interval'];h=r['direct_host_endpoint'];table.append(f"|PMAP/{r['name']}|未恢复|未恢复|[{d['lower']:.2f}, {d['upper']:.2f}]×10⁴|{h['operator']}{h['value']} {h['unit']}|")
(O/'ALL_15_LINKS.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
print('15-peptide source-linked report written. Stop bounded recovery; no classifier/3D training readiness.')
