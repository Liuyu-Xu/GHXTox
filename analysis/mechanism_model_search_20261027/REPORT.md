# 按GHXTox短板寻找生物机制候选模型

日期：2026-10-27。状态：联网初筛完成；未下载模型权重、未执行预测、未重训、未融合。证据为作者论文、代码仓库和官方模型卡；可使用入口不等于已通过本地复现。

## 出发点

复用 ghxtox_shortfall_registry_20261027：原标签低Cys正类421条中，新旧流共同漏151；富Cys负类445条中共同误64。共同错误不等于跨种子重复验证。原标签不等于统一实验终点金标准，尚无证据证明它们全是膜机制问题。不得将全部151条当溶血阳性，或把64条当实验确认安全。

## 候选与优先级

|顺序|候选|它实际学什么/生物学动机|公开资产核查|对应短板与限制|下一项最小工作|
|---|---|---|---|---|---|
|1|HemoPI2，2025|红细胞裂解分类、另有HC50回归；ESM2/传统模型及motif模块，提供宿主膜损伤端点|作者GitHub有Dataset、分类/回归代码；模型压缩包另在作者下载站，尚未下载校验；GPL-3.0|先看膜相关共漏及同类正确/负背景；不覆盖受体/凝血毒性，不提供膜脂物理相互作用证明；不是重做曲线拟合|先核对训练序列重叠及端点，核查分类权重；固定官方分类器对全6387诊断，分来源/家族/重叠层统计纠错添错|
|2|NTxPred2，2025|神经毒性分类；短肽ESM2-t30与较长蛋白ET分支|作者GitHub与HF公开；HF API确认config、tokenizer及model.safetensors存在，revision a50ea08f7aef4d6632d4a48ddef92bdaccfcefa5；GPL-3.0|对应通道/受体相关注释子类，但神经毒性并不等于具体通道或受体结合；未证明可解富Cys误报，反可能重复毒素家族偏好|核对长度与官方分支、训练负类及UniProt同源来源；通过后对全体运行，单列低Cys神经相关与富Cys添错|
|3|Protegrin-1深度突变扫描+膜选择性模型，2024|比较细菌与哺乳动物膜作用；研究结合态二级结构、芳香残基和Cys配对变化|Nature Biomedical Engineering原研究可核；论文有训练模型，通用可复用权重与完整代码入口本轮尚未确认|最贴近膜选择性生物问题；主要为Protegrin-1突变背景，不能泛化成所有低Cys毒肽教师，且其骨架本身富Cys|作为机制数据/设计参考，先查训练表与代码；仅在合适家族检验，跨骨架评价后才考虑辅助监督|
|4|HELM-BERT，2026；HemoPImod，2020作端点对照|前者用HELM表示修饰残基与连接形式；后者预测化学修饰肽溶血|HELM-BERT作者仓库/HF可查，54.8M，model.safetensors约219MB；MIT，revision 1820da61dbb306c556f7af88f19067c6f0fc7801；paper分支区别当前更新。HemoPImod作者论文声明standalone与下载数据，本轮未复现|解决输入化学身份缺失的可能性；HELM-BERT是表征模型非现成毒性分类器。化学形式未知就不能正确编码，无法自动修复完整毒素与亚基标签错配|先统计已有明确修饰/连接信息覆盖，不再只看H1/H2；具备输入才做冻结表征增量探针，保留Conantokin反例|

该顺序是可执行小试优先级，不是预期指标提升排名。尚不能估计任何模型在GHXTox上的实际收益。前两项可先做本地现成模型推理，不要求服务器或训练；确切显存、速度、兼容修复时间需实测。后两项数据/化学整理及再训练成本更高，不能承诺具体天数/费用。

## 值得借鉴但不直接当教师的方案

- 2022年《Function Prediction of Peptide Toxins with Sequence-Based Multi-Tasking PU Learning Method》：以UniProt毒肽为对象，正例—未标注学习预测溶血、神经、血管等功能。关键是“没有机制注释不等于没有该机制”。可借鉴有证据机制标签的掩码/PU辅助学习；它并不证明现有毒性负类都应当改为未标注。本轮没有确认可直接加载的作者权重，部分全文访问受限，暂不能列即用模型。
- AMPDeep有序列溶血迁移学习代码及数据，先经过分泌任务，但分泌不是宿主毒性机制；旧依赖且现成权重尚未确认，作为备用而非增加首轮模型数量。
- StrucToxNet是必要结构竞品：作者论文使用CAPTP数据，ProtT5+ESMFold+EGNN，与GHXTox信息重叠明显，且明确未建模细分机制/修饰；适合公平基线，不据此声称机制补强。PeptiTox作者仓库有代码及Data，但根目录未见即用权重/README；未进一步判定全仓无权重。
- SaProt保持已有诊断结论：结构表征候选，不重新作为默认赢家；先前小幅纠错与阈值效应不构成生物机制证明。

## 下一步执行门槛

1. 先取HemoPI2/NTxPred2训练数据与分类配置做来源、完全重复和近同源排查；区分有监督标签泄漏与普通自监督预训练接触。预训练接触也记录，但不能直接等同毒性标签泄漏。
2. 当前错误图谱已经参与候选选择，接下来其纠错统计只能用于发现/筛选。即便排除了教师训练重叠，也不是新的无偏确认。后续严格确认要在训练与候选选择均隔离的分组外层或真正未用于开发的外部集进行，Test1/2仍只作历史参照。
3. 现成教师无需为诊断全重训，但诊断必须包括全部6387以及151/64、同类正确池213/357和其他背景；不能只跑错题。排除教师监督训练重叠后的子集单独报告，不把混入训练样本的表现当泛化。
4. 首轮保留官方默认阈值/原始分数，报告纠错、添错、排序及来源/家族分层；不能直接拿溶血分数当总毒性概率。额外GHXTox分类头/校准器/门控均须训练选择评价隔离。
5. 至少证明模型针对某机制相关类有稳定增量且全体没有过度添错，再提出掩码机制监督、表征蒸馏或受限门控。纠错不等于因果解释；若增量可被组成/家族来源解释，不宣称发现新机制。

## 原始来源

- HemoPI2论文：https://www.nature.com/articles/s42003-025-07615-w
- HemoPI2作者代码：https://github.com/raghavagps/hemopi2
- NTxPred2论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC12267675/
- NTxPred2作者模型：https://huggingface.co/raghavagps-group/NTxPred2
- NTxPred2代码：https://github.com/raghavagps/ntxpred2
- 膜选择性突变研究：https://www.nature.com/articles/s41551-024-01243-1
- HELM-BERT作者代码：https://github.com/clinfo/HELM-BERT
- HELM-BERT权重：https://huggingface.co/Flansma/helm-bert
- HemoPImod论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC7045810/
- 机制PU学习论文：https://www.mdpi.com/2072-6651/14/11/811
- AMPDeep：https://github.com/milad73s/AMPDeep
- StrucToxNet论文：https://pmc.ncbi.nlm.nih.gov/articles/PMC12306123/
- PeptiTox代码：https://github.com/WYLqlit/PeptiTox

作者README与HF资产目录元数据已缓存于本目录，见source_manifest.json。仅缓存小型说明/元数据，未执行远程代码。
