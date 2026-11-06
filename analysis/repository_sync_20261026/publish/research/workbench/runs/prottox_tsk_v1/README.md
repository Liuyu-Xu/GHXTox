# ProtTox-TSK 独立路线：全部实验记录

更新时间：2026-09-05。本文档只记录独立的轻量序列模型路线，不改变项目根 README 中历史 GHXTox 的 1D+3D 主模型定义，也不涉及网站改动。

## 结论先行

目前唯一保留的默认模型是：

```text
Frozen ProtT5 residue embeddings
  → TriStat pooling（mean / max / std）
  → 9 个理化特征
  → MLP
  → toxicity probability
```

截至原始四组候选实验，均未通过晋级门槛；2026-09-05 追加的 outer-safe 3D-expert 蒸馏（尝试 5）首次通过训练 OOF 晋级条件。Test1/Test2 仍未读取，默认 MLP 尚未执行部署替换；判定与完整记录见文末“尝试 5”。

## 固定实验边界

- 训练集：6,387 条记录，4,569 条非毒、1,818 条有毒。
- 特征缓存：[features.pt](features.pt)；Frozen ProtT5 为 1024 维残基表示，TriStat 为 3,072 维，另加 9 个 PhysChem 特征。
- 折划分：[audited group-fold CSV](../../data/folds/prottox_tsk_train_prott5_official_mean_folds_v1.csv)。训练源、序列、标签和分组映射均逐条核验，无组泄漏。
- 所有正式新增候选：五折 OOF、20 epochs、learning rate `1e-3`、种子 `20260905` 与 `20260906`、固定阈值 `0.5`。
- 选择范围：训练 OOF。Test1/Test2 不得用于架构、阈值或种子选择。
- 通用晋级条件：两种子 MCC 均不低于基线，平均 MCC 至少提高 `+0.005`，平均 AUPRC 最多下降 `0.002`。CysConditional 另要求低 Cys 有毒肽敏感性每个种子至少提高 `+0.05`。

## 基线与头部替换

| 方法 | OOF MCC（20260905 / 20260906） | OOF AUPRC（20260905 / 20260906） | 结论 |
| --- | ---: | ---: | --- |
| TriStat-MLP（默认） | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 保留默认 |
| TriStat-KAN | 0.81491 / 0.80790 | 0.93208 / 0.92859 | 不晋级，保留消融 |

KAN 的 AUROC、AUPRC 与 Brier 有小幅改善，但两个种子的 MCC 都低于 MLP，平均 MCC 差为 `-0.00636`。这说明在当前固定特征上，KAN 没有带来稳定的主判别收益。详见 [MLP vs KAN 比较](comparison_mlp_vs_kan_v1/REPORT.md)。

## 诊断，不是候选模型

对默认 MLP 的全部训练 OOF 做了严格对齐的错误审计：缓存、OOF CSV、序列 SHA-256、标签和 fold 均须完全一致。

- 总错误：TP/TN/FP/FN = 1,558 / 4,353 / 216 / 260。
- FN 相比 TP：Cys 数量和 Cys 比例显著更低（SMD `-2.681` / `-2.378`，五折同方向）；FN 同时更短、疏水比例更高。
- FP 相比 TN：Cys 数量和 Cys 比例更高（SMD `+1.380` / `+1.534`，五折同方向）。
- Cys=0/1 的有毒肽 FN 率分别为 `0.739 / 0.727`；4+ Cys 的非毒肽 FP 率为 `0.256`。

该结论只用于提出可检验的模型假设，**不是**“按 Cys 数判毒”的规则。完整记录在 [MLP OOF error audit](mlp_oof_error_audit_v1/REPORT.md)。

## 尝试 1：谱特征 + 门控残差（SpectralRefine）

### 设计

将序列转成 6 条固定理化数值序列：疏水性、电荷、极性、侧链体积、柔性、芳香性。每条序列经标准化和 DCT-II，形成 66 维谱特征。候选用门控 logit 残差：

```text
Stage-1 MLP logit + sigmoid(gate) × spectral residual
```

残差输出层零初始化，起始预测严格等于 Stage-1 MLP。必要对照包括谱特征单独分类器和特征拼接。

### 结果

| 变体 | MCC（20260905 / 20260906） | 说明 |
| --- | ---: | --- |
| Spectral-only | 0.21138 / 0.19833 | 谱特征单独不足以判毒 |
| Concat | 0.81991 / 0.82324 | 有轻微波动，不是预设晋级候选 |
| Gated residual | 0.81511 / 0.82084 | 相对 MLP ΔMCC = -0.00066 / +0.00107 |

门控残差平均 MCC 仅 `+0.000205`，且第一个种子下降，未达到 `+0.005`。因此 `retain_as_ablation`。报告：[SpectralRefine](spectral_refine_comparison_v1/REPORT.md)。

## 尝试 2：Cys 区间条件残差（CysConditional）

### 设计

这是对错误审计的可学习回应，而不是硬规则：按已有 Cys count 仅作为条件输入分成 `0–1`、`2–3`、`4+`，每个区间由独立小型门控残差头修正 Stage-1 logit。另有一个参数量对照：不分区间的 shared residual。

### 结果

| 项目 | 20260905 | 20260906 | 平均 / 结论 |
| --- | ---: | ---: | --- |
| 条件残差 MCC | 0.81841 | 0.82013 | ΔMCC = +0.00150 |
| 条件残差 AUPRC | 0.92928 | 0.92821 | ΔAUPRC = -0.00006 |
| 低 Cys 有毒肽敏感性增量 | +0.00382 | +0.00382 | 要求 +0.05，未达到 |

尽管 MCC 没有下降，整体增量太小，且目标低 Cys 敏感性几乎未改善。因此 `retain_as_ablation`。报告：[CysConditional](cys_conditional_comparison_v1/REPORT.md)。

## 尝试 3：局部序列模式分支（LocalMotif）

### 审计依据

为了确认是否真的丢失了残基顺序信息，额外做了局部 motif OOF 审计。局部窗口特征先在每个 held-out fold 上用其余四折的无标签 ridge 回归，剔除原 9 个全局 PhysChem 特征可以解释的成分，之后才比较低 Cys FN 与 TP。

残差化后仍有稳定信号：C 端 5-residue 正电荷比例（SMD `+0.278`，5/5 同向）、7-residue 最大正电荷比例（SMD `-0.242`，4/5）、N 端 5-residue 疏水比例（SMD `+0.206`，4/5）。这支持尝试顺序敏感模型，但不构成人工 motif 判定规则。结果：[LocalMotif audit](mlp_local_motif_audit_v1/summary.json)。

### 设计

新建与原 TriStat 缓存 SHA 绑定的 residue cache：6,387 条序列、201,994 个残基、1024 维 float16 ProtT5 embedding，约 499 MB。原 TriStat 与 PhysChem 值逐条 bitwise 相同。

候选与对照：

```text
Frozen residue embeddings
  → 1024→96 projection
  ├─ control：masked global pooling → 非局部 MLP
  └─ candidate：3/5/7 1D CNN → masked attention pooling
  → 与 Stage-1 logit / PhysChem 融合
  → zero-initialized gated logit residual
```

两条局部分支参数量为 144,623 与 144,673，差异 50（0.035%），满足预设 0.2% 匹配条件。

### 结果

| 方法 | MCC（20260905 / 20260906） | AUPRC（20260905 / 20260906） | 结论 |
| --- | ---: | ---: | --- |
| 非局部参数匹配对照 | 0.82078 / 0.81722 | 0.93968 / 0.93677 | 种子不稳定 |
| CNN + attention 候选 | 0.82429 / 0.81191 | 0.94053 / 0.93811 | 不晋级 |

CNN 相对默认 MLP 的 MCC 增量为 `+0.00852 / -0.00786`，平均仅 `+0.00033`；AUPRC 虽两种子都上升，但 Brier 两种子都恶化（`+0.00499 / +0.00813`）。因此 `retain_as_ablation`，不读 Test1/Test2。报告：[LocalMotif](local_motif_comparison_v1/REPORT.md)。

## 后处理策略（不是架构改进）

默认 MLP 的 OOF 曾用于构建可选校准/阈值策略：screening、balanced、strict 分别优化 F2、MCC、F0.5。它不改变冻结编码器或 MLP 结构，且校准后的 Test1 Brier 略差于原始概率，所以校准保持可选。

- OOF 阈值：screening `0.199311`，balanced `0.486109`，strict `0.628709`。
- Test1 只在策略冻结后做历史审计：balanced MCC `0.8200`、ACC `0.9281`；strict precision `0.9385`；screening sensitivity `0.9125`。

策略文件：[deployment_policy_v1.json](mlp_fivefold_20epochs_v1/deployment_policy_v1.json)。这不是新的模型架构，也不能作为候选晋级的证据。

## 历史留出集审计：仅默认 MLP

默认 MLP 在所有独立候选开始前已冻结，随后做过一次只读评估：

| 数据集 | MCC | AUROC | AUPRC | Brier | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| Test1 | 0.82302 | 0.96471 | 0.93181 | 0.05989 | 与训练序列无重叠 |
| Test2 | 0.54288 | 0.92740 | 0.61935 | 0.05662 | 是 Test1 的带标签精确子集，不是独立验证 |

上述历史指标不可被用于选择 KAN、SpectralRefine、CysConditional 或 LocalMotif。所有未晋级候选均**没有**读取 Test1/Test2。

## 当前工程结论与分析提示

1. 默认 MLP 已是目前最稳健的独立轻量模型；不要将某一个较好的单种子 run 当作改进。
2. KAN、谱分支、Cys 条件修正、LocalMotif 都提供了有价值的失败证据：更复杂的头、频域特征、Cys 条件或局部卷积尚未稳定转化为 MCC 收益。
3. LocalMotif 的 OOF AUPRC 上升而 MCC/Brier 不稳定，说明它可能改变排序，但当前证据不足以证明更可靠、更可部署；不可据此替换模型。
4. 当前瓶颈更可能是标签机制异质性、训练数据覆盖或监督信号不足，而非继续叠加小模块即可解决。
5. 后续若继续，需先预注册新的数据/监督来源或一个与现有失败方向明显不同的假设；不要在当前数据上继续搜索阈值、种子、核宽度或 Cys 规则。

## 主要实现与结果入口

- 基线训练：[train_prottox_tsk.py](../../src/ghxtox/train_prottox_tsk.py)
- 谱特征：[spectral_refine.py](../../src/ghxtox/models/spectral_refine.py)
- Cys 条件残差：[cys_conditional.py](../../src/ghxtox/models/cys_conditional.py)
- LocalMotif 审计、缓存、训练和比较：[audit](../../src/ghxtox/prottox_local_motif_audit.py) · [cache](../../src/ghxtox/prottox_local_motif_cache.py) · [trainer](../../src/ghxtox/train_prottox_local_motif.py) · [comparison](../../src/ghxtox/prottox_local_motif_compare.py)
- LocalMotif 冻结配置：[prottox_local_motif_v1.json](../../configs/prottox_local_motif_v1.json)
- 回归测试：[test_prottox_tsk.py](../../tests/test_prottox_tsk.py)

最后验证状态：26 项相关单元/回归测试通过；所有结果在本机 RTX 4060 上执行；没有网站改动。

## 尝试 4：训练期结构质量辅助监督（ProtTox-StructGuide）

### 设计

这是一个独立轻量的内部两阶段模型，并不复用发表版 GHXTox 的在线 `1D + 3D` 融合：训练时把已有训练 OOF 的 `mean_plddt`、`min_plddt`、`low_plddt_fraction` 作为三个辅助回归目标；部署推理时仍只用 Frozen ProtT5 的 TriStat 和 9 个 PhysChem 特征。

```text
Sequence-derived frozen features
  → shared representation
  ├─ Stage 1: predicted structure-quality proxies
  └─ Stage 2: shared representation + predicted proxies → one toxicity probability
```

真实缓存包含 6,387 条训练记录，并严格通过 `historical_sample_id`、sequence、label、fold 的一对一审计。它不含 ESMFold 在线运行、坐标、3D 图、GHXTox 概率、teacher logit 或 Test1/Test2 数据。为了区分辅助监督的贡献，还运行了参数量完全一致但把 Huber 辅助损失设为零的对照。

### 结果

| 模型 | MCC（20260905 / 20260906） | AUPRC（20260905 / 20260906） | Brier（20260905 / 20260906） |
| --- | ---: | ---: | ---: |
| 保留 TriStat-MLP | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 0.05931 / 0.05893 |
| 同架构无辅助损失对照 | 0.80258 / 0.81236 | 0.92355 / 0.92526 | 0.07182 / 0.06841 |
| StructGuide 候选 | 0.80799 / 0.81259 | 0.92606 / 0.92736 | 0.06813 / 0.06817 |

候选相对保留 MLP 的 MCC 变化为 `-0.00779 / -0.00718`，平均 `-0.00748`；AUPRC 平均 `-0.00210`，Brier 平均 `+0.00903`。虽然它相对相同架构无辅助监督的对照有 `+0.00540 / +0.00023` MCC，但没有达到相对当前默认 MLP 的晋级标准。

结论：`retain_as_ablation`。冻结门槛要求两种子 MCC 均不降低、平均 MCC 至少 `+0.005`、平均 AUPRC 降低不超过 `0.002`、平均 Brier 增加不超过 `0.002`；本候选未满足。因此没有读取 Test1/Test2，也没有修改网站或替换默认模型。完整可复现报告见 [StructGuide report](structguide_comparison_v1/REPORT.md)。

实现入口：[模型](../../src/ghxtox/models/structure_quality_proxy.py)、[缓存](../../src/ghxtox/prottox_struct_guided.py)、[训练](../../src/ghxtox/train_prottox_struct_guided.py)、[比较](../../src/ghxtox/prottox_struct_guided_compare.py)、[配置](../../configs/prottox_struct_guided_v1.json)。

## 尝试 5：outer-safe 单结构 3D-expert 蒸馏（ProtTox-3D-Distill v1）

### 结论

`proxy_teacher_kd` 通过预注册训练 OOF 门槛，`mlp_teacher_kd` 也通过门槛。随后执行冻结的 head-to-head 统计比较（5000 次 paired bootstrap + 折级稳定性 + 三个预定义困难子集），proxy 相对 MLP+KD 无整体或困难子集上的可复现优势（MCC CI 含零、AUPRC 略差、Brier 更差且折级 Brier 2/10），因此**产品模型候选定为更简单的 `mlp_teacher_kd`**，`proxy_teacher_kd` 保留为消融/架构候选。未读取 Test1/Test2，默认 MLP 的部署替换需要后续完成 full-data refit 与运行器交接。

### 设计

```text
Sequence → Frozen ProtT5 TriStat + PhysChem
  ├─ 预测 mean/min pLDDT、low-pLDDT fraction（仅训练辅助）
  └─ outer-safe 单结构 3D teacher soft logit（T=2 蒸馏）
        ↓
     一个最终 toxicity logit
```

teacher 为五份严格外层 scratch GHXToxModel（`topology_strict_oof_v1_final.tar.gz`，fixed epoch 12、selection indices 为空）。每个 outer fold k 的目标表只含非 k 样本，覆盖 fitting / inner_validation / calibration 三类角色；外层折 k 不进入学生 k 训练。目标表 25,548 行，位于 `runs/prottox_3d_distill_v1/teacher_targets/`。

损失固定为 `L_tox + 0.10·L_proxy + 1.0·L_KD(T=2)`，其余超参与基线和 StructGuide 完全一致（20 epochs、lr 1e-3、batch 256、双种子、阈值 0.5）。

### OOF 结果

| 模型 | MCC（20260905/20260906） | AUPRC（20260905/20260906） | Brier（20260905/20260906） |
| --- | ---: | ---: | ---: |
| 保留 TriStat-MLP | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 0.05931 / 0.05893 |
| StructGuide v1（无 KD） | 0.80799 / 0.81259 | 0.92606 / 0.92736 | 0.06813 / 0.06817 |
| proxy_teacher_kd | 0.82173 / 0.82747 | 0.94028 / 0.94004 | 0.06113 / 0.06018 |
| mlp_teacher_kd | 0.82706 / 0.82392 | 0.94172 / 0.94173 | 0.05851 / 0.05943 |

proxy_teacher_kd 相对保留 MLP：MCC Δ = +0.00596 / +0.00770（均值 +0.00683），AUPRC Δ 均值 +0.01135，Brier Δ 均值 +0.00154，全部满足冻结门槛。相对 StructGuide v1，MCC Δ 均值 +0.01431、Brier Δ 均值 −0.00750，说明 3D teacher 软监督是主要收益来源。

### 完整记录

- 目标与实验目录：`runs/prottox_3d_distill_v1/`
- 报告：[REPORT.md](../prottox_3d_distill_v1/REPORT.md)
- 产品决定：[PRODUCT_MODEL_DECISION.md](../prottox_3d_distill_v1/PRODUCT_MODEL_DECISION.md)
- 机器判定：[decision.json](../prottox_3d_distill_v1/decision.json)、[promotion_report.json](../prottox_3d_distill_v1/promotion_report.json)、[head2head_analysis.json](../prottox_3d_distill_v1/head2head_analysis.json)
- 代码：[targets](../../src/ghxtox/prottox_3d_distill_targets.py) · [trainer](../../src/ghxtox/train_prottox_3d_distill.py) · [comparator](../../src/ghxtox/prottox_3d_distill_compare.py)
- 配置：[prottox_3d_distill_v1.json](../../configs/prottox_3d_distill_v1.json)
