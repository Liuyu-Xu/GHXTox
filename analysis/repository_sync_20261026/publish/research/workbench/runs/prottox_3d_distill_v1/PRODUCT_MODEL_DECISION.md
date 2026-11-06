# 产品模型决定：mlp_teacher_kd

日期：2026-09-05。本文件由冻结的 head-to-head 分析结果决定，只使用训练 OOF；Test1/Test2 未读取。

## 冻结比较口径

- 5000 次 label-stratified paired bootstrap，固定种子 20260909，同一重采样索引同时用于两模型
- 比较 proxy_teacher_kd − mlp_teacher_kd 的 ΔMCC、ΔAUPRC、ΔBrier
- 折级稳定性：5 folds × 2 seeds = 10 组
- 预定义困难子集（各 1,597 条，占 25%）：
  - 低 predicted mean-pLDDT（StructGuide v1 OOF proxy 预测 ≤ 25% 分位 0.6380）
  - 高 predicted low-pLDDT fraction（≥ 75% 分位 0.2250）
  - teacher 与保留 MLP 概率分歧 ≥ 75% 分位（0.0798）
- 决策规则：proxy+KD 需整体或至少两个困难子集出现 bootstrap 下界 >0 的 MCC（或 AUPRC 且 MCC ≥ −0.002），且 Brier 上界 ≤ +0.002

## 结果

整体（proxy − MLP+KD）：

| 指标 | 两 seed 点估计 | bootstrap 均值 | 95% CI |
| --- | --- | --- | --- |
| ΔMCC | −0.00532 / +0.00354 | −0.00092 | [−0.00809, +0.00607] |
| ΔAUPRC | −0.00150 / −0.00178 | −0.00167 | [−0.00431, +0.00089] |
| ΔBrier | +0.00262 / +0.00075 | +0.00170 | [+0.00000, +0.00339] |

折级：proxy 赢 MCC 6/10、AUPRC 3/10、Brier 2/10，整体没有稳定性优势。

困难子集：三个子集均无 bootstrap 下界 >0 的 MCC/AUPRC 优势；teacher 分歧子集上 proxy 反而更差（ΔMCC 均值 −0.0057，Brier 均值 +0.0059）。

## 决定

`mlp_teacher_kd` 为产品模型候选：Frozen ProtT5 → TriStat + PhysChem → MLP → outer-safe 单结构 3D-expert KD → 最终毒性概率。它更简单、整体指标更好、且是独立轻量线第一个通过 OOF 门槛的模型。

`proxy_teacher_kd` 保留为消融/架构候选：它相对保留 MLP 通过晋级门槛，但相对 MLP+KD 无可复现的额外技术效果，因此不因专利叙事强行保留 proxy 模块。

## 后续（未执行）

1. 冻结 `mlp_teacher_kd` full-data refit 协议（含 outer-safe teacher 全覆盖目标）
2. 生成部署 checkpoint 并交接本地 API/runner
3. 冻结后才可做 Test1/Test2 只读历史审计
4. 网站接入（另行单独处理，不在此仓库完成）

完整机器结果：[head2head_analysis.json](head2head_analysis.json)
