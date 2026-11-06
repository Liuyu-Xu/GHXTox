# V5 fold4 Root-Cause Diagnostic

日期：2026-09-05。只读诊断，不改 teacher、不改 v1、不读 Test1/Test2。

## 1. 工程/teacher 质量审计

- 五折 provenance/config 完全同口径：base_config SHA-256 相同，checkpoint SHA 与 manifest 一致。
- fold4 前向与已有 teacher target 生成一致：logit 最大差异 9.7e-6（两次独立推理的浮点级差异，非口径错误）。
- fold4 训练历史无崩溃/无 NaN；epoch 曲线正常。
- 但 fold4 teacher 本身是五折中偏弱的：
  - inner-validation 峰值 MCC：fold4 0.809（epoch 10），fold0–3 约 0.81–0.84；
  - 固定 epoch 12 的 val MCC：fold4 0.766，fold0–3 约 0.777–0.800。

## 2. 数据分布

| fold | n | positive % | 平均长度 | Cys≥2 | 平均 charge | 平均 pLDDT | low-pLDDT | 组平均大小 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1271 | 28.7 | 31.7 | 527 | 2.01 | 0.722 | 0.163 | 1.21 |
| 1 | 1229 | 29.3 | 31.4 | 500 | 1.68 | 0.723 | 0.172 | 1.17 |
| 2 | 1273 | 28.0 | 32.1 | 500 | 1.85 | 0.721 | 0.171 | 1.21 |
| 3 | 1291 | 28.0 | 31.1 | 513 | 2.04 | 0.715 | 0.174 | 1.23 |
| 4 | 1323 | 28.4 | 31.8 | 548 | 1.60 | 0.728 | 0.158 | 1.26 |

fold4 在类别比例、长度、Cys、电荷、pLDDT 与聚类结构上与其他折没有显著异常。

## 3. fold4 排序弱点定位

- fold4 teacher AUROC 0.958，低于同折 base（0.972），是唯一 teacher 低于 base 的折；
- teacher 在 fold4 改善了 FN（63→51），但 FP 明显增加（43→69）；fold0 同样 FP 增加（26→55），但 teacher 整体排序仍强；
- fold4 对 v1 排反 pair 的纠正率 45%，而 fold0–3 为 84–94%；
- fold4 teacher 对涉错 toxic 样本的平均概率低于 fold0（约 0.817 vs 0.866），说明该折 teacher 对困难 toxic 的置信不足。

## 判定

- 未发现明确工程错误（scaler/checkpoint/输入/hook/前向均一致）；
- 未发现 fold4 数据特殊异质性；
- teacher_4 训练未“明显失败”（无崩溃、曲线正常），只是五折中最弱；

按冻结决策规则：工程正常且差异不是由可修复技术错误造成 → 判断为 **teacher ranking capability 的跨折真实异质性**，**正式终止 V5**。不开 teacher-robustness 新协议。
