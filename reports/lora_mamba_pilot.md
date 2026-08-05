# 组合A 1D 模块：LoRA 微调 ProtT5 + 双向 Mamba — 本地试点结果

## 目标
替换默认 1D 四个模块中的两个以实现架构创新：
1. 蛋白质大模型：冻结 ProtT5 均值池化特征 → **ProtT5 编码器 + LoRA 任务自适应微调**
2. 分类网络：池化向量上的多尺度 TextCNN → **双向 Mamba（S6 状态空间）序列分类器**

数据增强：当前组合A 实现**未启用** BSMOTE（以 pos_weight 处理正负不平衡）；序列空间点突变增强作为受控实验另行验证（见 docs/README_lora_mamba_mutate_autodl.md）。多种子集成（多人投票）保留为 3-seed 集成。

## 实现（src/ghxtox/lora_mamba_1d.py）
- LoRA：自实现（无 peft 依赖），包裹 ProtT5 编码器全部线性层，冻结基座（含嵌入），
  仅训练低秩 A/B + 分类头。可训练参数 ≈ 1310 万。
- Mamba：纯 PyTorch S6 选择扫描（序列长度 ≤ 50，逐时间步循环即可），双向（前向+反向扫描），
  负特征值（A = -exp(log)）保证状态衰减稳定；fp32 扫描避免 fp16 溢出；掩码均值池化。
- 命令：cv（5折 OOF）/ full / predict，与 ghxtox.sequence_chemistry 接口对齐。

## 本地试点结果（组合A：LoRA-ProtT5 + BiMamba）
设置：600 条分层抽样（正样本 28.5%，与全量一致）× 5折，每折 3 epochs，1 seed（42），batch=4，fp16。
设备：RTX 5060 Laptop 8GB（WDDM，桌面占用约 3.6GB）。

| 指标 | 试点 OOF |
|---|---|
| Balanced Accuracy | 0.8843 |
| MCC | 0.7771 |
| Accuracy | 0.9100 |
| Sensitivity | 0.8246 |
| Specificity | 0.9441 |
| AUROC | 0.9267 |
| AUPRC | 0.8716 |

说明：试点为 600 条 × 3 epochs，仅验证架构端到端可学习，**不可**与全量 6387 × 30 epochs 的
基线（TextCNN BACC 0.9317 / MCC 0.8586）直接比较。损失 0.705 → 0.250 单调下降，预测概率分布合理。

## 速度测量（本地 RTX 5060）
- batch=4：约 102 ms/样本（fp16，含 ProtT5 前向+反向）
- batch=8：约 50 ms/样本，峰值显存 5.7GB

## 服务器/本地决策
- **本地试点**：已在本轮完成（约 17 分钟/600 条 × 3 epochs）。
- **全量 5折 OOF（6387 条 × 15 epochs）**：本地约 10+ 小时/seed；AutoDL RTX 4090
  预计 2~4 小时/seed（3 seeds 约 8~12 小时）→ 需要服务器。已打包 `lora_mamba_autodl_upload.zip`
  与一键脚本，见 docs/README_lora_mamba_autodl.md。


> 全量 3-seed + OOF 校准最终结果与配对自助检验见 `reports/combinationA_final_results.md` 与 `reports/mentor_report/REPORT.md` 第 3.1/5.2 节。
