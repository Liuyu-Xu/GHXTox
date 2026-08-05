![当前多维融合模型架构](figure6_current_architecture.png)

图 1. 当前多维融合模型架构：组合A 1D 序列专家（LoRA 微调 ProtT5 + 双向 Mamba，3-seed 集成 + 训练 OOF Platt 校准）与 pLDDT 感知全原子 3D 专家（五级尺度）按训练 OOF 冻结的 `0.62/0.38` 权重融合，决策阈值 `0.4977`。默认模型的上一代架构（冻结 ProtT5-TextCNN 15 成员 + 交互适配器 + BACC 决策残差）作为「之前多维融合模型」基线，见第 3.2 节对比。
