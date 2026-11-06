# 本目录试验已由新请求替代，勿混用A/B/C

2026-10-27。本目录旧A/B/C指样本权重1 / 低Cys正类1.5 / 再给MCD-NDBP总权重2.0。

完成了真实3D首折三组6epoch，留出1271条；未完成计划中的五折。数学检查发现固定旧1D、0.72/0.28和阈值时，11个MCD/NDBP原正类中8个即使3D概率为1仍无法判阳。因此在首折三组结束后停止其余扩跑，不能用该3D-only配置否定家族加权。

随后查明三个本地最终1D权重的LoRA B均为零，原梯度检查点设置可复现梯度断开。修复及真实ProtT5的三组梯度小步验证通过。当前用户要求先隔离修复效应，转至 ../lora_repair_control_20261027/。

新目录A/B/C是**原梯度机制 / 修复LoRA / 修复+低Cys正类1.5**，暂缓MCD/NDBP特定额外加权。两套命名不可直接比较。

本目录保留原protocol.json、范围修订protocol_amendment.json、3D首折结构诊断structural_fold0_preliminary.json、11条融合可达性与sequence_training_check.json。预检查verify_preflight.py已通过；原verify.py/summarize.py需要全五折，当前不应当运行后宣称全量完成。所有原模型权重、标签和Test输出未覆盖。
