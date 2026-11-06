# LoRA是否实际生效：当前可确认的证据

2026-10-27。不是通过性能猜测，而是检查保存的参数并复现梯度。

|已检查的本地最终权重|LoRA B矩阵数|非零B矩阵数|
|---|---:|---:|
|full_seed42.pt|144|0|
|full_seed123.pt|144|0|
|full_seed2025.pt|144|0|

本实现的附加输出为 `scaling * (dropout(hidden) @ lora_A.T) @ lora_B.T`。只要B为零，附加输出就为零。因此这三份实际保存的一维权重没有LoRA适配增量。它们的BiMamba、投影与分类头仍有训练参数，不能说整个模型没训练或所有旧指标失效。

代码中先冻结ProtT5 encoder（包括共享embedding）再加入可训练LoRA。当开启原默认reentrant activation checkpoint时，输入embedding没有梯度，检查点会断开这部分图。在相同路径的小型T5复现中，原模式12个LoRA B全部没有梯度，分类头有梯度；non-reentrant修复后12个都有非零梯度。

root `src/ghxtox/lora_mamba_1d.py`现已修复：新Transformers使用non-reentrant检查点；旧支持版本退回input-gradient hook，embedding权重本身仍冻结。两种API的回归测试均通过，验证LoRA更新、冻结参数无梯度，以及启用/停用检查点时eval输出完全一致。

真实供应ProtT5大模型的A/B/C加权小步验证也通过：修复后三组每组2个optimizer step，144个B矩阵均有非零梯度且实际变为非零，使用同一初始可训练参数和同三条训练角色记录。详见 `../group_weight_trial_20261027/sequence_training_check.json`；这只是梯度验证，不是性能结果。

### 结论边界

- 可以确认当前检查的3份最终权重LoRA附加输出为零。
- 可以确认当前复现环境中原训练设置会断开LoRA梯度，与保存权重现象一致；不据此声称已还原历史服务器每个操作。
- 没有全部历史fold checkpoint，不能把未检查的历史OOF适配器也写成逐份核实。
- 修复代码不会让旧B矩阵自动学到信息，需要重新训练。旧检查点/原预测均未覆盖。
- 公平三组必须用相同初始化和训练协议重跑：A明确复现旧机制，B只修复，C在B上加困难群体权重。不能直接拿历史A分数对比新B/C。
- 本轮首折/单种子不是已发表3种子集成，也不是新的独立测试。后续是否提升由三组实际结果决定。
