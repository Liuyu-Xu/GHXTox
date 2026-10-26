# Soft Cys topology：代码交付说明

实现位置：`D:/dachuang2026/GHXTox`。本次完成图片 A–C，以及 D–E 的训练和评估入口；未运行真实五折训练，未进入 F 重训或 G 平台/专利交底阶段。

## 已实现

- `src/ghxtox/soft_cys_topology.py`：复用化学位点中的 Cys SG，排除 Met；计算 SG/CA 距离、序列间隔、置信度、对称方向特征、局部化学类型环境。默认距离上限 8 Å，边界容差 1e-5 Å，均为候选定义，不是真实二硫键标注。
- `scripts/cache_soft_cys_topology.py`：生成新的缓存文件。缓存签名覆盖几何、氨基酸身份、掩码、pLDDT、特征版本和距离上限；不使用标签。仅 OOF 自定义 collate 消费该缓存，常规预测会按相同定义在线计算。
- `src/ghxtox/models/soft_cys_topology.py`：对称软配对、每个 Cys 的总容量约束和显式未配对状态；有界 beam 生成互不冲突的匹配，保留全未配对候选；逐候选非线性消息传递后加权聚合。返回期望配对 Cys 数、候选熵、未配对质量及候选证据。
- `src/ghxtox/models/dual_modal.py`：在化学残差之后、结构融合之前加拓扑残差；默认关闭。残差投影零初始化，支持 fusion / spatial_only，原融合层不变。
- `src/ghxtox/topology_oof.py`：固定训练轮数、冻结所有旧参数并保持旧模块 eval，只训练拓扑分支。五折检查身份、重复样本和组泄漏，要求检查点训练/选择来源覆盖全部监督祖先且排除当前留出折。
- `scripts/evaluate_topology_aware_oof.py`：完整 OOF 对齐、baseline/local/matching/topology/null 对照，MCC、BACC、AUROC、AUPRC、Brier、精确 Cys 数分层和标签标准化误差、按折分组的配对 bootstrap。拓扑稳定性采用 pLDDT 指导的坐标扰动，记录概率和配对矩阵变化。

软匹配使用对称容量投影，保证节点容量；它是连续松弛，不保证奇数集合约束。完整 topology 模式的输出是合法离散匹配的凸组合，补上这一限制。候选搜索是有界近似，不声称穷举或精确全局 top-K；候选选择离散，选中候选的评分、权重和消息可求导。local 对照有意不满足全局容量约束。

## 验证

使用已有 Python 3.10 / Torch 2.6.0 运行 11 项 unittest，包含几何不变性、缺失坐标、Met 排除、缓存失效、容量约束、互斥候选、梯度、无 Cys/单 Cys/补齐位零残差、旧检查点兼容、冻结训练、泄漏拒绝，以及合成数据五折训练→预测→稳定性→评估全流程。合成测试不证明真实预测性能。

零残差和旧参数不变按精确值验证；冻结参数可能改变底层计算路径，预测回退按 float32 容差验证（atol=1e-7、rtol=1e-6），评估报告同时输出 null 对照最大概率差。未运行整仓测试、GPU 测试或真实数据性能实验。

```powershell
$env:PYTHONPATH = 'D:\dachuang2026\GHXTox\src'
$pythonExe = 'D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe'
& $pythonExe -m unittest discover -s tests -p test_soft_cys_topology.py -v
```

## 真实实验前需要补齐的材料

`configs/topology_aware_structural_v1.json` 是模板，未冒充冻结协议。填写训练缓存、五折 CSV、文件 SHA-256、每折检查点与 provenance 后，将 status 设置为 `frozen_before_candidate_oof`。在查看候选 OOF 前确定轮数、距离、候选数、种子及晋级规则；当前 evaluator 只输出比较结果，不自动晋级。

CSV 必需列：`source_index,sample_id,sequence,label,fold,group`，fold 为 0–4。若原清单采用其他组列名，需显式保留来源后转换，不能自行重建不同划分。

每个 checkpoints 条目：`{"fold":0,"path":"...pt","provenance":"...json"}`。provenance 必需字段：

```json
{
  "checkpoint_sha256": "实际 SHA-256",
  "training_data_sha256": "实际 SHA-256",
  "fold_manifest_sha256": "实际 SHA-256",
  "external_test_labels_used": false,
  "all_supervised_ancestors_included": true,
  "fitting_indices": [1, 2],
  "selection_indices": []
}
```

上述索引只是格式示例，必须来自真实训练日志；selection_indices 包括早停、epoch/结构/参数选择用到的样本。若旧模型用当前外层留出折做过选择，不能靠补空列表通过，应取得相应训练折内部选择的检查点。程序检查申报内容和哈希，无法独立证明来源声明真实。

```powershell
& $pythonExe -m ghxtox.topology_oof --protocol configs/topology_aware_structural_v1.json --output-dir runs/soft_cys_v1 --preflight-only
& $pythonExe -m ghxtox.topology_oof --protocol configs/topology_aware_structural_v1.json --output-dir runs/soft_cys_v1 --device cuda
& $pythonExe scripts/evaluate_topology_aware_oof.py --oof runs/soft_cys_v1/oof_predictions.csv --protocol runs/soft_cys_v1/protocol.json --output runs/soft_cys_v1/evaluation.json
```

输出含每折每对照模型、完整 OOF CSV、候选证据 JSONL、协议及运行审计。JSONL 中 residue_indices 使用零基原序列索引，候选内配对使用紧凑 Cys 索引。记录总运行时间、各折对照时间、进程生命周期峰值 RSS 和 CUDA 分配峰值；RSS 包括整个 Python 进程，不是单分支额外开销。当前扰动同步移动每个残基的 CA/化学位点/骨架坐标，保留局部方向，不模拟侧链构象重建。

本机交接文档引用的 `data/folds/train_cdhit080_fallback_folds.csv` 在预期位置缺失。没有据此生成替代真实折划分，也没有读取 Test1/Test2 标签或开展真实训练。

## 已冻结的 AutoDL 五折协议

完整协议在 `configs/topology_strict_oof_v1.json`。它固定训练缓存、外层与嵌套折清单的 SHA-256；每折从零训练 12 个 epoch 的基础模型；随后冻结主干，对 local、matching、topology 与 null 各训练 8 个 epoch。拓扑分支只使用基础 provenance 中的 nested `train` 索引；内部 validation 只作基础模型诊断，calibration 角色不参与基础或拓扑拟合。

AutoDL 入口是 `scripts/run_topology_strict_oof_autodl.sh`。先将工作目录、训练缓存、外层折 CSV、以及已冻结的 `data/folds/topology_strict_nested_v1.csv` 和同名 JSON 一并传到服务器。不要在服务器重新生成嵌套清单；启动脚本会比对其哈希。

```bash
cd /root/autodl-tmp/GHXTox
export GHXTOX_PYTHON=python
export GHXTOX_TRAIN_CACHE=/root/autodl-tmp/data/train_chemical_sites_final_esm2_multisource_1_30_full2795.pt
export GHXTOX_OUTER_MANIFEST=/root/autodl-tmp/data/train_cdhit080_fallback_folds.csv
export GHXTOX_NESTED_MANIFEST=$PWD/data/folds/topology_strict_nested_v1.csv
bash scripts/run_topology_strict_oof_autodl.sh prepare
bash scripts/run_topology_strict_oof_autodl.sh all-base
bash scripts/run_topology_strict_oof_autodl.sh topology
```

如服务器中断，可逐折运行 `bash scripts/run_topology_strict_oof_autodl.sh base 0` 至 `base 4`。已完成且带 provenance 的折会被跳过；只有部分输出的目录会被拒绝覆盖，需先人工检查。`topology` 仅在五个基础 checkpoint 和 provenance 都通过哈希、训练/选择索引检查后运行。
