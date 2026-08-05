# Combination-A 1D 模块：LoRA 微调 ProtT5 + 双向 Mamba 分类器（服务器全量运行）

## 1. 上传文件（解压到 /root/autodl-tmp/GHXTox 仓库根目录）

**自包含包** `lora_mamba_autodl_upload.zip`（约 0.7MB，含运行所需全部代码与数据，
不依赖仓库其它 src 文件；新开实例也直接可用）：

| 文件 | 作用 |
|---|---|
| `src/ghxtox/lora_mamba_1d.py` | 组合A模块（自包含，仅依赖 torch/transformers/sklearn/numpy） |
| `configs/lora_mamba_1d_autodl.json` | 服务器配置（ProtT5 从 HF 拉取，batch=16，15 epochs） |
| `scripts/run_lora_mamba_full.sh` | 一键：5折OOF → 全量训练 → Test1/Test2 预测 → 集成 → 融合重冻结 → Test 融合指标 |
| `scripts/refreeze_lora_mamba_fusion.py` | 训练 OOF 上重选 1D/3D 融合权重 w 与阈值（按 sample_id 对齐，向量化快速扫描） |
| `scripts/evaluate_lora_mamba_fusion.py` | 用重冻结 w/阈值计算 Test1/Test2 融合指标（对比旧默认） |
| `scripts/ensemble_lora_mamba_predictions.py` | 多种子概率平均 |
| `data/folds/train_cdhit080_fallback_folds.csv` | 5 折分组 fold 清单（6387 行） |
| `data/processed/test1_lora_mamba_light.pt` 等 | Test1/Test2 轻量 adapter（仅 id/序列/标签） |
| `runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv` | 3D 训练 OOF（融合重冻结用） |
| `runs/default_3d_hybrid/test1_predictions.csv` 等 | 3D Test 概率 + 旧默认指标（对比用） |

## 2. 服务器命令（首次全量）

```bash
cd /root/autodl-tmp/GHXTox
unzip -o lora_mamba_autodl_upload.zip
python -c "import sentencepiece, transformers, sklearn" 2>/dev/null || pip install sentencepiece
export HF_ENDPOINT=https://hf-mirror.com
SEEDS=42 bash scripts/run_lora_mamba_full.sh
# 3 seeds（可选）：
# nohup env SEEDS="42 123 2025" bash scripts/run_lora_mamba_full.sh > runs/lora_mamba_full/full.log 2>&1 &
```

## 3. 快速恢复 Stage 3（OOF/全量训练/Test 预测已完成，只需补融合步骤）

若 Stage 1/2 已跑完（`runs/lora_mamba_full/` 已有 `oof_seed42.json`、
`test1/test2_lora_mamba_ensemble.csv`），**不要重跑整个脚本**，只执行：

```bash
cd /root/autodl-tmp/GHXTox
unzip -o lora_mamba_autodl_upload.zip      # 更新修复后的脚本（训练产物保留）
python -B scripts/refreeze_lora_mamba_fusion.py \
  --new-1d-oof runs/lora_mamba_full/oof_seed42.json \
  --fold-manifest data/folds/train_cdhit080_fallback_folds.csv \
  --three-d-oof runs/3d_v2_conformer_prott5_frozen_oof/predictions.csv \
  --output runs/lora_mamba_full/fusion_refreeze.json
python -B scripts/evaluate_lora_mamba_fusion.py \
  --refreeze runs/lora_mamba_full/fusion_refreeze.json \
  --test1-1d runs/lora_mamba_full/test1_lora_mamba_ensemble.csv \
  --test2-1d runs/lora_mamba_full/test2_lora_mamba_ensemble.csv \
  --hybrid-test1 runs/default_3d_hybrid/test1_predictions.csv \
  --hybrid-test2 runs/default_3d_hybrid/test2_predictions.csv \
  --old-test1-metrics runs/default_3d_hybrid/test1_metrics.json \
  --old-test2-metrics runs/default_3d_hybrid/test2_metrics.json \
  --output runs/lora_mamba_full/test_fusion_metrics.json
```

## 4. 产物（runs/lora_mamba_full/）

- `oof_seed{seed}.json`：5折 OOF 指标（BACC/MCC/AUROC/AUPRC 等）+ 阈值
- `full_seed{seed}.pt`：全量训练 checkpoint（仅 LoRA+头参数，约 50MB）
- `test1/test2_lora_mamba_seed{seed}.csv`、`test1/test2_lora_mamba_ensemble.csv`：测试概率
- `fusion_refreeze.json`：重选的融合权重 w 与阈值 + 新融合 OOF 指标
- `test_fusion_metrics.json`：Test1/Test2 新融合指标 vs 旧默认（含 delta）

## 5. 设计要点（供报告使用）

- 替换默认 1D 的「蛋白质大模型」与「分类网络」两个模块：
  1. 冻结 ProtT5 均值池化特征 → **ProtT5 编码器 + LoRA 任务自适应微调**
     （仅训练低秩 A/B 与分类头，参数量约 1310 万；LoRA r=8, alpha=16）
  2. 池化向量上的多尺度 TextCNN → **双向 Mamba（S6 状态空间）序列分类器**
     （纯 PyTorch 实现；前向/反向选择扫描，残差+层归一化，掩码均值池化）
- 数据增强（BSMOTE）与多种子集成（多人投票）保留在流水线外层。
