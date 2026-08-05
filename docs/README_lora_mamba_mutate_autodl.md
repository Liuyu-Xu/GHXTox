# 受控序列点突变增强实验（组合A + mutation augmentation）

在已定稿的组合A（LoRA 微调 ProtT5 + 双向 Mamba）基础上，验证**序列空间点突变增强**是否
能进一步优化正样本召回（尤其 Test2 与短肽层）。

## 协议（防泄漏）
- 仅对**正样本**做增强；每条正样本以 rate=0.2 概率随机 1~2 个点突变。
- **严格折内生成**：只在每个训练折的训练子集内突变，验证集样本不参与生成。
- **相似度过滤**：突变序列与**本折验证集**序列相似度 ≥ 0.8（同长用汉明、异长用编辑距离）
  即丢弃，保证 OOF 预测不被增强样本污染。
- **全量训练**（用于 Test 预测）不做任何过滤，也**不读取 Test1/Test2 序列**（遵循"测试集仅用于最终评估"）。
- 父序列自身不参与过滤（1~2 个点突变本就是其近拷贝，属预期）。
- 每折随机种子 = seed*1000+fold，完全可复现。

## 上传/运行

```bash
cd /root/autodl-tmp/GHXTox
unzip -o lora_mamba_mutate_upload.zip
export HF_ENDPOINT=https://hf-mirror.com
SEEDS=42 bash scripts/run_lora_mamba_mutate.sh          # 先 1 seed（约 2~4h）
# nohup env SEEDS="42 123 2025" bash scripts/run_lora_mamba_mutate.sh > runs/lora_mamba_mutate/full.log 2>&1 &
```

## 产物（runs/lora_mamba_mutate/）
- oof_seed*.json / full_seed*.pt / test ensemble / calibrated CSVs
- fusion_refreeze.json / test_fusion_metrics.json
- **length_stratified.json / paired_bootstrap.json / calibration_report/**（自动生成，可直接与
  runs/lora_mamba_full/（无增强基线）对比）

## 判定标准（预注册）
- 采用：Test2 SN/BACC 或短肽（11-15）层 BACC 提升，且 Test1 AUROC 不下降；
- 否则保留现状（无增强组合A 已满足要求，不强行采用）。
