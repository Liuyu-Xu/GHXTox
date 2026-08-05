# SCALAM 3-种子集成 + 适配器/BACC 补齐 — AutoDL 服务器包

目标：将 SCALAM（序列化学感知 1D）替换默认 1D，做 **3 种子集成全量训练 → Test1/Test2 预测 → 0.60/0.40 融合评估**，并可选**重训跨模态适配器与 BACC 决策残差**。

## 包结构
- `scripts/`：主流程 + 辅助脚本
- `configs/sequence_scalam.json`
- `data/`（内置小文件）：
  - `scalam3d_fusion_oof.csv`（6387 行，新融合 OOF 基线，适配器 precompute 用）
  - `teacher_oof.csv` / `base_oof.csv`（BACC 重筛输入表）
- `src/ghxtox/`：全套源码（含 sequence_chemistry / cross_modal_adapter / bacc_recall_residual）

## 第一阶段（必跑）：3 种子集成全量 + Test 融合评估
```bash
cd /root/autodl-tmp/GHXTox          # 或解压到独立目录后 cd 进去
bash autodl_scalam_refine/scripts/run_scalam_3seed_full.sh
```
产出：`runs/sequence_scalam_full/fusion_test_metrics_3seed.json`（3 种子集成后的 Test1/Test2 融合指标 vs 旧默认）。

## 第二阶段（可选）：适配器/BACC 重训
前置依赖（服务器需已有，或从本地上传）：
- `data/processed/train_chemical_sites_final_esm2.pt`
- `data/processed/prott5_tokens/train`（逐残基 ProtT5 token 缓存目录）
- `data/folds/train_cdhit080_fallback_folds.csv`
- `runs/chemical_site_ablation_include_same_residue_fold{0..4}/best_model.pt`
- 各数据集 processed + `*_prott5_structure_adapter.pt`

```bash
# 1) 生成 Test1/Test2 新融合基线（0.60*SCALAM + 0.40*3D）
python -B autodl_scalam_refine/scripts/make_fusion_test_csv.py \
  --scalam runs/sequence_scalam_full/test1_scalam_ensemble.csv \
  --hybrid runs/default_3d_hybrid/test1_predictions.csv \
  --output runs/sequence_scalam_full/test1_fusion_test.csv
python -B autodl_scalam_refine/scripts/make_fusion_test_csv.py \
  --scalam runs/sequence_scalam_full/test2_scalam_ensemble.csv \
  --hybrid runs/default_3d_hybrid/test2_predictions.csv \
  --output runs/sequence_scalam_full/test2_fusion_test.csv

# 2) 重训适配器 + BACC
bash autodl_scalam_refine/scripts/run_adapter_bacc_refit.sh
```
产出：`runs/scalam_adapter_refit/`（适配器 checkpoint + OOF）与 `runs/scalam_adapter_refit/bacc/decision.json`。

## 协议红线
- 不读 Test1/Test2 标签做参数选择（融合权重 0.60/0.40 与阈值 0.6013 已由训练 OOF 冻结）；
- 适配器/BACC 重训的基线全部来自新融合 OOF，不回流旧概率；
- 结果如实对比旧默认，不混用单项最优指标。