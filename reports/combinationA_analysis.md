# 组合A（LoRA-ProtT5 + 双向Mamba）1D 创新：结果分析与下一步决策

## 1. 服务器全量结果（seed42，全量 6387×15ep）

### 新 1D 单独（OOF）
- AUROC 0.9642 / AUPRC 0.9323 —— 排序能力与 SCALAM(0.971~0.973)、3D(0.9773) 同一量级
- 但默认阈值下 SN 0.966 / SP 0.686 → BACC 0.826：**概率刻度偏正（过度自信），校准差**

### 融合重冻结（w=0.18，阈值 0.5555，OOF 选择）
| 指标 | 3D 单独 OOF | 新融合 OOF |
|---|---|---|
| BACC | 0.9275 | **0.9318** |
| MCC | 0.8693 | **0.8721** |
| AUROC | 0.9773 | **0.9787** |

OOF 层面：新 1D 给融合带来小幅正贡献。

### Held-out（对比旧默认 0.62/0.38 融合）
| 数据集 | BACC Δ | MCC Δ | AUROC Δ |
|---|---|---|---|
| Test1 | -0.005 | -0.017 | -0.004 |
| Test2 | -0.016 | **-0.054** | -0.006 |

全部小幅保守化，Test2 更明显。

## 2. 诊断

1. **单 seed vs 旧默认 15 成员集成**（3 seeds × 5 folds）——对比方法学上不公平，是最主要的不确定性来源。
2. **新 1D 概率未校准**：SN/SP 严重偏移（0.97/0.69），导致融合阈值 0.5555 在 Test 上迁移差。
3. **序列建模饱和**：SCALAM（3-seed）与组合A 均呈现同一规律——1D 架构替换在 held-out 上不超旧默认；
   组合A 的 Test2 AUROC 0.9515 甚至好于 SCALAM 3-seed 的 0.9369，说明组合A 排序不弱，问题在校准与集成度。

## 3. 决策：一轮收敛性改进后定稿，不再换架构

1. **3-seed 集成**（主要杠杆）：`SEEDS="42 123 2025"`，后台 ~6-12h。预期 Test1 持平或微升（SCALAM 先例：Test1 MCC +0.006），Test2 仍小幅保守但较 -0.054 收窄。
2. **OOF 概率校准**（辅助，已实现并本地验证）：Platt 拟合于训练 OOF、应用到 Test，只动概率刻度（AUROC 不变），改善 Brier/ECE 与阈值迁移；失败自动回退原始概率，不阻塞。
3. **报告口径（无论结果）**：组合A 满足导师"1D 架构不同"；OOF 融合微升；held-out 持平或微降作为"序列建模饱和"对照验证；项目核心指标增益定位在 3D 多尺度信息提取与跨维融合，不夸大 1D 增益。

## 4. 执行命令（服务器，后台）

```bash
cd /root/autodl-tmp/GHXTox
unzip -o lora_mamba_autodl_upload.zip   # 已含校准脚本 + 修复后的 Stage3
nohup env SEEDS="42 123 2025" bash scripts/run_lora_mamba_full.sh > runs/lora_mamba_full/full.log 2>&1 &
tail -f runs/lora_mamba_full/full.log
```
产物：`runs/lora_mamba_full/` 下 oof_seed{42,123,2025}.json、3-seed 集成 Test CSV、
`fusion_refreeze.json`（校准后重冻结）、`test_fusion_metrics.json`（最终对比）。
