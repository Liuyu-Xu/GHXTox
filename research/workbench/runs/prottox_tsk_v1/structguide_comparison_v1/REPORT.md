# ProtTox-StructGuide v1：训练期结构质量辅助监督

## 结论

`retain_as_ablation`。候选未通过训练 OOF 晋级门槛，因此没有读取 Test1/Test2，也没有替换独立轻量默认 MLP 或修改网站。

## 固定设计

```text
冻结 ProtT5 → TriStat + 9 PhysChem
                    ↓
             shared representation
              ├─ 预测 mean/min pLDDT、low-pLDDT fraction（仅训练监督）
              └─ shared + 预测代理 → 最终毒性 logit
```

部署输入仍只有序列派生的冻结特征；没有 ESMFold、坐标、3D 图、GHXTox 概率或 teacher logit。辅助目标来自已有训练 OOF 结构质量表，全部 6,387 条记录经 `historical_sample_id`、序列、标签和 group-fold 精确一致性校验后生成缓存。

## 固定 OOF 结果

| 模型 | MCC（20260905 / 20260906） | AUPRC（20260905 / 20260906） | Brier（20260905 / 20260906） |
| --- | ---: | ---: | ---: |
| 保留 TriStat-MLP | 0.815772 / 0.819769 | 0.929270 / 0.928343 | 0.059310 / 0.058925 |
| 同架构、无辅助损失对照 | 0.802584 / 0.812361 | 0.923550 / 0.925260 | 0.071824 / 0.068406 |
| StructGuide 候选 | 0.807986 / 0.812591 | 0.926059 / 0.927361 | 0.068126 / 0.068173 |

候选相对保留 MLP 的 MCC 为 `-0.007786 / -0.007178`，平均 `-0.007482`；AUPRC 平均变化 `-0.002097`，Brier 平均增加 `+0.009032`。它相对同架构无辅助损失对照的 MCC 为 `+0.005401 / +0.000230`，说明辅助监督带来有限信号，但不足以抵消该新两阶段头相对于原 MLP 的性能损失。

冻结门槛要求两个种子 MCC 均不低于 MLP、平均 MCC 至少 `+0.005`、平均 AUPRC 下降不超过 `0.002`、平均 Brier 增加不超过 `0.002`。本实验四项均未满足，故停止于训练 OOF。

## 可复现入口

- 目标缓存：[structure_quality_proxy_v1.pt](../structure_quality_proxy_v1.pt)
- 候选实现：[model](../../../src/ghxtox/models/structure_quality_proxy.py)、[cache](../../../src/ghxtox/prottox_struct_guided.py)、[trainer](../../../src/ghxtox/train_prottox_struct_guided.py)、[comparator](../../../src/ghxtox/prottox_struct_guided_compare.py)
- 固定配置：[prottox_struct_guided_v1.json](../../../configs/prottox_struct_guided_v1.json)
- 机器可读结论：[promotion_report.json](promotion_report.json)
