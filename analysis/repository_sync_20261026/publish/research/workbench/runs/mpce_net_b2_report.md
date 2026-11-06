# MPCE-B2：Global Risk + Multi-Prototype Dual-Branch Fusion

日期：2026-09-05。v1 未修改；训练 OOF 双种子；未读 Test1/Test2。

## 结果

| 模型 | MCC（20260905/20260906） | AUPRC 均值 | Brier 均值 |
| --- | ---: | ---: | ---: |
| A1 v1（保底） | 0.82631 | 0.94294 | 0.05764 |
| MPCE-B（已有） | 0.83224 / 0.83350 | 0.93914 | 0.06198 |
| B2-global_only | 0.82879 / 0.82144 | 0.93612 | 0.06316 |
| B2-fixed | 0.83388 / 0.82758 | 0.93979 | 0.06293 |
| B2-gated | 0.83202 / 0.83765 | 0.93639 | 0.06196 |

## 判定

B2-gated 相对 A1：

- MCC Δ +0.00853（通过）
- AUPRC Δ −0.00655（未通过）
- Brier Δ +0.00433（未通过）

相对 MPCE-B：seed1 MCC 略低，AUPRC 更差，未证明优于原 B。
相对 fixed fusion：seed1 MCC 略低，未通过“均不低于”控制。

结论：`retain_as_ablation`。双分支融合没有同时修复 AUPRC/Brier；prototype 提升 MCC 但排序损失仍未解决。按协议不再做 parameter-efficient ensemble C，MPCE 主路线暂停。
