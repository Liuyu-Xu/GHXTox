# MPCE-3DKD lightweight metrics

设备：本地 RTX 4060 Laptop GPU；只做训练 OOF 后的产品审计，未读 Test1/Test2。

## 规模

| 项目 | 数值 |
| --- | ---: |
| ProtT5 encoder 参数 | 1,208,141,824 |
| MPCE head（单 seed） | 322,274 |
| MPCE head（双 seed ensemble） | 644,548 |
| 输入特征 | 3,072 TriStat + 9 PhysChem |

## 推理

为保证任意新序列与 canonical batch-size-1 训练缓存完全一致，ProtT5 按单序列独立前向。

| 指标 | 数值 |
| --- | ---: |
| 100 条 warm 后耗时（第二次） | 2.05s |
| 单条端到端 | 20.5370 ms |
| 吞吐 | 48.6925 seq/s |
| GPU 峰值显存 | 3090 MiB（含 encoder 与双 seed） |

## 部署结论

- 不运行 ESMFold；不运行 GHXTox 3D/EGNN/chemical-site；不需要 pLDDT、AutoDL 或远程 GPU。
- 最终推理链：sequence → Frozen ProtT5 → TriStat+PhysChem → MPCE dual-seed ensemble → probability。
- 展示概率可按已冻结 temperature T=2.6918 校准，模型输出本身不变。
