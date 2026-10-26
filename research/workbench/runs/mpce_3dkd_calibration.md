# MPCE-3DKD display calibration selection

范围：仅两 seed 平均训练 OOF；Test1/Test2 未读取；不改变模型与冻结阈值。

| policy | MCC | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| identity | 0.83888 | 0.94618 | 0.05573 | 0.04752 |
| temperature (T=2.6918) | 0.83888 | 0.94618 | 0.05056 | 0.00769 |

Brier gain +0.00517；ECE gain +0.03984。

**选定：temperature_scaling。**
