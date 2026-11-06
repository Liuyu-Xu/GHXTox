# ProtTox-3DKD-V2C：component intervention diagnostic（消融/负结论）

日期：2026-09-05。口径 A，只做诊断，未训练候选，未改 v1。

## 干预定义

- `z_full`：teacher_k 正常前向
- `z_no_chem`：关闭 chemical-site 分支（同一 checkpoint，strict 加载）
- `z_no_geo`：fusion 前将 spatial_h 置零（整个结构通道移除；因 chemical 残差嵌套在 spatial 通道内，因此这不是纯 geometry 干预）
- `z_no_chem_no_geo`：上述两者同时关闭（纯序列通路）
- Δchem = full − no_chem；Δstructure_total = full − no_geo；Δgeo_conditional = no_chem − no_chem_no_geo

术语统一为 teacher component intervention effect，不称为因果/生物学贡献。

## 1. 效应存在性

| 效应 | abs mean | abs median | abs p90 | abs max | nonzero |
| --- | ---: | ---: | ---: | ---: | ---: |
| Δchem | 0.168 | 0.032 | 0.396 | 4.46 | 6366/6387 |
| Δstructure_total | 0.835 | 0.117 | 3.18 | 14.45 | 6385/6387 |
| Δgeo_conditional | 0.820 | 0.149 | 2.77 | 13.30 | 6386/6387 |

结构通道干预效应明显大于化学位点干预效应；效应几乎对所有样本都非零。

## 2. 与 v1 错误的关系

| 效应 | v1 error abs mean | v1 correct abs mean | teacher-v1 disagreement Spearman |
| --- | ---: | ---: | ---: |
| Δchem | 0.175 | 0.167 | +0.199 |
| Δstructure_total | 0.697 | 0.845 | −0.057 |
| Δgeo_conditional | 0.667 | 0.831 | +0.015 |

结构通道效应在 v1 正确样本上反而更大（主要集中于 TN），在 v1 错误样本上没有系统性放大，与 teacher-v1 分歧的相关性也接近零或为负。

## 3. 从序列预测效应（cross-fitted）

- Δchem：Ridge/MLP R² 几乎全为负，Spearman 0.00–0.19，sign accuracy ≈0.49–0.55 → 基本不可预测。
- Δstructure_total：Spearman 0.13–0.46、sign accuracy 0.60–0.72，但 R² 在所有折均为大幅负值 → 方向有微弱排序信号，幅度完全不可预测。
- Δgeo_conditional：与 structure_total 类似但更弱，R² 大幅为负。

## 判定

`stop_v2c_component_line`。虽然组件干预效应真实存在且结构通道效应占主导，但它们与 v1 错误无正向关联，也无法从序列稳定预测；按冻结门控不再训练 component-effect student。v1 未修改，Test1/Test2 未读取。

## 产物

- 干预前向：`runs/protox_3d_distill_v1/v2c_component_effects/`
- 诊断结果：`runs/protox_3d_distill_v1/v2c_diagnostic.json`
- 协议：`configs/protox_3dkd_v2c_protocol_v1.json`
