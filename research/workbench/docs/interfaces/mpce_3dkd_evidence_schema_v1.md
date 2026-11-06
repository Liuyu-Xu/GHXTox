# MPCE-3DKD Evidence Schema v1.0（冻结接口合同）

状态：**冻结于 2026-09-05**。前端/API 只消费本文件定义的字段；字段语义、阈值与版本号不得因“展示效果不好看”随意修改。schema 机器可读版：[mpce_3dkd_evidence_schema_v1.json](mpce_3dkd_evidence_schema_v1.json)。

## 1. 顶层字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `sequence` | string | 输入肽序列 |
| `evidence_schema_version` | string | 固定 `1.0` |
| `model_architecture` | string | 固定 `MPCE-3DKD` |
| `model_version` | string | 产品 refit 版本 |
| `prototype_profile_version` | string/null | profile map 版本 |
| `prototype_profile_sha256` | string/null | profile map 文件 SHA-256 |
| `analysis_level` | `base`/`detailed` | base 不含扰动稳定性；detailed 包含 |
| `prediction` | `Toxic`/`Non-toxic` | 阈值 0.5 的标签 |
| `probability` | number | raw ensemble probability |
| `calibrated_probability` | number | 展示校准概率（T 已冻结） |
| `decision_threshold` | 0.5 | 冻结 |
| `prototype_evidence` | object | 见下 |
| `cross_view_evidence` | object | 见下 |
| `perturbation_stability` | object | 见下 |
| `evidence_record` | object | 见下 |
| `provenance` | object | 见下 |

## 2. prototype_evidence

逐 seed（`20260905`、`20260906`）输出：

- `toxic_pattern_id`：T1–T4；
- `toxic_similarity`：cosine；
- `toxic_pattern_support_count` / `mean_similarity` / `similarity_q25/q50/q75`：来自冻结 profile map；
- `toxic_pattern_profile`：model-associated profile 文本；
- non-toxic 侧同构字段；
- `prototype_affinity_logit`。

顶层聚合：

- `representative_seed`：成员概率最接近 ensemble 均值的 seed，平票取 20260905；
- `dominant_side` / `dominant_pattern_id` / `dominant_pattern_similarity` / `dominant_model_associated_pattern`。

## 3. cross_view_evidence

- `branch_removal_delta`：四组交互的 `p(去掉该组) − p(full)`；
- `dominant_interaction`：按 `|removal_delta|` 最大的组；
- `support_count/support_fraction`：移除此组会削弱当前预测方向的组数占比；
- 字段语义为 model-associated evidence，不是毒性原因。

## 4. perturbation_stability

判定规则冻结如下：

| label | 规则（mean_abs_probability_delta） |
| --- | --- |
| high | `< 0.02` |
| moderate | `0.02 <= x < 0.08` |
| low | `>= 0.08` |

`prediction_stability_score = max(0, 1 − mean_abs_probability_delta / 0.15)`。

固定输出：

- `max_abs_probability_shift`：max |delta|；
- `label_flip_count`：保守替换导致标签翻转的数量；
- `most_sensitive_positions`：|delta| 前 3；
- `largest_probability_shift` / `direction`；
- base 模式下 `analysis_performed=false`，不产生 position 成本。

## 5. evidence_record 判定规则

冻结规则：

```text
margin = |probability − 0.5|

consistency = high:
    margin >= 0.25
    AND cross_view_support_fraction >= 0.75
    AND perturbation stability != low

consistency = moderate:
    margin >= 0.10
    AND cross_view_support_fraction >= 0.50
    AND perturbation stability != low

else: low
```

`review_recommended=true` 当且仅当满足任一条件：

```text
consistency == low
OR margin < 0.15
OR (margin < 0.25 AND consistency == moderate)
```

base 模式（未跑扰动）将 perturbation stability 视为 `not_computed`，不因“未计算”直接触发 low。

## 6. provenance

强制字段：

```json
teacher_used_during_inference: false
structural_input_used_during_inference: false
esmfold_used_during_inference: false
```

## 7. 预算策略（冻结）

- Single Prediction 默认 `analysis_level=detailed`；
- Batch Prediction 默认 `analysis_level=base`：只返回 prediction + prototype/cross-view evidence；
- 用户点击 Detailed evidence 时，仅对单条序列调用 detailed；
- `max_perturbation_positions` 可选用于进一步限制单次 detailed 的 ProtT5 前向次数；当前默认不做长度截断，前端按此策略调用。

## 8. 禁止变更

- 不改字段名或单位；
- 不调 stability/review 阈值；
- 不把 `label_flip_count`、`max_abs_probability_shift` 之外的字段作为稳定性核心输出；
- 不在前端把 model-associated evidence 表述成实验因果。

