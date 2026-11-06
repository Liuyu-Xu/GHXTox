# ProtTox-3DKD v1 交付状态

日期：2026-09-05。模型、checkpoint、ensemble、threshold 均保持冻结。

## 已完成

1. Runner（已有并复核）：
   - `src/ghxtox/prottox_3dkd_predict.py`：单条 sequence → ProtT5 → TriStat+PhysChem → 双 seed MLP 概率平均
   - `src/ghxtox/web_prottox_3dkd_runner.py`：JSON-stdin 稳定 runner
2. FastAPI 最小交付接口：
   - `POST /api/predict`
   - 请求：`{"sequence": "..."}`
   - 响应：`toxicity_probability / prediction / confidence / sequence_length / model_version / sequence_evidence / warnings`
   - 实测：GIGKFLHSAKKFGKAFVGEIMNS → Non-toxic / High / 8.77e-05 / ProtTox-3DKD-v1
3. 前端单序列接入：
   - 请求地址改为 `/api/predict`
   - 用户可见品牌与 API 契约改为 ProtTox-3DKD v1
   - 错误提示、Result/Documentation/About 不再展示 teacher/KD/outer-fold 内部细节
   - `npm run build` 通过
4. Batch 批量接入：
   - 核心 `predict_batch`：长度排序分桶，ProtT5 单次加载，多序列一次前向
   - `POST /api/predict/batch`：上限 200、逐条 status（success/error）、单条错误隔离、保持输入顺序
   - 前端 Batch 一次调用批量接口；CSV/JSON 下载均已提供
   - 实测：3 条（2 有效 + 1 非法）返回 pep1/pep2 success、pep3 error

## 尚未完成（按用户顺序后续做）

- 部署测试矩阵：非法字符、空序列、超长、重复一致性、崩溃恢复、CPU/GPU、平均延迟
- 软著/专利材料组装（另行计划）
