# ProtTox-3D-Distill Product：cross-fitted refit 冻结记录

日期：2026-09-05。架构与统计口径在 refit 前已由用户确认并冻结，本文件记录执行参数与产物哈希。

## 产品架构（冻结，不再改）

```text
输入肽序列
  → Frozen ProtT5 残基表示
  → TriStat（mean/max/std）
  → 9 个固定理化描述符
  → TriStat-MLP head
  → 最终毒性概率
```

训练期额外接受 outer-safe 单结构 3D-expert soft supervision（KD，T=2.0，λ_KD=1.0）；部署期不调用 ESMFold、3D、GHXTox 概率或远程 GPU。

## Cross-fitted teacher targets（6387 行）

规则：样本属于 outer fold k 时，只使用 teacher_k 对该样本前向；teacher_k 训练时从未见过 outer-k，且为 fixed epoch（selection indices 为空）。

- 文件：`runs/protox_3d_distill_v1/teacher_targets/crossfit_teacher_targets_6387.csv`
- SHA-256：`5fef37e473cb8c984b08464d42fd941457c02e452f2a3f952a8b9c4812362374`
- 审计：6387 行、source_index 无重复无缺失、每条 teacher 的 outer fold = 样本自身 outer fold、role 全为 `outer_holdout`、logit/probability 全有限
- 保存字段：source_index / sample_id / sequence_hash / sequence / outer_fold / teacher_sample_role / teacher_probability / teacher_logit / checkpoint_sha256 / teacher_train_indices_sha256 / outer_manifest_sha256 / teacher_config_sha256

## Full-data refit（冻结参数）

- 变体：`mlp_teacher_kd_full_refit`
- seeds：`20260905` 与 `20260906`（不允许再选 seed）
- epochs=20、lr=1e-3、batch=256、hidden=192、dropout=0.10、T=2.0、λ_KD=1.0、阈值 0.5
- 训练集：全部 6,387 条真实标签 + 上述 cross-fitted teacher targets
- 部署 ensemble：两模型 sigmoid probability 的算术平均，不再选择单个 seed

## 产物

- `runs/protox_3d_distill_v1/product_full_refit_v1/full_model_seed20260905.pt`
  - SHA-256：`4089d38f3a360be1ac46bbc102fe067612863754d89d592d77a2e0802f63b9ea`
- `runs/protox_3d_distill_v1/product_full_refit_v1/full_model_seed20260906.pt`
  - SHA-256：`4f4a18908f822e127b103c9c3b5b40bbacdb58ed97f23be7f73cb8628b235a79`
- `runs/protox_3d_distill_v1/product_full_refit_v1/ensemble_manifest.json`

## 边界

- Test1/Test2 尚未读取；只在 API/checkpoint 交接完成后做一次冻结只读历史审计
- 网站前端未改动
- API 交接是下一步，不在本冻结文件内完成
