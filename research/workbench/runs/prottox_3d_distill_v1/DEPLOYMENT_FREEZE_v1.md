# ProtTox-3DKD v1.0 部署冻结记录

日期：2026-09-05。模型架构、ensemble、checkpoint 与阈值自本记录起冻结；后续只做 Test1/Test2 只读历史审计，不再改模型/阈值/ensemble。

## 冻结版本

```text
model_name: ProtTox-3DKD
version: v1.0
architecture: Frozen ProtT5 -> TriStat + PhysChem -> TriStat-MLP
training: outer-safe single-structure 3D-expert KD (T=2.0, lambda_KD=1.0)
ensemble: seed 20260905 + seed 20260906 mean probability
decision threshold: 0.5
deployment: sequence only; no ESMFold/3D/remote GPU
```

## Runner 入口

CLI：

```text
GHXTOX_PROTTOX_3DKD_MANIFEST=...\ensemble_manifest.json
GHXTOX_PROTTOX_3DKD_MODEL_PATH=<ProtT5 snapshot>
python -m ghxtox.web_prottox_3dkd_runner
```

FastAPI：

```text
GET  /v1/health
POST /v1/predictions/prottox-3dkd
POST /predict/prottox-3dkd
```

默认 `/v1/predictions` 仍指向旧 ProtTox-TSK 路由，等待后续网站侧显式切换；本仓库未改网站。

## 版本与溯源

- ensemble manifest SHA-256：`148ba9f77698bdb3d84ac084947ea07594c1a1dff095aeae21c594ee05628c3f`
- seed 20260905 checkpoint SHA-256：`4089d38f3a360be1ac46bbc102fe067612863754d89d592d77a2e0802f63b9ea`
- seed 20260906 checkpoint SHA-256：`4f4a18908f822e127b103c9c3b5b40bbacdb58ed97f23be7f73cb8628b235a79`
- crossfit teacher targets SHA-256：`5fef37e473cb8c984b08464d42fd941457c02e452f2a3f952a8b9c4812362374`
- 训练缓存 SHA-256：`9b89e8855ae44f0d93f6f2df8dc9ba0f739918bcb61923d70d1ef0a031c1120a`

## 验证

- CLI 单序列与重复预测一致（概率逐位相同）
- FastAPI health：`prottox_3dkd.state = ready`
- `POST /v1/predictions/prottox-3dkd` 返回 200 与正确概率
- golden 回归测试通过：[tests/test_prottox_3dkd_runner.py](../../tests/test_prottox_3dkd_runner.py)
- golden 值：[tests/golden/prottox_3dkd_v1_golden.json](../../tests/golden/prottox_3dkd_v1_golden.json)

## 后续（本文件不授权）

1. 网站侧将 Quick 后端替换为 `/v1/predictions/prottox-3dkd`
2. Test1/Test2 只读历史审计（MCC/BACC/AUROC/AUPRC/Brier/calibration，与原轻量 MLP 对比）
3. 任何结果不得回改 checkpoint、threshold 或 ensemble
