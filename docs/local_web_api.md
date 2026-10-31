# GHXTox 私有本地 Web API

本接口只绑定 `127.0.0.1`，用于本机 React 页面和 GHXTox 推理工程之间的边界。它不包含公网部署配置。

## 启动

在 `D:\dachuang2026\GHXTox` 下使用保留运行环境：

```powershell
D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m pip install -e ".[api]"
$env:HF_HOME = "D:\dachuang2026\ziliao\huggingface_cache"
$env:GHXTOX_WEB_QUICK_RUNNER = "D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m ghxtox.web_quick_runner"
$env:GHXTOX_WEB_FULL_RUNNER = "D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m ghxtox.web_canonical_runner"
D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m ghxtox.local_web_api
```

然后浏览 `http://127.0.0.1:8000/v1/health`。健康检查会分别列出新 ProtTox-TSK 与历史 Quick/Full 路径的就绪状态、资产及 SHA-256。

## 新项目默认：一次性 ProtTox-TSK

ProtTox-TSK 是独立的新模型路线：Frozen ProtT5 → TriStat（mean/max/std）→ 固定理化描述符 → 已训练的 MLP 或 KAN 小头。它不加载 LoRA、Mamba、3D 专家或融合权重，也没有 Quick→Full 工作流。训练出新的 `final_model.pt` 后配置：

```powershell
$env:GHXTOX_PROTTOX_TSK_CHECKPOINT = "D:\path\to\prottox_tsk\final_model.pt"
$env:GHXTOX_PROTTOX_TSK_RUNNER = "D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m ghxtox.web_prottox_tsk_runner"
```

- `POST /v1/predictions`：新项目默认的一次性 ProtTox-TSK 预测。
- `POST /v1/predictions/prottox-tsk`：显式的新模型端点。

未配置新的 checkpoint 时，这两个端点返回 `503`；不会拿历史 GHXTox checkpoint、Mock 值或单个旧分支替代。

## 两条计算路径

- `POST /v1/predictions/quick`：只运行冻结的 LoRA-ProtT5 + BiMamba 三 seed 1D 集成，不生成结构。它读取 `configs/web_quick_sequence_policy_v1.json`，其中阈值 `0.3920986716`、90% 训练 OOF 覆盖率拒识带及 Platt 校准均只由 sequence OOF 冻结；它不复用完整融合模型的阈值或拒识截断。接近 Quick 阈值的输出为 `Uncertain` 并建议 Full Analysis。
- `POST /v1/predictions/full`：运行 ESMFold、特征处理、1D/3D 专家和完整 canonical 融合。它是最终**计算**预测，仍不是实验毒性结论。

这些端点仅为历史 GHXTox 原型保留。新页面应调用一次性 ProtTox-TSK，不能向用户暴露 Quick/Full 工作流。

Quick policy 如需重新生成，只能在新增训练数据或新预冻结协议下运行：

```powershell
python scripts\freeze_web_quick_sequence_policy.py --manifest D:\dachuang2026\ziliao\train_cdhit080_fallback_folds.csv
```

不要在网页请求时重新计算阈值、拒识带或校准参数。

## 正式推理前提

资料目录已经保存了 checkpoint，但每条新序列还必须经过与正式默认模型一致的结构和特征预处理。上面配置的 runner 会在每个任务目录执行 ESMFold 结构生成、结构预处理、ESM2 残基特征、3-seed 1D 预测、3D 专家预测和 0.72/0.28 融合。首次使用还需要安装 `fair-esm[esmfold]` 和可用 GPU；接口不会偷偷改用启发式结构或单分支预测。

未配置 runner 时，`POST /v1/predictions` 返回 503 及完整审计状态，不会以 Mock 或单分支结果替代正式默认模型。

Soft-Cys 拓扑和结构效用路由均未通过预设 OOF 晋级条件，接口不会将它们启用为默认路径。

## 推荐运行位置

React 页面可以继续在 Windows 本机运行，但正式 runner 应运行在 AutoDL/Linux GPU。已验证 Windows 安装 `fair-esm[esmfold]` 会被其 `deepspeed`/`libaio` 依赖阻断。服务器上以 `127.0.0.1:8000` 启动 API 后，在本机建立 SSH 隧道：

```powershell
ssh -L 8000:127.0.0.1:8000 root@你的_AutoDL_服务器
```

浏览器仍访问本机前端，前端默认的 `http://127.0.0.1:8000` 会经隧道到服务器；无需开放 API 或网站到公网。

## MPCE-3DKD evidence 路由（当前前端默认）

当前 React 前端默认调用 MPCE-3DKD evidence 路由，不再调用旧 ProtTox-3DKD `/api/predict`。

```powershell
$env:GHXTOX_MPCE_3DKD_MANIFEST = "D:\dachuang2026\GHXTox\runs\mpce_3dkd_product_single_refit_v1\ensemble_manifest.json"
$env:GHXTOX_MPCE_3DKD_MODEL_PATH = "D:\dachuang2026\ziliao\huggingface_cache\hub\models--Rostlab--prot_t5_xl_half_uniref50-enc\snapshots\94a6abc029ae13029317b140b7424e012bf8dfbf"
$env:GHXTOX_MPCE_3DKD_RUNNER = "D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m ghxtox.web_mpce_3dkd_runner"
```

端点：

- `POST /api/predict/mpce-3dkd`：单条；默认 `analysis_level=detailed`，含 perturbation stability；
- `POST /api/predict/mpce-3dkd/batch`：批量；固定 `base` evidence，不跑扰动；
- `GET /v1/health`：新增 `mpce_3dkd` 状态。

旧 `POST /api/predict` 与 `/api/predict/batch` 仍绑定 ProtTox-3DKD v1，仅供历史接口保留；前端已切换到 MPCE-3DKD evidence 路由。
