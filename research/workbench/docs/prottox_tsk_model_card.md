# ProtTox-TSK v1 模型卡（训练前实现版本）

## 用途与状态

`ProtTox-TSK` 是与历史 GHXTox 隔离的新候选系统：Frozen ProtT5 残基表征、masked mean/max/std（TriStat）聚合、9 个固定序列理化描述符，以及可选择的 MLP 或 spline-KAN 小型分类头。

当前状态是**已实现、尚未训练和评估**。不得将任何历史 GHXTox 分数、阈值、LoRA 权重、Mamba 权重、3D 专家或融合结果归属给它。

## 不包含的历史路径

- 不训练 ProtT5、也不加载 LoRA。
- 不含 Bi-Mamba、旧 Projection、旧 Mean Pooling 或旧 Linear Head。
- 不含三 seed 集成、Platt 校准、Quick→Full 路由、ESMFold、ESM2 3D、结构专家、融合、Soft-Cys 或拓扑模块。

## 可复现训练顺序

先只做 128 条 smoke，不做性能宣称：

```powershell
python scripts\cache_frozen_prott5_features.py --input <processed_train.pt> --output runs\prottox_tsk_smoke\features.pt --limit 128 --stratify-labels --device cuda
python scripts\train_prottox_tsk.py --cache runs\prottox_tsk_smoke\features.pt --output-dir runs\prottox_tsk_smoke\mlp --head mlp --folds 5 --epochs 1 --device cuda
```

确认缓存哈希、显存和输出格式后，用既有审计分组 CSV 做全量缓存及同协议 5 折对照：

```powershell
python scripts\train_prottox_tsk.py --cache <full_features.pt> --fold-manifest <audited_folds.csv> --output-dir runs\prottox_tsk_mlp --head mlp --device cuda
python scripts\train_prottox_tsk.py --cache <full_features.pt> --fold-manifest <audited_folds.csv> --output-dir runs\prottox_tsk_kan --head kan --device cuda
```

比较同一折划分下的 MCC、AUROC、AUPRC、Brier、参数量、训练耗时与单条推理耗时；只有 KAN 有稳定收益且无明显不稳定性时才可选择它作为产品头部。

## 网页接口

训练产生的新 `final_model.pt` 后，设置：

```powershell
$env:GHXTOX_PROTTOX_TSK_CHECKPOINT = "D:\path\to\final_model.pt"
$env:GHXTOX_PROTTOX_TSK_RUNNER = "D:\dachuang2026\ziliao\ghxtox_runtime_py310\Scripts\python.exe -m ghxtox.web_prottox_tsk_runner"
```

`POST /v1/predictions` 与 `POST /v1/predictions/prottox-tsk` 都执行一次性最终计算预测。旧 `/quick` 和 `/full` 端点仅为历史 GHXTox 原型保留，不能作为新项目结果。
