# BioEmu-600 边界逐构象官能团图 — AutoDL 训练包

目标：在**已生成的 600 条 BioEmu 边界样本**（Phase33 资产，跨 11-15/16-30/31-50 aa）
上，复用 full2795 的"逐构象官能团图 + set-attention"架构，做**目标化二级残差训练**。
不生成任何新构象。GPU 训练在 AutoDL（RTX 4090）完成。

## 为什么本地不装 mdtraj

- PyPI（含清华镜像）**没有 mdtraj 的 Windows 预编译 wheel**，源码安装需要 MSVC（失败）；
- `conda install mdtraj` 会把 numpy 升到 2.2.6、重装 scipy/pandas，**可能破坏 torch 2.7.1 项目环境**；
- mdtraj 是 BioEmu 的依赖，**AutoDL 服务器环境自带 mdtraj**，所以"XTC 转换"这一步放在服务器做；
- 本地只需做纯 torch 的子集抽取（`subset_local.ps1`，不需要 mdtraj）。

## 包结构

```
autodl_bioemu600_conformer_graph/
├── README_AUTODL.md
├── scripts/
│   ├── subset_local.ps1                   # 本地：600 条子集 + folds + 参考 OOF（无 mdtraj）
│   ├── export_bioemu600_conformers.py     # XTC -> ensemble_summary.npz + conformer_*.npz（服务器，mdtraj）
│   ├── prepare_server.sh                  # 服务器：解压 -> 导出 -> attach
│   ├── run_autodl.sh                      # 服务器主流程：prepare -> 训练 -> OOF -> 门槛
│   ├── verify_package.py                  # 服务器自检
│   └── gate_validation.py                 # 冻结门槛判定
├── configs/conformer_graph_bioemu600_boundary.json
└── src/ghxtox/                            # ghxtox 源码
```

## 第一步：本地（约 20 分钟，无需 mdtraj）

```powershell
powershell -ExecutionPolicy Bypass -File autodl_bioemu600_conformer_graph\scripts\subset_local.ps1
```

产出（**需要上传**）：
| 文件 | 说明 | 大小 |
|---|---|---|
| `data/processed/bioemu_boundary_600_subset.pt` | 600 条基础张量子集 | ~100 MB |
| `data/folds/bioemu_boundary_600_folds.csv` | 600 条 fold manifest | <1 MB |
| `data/folds/bioemu_boundary_600_reference_oof.csv` | 父模型 600 条参考 OOF | <1 MB |

## 第二步：上传到 AutoDL `/root/autodl-tmp/GHXTox/`

1. 本包 `autodl_bioemu600_conformer_graph.zip`
2. 上面 3 个本地文件
3. **原始 BioEmu 输出** `bioemu_boundary_600_results_20260801.tar.gz`（53 MB，服务器导出用）
4. 父模型五折 checkpoint（各 ~5.9 MB，若服务器已有可跳过）：
   `runs/multisource_1_30_full2795_rank_stable_fold{0..4}/best_model.pt`

## 第三步：服务器（RTX 4090）

```bash
cd /root/autodl-tmp/GHXTox
unzip -o autodl_bioemu600_conformer_graph.zip

# 只需一条命令：服务器导出+attach+五折训练+OOF+门槛（约 1-4 GPU 小时）
bash scripts/run_autodl.sh
# 如需指定 python： PY=/root/.../python bash scripts/run_autodl.sh
```

分步执行（可选）：
```bash
bash scripts/prepare_server.sh          # 1) 解压+导出+attach（需 mdtraj）
PYTHONPATH="$PWD/src" python -B scripts/verify_package.py   # 2) 自检
bash scripts/run_autodl.sh              # 3) 训练+OOF+门槛
```

结果输出：
- `runs/bioemu600_conformer_graph_fold{0..4}/best_model.pt`
- `runs/bioemu600_conformer_graph_oof/predictions.csv`、`metrics.json`
- `runs/bioemu600_conformer_graph_oof/gate.json` ← **最终看这个（promotion_authorized）**

## 冻结门槛（结构专家级，预注册）

| 条件 | 要求 |
|---|---|
| 边界队列 MCC | 不下降 |
| 排序/BACC | AUROC 或 AUPRC 配对 95% CI >0，或 BACC 提高 |
| 五折 | ≥3/5 折 MCC 非降 |
| 长度层 | 各层 MCC 回退 ≤0.005 |
| 净纠错 | corrected − regressed ≥ 1 |

## 协议红线

- 不读取 Test1/Test2 标签；
- 训练不使用 Phase25/联合概率做监督（只用构象张量 + 训练标签）；
- 不在同一 OOF 上移动阈值/加深 MLP/搜索权重；
- 构象仅限已冻结的 600 条 BioEmu 边界，不授权新生成。