# GHXTox RTX 4080 Super 训练准备

本目录包含六种子一维 OOF 训练脚本、原始五折清单、历史三维 OOF 预测、模型源码及 pLDDT 消融开关。训练使用冻结 ProtT5 与无 LoRA 双向 Mamba，六种子各五折，完成后自动计算对照。对照结果属于开发集分析，历史三维 OOF 及校准流程不完全嵌套。

## 安装环境

安装 Python 3.10、支持 RTX 4080 Super 的 NVIDIA 驱动和 CUDA 版 PyTorch，确认 `python -c "import torch; print(torch.cuda.is_available())"` 输出 True。在本目录运行：

```bash
python -m pip install numpy scikit-learn transformers==4.46.3 sentencepiece protobuf networkx biopython
```

PyTorch 请使用目标电脑已验证的 CUDA 环境。依赖和 GPU 环境尚未在目标 4080 Super 上验证。

## ProtT5 权重

将 ProtT5 `prot_t5_xl_half_uniref50-enc` 的完整文件放在本目录的 `data/models/prot_t5_xl_half_uniref50-enc`，包括 `config.json`、`pytorch_model.bin`、`spiece.model`、tokenizer 配置。可通过环境变量 `GHX_PROTT5_DIR` 指向其他目录。权重不包含在 Git 或训练包中。本地试运行权重的 SHA256 为 `7f51ba885541c7dc569d46b796af57cc7a2ba7945107dced4f19d1b5ec091157`，使用不同权重须如实记录。

## Windows PowerShell

在本目录运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:GHX_PROTT5_DIR = 'D:\models\prot_t5_xl_half_uniref50-enc'
python analysis/reviewer_response_20261102/train_six_seed_1d.py --seed 42 --fold 0 --smoke --output-dir analysis/reviewer_response_20261102/six_seed_1d/smoke
python -u analysis/reviewer_response_20261102/run_six_seed_all.py
```

## Linux

在本目录运行：

```bash
export PYTHONPATH="$PWD/src"
export GHX_PROTT5_DIR=/path/to/prot_t5_xl_half_uniref50-enc
python analysis/reviewer_response_20261102/train_six_seed_1d.py --seed 42 --fold 0 --smoke --output-dir analysis/reviewer_response_20261102/six_seed_1d/smoke
python -u analysis/reviewer_response_20261102/run_six_seed_all.py
```

六种子脚本依次运行30折，每折完成后保存预测、权重和输入哈希。重启程序会核验并跳过已完成的折，未完成的折从头开始。输出位于 `analysis/reviewer_response_20261102/six_seed_1d/`。全部完成后生成 `comparison.json` 和 `comparison_predictions.csv`。脚本不会自动关机。

## pLDDT 消融

`prepare_plddt_ablations.py` 可生成 A/B/C/D 配置，源码保留消息加权及门控开关。示例配置仅用于检查，尚未确认历史 A 的权重谱系、五折初始化、训练/选模与完整匹配协议。需要先确认数据及对照后才能执行正式重训练，不能把示例结果作为审稿消融结果。

```bash
python analysis/reviewer_response_20261102/prepare_plddt_ablations.py --base-config configs/chemical_site_interaction_pareto_same_residue.json --output-dir plddt_example
```
