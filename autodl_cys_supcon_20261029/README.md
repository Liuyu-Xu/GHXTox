# AutoDL 一折试点：Cys 分层与监督对比学习

本包为研究试点，不是原论文完整三种子模型；不使用 Test1/Test2。已有全部训练集 6387 条的精简序列、原标签、序列组、疑点标记和历史 3D OOF 数据；模型权重需要额外准备。每折按原训练/验证/校准/留出角色隔离。历史 3D 与阈值有全局选择和非嵌套局限，不能作盲测结果。

## 训练组别

|组|设置|
|---|---|
|A|原始随机采样 + 分类 BCE|
|B|同批次分层采样 + BCE|
|C|同 B 采样 + 普通监督对比|
|D|同 C + 低Cys和高Cys各自组内监督对比|

每批16：低有毒2/低无毒3、中有毒2/中无毒3、高有毒4/高无毒2。B/C/D 每轮步数相同，均限制每批序列组不同，优先抽本轮使用较少的样本。A最后一批可能不足16；固定总batch次数。每轮未设置严格次数上限，应检查进度和采样；多看少数样本会减少其他样本出现次数。分类 BCE 的正类权重所有组统一使用原训练集负/正比例，因此与分层采样一起影响有效权重；不能把本方案称无加权。排除疑点仅用于对比，不从 BCE 删除。统一对比权重0.1、低和高各0.05、温度0.1，是先行试点值，不代表最优。学习率0.0003、最多8轮、验证BCE早停2轮。建议先一折一随机种子，历史fold3已探索，此包默认fold1，均非盲测。

## 环境

Linux + NVIDIA CUDA + 可用的 GPU 版 PyTorch；推荐干净 Python 3.10/3.11 环境。使用镜像自带与 GPU 驱动兼容的 PyTorch，不要盲目覆盖 CUDA 安装。安装其他依赖：

```bash
python -m pip install -r requirements.txt
python -c "import torch,numpy; print(torch.__version__,torch.cuda.is_available(),numpy.__version__)"
```

本地短测使用 torch 2.3.1+cu118、transformers 4.57.6 和 numpy 1.26.4；AutoDL 镜像尚未实测。缺少依赖时以错误日志排查，勿把本机Windows的 site-packages 复制到Linux。

## 权重和上传

ProtT5 目录需包括 config.json、pytorch_model.bin、spiece.model、tokenizer_config.json、special_tokens_map.json；这些权重未包含在本包（权重 bin 约2.42 GB）。本机位置：
`D:/dachuang2026/GHXTox/analysis/upstream_adoption_20261027/handoff/GHXTox_team_handoff_20261027/Rostlab/prot_t5_xl_half_uniref50-enc/`
请将整个目录传到 AutoDL，例如 `/root/autodl-tmp/prot_t5`。这只是示例，以实例实际数据盘为准；`--model-dir`填真实路径。无需上传原论文的3D权重，本试验复用 data/samples.json 内的历史 3D 折外概率，不重新训练三维模型。

## 短测和训练

在包目录打开终端，先短测两批与少量验证，不产生正式模型：

```bash
python train.py --model-dir /root/autodl-tmp/prot_t5 --fold 1 --arm D --smoke
```

成功后分别运行，每个命令是实际训练、耗费GPU时间；先后顺序不限：

```bash
python train.py --model-dir /root/autodl-tmp/prot_t5 --fold 1 --arm A
python train.py --model-dir /root/autodl-tmp/prot_t5 --fold 1 --arm B
python train.py --model-dir /root/autodl-tmp/prot_t5 --fold 1 --arm C
python train.py --model-dir /root/autodl-tmp/prot_t5 --fold 1 --arm D
python compare.py --fold 1
```

输出 `runs/fold1/<组>/`：progress.json、best_model.pt、predictions.json、result.json；compare.py保存逐样本纠错/添错统计。训练已经完成时不覆盖result.json；中断任务未实现优化器断点续训，应保护已有目录后再重新运行。短测目录为 smoke/。只训练新一维头，ProtT5冻结；序列概率在该折calibration角色拟合Platt，融合0.72新一维+0.28历史3D，使用历史阈值。和论文完整模型不能直接当同预算消融。校准、历史阈值和已用过数据均限制结论。

## 注意

这是一套可运行的试点配置，尚未在 AutoDL 镜像上实测；不能保证数据在网络下载站可访问。保留实验输出、日志和环境版本；不要把样本原标签或疑点标记作为推理输入。对比损失权重、分层方案及评价指标不能看留出折结果后反复调整。只看原标签的错误并非确认标签错误；需独立外部数据验证。
