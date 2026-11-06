# GHXTox 原版对照：B/C/D 训练准备包

## 当前状态：短测试通过，完整端到端尚未验证

本包用于准备后续正式对照，**不能声称新训练已与历史训练完全等价**。本机无 LoRA 的 B、D 各跑完2批真实GPU前向/反向；所有5折×3种子的分层采样及组隔离已模拟。尚未跑完整15轮、15个OOF模型、3个完整模型，也未在AutoDL Linux上实测 evaluate.py。根据用户要求，先只完整训练 A；与历史 A 比较并审核后，再决定是否训练 B/C/D。自动脚本只运行 A。

## A/B/C/D 不改名

A 普通随机采样+BCE；B Cys分层采样+BCE；C B+普通SupCon；D C+低/高Cys组内SupCon。B/C/D每批16：低原正2、低原负3、中原正2、中原负3、高原正4、高原负2；BCE保留全部标签样本，原有positive weight=训练折原负/原正；C/D对比项不含疑点标记样本。C普通lambda 0.1，D额外低/高各0.05，温度0.1。各组同折相同样本、相同轮数15、batch16、lr0.0003、AdamW 0.01及正类权重；各组采样必然不同。A只是从头重训的可比较基线；历史A须用既有3份原权重与原缓存另作参考。

新训练统一使用 **rank 0，不创建 LoRA 层**，ProtT5 冻结，A/B/C/D 的结构保持一致。三份历史完整权重中 LoRA B 为零，之前验证移除零 LoRA 对已测样本推理等价；不过从头训练的随机流程、旧训练命令和环境不完全已知，不能保证新训练 A 与历史权重训练轨迹相同。历史 A 权重只用于单独的参考推理。

## 数据和校准

原始data/folds.csv与6387条注释ID/标签/序列/组已核对；fold训练为非留出4折，full为6387条全量；本包不包含ProtT5权重，外部输入。每组42、123、2025各5折OOF+3份full，总18次。评价先把3 seed OOF概率取均值、logit、C=1e6 Platt；测试时3个full模型输出概率均值后使用该组校准器；统一复用历史单结构3D概率，固定0.72/0.28与0.47022256531284556。OOF校准本身使用全OOF标签，仅可用于开发，不能把OOF效果当严格独立评价。Test2是Test1子集，均以前用于开发，非新盲测。

数据中有样本标签和历史测试分数，仅用于预定义评价；不要根据它们调整候选及选择阈值。

## 安装

先使用CUDA可用的PyTorch镜像（本地测torch2.3.1+cu118、numpy1.26.4）。确保python3.10/3.11与GPU驱动兼容：

```bash
python -m pip install -r requirements.txt
python -c "import torch;print(torch.__version__,torch.cuda.is_available())"
```

服务器权重目录需包含 config.json、pytorch_model.bin、spiece.model 等ProtT5文件；原版服务器通常在 `/root/autodl_cys_supcon_20261029`，以实际目录为准。原版权重和ProtT5基座不包含在本小包；历史校准和预测参考已包含。

## 短测试：不启动正式训练

```bash
python train.py --model-dir /root/autodl_cys_supcon_20261029 --arm D --seed 42 --fold 0 --smoke
```

输出smoke_no_lora/D/seed42/fold0，只运行2批；禁止把smoke当正式结果。若结果目录已存在，为保护数据脚本拒绝覆盖。

## 单个正式试点

```bash
python train.py --model-dir /root/autodl_cys_supcon_20261029 --arm B --seed 42 --fold 0
```

脚本固定15轮、无早停、记录每轮steps和anchors，runs/B/seed42/fold0保留checkpoint、OOF、训练日志。单fold结束不足以评估测试集。先核查日志、训练时长、以及相同初始化和运行环境；不要根据这折分数改权重。

完成某组全部折及完整模型后：

```bash
python evaluate.py --model-dir /root/autodl_cys_supcon_20261029 --arm B
```

此命令要求该组15个OOF文件和3个full checkpoint全部存在；输出校准器、OOF融合预测及Test1/Test2预测。先在非正式小范围验证此流程，一旦使用测试集，不能据此调参。需要复现原版权重，请从 D:/dachuang2026/GHXTox-ziliao 上传full_seed42、123、2025以及 best_model(1).pt；本包历史测试比较使用已保存三维预测，不能声称本次执行了3D前向。

## 限制

当前代码不是已证实历史完全复现的原训练脚本；A的shuffle与历史原始loader未逐批验证；B/C/D的分层batch及对比均为新方法。历史checkpoint只记录配置，不能排除训练时覆盖15轮。原标签疑点可能是来源/实验差异；不据模型错误自动删除样本。科学结论需要匹配同环境同初始化、训练预算与calibration的baseline，并最终独立数据/严谨嵌套确认。

## 本轮只跑 A

`bash run_all.sh` 只运行 A 的3个种子、5个OOF折和3个完整模型，并在成功后执行A的评价；遇到错误立即停止，已完成结果经过检查后复用。它不会训练B/C/D。每轮固定15 epochs，72是全部四组的远期总量，本轮是18个一维模型。日志不保存训练优化器状态，中断未完成模型时需从头开始该模型。完成后下载包含OOF/日志/预测而不含 checkpoint.pt 的 runs/A；模型权重另行保存。

本机以模拟文件做过evaluate.py从OOF校准到历史测试输入的流程通路测试，该测试的模型权重与分数是模拟、不可作实验结果；没有做真实15轮端到端，也没有做服务器Linux完整训练。
