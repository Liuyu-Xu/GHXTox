# ProtTox-3D-Distill v1：outer-safe 单结构 3D-expert 蒸馏

日期：2026-09-05。本文档只记录独立轻量线的第二级实验，不涉及网站和发表版主线。

## 结论

`proxy_teacher_kd`（两段结构质量代理 + 3D teacher 蒸馏）通过预注册训练 OOF 晋级门槛，判定 `promote_proxy_teacher_kd`；`mlp_teacher_kd`（仅 MLP + 同一 teacher）同样通过门槛，作为消融/对照记录。两者都没有读取 Test1/Test2。正式默认 MLP 的部署替换需要后续单独完成 full-data refit 与运行器交接，本实验只做 OOF 晋级判定。

## Teacher 与目标表

- 来源：`D:\dachuang2026\ziliao\topology_strict_oof_v1_final.tar.gz`（SHA-256 `606f042a...efce3`）
- 五份 checkpoint：严格外层 scratch GHXToxModel（单结构缓存训练、固定 12 epoch、selection indices 为空）
- 每份 teacher 对“非自身外层折”的全部样本前向，输出 probability 与 logit
- 输出：`runs/prottox_3d_distill_v1/teacher_targets/teacher_outer0..4_targets.csv`，共 25,548 行
- 每行记录 role：fitting / inner_validation / calibration，并绑定 checkpoint、训练索引、外层与嵌套 manifest 的 SHA-256

角色覆盖（按 outer fold 0–4）：fitting 3,830/3,884/3,809/3,816/3,840；inner_validation 616/641/637/644/600；calibration 670/633/668/636/624。

## OOF 结果（训练 OOF，阈值 0.5，双种子）

| 模型 | MCC（20260905/20260906） | AUPRC（20260905/20260906） | Brier（20260905/20260906） |
| --- | ---: | ---: | ---: |
| 保留 TriStat-MLP | 0.81577 / 0.81977 | 0.92927 / 0.92834 | 0.05931 / 0.05893 |
| StructGuide v1（proxy，无 KD） | 0.80799 / 0.81259 | 0.92606 / 0.92736 | 0.06813 / 0.06817 |
| proxy_teacher_kd（候选） | 0.82173 / 0.82747 | 0.94028 / 0.94004 | 0.06113 / 0.06018 |
| mlp_teacher_kd（KD 对照） | 0.82706 / 0.82392 | 0.94172 / 0.94173 | 0.05851 / 0.05943 |

候选相对保留 MLP：

- proxy_teacher_kd：MCC Δ = +0.00596 / +0.00770（均值 +0.00683）；AUPRC Δ 均值 +0.01135；Brier Δ 均值 +0.00154
- mlp_teacher_kd：MCC Δ = +0.01129 / +0.00415（均值 +0.00772）；AUPRC Δ 均值 +0.01292；Brier Δ 均值 −0.00015

相对 StructGuide v1（同一两段结构、无 KD），proxy_teacher_kd 平均 MCC +0.01431、AUPRC +0.01345、Brier −0.00750：KD 是主要增量来源。

## 口径说明

- 学生 outer-k 只读取 `teacher_outer{k}_targets.csv`，其内容为该 teacher 对非 k 全部样本的 soft target；teacher 从未见过 k，因此对学生验证折外层安全
- teacher 对 fitting 样本是 in-sample、对 inner_validation/calibration 是 out-of-fit；两种角色都允许出现在学生训练集中
- 损失：L_tox + 0.10·L_proxy(Huber) + 1.0·L_KD(T=2)，与冻结协议一致
- 不包含 ESMFold 在线运行、坐标输入、GHXTox 概率或远程 GPU；部署形态仍为 sequence → student → final probability

## 可复现入口

- teacher 目标：[prottox_3d_distill_targets.py](../../src/ghxtox/prottox_3d_distill_targets.py)
- 学生训练：[train_prottox_3d_distill.py](../../src/ghxtox/train_prottox_3d_distill.py)
- 比较判定：[prottox_3d_distill_compare.py](../../src/ghxtox/prottox_3d_distill_compare.py)
- 冻结协议：[prottox_3d_distill_v1.json](../../configs/prottox_3d_distill_v1.json)
- 机器可读决定：[promotion_report.json](promotion_report.json)

下一步（未在本实验中执行）：若决定替换默认，需冻结 full-data refit 协议并生成 outer-holdout teacher targets 作为完整部署训练集，再接 API checkpoint 与版本记录；Test1/Test2 只能在该协议冻结后做只读历史审计。
