# 研究复现与资产说明

1. 根目录是同步的上游公开模型；研究工作台是独立代码快照，不能混用checkpoint、数据终点或阈值。默认完整模型与轻量分类试验分开报告。
2. 每个分析目录保留协议、脚本和紧凑结果。大文本为gzip；先在本目录运行`python restore_assets.py --verify-only`校验，再运行`python restore_assets.py`恢复原路径。恢复只创建不存在的文件，不覆盖已有不同内容。
3. `ASSET_MANIFEST.json`记录原文件及发布文件SHA256。Markdown仅调整发布副本的本地链接，原备份未变。原README保存为HISTORICAL_README.md。
4. 此快照不是全量训练数据包。权重/PLM特征/ESMFold与原始骨架缓存不上传；部分第三方原始下载与中间文件也不上传。不能声称仅clone即可重训所有历史实验。旧source路径、D盘绝对路径和Python环境名保留为溯源，跨机器执行需要按脚本数据路径配置；不要对缺失实验输入用伪数据替代。
5. SaProt关键逐样本概率、纠错/添错、bootstrap及核验结果已保留；没有重新训练或挑选新种子。原标签疑点并非确认错标，旧测试集不是盲外部队列。
6. 根目录依赖与研究版pyproject/requirements有区别，应在独立虚拟环境检查研究模块，避免覆盖已有科学计算环境。研究版代码以`PYTHONPATH=research/workbench/src`运行；这不会自动验证未提供资产的训练任务。
7. 下载论文PDF/Word、模型二进制、原始第三方源码目录、临时规划、私人备份与日志不进入发布成果。历史fork独有工具另在../legacy_hypeptox，不能覆盖根目录最新版。
