from pathlib import Path
import json
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
m=read(P/'metrics.json');cs=read(P/'comparisons.json');runtime=read(P/'runtime.json');manifest=read(P/'run_manifest.json')
ARMS=read(P/'execution_status.json')['expected_completed_arms'] if (P/'execution_status.json').exists() else ['A','B','C']
lines=['# LoRA修复A/B首轮对照结果','','2026-10-27。按用户截图先隔离修复与加权；用户随后要求C先不训练、B完成即汇报。因此本次仅A/B完成，C未训练。实际ProtT5-LoRA+BiMamba重训，非小型替代模型。','',
'当前结论：本轮原配置直接启用LoRA未提升；真实梯度已修复，但单折序列MCC/AUPRC及条件融合MCC均下降。不能将其扩展成LoRA普遍无用，也不能宣布已证明下降原因。','',
'## 三组定义与比较范围','',
'|组|LoRA梯度机制|额外样本权重|','|---|---|---|',
'|A|复现原reentrant+冻结embedding，B矩阵无梯度|全部1|',
'|B|修复LoRA梯度路径|全部1|',
'|C（未训练）|计划同B修复|计划低Cys原正类1.5，其余1；本次暂缓|','',
'同一outerfold0、seed42，15epoch、batch16、AdamW lr0.0003/wd0.01、相同初始参数/训练顺序/训练角色class_pos_weight。已运行A/B均执行15epoch，按同一未加权validationBCE规则选checkpoint，只用独立calibration角色拟合标量Platt，序列阈值0.5固定。A/B仅改变LoRA梯度设置。','',
'A复现的是旧梯度机制，不是历史论文训练逐位复现：共同采用嵌套角色、validation checkpoint选择、clip1和AMP初始scale128；不能将本次A绝对指标直接与旧OOF或论文三种子集成排名。','',
'## 留出序列专家与条件融合','',
'|输出|组|n|MCC|AUPRC|AUROC|FN|FP|','|---|---|---:|---:|---:|---:|---:|---:|']
for mode,title in [('sequence','单种子一维'),('fusion','条件融合')]:
 for arm in ARMS:
  q=m[mode][arm]['all'];lines.append(f"|{title}|{arm}|{q['n']}|{q['MCC']:.5f}|{q['AUPRC']:.5f}|{q['AUROC']:.5f}|{q['FN']}|{q['FP']}|")
lines+=['','条件融合固定为0.72×本次单个一维成员+0.28×同一份历史三维OOF，阈值0.4702225653。它用于检查当前固定融合下的变化，不是原三种子完整部署模型，也不是全流程严格嵌套的新盲测。没有用Test1/Test2，没有扫权重/阈值。','',
'## 修复与加权是否有净收益','',
'|输出|对照|ΔMCC|95%组bootstrap区间|纠错|添错|净少错|','|---|---|---:|---|---:|---:|---:|']
for mode,title in [('sequence','单种子一维'),('fusion','条件融合')]:
 for q in cs[mode]:
  if (q['control'],q['treatment']) not in [('A','B'),('B','C')]:continue
  low,high=q['CI95_group_bootstrap'];lines.append(f"|{title}|{q['treatment']}-{q['control']}|{q['delta_MCC']:+.5f}|[{low:+.5f}, {high:+.5f}]|{q['corrected']}|{q['harmed']}|{q['net_fewer_errors']:+d}|")
lines+=['','本轮只有B-A比较；C-B尚未执行。2000次按原序列group成组重采样，报告95%区间。区间仅描述固定这一个折/种子的预测不确定性，不包含重训随机种子方差、折间差异或历史选择偏差。即使区间不跨0也不能宣布模型普遍优劣。','',
'## 四类与证据分层（条件融合）','',
'|分层|n|A错数|B错数|','|---|---:|---:|---:|']
for key,title in [('physical_0','低Cys原正类'),('physical_2','富Cys原正类'),('physical_1','富Cys原负类'),('physical_3','低Cys原负类'),('flagged','原主疑点标记'),('unflagged_not_gold','未带主标记，不是可靠金标签'),('MCD','MCD（仅本折留出成员）'),('NDBP','NDBP（仅本折留出成员）')]:
 q=[m['fusion'][a][key] for a in ARMS];lines.append(f"|{title}|{q[0]['n']}|{q[0]['errors']}|{q[1]['errors']}|")
lines+=['','全部家族背景而非仅MCD/NDBP均在metrics.json逐类保留。没有删掉标签疑点或未匹配家族；“错/对”均相对原标签。家族留出样本很少，不能把三条或少量变化当整类泛化结论。','',
'## 梯度与预算核对','',
'|组|选择epoch|实际训练秒数|峰值CUDA GiB|','|---|---:|---:|---:|']
for a,r in runtime.items():lines.append(f"|{a}|{r['selected_epoch']}|{r['seconds']:.1f}|{r['peak_cuda_bytes']/1024**3:.2f}|")
lines+=['','A每轮第一batch的144个LoRA B均无梯度、训练末仍全零；B每轮均有144个B获得梯度并实际更新。两组分类头都有梯度。检查点从同一预训练ProtT5与相同随机头/LoRA初始化开始，没有加载全数据任务权重初始化。两组同样的batch顺序和优化尝试次数；AMP可能按溢出情况跳过更新，本轮未逐步记录成功更新计数。','',
'旧三个最终权重LoRA增量为零的审计见LORA_AUDIT.md；没有全部旧fold权重，未检查者不作断言。旧权重/预测未覆盖，训练入口修复只影响以后重新训练。','',
'## 文件','',
'- protocol.json、run_manifest.json：定义、共同适配、角色、源代码与初始化身份。',
'- runs/A,B,C：每轮日志、选择epoch、真实LoRA更新检查、calibration记录、逐样本留出预测及研究检查点。',
'- metrics.json：序列/条件融合、四类、疑点状态和全部家族分层。',
'- comparisons.json：B-A、C-B及次要C-A的纠错/添错名单和条件bootstrap区间。',
'- verify.py、verification.json：独立重算判定、角色隔离、训练预算、真实LoRA状态。','',
'## 训练曲线与下一步','',
'A的最佳validationBCE在epoch4（约0.2236），B在epoch1（约0.2367）；B后续训练损失持续下降、validation损失回升。这支持存在过拟合迹象，不能单独证明学习率过大或标签噪声导致下降。原来未真正训练LoRA的设置不等于已为真实LoRA优化过。','',
'本轮A/B均用原统一lr0.0003，未单独给新增有效的LoRA设低学习率。下一项若继续应优先在训练/验证内部检验LoRA学习率、训练轮数/早停及正则，而非立即叠加困难群体权重。C依用户要求保持未训练。没有开始额外调参或测试集预测。','',
'本次结果为一个预先固定的折和种子的方向试验，不能自动更新默认模型。完整五折、多种子和新独立评价尚需单独完成。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('REPORT written',flush=True)
