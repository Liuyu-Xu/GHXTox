from pathlib import Path
import json,collections
O=Path(__file__).resolve().parent
read=lambda n:json.loads((O/n).read_text(encoding='utf-8-sig'))
rows=read('all_training_category_index.json');summary=read('category_matrix.json');pairs=read('existing_pairs_category_overlay.json')
target=[r for r in rows if r['label']==1 and 'membrane_or_lysis' in r['categories']]
assert len(target)==76
controls=[r for r in target if r['fusion_category']=='TP'];cases=[r for r in target if r['fusion_category']=='FN']
assert len(controls)==33 and len(cases)==43
pool=dict(scope='candidate comparison pool, not 43 independent mechanistic failures or 33 gold matched controls',all76=target,correct33_ids=[r['sample_id'] for r in controls],error43_ids=[r['sample_id'] for r in cases],priority14_fusion_harms_ids=[r['sample_id'] for r in target if r['existing_pattern']=='A_D_一维对三维错融合错'],priority28_both_wrong_ids=[r['sample_id'] for r in target if r['existing_pattern']=='B_两路及融合都错'],remaining1_ids=[r['sample_id'] for r in target if r['existing_pattern']=='D其余_一维错三维对融合错'])
(O/'membrane76_comparison_pool.json').write_text(json.dumps(pool,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 毒性分类：生物学线索与模型表现总表','',
'日期2026-10-26。复用旧6387条训练OOF、已有功能注释、73行公开溶血关联和27对探索性对照。没有重算旧基础分组、重新配对、模型拟合或测试集分析。',
'','## 主要判断','',
'优先关注膜作用/溶血相关候选类别。已有缓存提示其分类短板分布于整类和五折，并非只由Na1b等个案构成；但尚不能证明缺失的真实生物机制，更不能保证改网络即可提高独立测试指标。当前不继续优化溶血曲线。',
'','## 类别 × 原标签 × 分支表现','',
'以下为缓存注释线索，不是经过统一实验阈值确认的机制类别。一个样本可属于多个类别，不可相加；原标签为0并不等于实验无溶血。表格错误数只对应原广义毒性标签。',
'','|注释线索|原标签|条数|一维错误|三维错误|融合错误|','|---|---:|---:|---:|---:|---:|']
for s in summary:
 if s['category'] in ['membrane_or_lysis','ion_channel','receptor_or_signaling','hemostasis_or_proteolysis','antimicrobial','hemolysis_text']:
  a=s['all_cached'];lines.append(f"|{s['title']}|{s['original_label']}|{a['n']}|{a['seq_errors']}|{a['struct_errors']}|{a['fusion_errors']}|")
lines+=['','## 膜作用候选整类：76条原阳性','',
'43条融合漏判，33条融合正确，共70个旧序列群。43条漏判覆盖39个群；正确33条覆盖31个群，均不等于70个生物家族。五折样本数/融合漏判分别15/7、19/12、11/6、17/10、14/8。这是跨折诊断分布，不是重新冻结的独立验证。',
'','|互斥旧模式|条数|下一步关注|','|---|---:|---|',
'|一维对、三维错、融合错|14|结构信息或融合对这一候选类别的处理为何不合适|',
'|两路及融合都错|28|当前两路共同未利用的信息，或原标签/对象边界问题|',
'|一维错、三维对、融合仍错|1|不能靠上述两种解释强行归类|',
'|三路都对|26|正确比较池|',
'|一维对、三维错、融合对|7|同样三维出错但融合未受损的比较池|',
'',
'一维29次漏判、融合43次，净多14次错误。不能因此给这76条加权或按已知注释修改预测；注释是诊断信息，新样本通常没有该注释。旧OOF还包含全局校准/选择影响，不能把这些计数当新方法的无偏泛化评估。',
'',
'这76条里25条有旧否定/弱活性提醒。注释提到溶血也可能是在说“溶血很弱”，或同时讨论其他机制。除去片段与跨来源标签冲突标记后仍有38条、26次融合漏判；这是敏感性描述，不是38条金标准。Ectatomin这样的分子完整性问题也可能没有被简单fragment字段捕获。',
'','## 正确对照与证据强度','',
'已保存全部33条正确候选，没有只分析43条错题。但这些正确样本尚非43条错题的逐条匹配实验对照。复用的27对旧背景对照中，5对共享膜作用线索、4对共享溶血文字线索。膜组仅2对两边均无简单片段/标签冲突标记：Ectatomin A/B仍有完整分子问题，H1/同家族仍有化学状态争议。其他三对含片段或标签冲突。旧对照不能充当可靠机制验证。',
'',
'73行公开溶血分子形式关联折叠到68个训练样本：原阳性7条全为融合FN，原阴性61条为59TN、2FP。没有TP，不能用这批算出“溶血阳性漏判率100%”；它只覆盖旧筛选范围，实验测量值也不是统一溶血阳性定义。按序列重叠不代表化学形式严格一致。',
'',
'旧人工记录中已查到直接溶血测量的6个样本为94、249、444、463、1434、3912。前5个原阳性均融合漏判，但病例原本按问题选择，不能估计总体错误率；3912原阴性却有宿主细胞作用，不能作为安全阴性对照。Nc2a等弱溶血也不能自动重新标成同一强度。1434有两篇原文记录，按样本只计一次。288的2a直接溶血对照认定仍撤回，不沿用2b的曲线。',
'',
'显式二硫键注释85条、化学特征记录62条，是“详细缓存存在”标记，缓存主要围绕错误样本补全；它们的高错误占比不能作为拓扑/修饰的致错证据。Conantokin旧补齐已证明正确与错误肽均有Gla，不重启Gla有无纠错分支。',
'','## 接下来的具体任务与门槛','',
'1. **优先膜相关分类诊断，固定76条与33条正确比较池。** 将功能原句区分为肯定效应、弱/无效应、对象不对应和未知。复用已有完整文献与身份缓存，保留全部样本，不以Cys或长度过滤。14条融合拖错和28条两路都漏分开研究，不能只救14条后声称补齐共同机制。',
'2. **对这些候选复用已有空间电荷/疏水与结构特征。** 只做在同标签、可比背景下的增量信息检验；必须同时纳入正确池，并检查普通序列理化特征能否解释。前面旧52对/27对已完成的检验不重复，新增仅围绕新冻结的整类问题与可用对照。若缺严格对照，结果只能探索，不能宣称机制发现。',
'3. **有可重复的额外信息才做一次小规模分类试验。** 目标是原毒性标签，比较同协议简单序列/结构特征基线；同时看整体验证MCC/AUPRC、膜候选子组和其余样本，防止只救一类而损失全局。分类阈值、调参与候选选择仅在内层，外层按群留出。注释、原文结果、溶血实测数值和错误分组不作为推理输入。暂不指定新网络。',
'4. **再决定三维改法。** 如果空间关系提供稳定增量才考虑针对性编码；如只有融合问题，先用已做路由/阈值诊断作比较，不把复杂门控当默认方案。已做严格OOF的拓扑方案未胜过匹配基线，不能换个名字重复；最后用独立评估验证分类收益。',
'',
'通道类别395条原阳性只有21条融合漏判，当前不优先全面改通道分支；受体类140条有27次漏判可列第二候选。凝血类原阳性112条中仅9条无简单片段/冲突标记，先处理对象边界。以上只是按现有诊断选择工作顺序，非不同生物机制性能的公平排名。',
'','## 产物与复核','',
'all_training_category_index.json为6387条逐条证据及原概率/旧模式；category_matrix.json为按原标签分层矩阵；membrane76_comparison_pool.json含固定76条、33正确池、14/28/1诊断清单；existing_pairs_category_overlay.json复用27对而不重新匹配；verification.json记录输入哈希与0次训练。所有类别是旁表，未改原标签、模型或数据集。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
assert len(pool['priority14_fusion_harms_ids'])==14 and len(pool['priority28_both_wrong_ids'])==28
print('Report and complete 76-sample comparison pool written; partition 14+28+1+33 verified.')
