from pathlib import Path
import json
import pandas as pd
OUT=Path(__file__).resolve().parent

def run():
    stats=pd.read_csv(OUT/'association_statistics.csv')
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf8'))
    adj=pd.read_csv(OUT/'class_adjusted_association.csv')
    text=['# Test1 / Test2：待核查证据与预测错误的关联','',
        '结论：按预先固定的两种证据标记初筛，Test1的错题中有标记者明显富集；Test2总体比较证据较弱。按原有毒/无毒类别分层后，两套均提示关联。这支持优先核查数据，不能证明标签错误，也不能证明清理数据重训后一定提高。','',
        '## 定义与分析边界','',
        '- “有待核查标记”：精确匹配当前UniProt Fragment条目，或相同序列在已固定的其他来源中出现相反的广义/操作性二分类标签。两类取并集，重叠只算一次。',
        '- 片段标记提示完整分子与输入短序列可能不一致，不等于标签一定错误。相反标签也可能来自实验对象、终点或条件不同。',
        '- “未发现上述标记”不等于实验证明清楚；本轮未完成逐条论文全文、物种、剂量、时间与化学修饰核查。未匹配到当前数据库身份者单独报告，主分析没有自动把“未查到”算成“有问题”。',
        '- 标记规则不使用模型预测概率或对错；预测来自已保存的 canonical 单结构融合结果。固定阈值不变，没有重训、删除、重标或新推理。',
        '- Test2全部包含在Test1中，同序列的原标签、标记和预测决策一致。以下为分别描述，不能合并总数或视为独立重复验证。','']
    for s in summary:
        split=s['split'];r=stats[(stats.split==split)&(stats.definition=='primary_review_flag')].iloc[0];a=adj[adj.split==split].iloc[0]
        text += [f'## {split.upper()}','',
            '| 证据状态 | 模型做错 | 模型做对 | 合计 | 错误率 |',
            '|---|---:|---:|---:|---:|',
            f'| 有待核查标记（初筛不确定） | {int(r.flagged_error)} | {int(r.flagged_correct)} | {int(r.flagged_n)} | {100*r.flagged_error_rate:.2f}% |',
            f'| 未发现上述标记 | {int(r.unflagged_error)} | {int(r.unflagged_correct)} | {int(r.unflagged_n)} | {100*r.unflagged_error_rate:.2f}% |',
            f'| 合计 | {s["errors"]} | {s["n"]-s["errors"]} | {s["n"]} | {100*s["errors"]/s["n"]:.2f}% |','',
            f'- 全测试集有标记：{int(r.flagged_n)}/{s["n"]} = **{100*r.flagged_fraction:.1f}%**。',
            f'- 错题中有标记：{int(r.flagged_error)}/{s["errors"]} = **{100*r.flagged_fraction_of_errors:.1f}%**。',
            f'- 有标记者错误率是另一组的 **{r.error_risk_ratio:.2f}倍**，95%置信区间 {r.risk_ratio_CI95_low:.2f}–{r.risk_ratio_CI95_high:.2f}。',
            f'- 双侧Fisher检验p={r.fisher_two_sided_p:.6g}；两套主检验的Holm校正p={r.primary_two_tests_Holm_p:.6g}。',
            f'- 组成：片段标记{s["fragment"]}条、来源标签相反{s["label_disagreement"]}条、两者重叠{s["both_flags"]}条，并集{s["primary_flagged"]}条。',
            f'- 当前身份未恢复{s["unresolved_identity"]}条，其中{s["unresolved_without_primary_flag"]}条没有上述两种主标记，不能据此确认其标签可靠。',
            f'- 原标签类别分层后的合并优势比OR={a.pooled_odds_ratio:.2f}，95%CI {a.CI95_low:.2f}–{a.CI95_high:.2f}，CMH p={a.CMH_p:.6g}。OR不是错误率倍数。','']
    text += ['## 为什么Test2分层前后结论不同','',
        'Test2有毒46条、无毒536条，比例很不均衡。待核查标记在无毒样本中更多，而有毒样本本身更容易漏判，直接混在一起会遮住一部分关系。','',
        '| 数据集与原标签 | 有标记：错/总数 | 无标记：错/总数 | 错误率比较 |',
        '|---|---:|---:|---|',
        '| Test1 无毒 | 18/279 | 8/527 | 6.45% vs 1.52% |',
        '| Test1 有毒 | 15/65 | 18/255 | 23.08% vs 7.06% |',
        '| Test2 无毒 | 13/206 | 7/330 | 6.31% vs 2.12% |',
        '| Test2 有毒 | 2/5 | 9/41 | 40.00% vs 21.95% |','',
        'Test2有标记的有毒样本只有5条，不能据此声称在所有毒肽类别都存在稳定关系。分层结果为探索性敏感性分析，不替代主检验，也没有控制家族、来源或相似序列的相关性。','',
        '## 两个标记分开看与未恢复身份的敏感性分析','',
        '| 定义 | Test1：标记数/错题中有标记数 | Test1 p | Test2：标记数/错题中有标记数 | Test2 p |',
        '|---|---:|---:|---:|---:|']
    for col,label in [('fragment_review_flag','仅片段标记'),('cross_source_label_disagreement','仅来源标签相反'),('identity_not_recovered','仅当前身份未恢复'),('expanded_review_flag','主标记或身份未恢复')]:
        r1=stats[(stats.split=='test1')&(stats.definition==col)].iloc[0];r2=stats[(stats.split=='test2')&(stats.definition==col)].iloc[0]
        text.append(f'| {label} | {int(r1.flagged_n)}/{int(r1.flagged_error)} | {r1.fisher_two_sided_p:.5g} | {int(r2.flagged_n)}/{int(r2.flagged_error)} | {r2.fisher_two_sided_p:.5g} |')
    text += ['', '以上单标记与扩大范围结果均为补充分析，不能选择最显著的一项充当预先规定的结论。身份未恢复是资料缺失，不是确认标签含混；主分析仍使用原来的双标记并集。','',
        '## 对下一步的实际意义','',
        '优先复核Test1的33条有标记错题，以及有标记但模型做对的对照样本；Test2的15条属于同一批历史序列，避免重复算研究证据。先核对完整肽/片段、原始效应和实验条件，不按模型意见改标签。若后续据这些案例修改方法，历史测试结果只能作为开发/回顾性结果，最终方法仍需新留出的独立评价。','',
        '本次分析需要的只是保存的预测与证据表，没有使用训练服务器。结果表明数据证据问题值得核查，但尚不足以决定整体更换数据集；更不能把剔除后重新计算的分数解释为模型能力提升。','',
        '## 可复核文件','',
        '- test1_2x2.csv / test2_2x2.csv：用户请求的两张表。',
        '- test1_sample_audit.csv / test2_sample_audit.csv：逐样本证据标记、accession、原标签与固定预测。',
        '- evidence_flags_before_outcome_join.csv / protocol.json：不依赖预测结果的规则和证据表。',
        '- association_statistics.csv：主分析与补充分析；class_stratified_association.csv、class_adjusted_association.csv：类别分层。',
        '- input_manifest.json、verification.json：输入哈希及核对结果。',
        '- 原始来源与固定版本沿用 ../dataset_comparison_20261025/REPORT.md 和 ../ghxtox_error_atlas_20261025/evidence_phase2/EVIDENCE_REPORT.md。当前数据库身份恢复不是原始标签来源的完整重建。','']
    (OUT/'REPORT.md').write_text('\n'.join(text),encoding='utf8')

if __name__=='__main__':run()
