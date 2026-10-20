#!/usr/bin/env python3
"""Join AL label-utility checks and the observed training coverage census."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.summarize_six_pattern_audit import markdown_table

CODES=['P1','P2','P3','P4','P5a','P5b']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir',type=Path,required=True)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();base=args.analysis_dir;comp=base/'label_comparison';snap=base/'training_snapshots'
    if json.loads((snap/'errors.json').read_text()):raise ValueError('Incomplete training census')
    frame=pd.read_csv(comp/'all_real_evaluations.csv.gz');primary=frame[frame['round'].eq(2)].copy()
    strategies=sorted(primary.strategy.unique())
    train_counts=pd.read_csv(snap/'training_pattern_counts.csv')
    cumulative=train_counts[train_counts.population.eq('synthetic')].copy()
    cumulative.to_csv(base/'training_coverage_by_round.csv',index=False)
    counts=cumulative[cumulative['round'].eq(2)].pivot(index='strategy',columns='pattern_code',values='count')[CODES]
    # Compare every cumulative snapshot prefix before interpreting differences
    # as newly acquired data rather than reordering/resampling old examples.
    added=[];support=[]
    for directory in sorted(snap.glob('al4pde_*')):
        previous=None
        for file in sorted(directory.glob('al_iter*/samples.csv.gz')):
            f=pd.read_csv(file);f=f.loc[~f.is_real].reset_index(drop=True)
            round_id=int(file.parent.name.split('_')[1].replace('iter',''))
            offset=0
            if previous is not None:
                offset=len(previous);prefix=f.iloc[:offset]
                for col in ('mutant_name','original_pattern','pattern_code','is_bifurcation_critical'):
                    if prefix[col].tolist()!=previous[col].tolist():raise ValueError(f'Changed snapshot prefix: {file}, {col}')
                np.testing.assert_allclose(prefix[['max_clb2','max_ori','max_spn']],
                    previous[['max_clb2','max_ori','max_spn']],rtol=1e-12,atol=1e-12)
            new=f.iloc[offset:]
            for code in CODES:
                part=new[new.pattern_code.eq(code)]
                added.append(dict(strategy=directory.name,round=round_id,pattern_code=code,
                    count=len(part),critical_count=int(part.is_bifurcation_critical.sum()),total_added=len(new)))
            z=pd.read_csv(file.parent/'zero_face_support.csv')
            z['strategy']=directory.name;z['round']=round_id;support.append(z)
            previous=f
    pd.DataFrame(added).to_csv(base/'new_acquisition_patterns.csv',index=False)
    supports=pd.concat(support,ignore_index=True)
    supports.to_csv(base/'zero_parameter_face_support_by_round.csv',index=False)
    support2=supports[supports['round'].eq(2)]
    coverage=[]
    for ident,code in (('15','P3'),('22','P2'),('42','P2'),('124','P5b')):
        sub=primary[primary.mutant_name.str.split('_').str[0].eq(ident)]
        z=support2[support2.mutant_name.str.split('_').str[0].eq(ident)].set_index('strategy')
        for row in sub.itertuples():
            coverage.append(dict(mutant_id=ident,mutant_name=row.mutant_name,strategy=row.strategy,
                pattern_code=code,train_same_pattern=int(counts.loc[row.strategy,code]),
                train_same_zero_parameter_face=int(z.loc[row.strategy,'train_same_zero_parameter_face']),
                MAE=row.MAE,MSE=row.MSE,Relative_L2=row.Relative_L2))
    pd.DataFrame(coverage).to_csv(base/'key_badcase_coverage_and_error.csv',index=False)
    counts.to_csv(base/'round2_training_six_patterns.csv')
    # Budget differences are recorded, not silently treated as matched cost.
    budgets=primary.groupby('strategy').train_samples.first()
    benchmark=primary.groupby('strategy')[['MAE','MSE','Relative_L2','Correlation']].mean()
    benchmark.insert(0,'train_samples',budgets);benchmark.to_csv(base/'round2_per_mutant_macro_errors.csv')
    utility=pd.read_csv(comp/'round2_crossvalidated_utility.csv')
    table=utility[utility.scheme.ne('GLOBAL')].pivot(index='scheme',columns='metric',values='rmse_improvement_vs_ABC_percent')
    table=table[['MAE','MSE','Relative_L2','composite_error_rank']].round(2)
    table.to_csv(base/'label_utility_quick_comparison.csv')
    flags=pd.read_csv(comp/'critical_flag_badcase_coverage.csv')
    persistent=pd.read_csv(comp/'persistent_top20.csv')
    small=persistent[['mutant_name','top20_frequency','pattern_code','median_Relative_L2']].copy()
    small['mutant_id']=small.mutant_name.str.split('_').str[0]
    small=small[['mutant_id','top20_frequency','pattern_code','median_Relative_L2']].round(4)
    p3=primary[primary.pattern_code.eq('P3')].groupby('mutant_name').Relative_L2.agg(['median','min','max'])
    p3.to_csv(base/'p3_within_group_relative_errors.csv')
    boot=pd.read_csv(comp/'round2_mutant_cluster_bootstrap.csv')
    perm=pd.read_csv(comp/'cardinality_matched_permutation_tests.csv')
    def result(scheme,parent,metric):
        v=boot[boot.scheme.eq(scheme)&boot.parent.eq(parent)&boot.metric.eq(metric)].iloc[0]
        return f'{v.rmse_improvement_percent:+.2f}%（条件95%区间{v.ci_low:+.2f}%至{v.ci_high:+.2f}%）'
    def pvalue(comparison,metric):
        return perm.loc[perm.comparison.eq(comparison)&perm.metric.eq(metric),'p_value'].iloc[0]
    retrieval=pd.read_csv(comp/'round2_top20_retrieval.csv').groupby('scheme').expected_top20_hits.mean()
    retrieval['uniform_random_20_of_126']=400/126
    retrieval.to_csv(base/'expected_top20_hits_with_random_control.csv')
    manifest=json.loads((comp/'evaluation_manifest.json').read_text())
    quality=pd.read_csv(snap/'snapshot_quality.csv')
    for scheme in ('SIX','ABC_BIO','SIX_CRITICAL'):
        assert scheme in table.index
    text=['# 主动学习实验：新旧 Pattern 标签是否更有用','',
        '**结论：六分类有助于区分绝对误差风险，但没有显示全面优于现有标签组合；当前临界标记没有附加收益。不能将本统计解释为新标签已经提高了主动学习精度。**','',
        '## 1. 实际纳入了哪些结果','',
        f'- 10种策略目录；第0轮评测完全相同，去重后保留1份。第1轮有8种策略，第2轮有7种。',
        '- `bifurcation`仅到第1轮，`event_entropy`和`fisher_oed`只有第0轮，不能作为完成后的策略比较；`seed_42_failed_v1`单列排除。',
        '- 根目录的7份最终评测与对应轮次重复，不重复计数。原始路径和SHA256见`label_comparison/evaluation_manifest.json`。',
        f'- 实际读取25个物理训练快照、{int((quality.n_real+quality.n_synthetic).sum()):,}行。每个快照的126个real按名字与整条float32轨迹SHA256核对，全部等于canonical MA目标；real均排除在训练覆盖计数外。',
        f'- 主比较是第2轮7种策略，训练量为{int(budgets.min()):,}–{int(budgets.max()):,}。预算不齐，仅用于检查标签的跨策略诊断能力，不据此宣称某个AL算法在同成本下更优。',
        '- 全部是seed42，初始模型和基准共享；不同策略、轮次不是新的独立生物样本。仅使用逐mutant CSV，未混入JSON中的pooled relL2/correlation。','',
        '## 2. 用什么标准判断“优于”','',
        '每次评测内，将MAE、MSE、relL2转为0–1误差名次，三者等权平均为主综合指标。只用标签预测该误差风险：留出目标AL算法，同时按突变体身份做5折留出、重复10次；同一个被留出的突变体在所有其他算法中的误差也禁止参与拟合。',
        '用其他算法、其他突变体的组均值预测，细组向父组收缩5个突变体的权重；未细分的P1/P2严格继承A/B预测，防止额外正则化带来假收益。另报告0/2/5/10收缩敏感性。',
        '以下数值是**误差名次预测RMSE相对ABC的降低百分比**，正数更好、负数更差；它们不是神经网络MAE/MSE/relL2降低了多少。BIO是现有生物存活标签，126个real实际出现5类；ABC×BIO出现7组。','',
        markdown_table(table.reset_index()),'',
        f'六分类相对ABC：MAE风险预测{result("SIX","ABC","MAE")}；MSE {result("SIX","ABC","MSE")}；relL2 {result("SIX","ABC","Relative_L2")}；综合{result("SIX","ABC","composite_error_rank")}。正数改善，负数恶化。',
        f'六分类单独替换现有ABC×BIO：综合{result("SIX","ABC_BIO","composite_error_rank")}，不能证明胜出。更公平的补充对照是保留BIO、只换pattern字段：六分类×BIO对ABC×BIO的MSE风险预测{result("SIX_BIO","ABC_BIO","MSE")}；relL2 {result("SIX_BIO","ABC_BIO","Relative_L2")}；综合{result("SIX_BIO","ABC_BIO","composite_error_rank")}。仍然没有全面优势。',
        f'类别数量对照：在ABC内部随机打乱六分类、保持各组大小，条件置换中，MAE/MSE的p分别为{pvalue("SIX_vs_ABC","MAE"):.3f}/{pvalue("SIX_vs_ABC","MSE"):.3f}，主综合p={pvalue("SIX_vs_ABC","composite_error_rank"):.3f}。在ABC×BIO内部保持分组大小随机细分，六分类×BIO的综合p={pvalue("SIX_BIO_vs_ABC_BIO","composite_error_rank"):.3f}。这不消除在同一126基准上提出阈值的事后选择偏差；保留BIO的组合对照为补充分析。',
        '区间按126个突变体聚类重采样已生成的交叉验证误差，条件于当前策略/划分；不是多seed独立重复训练的区间。','',
        '## 3. 临界标记与坏例','',
        f'7个真实条件被标True。7种第2轮策略各取20个坏例，共140个“策略×坏例”记录，标记命中{int(flags.top20_critical_hits.sum())}个（{100*flags.top20_critical_hits.sum()/140:.2f}%）；重复mutant不视为独立样本。#15/#22/#42/#124一直为False。',
        f'覆盖率低本身不能否定物理临界性：被标记群体原本也只占7/126。更直接的证据是，六分类再加该标记，主综合风险预测变化为{result("SIX_CRITICAL","SIX","composite_error_rank")}；相同大小的随机标记对照也未支持额外收益。不要把该标记与名字含bifurcation的现有AL策略混为一谈；这些历史实验未使用本轮新标记。','',
        '## 4. 稳定坏例：新的大类仍然太粗','',markdown_table(small),'',
        '#15和#124在7/7策略中进入Top20，#22/#42为6/7。#15的relL2范围6.11–9.00；同P3另外6个条件的各自中位relL2仅0.0089–0.0898，因此不能把整个P3都视为同等困难。',
        f'严格留出检验中，仅靠六分类挑20条的平均预期命中数为{retrieval["SIX"]:.2f}/20，ABC为{retrieval["ABC"]:.2f}/20；均匀随机选择20条的预期为{400/126:.2f}/20。六分类用于定位具体坏例的绝对能力仍很有限，不能只强调相对ABC的比例提升。','',
        'P2与原B完全相同，不能细分#22/#42与其他单次分裂条件；P1与原A也相同。六分类主要增加的是零分裂C内部的信息。',
        'P5b的合并逐样本中位MSE为1.414，而P5a为0.032；但P5b中位relL2只有0.0367。它更像幅值/绝对误差风险组，不能直接称为普遍的波形预测崩塌。','',
        '## 5. 实际训练覆盖','',markdown_table(counts.reset_index()),'',
        '#15所在P3已存在1,696–7,393条训练样本，其精确零参数面支持数为15–62，仍普遍失败；“没有P3”不能解释这些坏例。',
        '#124在所有7种策略中的精确零参数面支持数均为0，即使residual已有9,457条P5b；random有2,978条P5b，#124的MSE反而更低（9.51 vs 43.15）。这不是公平预算下的算法因果比较，但说明大类样本数不够描述局部覆盖。',
        '#42的精确零参数面支持数：6种策略为0、uncertainty为2。零面数只表示部分坐标恰为0，不保证其他参数/初值接近；缺少精确0也不证明近零样本无用，需结合邻域距离和扰动响应。',
        '所有累积快照的既有样本前缀经名称、原标签、新标签、临界标记和Clb2/ORI/SPN最大值核对一致；新增样本计数见`new_acquisition_patterns.csv`。','',
        '## 6. 生物学合理性与修改建议','',
        '- P2的“衰竭”命名已被上轮长时积分反例否定，先用“当前窗口单次分裂”。P1也仅能称为窗口多次分裂，不能直接证明稳定极限环。',
        '- P4的5个real中，#37/#77/#110的ORI最大值分别约0.712/0.644/0.540，未达到模型的DNA起始事件阈值1；“DNA已复制”的解释不成立。#16/#35达到了ORI=1。后续机制标签应明确是否跨过真实ORI/SPN事件，DNA起始也不等于复制完成。',
        '- #26因终点Esp1低被分为P5a，但Clb2从未超过约0.147；“进入M期后退出失败”并不能由此证明。保留阈值分箱与机制证据两个字段。',
        '- 建议保留ABC与生物存活标签，在此之外加入中性六分类；将高Clb2/高ORI作为跨所有分裂类的幅值标记，加入首次事件时间、事件顺序/缺失、窗口末尾是否仍在变化、参数/初值邻域覆盖。暂不把当前布尔区间标记当成分岔证书。',
        '- 对#15这类真值轨迹与部分简单G1样本近似、输入映射却难学的条件，还要检查参数条件化、假激活与局部敏感性；单靠继续细分真实输出pattern未必能解决。','',
        '## 7. 还缺少什么证据','',
        '现有实验都使用原标签，不能反推“新标签驱动AL已经胜出”。直接验证需在相同初始训练集/候选池、相同成功ODE标注预算下，对照ABC、ABC+BIO、六分类、六分类+经过验证的附加标记；固定归一化政策，至少多个seed，并另留未参与标签设计的基因型/条件作为评测。分别看MAE/MSE/relL2、尾部误差和事件预测，而不只看类别是否均衡。',
        '本轮未改生成、采样、损失、训练或原评测结果。所有CSV与统计图均是独立分析产物。','']
    (base/'REPORT.md').write_text('\n'.join(text))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        selected=table.loc[['BIO','ABC_BIO','SIX','SIX_CRITICAL','SIX_BIO']]
        for i,(scheme,row) in enumerate(selected.iterrows()):
            ax.bar(np.arange(4)+(i-2)*.16,row.to_numpy(),width=.16,label=scheme)
        ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(np.arange(4),['MAE rank','MSE rank','relL2 rank','Composite rank'])
        ax.set_ylabel('Reduction in held-out risk-rank RMSE vs ABC (%)')
        ax.set_title('Diagnostic label utility — not surrogate-model error improvement')
        ax.legend(ncols=5,fontsize=8);fig.tight_layout();fig.savefig(base/'label_utility.png',dpi=180);fig.savefig(base/'label_utility.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.4,1]})
        left=np.zeros(len(counts));colors=['#2c7bb6','#74add1','#abd9a9','#fee090','#fdae61','#d73027']
        for code,color in zip(CODES,colors):
            width=counts[code].to_numpy()/counts.sum(1).to_numpy()*100
            axes[0].barh(np.arange(len(counts)),width,left=left,label=code,color=color);left+=width
        short=[s.removeprefix('al4pde_') for s in counts.index]
        axes[0].set_yticks(np.arange(len(counts)),short);axes[0].invert_yaxis();axes[0].set_xlim(0,100)
        axes[0].set_xlabel('Round-2 synthetic training rows (%)');axes[0].legend(ncols=6,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.12))
        error15=primary[primary.mutant_name.str.startswith('15_')].set_index('strategy').loc[counts.index,'Relative_L2']
        axes[1].barh(np.arange(len(counts)),error15.to_numpy(),color='#b2182b');axes[1].invert_yaxis()
        axes[1].set_yticks(np.arange(len(counts)),short);axes[1].set_xlabel('Mutant #15 relative L2');axes[1].set_xlim(0,10)
        fig.tight_layout();fig.savefig(base/'training_coverage_and_mutant15.png',dpi=180);fig.savefig(base/'training_coverage_and_mutant15.pdf');plt.close(fig)
    print(table.to_string());print(f'Wrote {base / "REPORT.md"}')


if __name__=='__main__':main()
