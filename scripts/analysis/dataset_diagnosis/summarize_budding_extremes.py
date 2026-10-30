#!/usr/bin/env python3
"""Compare random and designed in-support Budding extremes with real trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',required=True);p.add_argument('--targeted',required=True)
    p.add_argument('--reference',required=True)
    args=p.parse_args();pilot=Path(args.pilot);targeted=Path(args.targeted)
    assert (pilot/'completion.json').exists() and (targeted/'completion.json').exists()
    ref=json.loads(Path(args.reference).read_text());real=np.array(ref['real_maxima'])
    manifest=json.loads((pilot/'manifest.json').read_text());variables=manifest['variables']
    random=read_lines(pilot/'diagnostics.jsonl');design=read_lines(targeted/'diagnostics.jsonl')
    candidates=read_lines(targeted/'candidates.jsonl')
    rr=[r for r in random if r['status']=='success'];dd=[r for r in design if r['status']=='success']
    maxima=np.array([r['maxima'] for r in rr]);ratios=maxima/real
    columns=[];examples=[]
    for i,var in enumerate(variables):
        ordinary=max((r for r in dd if r['scenario']=='one_high_q90'),key=lambda r:r['maxima'][i])
        extreme=max(dd,key=lambda r:r['maxima'][i])
        row=dict(variable=var,real_max=real[i],random_median=np.median(maxima[:,i]),random_p99=np.quantile(maxima[:,i],.99),
            random_max=maxima[:,i].max(),random_max_ratio=ratios[:,i].max(),random_above_real=int((ratios[:,i]>1).sum()),
            random_above_2x_real=int((ratios[:,i]>2).sum()),q90_case_peak=ordinary['maxima'][i],
            q90_case_ratio=ordinary['maxima'][i]/real[i],q90_case_skeleton=ordinary['skeleton'],q90_case_profile=ordinary['profile'],
            diagnostic_max=extreme['maxima'][i],diagnostic_max_ratio=extreme['maxima'][i]/real[i])
        columns.append(row)
        examples.append(dict(variable=var,**ordinary,candidate=candidates[ordinary['index']]))
    table=pd.DataFrame(columns);table.to_csv(pilot/'variable_comparison.csv',index=False)
    (pilot/'q90_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    negative=[]
    for directory,rows in ((pilot,random),(targeted,design)):
        y=np.load(directory/'trajectories.npy',mmap_mode='r')
        for r in rows:
            if r['status']!='success' or min(r['minima'])>=0:continue
            trajectory=y[r['index']];v,t=np.unravel_index(np.argmin(trajectory),trajectory.shape)
            negative.append(dict(arm=directory.name,index=r['index'],name=r['name'],profile=r['profile'],level=r['level'],
                minimum=float(trajectory[v,t]),variable=variables[v],time_min=float(t*210/499),negative_points=r['negative_points']))
    pd.DataFrame(negative).to_csv(pilot/'negative_samples.csv',index=False)
    runtime={}
    for label,rows in (('random',random),('targeted',design)):
        times=np.array([r['wall_seconds'] for r in rows])
        runtime[label]=dict(total=len(rows),success=sum(r['status']=='success' for r in rows),timeouts=sum(r['status']=='timeout' for r in rows),
                           p50=float(np.median(times)),p95=float(np.quantile(times,.95)),p99=float(np.quantile(times,.99)),maximum=float(times.max()))
    summary=dict(runtime=runtime,negative_rows=len(negative),minimum=min((r['minimum'] for r in negative),default=0),
        random_above_2x_any_nonmass=int((ratios[:,1:]>2).any(axis=1).sum()),
        profiles={profile:dict(count=len(rows),maxima=np.max([r['maxima'] for r in rows],axis=0).tolist())
                  for profile in ('mass','mutant_aware') if (rows:=[r for r in rr if r['profile']==profile])})
    (pilot/'comparison_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    selected=['CLB2','SIC1','CDC6','CDC14','CDC14T','TEM1','CDC15','ORI','BUD','SPN']
    display=table.set_index('variable').loc[selected]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');x=np.arange(len(display))
    ax.bar(x-.18,display.random_max_ratio,.36,label='Random pilot maximum / real maximum')
    ax.bar(x+.18,display.q90_case_ratio,.36,label='One high Q90 diagnostic / real maximum')
    ax.axhline(1,color='black',ls='--',lw=1);ax.set_xticks(x,display.index)
    ax.set_ylabel('Ratio to real-mutant channel maximum');ax.legend(fontsize=9)
    ax.set_title('In-support Budding extremes (diagnostic genotypes are not frequency estimates)')
    fig.savefig(pilot/'comparison.png',dpi=180);fig.savefig(pilot/'comparison.pdf');plt.close(fig)
    convergence=json.loads((targeted/'convergence_checks.json').read_text())
    lines=['# Budding 分布内大值实验（2026-10-22）','',
    '本次不修改生产代码、现有分布、ODE、初值或 ours 策略。仅在独立内存函数中去掉 terminal MASS 拒绝，外层使用 30 秒标定截止；不加浓度阈值、负值拒绝或 ours 参数门控。数值失败仍拒绝。',
    '',f'当前两套分布各 2500 条随机样本，L1/L2/L3 = 5/40/55；另有 {len(design)} 组诊断组合，覆盖 29 种合法骨架、两套分布和 median/one-high-Q90/joint-Q90/joint-Q95/joint-Q99。',
    '', '诊断组合记录每个参数的分布 CDF 坐标；one-high-Q90 只将一个表达强度设到第90百分位，其余可变强度取中位数。骨架、KO和复合模式本身仍需条件选择，因此这不是整组组合的发生概率。随机试验与诊断设计不能混算为自然发生率。',
    '', 'real 最大值来自同一模型、210分钟、500点的历史126 real轨迹。不同变量分别比较，不把状态指标与蛋白浓度混为一种单位。','',
    '| 变量 | real最大值 | 随机试验最大值 | 单参数Q90案例最大值 | Q90/real |','|---|---:|---:|---:|---:|']
    for var,row in display.iterrows():lines.append(f'| {var} | {row.real_max:.6g} | {row.random_max:.6g} | {row.q90_case_peak:.6g} | {row.q90_case_ratio:.3f} |')
    lines+=['','## 耗时与负值','',json.dumps(summary['runtime'],ensure_ascii=False),
    '',f'负值样本（两组总计）：{len(negative)}；最小值：{summary["minimum"]:.9g}。`negative_samples.csv` 逐条记录幅度、变量、时间点和负值点数，不作为拒绝条件。',
    '',f'随机样本中任一非MASS变量超过2倍real最大值：{summary["random_above_2x_any_nonmass"]}/{len(rr)}。',
    '', '## 加严容差复算','',f'对 {len(convergence)} 个Q90代表案例，将 rtol/atol 从1e-4/1e-6收紧到1e-7/1e-9；最大所选峰值相对变化为 {max(r["relative_peak_change"] for r in convergence):.6g}。详见旁边targeted目录的convergence_checks.json。这支持这些案例的大幅值不是单纯由默认积分精度造成，但不证明所有组合都有实验生物学依据。',
    '', '## 用途与限制','',
    '本实验支持不以2倍real最大值作为通用拒绝条件。保留超时、数值失败等拒绝；保留大幅值和负值诊断。候选超时阈值需要考虑并发负载与更广采样，不能将本次30秒标定上限误写成已验证最优阈值。',
    '', '未进行神经网络训练，因此不能据此证明移除幅值过滤提高或降低real预测表现。应使用相同初始集、预算、种子与候选顺序，比较有/无幅值过滤，再报告real与分组误差；ours门控当前保持不变。',
    '', '原始轨迹：trajectories.npy（随机5000条）；targeted目录另存290条。参数覆盖在candidates.jsonl，逐条耗时、状态与变量极值在diagnostics.jsonl。所有生产文件哈希已在completion.json记录核验结果。']
    (pilot/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(display[['real_max','random_max','q90_case_peak','q90_case_ratio']].to_string())
    print(json.dumps(summary['runtime'],ensure_ascii=False))


if __name__=='__main__':main()
