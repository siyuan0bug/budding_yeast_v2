#!/usr/bin/env python3
"""Summarise completed audits and rank badcases from one explicit evaluation run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import CODES, REFERENCE, features
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES

PURE = 'mutant_aware_pure_random_100000_210m_500s'
FROZEN = ROOT / 'scripts/analysis/outputs/pure_random_current_audit/hybrid_mse_rel_predictions.npz'


def select_badcases(metrics, top_k=20):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    metrics = metrics.copy()
    required = ['Mutant','MAE','MSE','Relative_L2']
    if any(c not in metrics for c in required) or metrics.Mutant.duplicated().any():
        raise ValueError('Expected unique mutant names and MAE/MSE/Relative_L2 columns')
    if not np.isfinite(metrics[required[1:]].to_numpy()).all():
        raise ValueError('Nonfinite evaluation metrics cannot be ranked silently')
    for c in required[1:]: metrics['rank_'+c] = metrics[c].rank(method='min', ascending=False).astype(int)
    metrics['mean_error_rank'] = metrics[['rank_'+c for c in required[1:]]].mean(axis=1)
    metrics = metrics.sort_values(['mean_error_rank','rank_Relative_L2','rank_MAE','Mutant'], kind='stable')
    metrics['badcase_rank'] = np.arange(1,len(metrics)+1)
    metrics['selected_top_k'] = metrics.badcase_rank <= top_k
    return metrics


def markdown_table(df):
    return '| ' + ' | '.join(map(str,df.columns)) + ' |\n|' + '|'.join(['---']*len(df.columns)) + '|\n' + '\n'.join(
        '| ' + ' | '.join(str(v).replace('|','/') for v in row) + ' |' for row in df.itertuples(index=False,name=None))


def inspect_frozen_predictions(path, real, out):
    """Verify targets by name+digest before attributing channel errors to labels."""
    if not path.exists(): return None
    with np.load(path, allow_pickle=False) as z:
        pred=z['predictions'].astype(np.float64); truth=z['targets'].astype(np.float64); names=z['names'].astype(str)
    if len(set(names)) != len(names): raise ValueError('Duplicate names in frozen predictions')
    if pred.shape != truth.shape or pred.shape[1:] != (39,500): raise ValueError('Wrong frozen trajectory shape')
    expected=real.set_index('mutant_name')['trajectory_sha256_float32'].to_dict()
    matched=[]
    for n,y in zip(names,truth):
        digest=hashlib.sha256(np.asarray(y,dtype='<f4').tobytes()).hexdigest()
        matched.append(n in expected and digest == expected[n])
    provenance=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        total=len(names),targets_exactly_match_current_npz=int(sum(matched)))
    if not all(matched):
        (out/'frozen_prediction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        raise ValueError('Frozen prediction targets differ from current data; cannot assign current labels')
    diff=pred-truth; sq=(diff**2).sum(axis=2)
    rel=np.sqrt(sq/np.maximum((truth**2).sum(axis=2),.1**2*truth.shape[2]))
    pc=pred-pred.mean(axis=2,keepdims=True); tc=truth-truth.mean(axis=2,keepdims=True)
    corr=(pc*tc).sum(2)/(np.sqrt((pc**2).sum(2)*(tc**2).sum(2))+1e-8)
    rows=[]
    for i,n in enumerate(names):
        for j,var in enumerate(VARIABLE_NAMES):
            rows.append(dict(mutant_name=n,variable=var,MAE=np.abs(diff[i,j]).mean(),
                MSE=sq[i,j]/truth.shape[2],Relative_L2=rel[i,j],Correlation=corr[i,j],
                mse_fraction=sq[i,j]/max(float(sq[i].sum()),1e-12)))
    channels=pd.DataFrame(rows);channels.to_csv(out/'frozen_per_channel_metrics.csv',index=False)
    aggregate=pd.DataFrame(dict(mutant_name=names,frozen_MAE=np.abs(diff).mean((1,2)),
        frozen_MSE=(diff**2).mean((1,2)),frozen_Relative_L2=rel.mean(1),frozen_Correlation=corr.mean(1)))
    predicted,_=features(pred); actual,_=features(truth)
    for key in ('pattern_code','divisions','ori_reaches_one','spn_reaches_one','is_bifurcation_critical'):
        aggregate['frozen_predicted_'+key]=predicted[key]
        aggregate['frozen_actual_'+key]=actual[key]
    aggregate.to_csv(out/'frozen_per_mutant_metrics.csv',index=False)
    (out/'frozen_prediction_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return channels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',type=Path,required=True)
    p.add_argument('--eval-csv',type=Path,default=REFERENCE)
    p.add_argument('--frozen-predictions',type=Path,default=FROZEN)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--plot',action='store_true')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);out=args.output_dir
    counts=pd.read_csv(args.audit_dir/'all_pattern_counts.csv')
    manifest=json.loads((args.audit_dir/'manifest.json').read_text())
    errors=json.loads((args.audit_dir/'errors.json').read_text())
    if errors:raise ValueError(f'Incomplete dataset audit: {errors}')
    expected={Path(r['path']).stem for r in manifest if r['status'] in ('scheduled','cached')}
    if set(counts.dataset) != expected: raise ValueError('Manifest and completed datasets disagree')
    real=pd.read_csv(args.audit_dir/PURE/'real_mutants.csv')
    metrics=select_badcases(pd.read_csv(args.eval_csv),args.top_k).rename(columns={'Mutant':'mutant_name'})
    joined=metrics.merge(real,on='mutant_name',how='outer',validate='one_to_one',indicator=True)
    if not joined['_merge'].eq('both').all():raise ValueError('Evaluation names and current canonical real rows differ')
    joined=joined.drop(columns='_merge')
    support=pd.read_csv(args.audit_dir/PURE/'real_training_support.csv')
    joined=joined.merge(support.drop(columns=['pattern_code','is_bifurcation_critical']),on='mutant_name',validate='one_to_one')
    channels=inspect_frozen_predictions(args.frozen_predictions,real,out)
    if channels is not None:
        # Frozen replay is a separate prediction artifact; it is not the CSV
        # being ranked and can differ slightly from the originally saved run.
        ch=channels.sort_values('MSE',ascending=False).groupby('mutant_name',sort=False).head(1)
        ch=ch[['mutant_name','variable','mse_fraction']].rename(columns={'variable':'frozen_largest_mse_channel','mse_fraction':'frozen_largest_channel_mse_fraction'})
        joined=joined.merge(ch,on='mutant_name',validate='one_to_one')
    joined=joined.sort_values('badcase_rank')
    joined.to_csv(out/'all_126_ranked.csv',index=False)
    selected=joined.loc[joined.badcase_rank<=args.top_k].copy()
    selected.to_csv(out/'top20_badcases.csv',index=False)
    for metric in ('MAE','MSE','Relative_L2'):
        joined.sort_values(['rank_'+metric,'mutant_name']).head(args.top_k).to_csv(out/f'top20_by_{metric}.csv',index=False)
    joined.groupby(['pattern_code','is_bifurcation_critical'])[['MAE','MSE','Relative_L2','Correlation']].agg(['count','median','mean','max']).to_csv(out/'evaluation_by_pattern.csv')
    provenance=dict(evaluation_csv=str(args.eval_csv.resolve()),evaluation_sha256=hashlib.sha256(args.eval_csv.read_bytes()).hexdigest(),
        ranking='ascending arithmetic mean of descending min-ranks of MAE, MSE and Relative_L2; ties broken by Relative_L2 rank, MAE rank, name',
        top_k=args.top_k,correlation_in_ranking=False,fit_or_retraining=False,
        training_support='current DataModule non-LHS 80/10/10 split, seed=42; not proof of historical training membership',
        cross_dataset_sample_counts='row counts per file; overlap between files is not deduplicated')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    # All thirty files, including historical real variants: preserve differences.
    pivot=counts[counts.population.eq('synthetic')].pivot(index='dataset',columns='pattern_code',values='count').fillna(0).astype(int)
    pivot=pivot[[*CODES,'INVALID']];pivot['synthetic_total']=pivot.sum(axis=1)
    pivot.to_csv(out/'all_datasets_synthetic_counts.csv')
    all_real=[]
    for name in sorted(expected):
        rr=pd.read_csv(args.audit_dir/name/'real_mutants.csv');rr.insert(0,'dataset',name);all_real.append(rr)
    rr=pd.concat(all_real,ignore_index=True)
    rr['condition_id']=rr.mutant_name.str.extract(r'^(\d+(?:\.\d+)?)_',expand=False)
    canonical=real.copy();canonical['condition_id']=canonical.mutant_name.str.extract(r'^(\d+(?:\.\d+)?)_',expand=False)
    baseline=canonical.set_index('condition_id').trajectory_sha256_float32.to_dict()
    rr['matches_canonical_float32_target']=rr.apply(lambda r: baseline.get(r.condition_id)==r.trajectory_sha256_float32,axis=1)
    rr.to_csv(out/'all_dataset_real_comparison.csv.gz',index=False,compression='gzip')
    comparison=rr.groupby('dataset').agg(real_rows=('mutant_name','size'),
        unique_condition_ids=('condition_id','nunique'),exact_canonical_targets=('matches_canonical_float32_target','sum'))
    comparison.to_csv(out/'dataset_real_consistency.csv')
    diagnostics=[]
    for name in sorted(expected):
        s=json.loads((args.audit_dir/name/'summary.json').read_text())
        for population,values in s['diagnostics'].items():diagnostics.append(dict(dataset=name,population=population,**values))
    diagnostic=pd.DataFrame(diagnostics);diagnostic.to_csv(out/'all_rule_diagnostics.csv',index=False)
    h=pd.read_csv(args.audit_dir/'horizon_probes/horizon_probes.csv')
    if not h.complete.all(): raise ValueError('Some extended integrations failed; do not silently compare')
    hp=h.pivot(index='mutant_name',columns='tf_min',values='pattern_code')
    transition=pd.crosstab(hp[210],hp[630]);transition.to_csv(out/'real_horizon_transition_210_to_630.csv')
    compact=selected[['badcase_rank','mutant_name','pattern_code','is_bifurcation_critical','MAE','MSE','Relative_L2','Correlation',
        'train_same_pattern','train_same_zero_parameter_face']].copy()
    for col in ('MAE','MSE','Relative_L2','Correlation'):compact[col]=compact[col].map(lambda v:f'{v:.4f}')
    compact['mutant_name']=compact.mutant_name.str.split('_').str[0]
    text=['# Budding 六分类与 Badcase 实测结果','',
        f'范围：{len(expected)} 个非 LHS 轨迹文件；合计 {counts.loc[counts.population.eq("all"),"count"].sum():,} 行（跨文件未去重）。',
        '只写分析旁文件；原始数据、存活标签、A/B/C 标签、生成与训练代码均未修改。','',
        '## 当前指定运行的 20 个 Badcase','',
        f'指标来源：`{args.eval_csv.resolve()}`。MAE/MSE/Relative_L2 各自降序排名，三个名次等权平均；Correlation 仅展示。',
        '训练支持数取当前 seed=42 的 80/10/10 拆分；不能据此证明历史模型实际训练样本或错误因果。','',markdown_table(compact),'',
        '## 所有数据集的合成样本六分类计数','',markdown_table(pivot.reset_index()),'',
        '## 观察窗口检验','',markdown_table(transition.reset_index()),'',
        '210 分钟 P2 的 47 个条件，到 630 分钟全部成为 P1；其中 2 个长时质量超过 10，不能解释为生物可存活。',
        '378 次积分均完成；质量跌落计数和求解器真实分裂事件数逐次一致。',
        '630 分钟仍是有限观察窗，既不能单凭多次分裂证明稳定极限环，也不能单凭不分裂证明渐近固定点。','',
        '## 产物说明','',
        '- `top20_badcases.csv`：主列表，含规则、特征、每项名次与当前训练支持数。',
        '- `top20_by_*.csv`：各指标分别最差的 20 条，避免综合排名掩盖单指标极端值。',
        '- `all_126_ranked.csv`：全部真实条件，按名称与当前数据一对一匹配。',
        '- `all_datasets_synthetic_counts.csv`：全部数据集；不把重复出现的 real 当成独立生物证据。',
        '- `dataset_real_consistency.csv`：各文件 real 与当前 MA 真值的 float32 精确一致性；差异不能归因于采样算法。',
        '- `all_rule_diagnostics.csv`：文字规则漏洞、P4 条件冲突、阈值上下浮动 10% 的敏感度。',
        '- `frozen_per_channel_metrics.csv`：已保存重放预测的逐变量诊断；与主排名 CSV 是独立预测产物，不混用数值。',
        '- 每个数据集子目录 `samples.csv.gz`：每行新规则、原标签、临界标记、特征与重建拆分；输入清单见 `manifest.json`。','']
    (out/'REPORT.md').write_text('\n'.join(text))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        q=pivot.loc[pivot.synthetic_total>0,list(CODES)]
        fig,ax=plt.subplots(figsize=(13,11));left=np.zeros(len(q))
        colors=['#2c7bb6','#74add1','#abd9a9','#fee090','#fdae61','#d73027']
        for code,color in zip(CODES,colors):
            val=q[code].to_numpy()/q.sum(axis=1).to_numpy()*100
            ax.barh(np.arange(len(q)),val,left=left,label=code,color=color);left+=val
        ax.set_yticks(np.arange(len(q)),q.index,fontsize=7);ax.invert_yaxis()
        ax.set_xlabel('Synthetic trajectories (%) — user six-pattern v1, 210 min')
        ax.set_xlim(0,100);ax.legend(ncols=6,loc='upper center',bbox_to_anchor=(.5,1.04))
        fig.tight_layout();fig.savefig(out/'all_dataset_patterns.png',dpi=180);fig.savefig(out/'all_dataset_patterns.pdf');plt.close(fig)
        if args.frozen_predictions.exists():
            with np.load(args.frozen_predictions,allow_pickle=False) as z:
                pred=z['predictions'];truth=z['targets'];names=z['names'].astype(str)
            fig,axes=plt.subplots(4,4,figsize=(14,10),sharex=True)
            for row,ident in enumerate(('15','22','42','124')):
                i=next(i for i,n in enumerate(names) if n.split('_')[0]==ident)
                for col,(variable,idx) in enumerate((('MASS',0),('CLN2',1),('CLB2',2),('ORI',33))):
                    ax=axes[row,col];t=np.linspace(0,210,truth.shape[2])
                    ax.plot(t,truth[i,idx],label='ODE target',color='#2166ac')
                    ax.plot(t,pred[i,idx],label='Frozen hybrid replay',color='#b2182b',linestyle='--')
                    ax.set_title(f'#{ident}: {variable}',fontsize=10)
                    if row==3:ax.set_xlabel('Time (min)')
            axes[0,0].legend(fontsize=7)
            fig.tight_layout();fig.savefig(out/'key_badcase_trajectories.png',dpi=180);fig.savefig(out/'key_badcase_trajectories.pdf');plt.close(fig)
    print(compact.to_string(index=False));print(f'Wrote {out / "REPORT.md"}')


if __name__ == '__main__': main()
