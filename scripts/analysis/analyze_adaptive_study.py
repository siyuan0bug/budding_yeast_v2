"""Audit completed adaptive multi-seed generation, terminal metrics and query feedback."""
import argparse,csv,json,hashlib,statistics
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.stats import spearmanr
from budding_yeast_v2.data.dataset_archive import file_sha256,TrajectoryStore
from budding_yeast_v2.utils.sampling.ours.adaptive_generation.features import context_key
from budding_yeast_v2.utils.sampling.ours.adaptive_generation.feedback import Feedback

BASELINES=['random','power','lcmd','physics_pre','low_rank','peaks','ogas']
OURS=['ours_response_generate','ours_event_generate','ours_progress_generate']
KEYS=['MAE','MSE','Relative L2','Correlation']

def flatten(v,prefix=''):
    if isinstance(v,dict):return {k:x for a,b in v.items() for k,x in flatten(b,f'{prefix}.{a}' if prefix else a).items()}
    return {prefix:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v}

def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((a.batch/'tasks.json').read_text());assert len(tasks)==9 and all(t['status']=='complete' and t['exit_code']==0 for t in tasks)
    records=[];all_rows=[];rounds=[];mutants=[];diagnostics=[];feedback_rounds=[];paired=[]
    for method in BASELINES+OURS:
        for seed in [42,123,1024]:
            matches=list((a.root/'fno/mutation'/method).glob(f'*/seed_{seed}/eval/global_metrics_denorm.json'));assert len(matches)==1
            path=matches[0];d=path.parent.parent;metrics=json.loads(path.read_text());config=json.loads((d/'config.json').read_text())
            identity=dict(method=method,seed=seed,directory=str(d))
            assert metrics['train_pool']=='mutation' and metrics['real_mutants']['count']==126
            if method in OURS:assert metrics['round']==4
            for cohort in ('real_mutants','synth_test','cross_pool_test'):
                assert metrics[cohort]['metric_space']=='physical' and all(np.isfinite(metrics[cohort]['all_39_vars'][k]) for k in KEYS)
                all_rows.append(dict(**identity,cohort=cohort,**flatten(metrics[cohort])))
            for rp in sorted(d.glob('rounds/round_*/eval/global_metrics_denorm.json')):
                mv=json.loads(rp.read_text())
                for cohort in ('real_mutants','synth_test','cross_pool_test'):
                    if cohort in mv:rounds.append(dict(**identity,round=mv['round'],train_samples=mv['train_samples'],cohort=cohort,**flatten(mv[cohort])))
            samples=read_csv(d/'eval/per_mutant_metrics.csv');assert len(samples)==126
            mutants.extend(dict(**identity,**s) for s in samples)
            record=dict(**identity,metrics=metrics,metrics_sha256=file_sha256(path),config=config)
            if method in OURS:
                done=json.loads((d/'_state/completion.json').read_text());assert done['status']=='complete' and done['accepted']==10000 and done['newly_accepted']==8000
                assert done['solver_attempts_exact'] and done['sources_unchanged']
                assert not list((d/'rounds').glob('round_0[0-3]/eval/real_mutants/*.png'))
                assert len(list((d/'rounds/round_04/eval/real_mutants').glob('*.png')))==126
                assert len(list((d/'eval/real_mutants').glob('*.png')))==126
                state=json.loads((d/'_state/generation.json').read_text());assert file_sha256(state['policy_path'])==state['policy_sha256']
                policy=torch.load(state['policy_path'],map_location='cpu',weights_only=False)['controller']
                fb=Feedback();fb.load_state_dict(policy['feedback']);assert len(fb.rows)==8000 and not policy['pending']
                meta=[]
                # Read only metadata members, never decompress trajectories.
                for chunk in sorted((d/'data/chunks').glob('*.npz')):
                    with np.load(chunk,allow_pickle=False) as z:meta.extend(json.loads(str(x)) for x in z['candidate_metadata_json'])
                assert len(meta)==8000
                controlled=sum(m.get('sampling_paradigm')=='single_axis_canonical_mutation_redraw' for m in meta)
                snapshot=json.loads((d/'rounds/round_00/dataset.json').read_text())
                initial=TrajectoryStore(snapshot['source']).read(snapshot['rows'],('candidate_metadata_json','overrides_json'))
                known=set()
                for mj,oj in zip(initial['candidate_metadata_json'],initial['overrides_json']):
                    m=json.loads(str(mj));m['_effective_overrides']=json.loads(str(oj));known.add(context_key(m))
                unknown=[]
                for round_index in sorted({r['round'] for r in fb.rows}):
                    current=[r for r in fb.rows if r['round']==round_index]
                    unknown.extend(r for r in current if r['key'] not in known)
                    known.update(r['key'] for r in current)
                assert all(context_key(m)==r['key'] for m,r in zip(meta,fb.rows))

                acquisitions=[json.loads(x) for x in (d/'_state/acquisition.jsonl').read_text().splitlines()];assert len(acquisitions)==500
                diag=dict(**identity,seconds=done['seconds'],attempts=done['attempted_inputs'],failed=done['failed_inputs'],
                    selected_interventions=controlled,selected_global=8000-controlled,
                    unknown_context_queries=len(unknown),unknown_context_score_one=sum(r['score']==1 for r in unknown),
                    singleton_feedback_contexts=sum(len(g)==1 for g in fb.grouped().values()),
                    mean_multiple_anchor_fraction=statistics.mean(x['local_multiple_anchor_rows']/128 for x in acquisitions),
                    feedback_spearman=spearmanr([r['score'] for r in fb.rows],[r['error'] for r in fb.rows]).statistic,
                    progress_non_neutral_contexts=sum(abs(fb.weight(k)-1)>1e-8 for k in fb.grouped()),
                    feedback_contexts=len(fb.grouped()),
                    actual_prequery_feedback=8000,initial_shared=2000,
                    **{'proposals.'+k:v for k,v in done['proposal_counts'].items()})
                diagnostics.append(diag)
                for round_index in sorted({r['round'] for r in fb.rows}):
                    g=[r for r in fb.rows if r['round']==round_index]
                    correlation=spearmanr([r['score'] for r in g],[r['error'] for r in g]).statistic
                    feedback_rounds.append(dict(**identity,round=round_index,count=len(g),spearman=correlation,mean_prequery_error=statistics.mean(r['error'] for r in g)))
                record['completion']={k:v for k,v in done.items() if k!='artifacts'}
                record['feedback_diagnostics']=diag
            records.append(record)
    # Verify common training contract, metrics and initial dataset identity per seed.
    assert len({json.dumps({k:v for k,v in r['config']['training'].items() if k!='seed'},sort_keys=True) for r in records})==1
    assert len({r['metrics']['real_mutants']['metric_version'] for r in records})==1
    for method in OURS:
        for seed in [42,123,1024]:
            own=[r for r in mutants if r['method']==method and r['seed']==seed]
            reference={r['mutant_name']:r for r in mutants if r['method']=='power' and r['seed']==seed}
            for row in own:
                base=reference[row['mutant_name']]
                paired.append(dict(method=method,seed=seed,mutant_name=row['mutant_name'],pattern_group=row.get('pattern_group'),
                    **{k+'_ours':float(row[k]) for k in ['MAE','MSE','rel_l2']},
                    **{k+'_power':float(base[k]) for k in ['MAE','MSE','rel_l2']},
                    **{k+'_delta':float(row[k])-float(base[k]) for k in ['MAE','MSE','rel_l2']}))
    aggregate=[]
    for method in BASELINES+OURS:
        g=[r for r in records if r['method']==method]
        for cohort in ['real_mutants','synth_test','cross_pool_test']:
            value=dict(method=method,cohort=cohort,n=3)
            for k in KEYS:
                x=[r['metrics'][cohort]['all_39_vars'][k] for r in g];value[k+'_mean']=statistics.mean(x);value[k+'_std']=statistics.stdev(x)
            aggregate.append(value)
    for name,data in [('all_final_metrics',all_rows),('all_round_metrics',rounds),('mean_std',aggregate),('per_mutant_metrics',mutants),('paired_vs_power',paired),('generation_diagnostics',diagnostics),('feedback_by_round',feedback_rounds)]:write_csv(a.output/(name+'.csv'),data)
    (a.output/'inventory.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    lines=['# 三种自适应生成方法：三seed终评（2026-11-22）','','9项全部complete，exit0；每项共享初始2000，新增8000，总训练10000。所有数值取根eval（最终round04，200epochs训练中的归一化验证best），不把中间轮当终评。三seed是42/123/1024，std为样本标准差。',
        '','本次只比较mutation来源；不把LHS旧结果混入。本次7类baseline与3类ours共30项。完整所有指标（39/9变量、轨迹、分裂/Pattern及混淆矩阵）在all_final_metrics.csv，具体路径及原JSON在inventory.json。',
        '','相同训练配置不等于相同算法总算力：PEAKS/OGAS持续训练，Low-Rank带权有放回；生成式新ODE成本与固定池标签访问分开看。', '']
    for cohort in ['real_mutants','synth_test','cross_pool_test']:
        lines += [f'## {cohort}','','|方法|MAE↓|MSE↓|Relative↓|Corr↑|','|---|---|---|---|---|']
        for r in aggregate:
            if r['cohort']==cohort:lines.append('|'+ '|'.join([r['method']]+[f"{r[k+'_mean']:.5f} ± {r[k+'_std']:.5f}" for k in KEYS])+'|')
        lines+=['']
    lines+=['## 各实验real终评','','|方法|seed|MAE|MSE|Relative|Corr|','|---|---|---|---|---|---|']
    for r in records:
        lines.append('|'+ '|'.join([r['method'],str(r['seed'])]+[f"{r['metrics']['real_mutants']['all_39_vars'][k]:.5f}" for k in KEYS])+'|')
    lines+=['','## 查询诊断','','|方法|seed|完整小时|选中干预数/8000|多锚点候选比例|全反馈Spearman|','|---|---|---|---|---|---|']
    for r in diagnostics:lines.append(f"|{r['method']}|{r['seed']}|{r['seconds']/3600:.2f}|{r['selected_interventions']}|{r['mean_multiple_anchor_fraction']:.2%}|{r['feedback_spearman']:.4f}|")
    lines+=['','Spearman仅针对被选中样本，分数/实际误差均为各方法自身的asinh或事件特征尺度，不等同于主Relative，也不是候选池全域或real上的校准证明。分轮结果另存feedback_by_round.csv。','',
        '## 每项目录、预测图和日志','']
    for r in records:
        d=Path(r['directory']);lines.append(f"- {r['method']} / seed{r['seed']}：[目录]({d})；[预测图]({d}/eval/real_mutants)；[原JSON]({d}/eval/global_metrics_denorm.json)；[日志]({d}/train.log)。")
    (a.output/'report.md').write_text('\n'.join(lines))
    print('Verified 9 completed adaptive experiments and 21 baseline comparators; exported full metrics.')
    for r in diagnostics:print(r['method'],r['seed'],'interventions',r['selected_interventions'],'multiple',round(r['mean_multiple_anchor_fraction'],3),'rho',round(r['feedback_spearman'],3),'progress_active',r['progress_non_neutral_contexts'],'/',r['feedback_contexts'])

if __name__=='__main__':main()
