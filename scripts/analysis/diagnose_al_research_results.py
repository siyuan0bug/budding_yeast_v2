"""Post-hoc AL outcome, selected-input coverage and temporal feedback diagnosis.

Never changes acquisition, fits a surrogate, queries ODEs, or uses real labels
inside acquisition. Auxiliary trees only replay the already-used error predictor.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.scripts.analysis.audit_real_mutant_module_counts import count_rows


def read(path):
    return json.loads(path.read_text())


def concentration(values):
    counts=np.array(list(Counter(values).values()),dtype=float)
    p=counts/counts.sum()
    return dict(unique=len(counts),effective=np.exp(-np.sum(p*np.log(p))),
                top10_fraction=np.sort(p)[-10:].sum(),largest_count=counts.max())


def ids_for(action):
    return [int(e['candidate_id']) if e['kind']=='pool' else -1-int(e['training_row'])
            for e in action['endpoints']]


def acquisition(run, strategy, output, seed=42):
    rows, observations, graph_rows, feedback_rows, histories=[],[],[],[],[]
    feature_stats=None
    for directory in sorted((run/'acquisitions').glob('after_round_*')):
        design_path=directory/'query_design.json'
        if not design_path.exists():continue
        r=int(directory.name.rsplit('_',1)[1]);design=read(design_path)
        seen=set();kinds={};costs=Counter();actions=[]
        for batch in design['batches']:
            for action in batch['choices']:
                actions.append(action)
                for endpoint in action['endpoints']:
                    if endpoint['kind']!='pool' or endpoint['status']!='accepted':continue
                    cid=int(endpoint['candidate_id'])
                    if cid not in seen:
                        seen.add(cid);costs[action['kind']]+=1;kinds[cid]=action['kind']
        with np.load(directory/'added_samples.npz',allow_pickle=False) as archive:
            assert seen==set(map(int,archive['candidate_ids']))
        for kind,cost in costs.items():
            rows.append(dict(strategy=strategy,acquisition=r,kind=kind,accepted_queries=cost,
                             accepted_fraction=cost/len(seen),accepted_total=len(seen)))
        contrast=[a for a in actions if a['kind'] in ('response','interaction','invariance')]
        degree=Counter(i for a in contrast for i in ids_for(a) if i>=0)
        anchors=Counter(i for a in contrast for i in ids_for(a) if i<0)
        graph_rows.append(dict(strategy=strategy,acquisition=r,actions=len(contrast),
            unique_pool_endpoints=len(degree),unique_training_anchors=len(anchors),
            max_pool_degree=max(degree.values(),default=0),
            top10_endpoint_incidence_fraction=sum(n for _,n in degree.most_common(10))/max(1,sum(degree.values())),
            zero_response_variance_fraction=np.mean([a['response_variance']<=1e-12 for a in contrast]) if contrast else None))
        path=directory/'prequery_feedback.npz'
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as archive:
            current={k:archive[k] for k in ('candidate_ids','input_features','point_error',
                'response_error','response_observation_count','actual_log1p')}
            stats=(archive['input_mean'],archive['input_std'])
        if feature_stats is None:feature_stats=stats
        elif not all(np.array_equal(a,b) for a,b in zip(feature_stats,stats)):
            raise ValueError('Feedback input encoding changed across acquisitions')
        random=np.array([kinds[int(cid)]=='random' for cid in current['candidate_ids']])
        target='response_error' if strategy=='ours_response_feedback' else 'point_error'
        if histories:
            old_ids=set(int(i) for h in histories for i in h['candidate_ids'])
            if old_ids.intersection(map(int,current['candidate_ids'])):
                raise ValueError('Temporal feedback test contains earlier queried candidates')
            x=np.concatenate([h['input_features'] for h in histories])
            y=np.concatenate([h[target] for h in histories])
            model=ExtraTreesRegressor(n_estimators=32,max_depth=12,min_samples_leaf=8,
                                     n_jobs=2,random_state=seed+97*r)
            model.fit(x,np.log1p(y))
            prediction=np.maximum(np.expm1(model.predict(current['input_features'])),0)
            for scope,mask in (('all_current_queries',np.ones(len(random),dtype=bool)),('random_branch',random)):
                truth=current[target][mask];pred=prediction[mask]
                mse=np.mean((pred-truth)**2);constant=np.mean((y.mean()-truth)**2)
                feedback_rows.append(dict(strategy=strategy,acquisition=r,scope=scope,target=target,
                    training_rows=len(y),test_rows=len(truth),spearman=float(spearmanr(pred,truth).statistic),
                    mse=float(mse),old_mean_constant_mse=float(constant),
                    relative_mse_vs_old_mean=float(mse/max(constant,1e-30)),
                    predicted_mean=float(pred.mean()),actual_mean=float(truth.mean())))
        feedback_rows.append(dict(strategy=strategy,acquisition=r,scope='observed_feedback',target=target,
            test_rows=len(random),response_observation_fraction=float(np.mean(current['response_observation_count']>0)),
            point_error_median=float(np.median(current['point_error'])),point_error_q90=float(np.quantile(current['point_error'],.9))))
        outcome_path=directory/'response_feedback.json'
        if outcome_path.exists():
            outcomes={tuple(a['endpoints']):a['observed_response_error'] for a in read(outcome_path)['actions']}
            actual={int(cid):row for cid,row in zip(current['candidate_ids'],current['actual_log1p'])}
            for action in contrast:
                nodes=ids_for(action);error=outcomes.get(tuple(nodes))
                if error is None:continue
                energy=None
                if all(i in actual for i in nodes):
                    delta=actual[nodes[1]]-actual[nodes[0]]
                    if len(nodes)==4:delta=actual[nodes[3]]-actual[nodes[2]]-delta
                    energy=float(np.mean(delta**2))
                observations.append(dict(strategy=strategy,acquisition=r,kind=action['kind'],
                    predicted_response_variance=action['response_variance'],observed_response_error=error,
                    true_log_response_energy=energy,training_anchor=any(i<0 for i in nodes)))
        # Keep feedback labels/features only. Predictions/trajectories are not retained.
        histories.append({k:v for k,v in current.items() if k!='actual_log1p'})
    return rows,observations,graph_rows,feedback_rows


def coverage(run,strategy,definitions):
    with open_dataset(run/'rounds/round_03/dataset.json',allow_pickle=False) as archive:
        train=archive['split_train'];real=archive['split_real'];p=archive['params'];y0=archive['y_init']
        names=archive['mutant_names'];encoded=archive['mutation_pairs_json'];levels=archive['levels']
        groups=archive['pattern_group_codes']
    modules=[];modes=[]
    for row in train:
        pairs=[(g,m) for g,m in json.loads(str(encoded[row])) if g!='MDT']
        modules.append(tuple(sorted(set(g for g,m in pairs))));modes.append(tuple(sorted(pairs)))
    counts=Counter(modules);pc=p[train];ic=y0[train];mdt=PARAM_KEYS.index('mdt')
    defaults=BuddingSystem().defaults();wt=np.array([defaults[k] for k in PARAM_KEYS])
    rows=[]
    for row in real:
        identity=str(names[row]).split('_',1)[0]
        modules_real=tuple(definitions[identity]['modules'].split(';')) if definitions[identity]['modules'] else ()
        width=np.maximum(.25*np.abs(p[row]),.05*np.maximum(np.abs(wt),1e-8))
        soft=(np.abs(pc-p[row])<=width+1e-10).all(1)
        soft&=np.isclose(pc[:,mdt],p[row,mdt],rtol=0,atol=1e-8)
        local=soft.copy()
        local&=(pc[:,p[row]==0]==0).all(1)
        local&=np.isclose(pc[:,mdt],p[row,mdt],rtol=0,atol=1e-8)
        same_initial=np.isclose(ic,y0[row],rtol=1e-8,atol=1e-10).all(1)
        rows.append(dict(strategy=strategy,Mutant=str(names[row]),mutant_id=identity,
            modules=';'.join(modules_real),same_module_set=counts[modules_real],
            local_parameter_box25=int(local.sum()),local_parameter_box25_same_ic=int((local&same_initial).sum()),
            soft_zero_parameter_box25=int(soft.sum()),soft_zero_parameter_box25_same_ic=int((soft&same_initial).sum()),
            caveat='Parameter/IC geometry only; not event-switch equivalence or a causal accuracy explanation'))
    summary=dict(strategy=strategy,**{'modules_'+k:v for k,v in concentration(modules).items()},
        **{'modes_'+k:v for k,v in concentration(modes).items()})
    hist=[dict(strategy=strategy,kind='L',value=str(k),count=n) for k,n in Counter(levels[train]).items()]
    hist += [dict(strategy=strategy,kind='window_group',value=str(k),count=n) for k,n in Counter(groups[train]).items()]
    return rows,summary,hist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    definitions={r['identity']:r for r in count_rows('budding')[0]}
    rounds,mutants,status=[],[],[]
    query,observations,graphs,calibration,support,concentration_rows,histograms=[],[],[],[],[],[],[]
    for task in read(args.manifest)['tasks']:
        run=Path(task['experiment_dir']);strategy=task['strategy'];records=read(run/'summary/rounds_summary.json')
        complete=len(records)==4 and (run/'rounds/round_03/eval/metrics_denorm.json').exists()
        status.append(dict(strategy=strategy,completed_rounds=len(records),complete=complete,run=str(run)))
        for record in records:
            r=record['round'];metrics=read(run/f'rounds/round_{r:02d}/eval/metrics_denorm.json')
            rounds.append(dict(strategy=strategy,round=r,**{k:v for k,v in record.items() if k!='round'}))
            mutants.extend(dict(strategy=strategy,round=r,**m) for m in metrics['per_mutant'])
        if strategy.startswith('ours_'):
            a,b,c,d=acquisition(run,strategy,args.output,seed=int(task.get('seed',42)));query+=a;observations+=b;graphs+=c;calibration+=d
        if complete:
            a,b,c=coverage(run,strategy,definitions);support+=a;concentration_rows.append(b);histograms+=c
        print(strategy,'complete',complete,flush=True)
    frame=pd.DataFrame(rounds);per=pd.DataFrame(mutants);final=per[per['round']==3]
    reference=final[final.strategy=='al4pde_random'].set_index('Mutant')
    top20=reference[['MAE','MSE','Relative_L2']].rank(pct=True).mean(1).nlargest(20).index
    final[final.Mutant.isin(top20)].to_csv(args.output/'top20_badcases_all_methods.csv',index=False)
    robustness=[];contributions=[]
    for method,local in final.groupby('strategy'):
        local=local.set_index('Mutant').loc[reference.index]
        row=dict(strategy=method)
        for metric in ('MAE','MSE','Relative_L2'):
            row.update({metric+'_median':local[metric].median(),metric+'_q90':local[metric].quantile(.9),
                metric+'_q95':local[metric].quantile(.95),metric+'_max':local[metric].max(),
                metric+'_improved_count':int((local[metric]<reference[metric]).sum()),
                metric+'_top20_mean':local.loc[top20,metric].mean()})
            diff=(local[metric]-reference[metric])/126
            for name,value in diff.items():contributions.append(dict(strategy=method,Mutant=name,metric=metric,delta_global_arithmetic_mean=value))
        row['relative_l2_above_1']=int((local.Relative_L2>1).sum())
        local_rounds=frame[frame.strategy==method].sort_values('train_samples')
        for metric in ('real_mutants_mae','real_mutants_mse','real_mutants_rel_l2'):
            row[metric+'_normalized_AULC']=trapezoid(local_rounds[metric],local_rounds.train_samples)/60000
        robustness.append(row)
    for name,data in (
        ('status',status),('round_metrics',rounds),('per_mutant',mutants),('accepted_query_budget',query),
        ('response_observations',observations),('contrast_graph',graphs),('feedback_temporal_calibration',calibration),
        ('real_input_support',support),('selected_input_concentration',concentration_rows),
        ('training_histograms',histograms),('robustness_and_learning_curve',robustness),('mutant_metric_contributions',contributions)):
        pd.DataFrame(data).to_csv(args.output/(name+'.csv'),index=False)
    frame[frame['round']==2].to_csv(args.output/'common_40000_comparison.csv',index=False)
    print(args.output,flush=True)


if __name__=='__main__':main()
