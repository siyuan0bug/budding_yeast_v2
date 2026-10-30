"""Explain completed response-AL queries from saved inputs and action traces.

No training, no new oracle queries, no use of real targets to select candidates.
Reconstructed point scores cover only trace components with known point/anchor
scores; report their coverage rather than imputing scores for other components.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from budding_yeast_v2.data.dataset_archive import open_dataset
from budding_yeast_v2.scripts.analysis.audit_real_mutant_module_counts import count_rows


def keys(encoded):
    pairs=[(str(g),str(m)) for g,m in json.loads(str(encoded)) if g!='MDT']
    return ';'.join(sorted({g for g,m in pairs})), ';'.join(f'{g}[{m}]' for g,m in sorted(pairs))


def distributions(counter):
    p=np.asarray(list(counter.values()),dtype=float)
    p/=p.sum()
    return dict(unique=len(counter),effective_count=float(np.exp(-np.sum(p*np.log(p)))),
                top10_fraction=float(np.sort(p)[-10:].sum()),largest_count=max(counter.values()))


def trace_statistics(path, output):
    obj=json.loads(path.read_text());actions=obj['choices']
    known={};adj=defaultdict(list)
    for action in actions:
        pool=[e['candidate_id'] for e in action['endpoints'] if e['kind']=='pool']
        if action['kind']=='point':known[pool[0]]=action['score']
        elif action['kind']=='response':
            total=len(pool)*action['score']-action['response_variance']/(1+action['distance'])
            if len(pool)==1:known[pool[0]]=total
            else:
                a,b=pool;adj[a].append((b,total));adj[b].append((a,total))
    queue=deque(known)
    while queue:
        a=queue.popleft()
        for b,total in adj[a]:
            if b not in known:
                known[b]=total-known[a];queue.append(b)
    residuals=[abs(known[a]+known[b]-total) for a,edges in adj.items()
               for b,total in edges if a in known and b in known]
    query_counts=Counter();action_counts=Counter();degrees=Counter();anchors=Counter()
    seen=set();reused_queries=0;response_shares=[];reused_shares=[];distances=[];records=[]
    for index,action in enumerate(actions):
        kind=action['kind'];action_counts[kind]+=1;query_counts[kind]+=action['new_queries']
        pool=[e['candidate_id'] for e in action['endpoints'] if e['kind']=='pool']
        new=set(pool)-seen;old=set(pool)&seen
        assert len(new)==action['new_queries']
        row=dict(action=index,kind=kind,new_queries=len(new),reused_pool_endpoints=len(old))
        if kind=='response':
            degrees.update(pool);anchors.update(e['candidate_id'] for e in action['endpoints'] if e['kind']=='training_anchor')
            distances.append(action['distance'])
            fraction=action['response_variance']/(1+action['distance'])/(len(pool)*action['score'])
            response_shares.append(fraction)
            row.update(distance=action['distance'],response_fraction=fraction,
                       training_anchor=any(e['kind']=='training_anchor' for e in action['endpoints']))
            if old:
                reused_queries+=len(new)
                if all(i in known for i in pool):
                    share=sum(known[i] for i in old)/(len(pool)*action['score'])
                    reused_shares.append(share);row['already_selected_point_reward_fraction']=share
        seen.update(pool);records.append(row)
    report=dict(round=int(path.parent.name.rsplit('_',1)[-1]),query_counts=dict(query_counts),
        action_counts=dict(action_counts),unique_queries=len(seen),response_actions=obj['response_actions'],
        reused_endpoint_new_queries=reused_queries,reused_endpoint_fraction=reused_queries/len(seen),
        unique_training_anchors=len(anchors),training_anchor_actions=sum(anchors.values()),
        max_pool_endpoint_degree=max(degrees.values()),top_hubs=degrees.most_common(10),
        distance_p10_p50_p90=np.quantile(distances,[.1,.5,.9]).tolist(),
        distance_above_half_fraction=float(np.mean(np.asarray(distances)>.5)),
        response_reward_p10_p50_p90=np.quantile(response_shares,[.1,.5,.9]).tolist(),
        reconstructed_point_scores=len(known),reused_reward_evaluable_actions=len(reused_shares),
        reused_old_point_reward_p10_p50_p90=np.quantile(reused_shares,[.1,.5,.9]).tolist(),
        reconstructed_equation_max_residual=max(residuals,default=0.),timings=obj['timings'])
    pd.DataFrame(records).to_csv(output/f'ours_actions_round_{report["round"]:02d}.csv',index=False)
    return report


def load_inputs(path):
    with open_dataset(path,allow_pickle=False) as z:
        return {k:z[k] for k in ('params','y_init','mutation_pairs_json','candidate_ids',
            'module_counts','pattern_group_codes','split_train','split_real','mutant_names','param_names')}


def local_box(params,query,wt,mdt):
    width=np.maximum(.25*np.abs(query),.05*np.abs(wt))
    zero=query==0;width[zero]=0.
    mask=(np.abs(params-query)<=width+1e-10).all(1)
    mask&=(params[:,zero]==0).all(1)
    mask&=np.isclose(params[:,mdt],query[mdt],rtol=0,atol=1e-5)
    return mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/users/zsy/mydata/budding_yeast/experiments/al_budding/fno'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-top20',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths={d.parent.parent.name.removeprefix('hybrid_mse_rel_'):d for d in args.root.glob('*/*_refit_gene_ranges/seed_42')}
    ours=paths['ours_response']
    traces=[trace_statistics(p,args.output) for p in sorted(ours.glob('acquisitions/*/response_pairs.json'))]
    (args.output/'response_trace_summary.json').write_text(json.dumps(traces,indent=2))
    acquisitions={};coverage=[];group_rows=[];phase_rows=[];supports=[]
    reference=pd.read_csv(args.reference_top20)['Mutant'].tolist()
    real_definitions={r['identity']:r for r in count_rows('budding')[0]}
    for strategy,run in sorted(paths.items()):
        print('Inputs:',strategy,flush=True)
        saved=load_inputs(run/'rounds/round_03/dataset.npz')
        params=saved['params'];initial=saved['y_init'];train=saved['split_train'];real=saved['split_real']
        real_lookup={str(saved['mutant_names'][i]):i for i in real}
        wt=params[next(i for i in real if str(saved['mutant_names'][i]).startswith('1_'))]
        mdt=list(saved['param_names']).index('mdt')
        modules,alleles=zip(*(keys(v) for v in saved['mutation_pairs_json']))
        modules=np.asarray(modules);alleles=np.asarray(alleles)
        for name in reference:
            q=real_lookup[name];mask=local_box(params[train],params[q],wt,mdt)
            initial_match=np.isclose(initial[train],initial[q],rtol=1e-5,atol=1e-8).all(1)
            identity=re.match(r'^(\d+(?:\.\d+)?)_',name).group(1)
            # Existing real rows store [] rather than parsed alleles. An empty
            # annotation is not evidence that a mutant has no mutated modules.
            real_modules=real_definitions[identity]['modules']
            known_alleles=bool(alleles[q]) or not real_modules
            supports.append(dict(strategy=strategy,Mutant=name,same_modules=int((modules[train]==real_modules).sum()),
                real_module_source='canonical_name_modules',real_alleles_available=known_alleles,
                same_alleles=int((alleles[train]==alleles[q]).sum()) if known_alleles else None,local_parameter25=int(mask.sum()),
                local_parameter25_same_initial=int((mask&initial_match).sum())))
        previous=set()
        for index,n in enumerate((10000,20000,40000,70000)):
            rows=train[:n];mc=Counter(modules[rows]);ac=Counter(alleles[rows]);codes=Counter(saved['pattern_group_codes'][rows])
            coverage.append(dict(strategy=strategy,round=index,train_samples=n,
                **{'module_'+k:v for k,v in distributions(mc).items()},
                **{'allele_'+k:v for k,v in distributions(ac).items()},
                levels=dict(Counter(map(int,saved['module_counts'][rows]))),groups=dict(codes)))
            for kind,counter in [('modules',mc),('alleles',ac),('pattern',codes)]:
                group_rows.extend(dict(strategy=strategy,round=index,kind=kind,key=str(k),count=int(v)) for k,v in counter.items())
        for index in range(3):
            report=json.loads((run/f'acquisitions/after_round_{index:02d}/report.json').read_text())
            acquisitions[strategy,index]=set(report['candidate_ids'])
        # New runs have stage durations. Do not infer missing old timings.
        for line in (run/'train.log').read_text(errors='replace').splitlines():
            m=re.search(r'\[END\] (.+?) \| (.*)elapsed_s=([\d.]+)',line)
            if m:phase_rows.append(dict(strategy=strategy,phase=m.group(1),fields=m.group(2),seconds=float(m.group(3))))
    overlaps=[]
    for i in range(3):
        a=acquisitions['ours_response',i]
        for strategy in paths:
            b=acquisitions[strategy,i]
            overlaps.append(dict(round=i,strategy=strategy,intersection=len(a&b),
                ours_fraction=len(a&b)/len(a),jaccard=len(a&b)/len(a|b)))
    frame=pd.DataFrame(coverage)
    for name,rows in [('coverage',coverage),('genotype_groups',group_rows),('stage_timings',phase_rows),
                      ('badcase_input_support',supports),('ours_selection_overlap',overlaps)]:
        pd.DataFrame(rows).to_csv(args.output/f'{name}.csv',index=False)
    (args.output/'coverage.json').write_text(json.dumps(coverage,indent=2,default=lambda x:x.item()))
    final=pd.read_csv(args.output/'per_mutant.csv')
    final=final[(final.cohort=='gene_ranges')&(final['round']==3)]
    comparisons=[]
    a=final[final.strategy=='ours_response'].set_index('Mutant')
    for strategy in paths:
        b=final[final.strategy==strategy].set_index('Mutant').loc[a.index]
        for metric in ['MAE','MSE','Relative_L2','Correlation']:
            delta=a[metric]-b[metric]
            comparisons.append(dict(baseline=strategy,metric=metric,
                ours_better=int(((delta>0) if metric=='Correlation' else (delta<0)).sum()),
                ours_worse=int(((delta<0) if metric=='Correlation' else (delta>0)).sum()),
                ours_mean=float(a[metric].mean()),baseline_mean=float(b[metric].mean()),
                ours_median=float(a[metric].median()),baseline_median=float(b[metric].median())))
    pd.DataFrame(comparisons).to_csv(args.output/'ours_paired_metrics.csv',index=False)
    plot(traces,frame,pd.DataFrame(overlaps),args.output)


def plot(traces,coverage,overlap,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    x=np.arange(3)
    bottom=np.zeros(3)
    for kind in ['point','response','random']:
        values=np.array([v['query_counts'].get(kind,0)/v['unique_queries'] for v in traces])
        axes[0].bar(x,values,bottom=bottom,label=kind);bottom+=values
    axes[0].set(xticks=x,xticklabels=['10k added','20k added','30k added'],ylabel='New-query fraction',title='Ours query accounting');axes[0].legend()
    axes[1].plot(x,[v['distance_p10_p50_p90'][1] for v in traces],marker='o',label='Median CDF distance')
    axes[1].plot(x,[v['response_reward_p10_p50_p90'][1] for v in traces],marker='o',label='Median response share of score')
    axes[1].plot(x,[v['reused_endpoint_fraction'] for v in traces],marker='o',label='Queries through reused pool endpoints')
    axes[1].set(xticks=x,xticklabels=['Acq 1','Acq 2','Acq 3'],ylim=(0,1),title='Response mechanism diagnostics');axes[1].legend(fontsize=7)
    final=coverage[coverage['round']==3].sort_values('module_effective_count')
    axes[2].barh(final.strategy.str.removeprefix('al4pde_'),final.module_effective_count)
    axes[2].set(xlabel='exp(entropy of module-set counts)',title='Final training module coverage')
    for suffix in ['png','pdf']:fig.savefig(output/f'ours_mechanism_diagnostics.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
