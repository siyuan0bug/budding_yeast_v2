"""Paired ODE experiment: exact channel closure versus current MA residuals.

Runs the production event/reset/initializer code without modifying production
files. A worker-local solver wrapper tightens tolerances for an independent
numerical check. All backgrounds, fractions and outputs are saved for review.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding import simulation as sim
from budding_yeast_v2.data.systems.budding.parameters import get_default_params
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.data.systems.budding.real_mutants import is_duplicate_real_mutant
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.data import simulation_guards

CORE = [0, 1, 2, 3, 4, 20, 22, 33, 35]
# All target rates have residual fraction alpha~U(0,.01) relative to WT.
# A target is separate from its context; multirate targets share one alpha.
MECHANISMS = {
    'CLB5_db': dict(target=['kdb5pp'], channel=3, contexts=[
        ('WT', []), ('GAL_CLB5', [('CLB5','GAL-OE',.016)]),
        ('GAL_CLB5_CDH1_KO', [('CLB5','GAL-OE',.016),('CDH1','KO',0.)])]),
    'SIC1_db': dict(target=['kd3c1'], channel=4, contexts=[
        ('WT', []), ('GAL_SIC1', [('SIC1','GAL-OE',.18)]),
        ('GAL_SIC1_CDH1_KO', [('SIC1','GAL-OE',.18),('CDH1','KO',0.)])]),
    'PDS1_db': dict(target=['kd2pdspp','kd3pdspp'], channel=31, contexts=[
        ('WT', []), ('GAL_PDS1', [('PDS1','GAL-OE',.14)]),
        ('GAL_PDS1_ESP1_ts', [('PDS1','GAL-OE',.14),('ESP1','ts',.003)])]),
    'CDC20_ts': dict(target=['ks20p','ks20pp'], channel=20, contexts=[
        ('WT', []), ('GAL_SIC1', [('SIC1','GAL-OE',.18)]),
        ('PDS1_KO_CLB5_KO', [('PDS1','KO',0.),('CLB5','KO',0.)])]),
    'CDC15_ts': dict(target=['kkpnetpp'], channel=29, contexts=[
        ('WT', []), ('NET1_TAB6', [('NET1','TAB6-1',.065)]),
        ('MC_CDC14', [('CDC14','MC',3.)])]),
    'APC_Core_ts': dict(target=['ks20p','ks20pp','kscdh'], channel=20, contexts=[
        ('WT', []), ('MC_SIC1', [('SIC1','MC',7.)]),
        ('CLN3_KO_GAL_CLB2', [('CLN3','KO',0.),('CLB2','GAL-OE',.18)])]),
    'CDC14_ts_synthesis': dict(target=['ks14'], channel=24, contexts=[
        ('WT', []), ('GAL_SIC1', [('SIC1','GAL-OE',.18)]),
        ('CDH1_KO', [('CDH1','KO',0.)])]),
    'APC_A': dict(target=['ka20pp'], channel=20, contexts=[
        ('WT', []), ('CDH1_KO', [('CDH1','KO',0.)]),
        ('GAL_CLB2', [('CLB2','GAL-OE',.18)])]),
}


def make_tasks(horizon, steps, fractions):
    backend = BuddingSystem()
    defaults = get_default_params()
    tasks = []
    for mechanism, definition in MECHANISMS.items():
        for context, instructions in definition['contexts']:
            base = {}
            for gene, mode, alpha in instructions:
                backend.apply_gene_mutation(gene,mode,alpha,backend.defaults(),{},base,
                                            rng=np.random.RandomState(42))
            base['mdt'] = 150. if any(m in ('GAL-OE','GAL-MC') for _,m,_ in instructions) else 90.
            if mechanism == 'APC_Core_ts':
                base.update(init_CDH1T=0., init_CDH1=0.)
            initial = sim.prepare_initial_conditions(base)[1]
            for fraction in fractions:
                alpha = .01*fraction
                override = dict(base, **{key: defaults[key]*alpha for key in definition['target']})
                np.testing.assert_array_equal(initial, sim.prepare_initial_conditions(override)[1])
                for precision, rtol, atol, timeout in [('production',1e-4,1e-6,5.),('tight',1e-8,1e-10,20.)]:
                    tasks.append(dict(mechanism=mechanism,context=context,alpha=alpha,
                        fraction_of_residual_upper=fraction, precision=precision,rtol=rtol,atol=atol,
                        timeout_s=timeout, overrides=override,target=definition['target'],
                        context_instructions=instructions,horizon=horizon,steps=steps,
                        complete_input_matches_real=is_duplicate_real_mutant(override,backend.defaults())))
    return tasks


def run(task):
    started = time.monotonic()
    native_solve = sim.solve_ivp
    events = []
    def solve(*args, **kwargs):
        kwargs.update(rtol=task['rtol'],atol=task['atol'])
        result = native_solve(*args, **kwargs)
        if result.status == 1:
            index = next((i+1 for i,times in enumerate(result.t_events)
                          if len(times) and np.isclose(times[-1],result.t[-1])),None)
            events.append((index,float(result.t[-1])))
        return result
    with patch.object(sim,'solve_ivp',solve), patch.object(simulation_guards,'DEFAULT_SIM_TIMEOUT',task['timeout_s']):
        _, trajectory, bio, abc = sim.simulate_mutant('zero_residual_diagnostic',task['overrides'],
                                                     tf=task['horizon'],ns=task['steps'])
    row = dict(task, success=trajectory is not None,seconds=time.monotonic()-started,
               biological_label=bio, abc_label=abc, native_division_times=[t for e,t in events if e==1])
    if trajectory is not None:
        groups = pattern_group_metadata(trajectory)
        row.update(group=str(groups['pattern_group_codes'][0]),
                   sampled_divisions=int(groups['observed_division_counts'][0]),
                   native_divisions=len(row['native_division_times']),
                   minimum=float(trajectory.min()),maximum=float(trajectory.max()),
                   clb2_peak=float(trajectory[2].max()))
    return row, trajectory


def rel_l2(value,reference):
    return float(np.linalg.norm(value-reference)/max(float(np.linalg.norm(reference)),1e-12))


def comparisons(results):
    lookup={(r['mechanism'],r['context'],r['alpha'],r['precision']):(r,y) for r,y in results}
    rows=[]
    for row,y in results:
        if not row['success'] or row['alpha']==0:continue
        zero,z = lookup[row['mechanism'],row['context'],0.,row['precision']]
        if z is None:continue
        log_y,log_z=np.log1p(np.maximum(y,0)),np.log1p(np.maximum(z,0))
        delta=np.abs(y-z);channel,index=np.unravel_index(delta.argmax(),delta.shape)
        rows.append(dict(mechanism=row['mechanism'],context=row['context'],alpha=row['alpha'],
            precision=row['precision'],rel_l2_all=rel_l2(y,z),rel_l2_core=rel_l2(y[CORE],z[CORE]),
            log_rel_l2_all=rel_l2(log_y,log_z),mae_all=float(delta.mean()),max_abs_difference=float(delta.max()),
            worst_variable=VARIABLE_NAMES[channel],worst_time_min=float(index*row['horizon']/(row['steps']-1)),
            zero_divisions=zero['native_divisions'],residual_divisions=row['native_divisions'],
            zero_group=zero['group'],residual_group=row['group'],
            zero_bio=zero['biological_label'],residual_bio=row['biological_label'],
            zero_clb2_peak=zero['clb2_peak'],residual_clb2_peak=row['clb2_peak'],
            event_or_group_changed=(zero['native_divisions']!=row['native_divisions'] or zero['group']!=row['group']),
            bio_changed=zero['biological_label']!=row['biological_label']))
    numeric=[]
    for row,y in results:
        if row['precision']!='production' or y is None:continue
        tight,t=lookup[row['mechanism'],row['context'],row['alpha'],'tight']
        if t is None:continue
        numeric.append(dict(mechanism=row['mechanism'],context=row['context'],alpha=row['alpha'],
            production_vs_tight_rel_l2=rel_l2(y,t),
            production_divisions=row['native_divisions'],tight_divisions=tight['native_divisions'],
            production_group=row['group'],tight_group=tight['group']))
    return rows,numeric,lookup


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_key_examples(output,lookup,fractions,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    colors=['#151515','#237cb7','#df8e17','#bd4d71','#23a187']
    examples=[('SIC1_db','WT',4,(0,horizon),'SIC1-db: concentration changes'),
              ('PDS1_db','WT',0,(184,195),'PDS1-db: division-time shift (zoom)'),
              ('CDC14_ts_synthesis','GAL_SIC1',24,(0,horizon),'CDC14 synthesis loss: almost coincident')]
    for ax,(mechanism,context,channel,limits,title) in zip(axes,examples):
        for index,fraction in reversed(list(enumerate(fractions))):
            _,y=lookup[mechanism,context,.01*fraction,'tight']
            if y is None:continue
            times=np.linspace(0,horizon,y.shape[-1])
            ax.plot(times,y[channel],color=colors[index],ls='--' if fraction==0 else '-',
                lw=2 if fraction==0 else 1.4,label='Exact zero' if fraction==0 else f'alpha={.01*fraction:g}')
        ax.set_xlim(*limits);ax.set_title(title,fontsize=10)
        ax.set_xlabel('Time (min)');ax.set_ylabel(VARIABLE_NAMES[channel]);ax.grid(alpha=.2)
        ax.spines[['right','top']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Matched initial states and environment; only the target reaction rate changes',fontsize=12)
    fig.savefig(output/'key_examples.png',dpi=180);plt.close(fig)


def plot(output,results,stats,lookup,fractions,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'savefig.facecolor':'white'})
    colors=['#151515','#237cb7','#df8e17','#bd4d71','#23a187']
    times=np.linspace(0,horizon,results[0][0]['steps'])
    def curves(ax,mechanism,context,channel,precision='tight'):
        for i,f in reversed(list(enumerate(fractions))):
            row,y=lookup[mechanism,context,.01*f,precision]
            if y is None:continue
            label='Exact zero' if f==0 else f'alpha={.01*f:g}'
            ax.plot(times,y[channel],color=colors[i],lw=2.0 if f==0 else 1.25,
                    ls='--' if f==0 else '-',label=label,zorder=4 if f==0 else 2)
        ax.set_xlim(0,horizon);ax.set_xlabel('Time (min)');ax.set_ylabel(VARIABLE_NAMES[channel])
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
    plots=output/'plots';plots.mkdir(exist_ok=True)
    with PdfPages(output/'all_24_contexts.pdf') as pdf:
        for mechanism,definition in MECHANISMS.items():
            for context,_ in definition['contexts']:
                channels=list(dict.fromkeys([0,2,definition['channel'],1,3,4,20,22,32,33,35,26]))[:9]
                fig,axes=plt.subplots(3,3,figsize=(13.5,9.7),layout='constrained')
                for ax,ch in zip(axes.flat,channels):curves(ax,mechanism,context,ch)
                axes[0,0].legend(fontsize=8)
                zero,_=lookup[mechanism,context,0.,'tight']
                local=[r for r in stats if r['mechanism']==mechanism and r['context']==context and r['precision']=='tight']
                largest=max((r['rel_l2_all'] for r in local),default=float('nan'))
                fig.suptitle(f'{mechanism} | context: {context} | matched initial states and environment\n'
                             f'Tight BDF rtol=1e-8, atol=1e-10; zero: {zero.get("group","FAILED")}, '
                             f'{zero.get("native_divisions","?")} divisions; max paired RelL2={largest:.2%}',fontsize=12)
                fig.savefig(plots/f'{mechanism}__{context}.png',dpi=160);pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(8,3,figsize=(12,23),layout='constrained')
    for index,(mechanism,definition) in enumerate(MECHANISMS.items()):
        for ax,ch in zip(axes[index],[0,2,definition['channel']]):curves(ax,mechanism,'WT',ch)
        axes[index,0].set_title(mechanism,loc='left',fontweight='bold')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Exact zero vs MA residual: all eight mechanisms in WT background',fontsize=14)
    fig.savefig(output/'all_mechanisms_WT.png',dpi=150);plt.close(fig)
    # Show the worst context for each of three distinct mechanisms; the full
    # PDF includes every predefined context, including nearly equal curves.
    best={}
    for r in stats:
        key=r['mechanism']
        if r['precision']=='tight' and (key not in best or r['rel_l2_all']>best[key]['rel_l2_all']):best[key]=r
    top=sorted(best.values(),key=lambda r:r['rel_l2_all'],reverse=True)[:3]
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for index,r in enumerate(top):
        definition=MECHANISMS[r['mechanism']]
        channels=[0,2,VARIABLE_NAMES.index(r['worst_variable'])]
        for ax,ch in zip(axes[index],channels):curves(ax,r['mechanism'],r['context'],ch)
        axes[index,0].set_title(f'{r["mechanism"]} / {r["context"]}\n'
                              f'largest RelL2={r["rel_l2_all"]:.1%}; divisions {r["zero_divisions"]} -> {r["residual_divisions"]}',
                              loc='left',fontsize=9)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Three distinct mechanisms: largest differences among the 24 predefined contexts\n'
                 'All five strengths shown; tight-solver results; other contexts available in full PDF',fontsize=12)
    fig.savefig(output/'largest_differences.png',dpi=180);plt.close(fig)
    plot_key_examples(output,lookup,fractions,horizon)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--horizon',type=float,default=210.)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--plot-only',action='store_true',help='Reuse saved trajectories without simulating again')
    parser.add_argument('--verify-ultra',action='store_true',help='Verify saved tight runs at rtol=1e-10, atol=1e-12')
    parser.add_argument('--examples-only',action='store_true',help='Draw only the compact three-example figure from saved data')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fractions=[0.,.1,.5,.9,1.]
    if args.plot_only or args.verify_ultra or args.examples_only:
        saved=json.loads((args.output/'runs.json').read_text())
        with np.load(args.output/'trajectories.npz') as archive:
            values={int(i):y for i,y in zip(archive['run_indices'],archive['data'])}
        existing=[(r,values.get(i)) for i,r in enumerate(saved)]
        if args.examples_only:
            _,_,lookup=comparisons(existing)
            plot_key_examples(args.output,lookup,fractions,args.horizon)
            return
        if args.plot_only:
            rows,_,lookup=comparisons(existing)
            plot(args.output,existing,rows,lookup,fractions,args.horizon)
            return
        tasks=[dict(r,precision='ultra',rtol=1e-10,atol=1e-12,timeout_s=30.)
               for r,_ in existing if r['precision']=='tight']
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
            verified=list(executor.map(run,tasks))
        tight={(r['mechanism'],r['context'],r['alpha']):(r,y) for r,y in existing if r['precision']=='tight'}
        numeric=[]
        for r,y in verified:
            old,reference=tight[r['mechanism'],r['context'],r['alpha']]
            if y is None or reference is None:
                numeric.append(dict(mechanism=r['mechanism'],context=r['context'],alpha=r['alpha'],
                                    success=False,rel_l2=float('nan'),division_agreement=False,group_agreement=False))
            else:
                numeric.append(dict(mechanism=r['mechanism'],context=r['context'],alpha=r['alpha'],
                    success=True,rel_l2=rel_l2(reference,y),
                    division_agreement=old['native_divisions']==r['native_divisions'],group_agreement=old['group']==r['group']))
        write_csv(args.output/'ultra_verification.csv',numeric)
        zero={(r['mechanism'],r['context']):y for r,y in verified if r['alpha']==0.}
        paired=[dict(mechanism=r['mechanism'],context=r['context'],alpha=r['alpha'],
                     rel_l2_all=rel_l2(y,zero[r['mechanism'],r['context']]))
                for r,y in verified if r['alpha']>0 and y is not None and zero.get((r['mechanism'],r['context'])) is not None]
        write_csv(args.output/'ultra_paired_comparisons.csv',paired)
        np.savez_compressed(args.output/'ultra_trajectories.npz',
            data=np.array([y for _,y in verified if y is not None]),
            run_indices=np.array([i for i,(_,y) in enumerate(verified) if y is not None]))
        (args.output/'ultra_runs.json').write_text(json.dumps([r for r,_ in verified],indent=2))
        report=dict(runs=len(verified),successful=sum(r['success'] for r,_ in verified),
            median_tight_vs_ultra_rel_l2=float(np.nanmedian([r['rel_l2'] for r in numeric])),
            maximum_tight_vs_ultra_rel_l2=float(np.nanmax([r['rel_l2'] for r in numeric])),
            division_disagreements=sum(not r['division_agreement'] for r in numeric),
            group_disagreements=sum(not r['group_agreement'] for r in numeric))
        (args.output/'ultra_summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        return
    tasks=make_tasks(args.horizon,args.steps,fractions)
    (args.output/'design.json').write_text(json.dumps(dict(tasks=tasks,fractions=fractions,
        note='Hand-defined matched diagnostic backgrounds; no training or acquisition tuning'),indent=2))
    started=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        pending={executor.submit(run,task):i for i,task in enumerate(tasks)}
        indexed={}
        for future in as_completed(pending):
            indexed[pending[future]]=future.result()
            if len(indexed)%20==0:print(f'Solved {len(indexed)}/{len(tasks)}',flush=True)
        results=[indexed[i] for i in range(len(tasks))]
    rows,numeric,lookup=comparisons(results)
    write_csv(args.output/'comparisons.csv',rows);write_csv(args.output/'numerical_checks.csv',numeric)
    (args.output/'runs.json').write_text(json.dumps([r for r,_ in results],indent=2))
    good=[(i,r,y) for i,(r,y) in enumerate(results) if y is not None]
    np.savez_compressed(args.output/'trajectories.npz',run_indices=np.array([i for i,_,_ in good]),
        data=np.array([y for _,_,y in good]),time_grid=np.linspace(0,args.horizon,args.steps),
        variable_names=VARIABLE_NAMES,artifact_kind='zero_residual_diagnostic_not_training_data')
    per_mechanism=[]
    for mechanism in MECHANISMS:
        local=[r for r in rows if r['mechanism']==mechanism and r['precision']=='tight']
        worst=max(local,key=lambda r:r['rel_l2_all'])
        per_mechanism.append(dict(mechanism=mechanism,paired_comparisons=len(local),
            median_rel_l2=float(np.median([r['rel_l2_all'] for r in local])),
            maximum_rel_l2=worst['rel_l2_all'],worst_context=worst['context'],
            event_or_group_changes=sum(r['event_or_group_changed'] for r in local),
            bio_changes=sum(r['bio_changed'] for r in local)))
    summary=dict(runs=len(tasks),successful=len(good),failed=[r for r,y in results if y is None],
        contexts=24,strengths=5,paired_comparisons_tight=sum(r['precision']=='tight' for r in rows),
        horizon=args.horizon,steps=args.steps,elapsed_s=time.monotonic()-started,
        mechanisms=per_mechanism,numerical_disagreements=[r for r in numeric
            if r['production_divisions']!=r['tight_divisions'] or r['production_group']!=r['tight_group']],
        caution='Finite-window handpicked diagnostic contexts, not estimated prevalence or proof of asymptotic bifurcation',
        production_files_modified=False,training_or_pool_generation_started=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    plot(args.output,results,rows,lookup,fractions,args.horizon)
    print('All plots saved',flush=True)


if __name__=='__main__':main()
