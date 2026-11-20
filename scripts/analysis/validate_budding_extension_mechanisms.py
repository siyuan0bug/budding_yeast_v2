"""Mechanistic diagnostics for synthetic DAmP, SIC1-PM and related extensions.

No production files or datasets are modified. Hypothetical PM definitions and
depleted initial states are diagnostic controls, never candidate-pool recipes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data import simulation_guards
from budding_yeast_v2.data.systems.budding import simulation as sim
from budding_yeast_v2.data.systems.budding.distributions.mutant_aware import sample_scaling_factor
from budding_yeast_v2.data.systems.budding.mutations.mutant_aware import apply_gene_mutation
from budding_yeast_v2.data.systems.budding.ode import eqns
from budding_yeast_v2.data.systems.budding.parameters import get_default_params
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES

SIC1_FORMS = [4, 6, 7, 10, 11, 12]
CDC6_FORMS = [5, 8, 9, 13, 14, 15]
DAMP = {
    'CDC6': [('WT', []), ('CLN3_BCK2_KO', [('CLN3','KO',0),('BCK2','KO',0)]),
             ('CDH1_KO', [('CDH1','KO',0)])],
    'CDC20': [('WT', []), ('CDH1_KO', [('CDH1','KO',0)]), ('PDS1_KO',[('PDS1','KO',0)])],
    'CDC14': [('WT', []), ('CDH1_KO', [('CDH1','KO',0)]), ('NET1_TAB6',[('NET1','TAB6-1',.05)])],
    'NET1': [('WT', []), ('TEM1_KO',[('TEM1','KO',0)]), ('CDC15_KO',[('CDC15','KO',0)])],
}
PM_CONTEXTS = [('WT', []), ('CLN3_MC2',[('CLN3','MC',2)]), ('CLB5_KO',[('CLB5','KO',0)])]
PM_DEFINITIONS = {
    'WT': {}, 'db_only': {'kd3c1':0.}, 'current_PM': {'kd2c1':0.,'kd3c1':0.},
    'CDK_phosphorylation_off': {'kd2c1':0.},
    'all_phosphorylation_off': {'kd1c1':0.,'kd2c1':0.},
    'all_phosphorylation_and_decay_off': {'kd1c1':0.,'kd2c1':0.,'kd3c1':0.},
}


def assemble(instructions):
    defaults=get_default_params(); override={}
    for gene,mode,strength in instructions:
        apply_gene_mutation(gene,mode,strength,defaults,{},override,rng=np.random.RandomState(42))
    override['mdt']=150. if any(m in ('GAL-OE','GAL-MC') for _,m,_ in instructions) else 90.
    return override


def depleted_initial(y, gene, alpha):
    """Mass-consistent sensitivity control, not a mutant steady-state estimate."""
    y=y.copy()
    if gene=='CDC6':
        y[2]+=(1-alpha)*(y[8]+y[14]); y[3]+=(1-alpha)*(y[9]+y[15])
        y[CDC6_FORMS]*=alpha
    elif gene=='CDC20':
        y[[19,20]]*=alpha
    elif gene=='CDC14':
        # Release Net1 from removed unphosphorylated RENT.
        y[26]+=(1-alpha)*y[27]; y[[23,24,27]]*=alpha
    elif gene=='NET1':
        # Release Cdc14 from both RENT and implicit RENTP.
        y[24]+=(1-alpha)*(y[23]-y[24]); y[[25,26,27]]*=alpha
    elif gene=='SIC1':
        y[4]+=y[10]; y[6]+=y[11]; y[7]+=y[12]; y[[10,11,12]]=0.
    return y


def tasks(horizon, steps):
    result=[]
    def add(family,gene,context,variant,strength,override,initial='canonical',long=False):
        for precision,rtol,atol,timeout in [('production',1e-4,1e-6,5.),('tight',1e-8,1e-10,30.)]:
            if long and precision!='tight': continue
            result.append(dict(family=family,gene=gene,context=context,variant=variant,
                strength=float(strength),overrides=override,initial=initial,
                horizon=float(horizon*4 if long else horizon),steps=steps*4 if long else steps,
                precision=precision,rtol=rtol,atol=atol,timeout_s=timeout))
    for gene,contexts in DAMP.items():
        for context,instructions in contexts:
            for alpha in [0.,.05,.1,.2,.3,.4,1.]:
                # Zero and one are boundary controls, not DAmP draws.
                override=assemble(instructions+[(gene,'DAmP',alpha)])
                add('DAmP',gene,context,'expression_scale',alpha,override)
                if alpha==.2:
                    add('DAmP',gene,context,'expression_scale',alpha,override,'depleted')
                    for initial in ['canonical','depleted']:
                        add('DAmP',gene,context,'expression_scale',alpha,override,initial,True)
            add('DAmP',gene,context,'expression_scale',1.,assemble(instructions),long=True)
    for context,instructions in PM_CONTEXTS:
        for variant,changes in PM_DEFINITIONS.items():
            override=dict(assemble(instructions),**changes)
            add('SIC1_PM','SIC1',context,variant,1.,override)
            if variant in ('current_PM','db_only','all_phosphorylation_off'):
                add('SIC1_PM','SIC1',context,variant,1.,override,long=True)
                add('SIC1_PM','SIC1',context,variant,1.,override,'depleted')
    for gene,mode,values in [('CLN2','ts',[.01,.05,.1,.2,1.]),
                             ('CLB2','ts',[.01,.05,.1,.2,1.]),
                             ('SWI5','GAL-OE',[.02,.04,.06,.1]),
                             ('CLB2','pLOF',[.1,.2,.33,.5,1.])]:
        for alpha in values:
            add('other_extensions',gene,'WT',mode,alpha,assemble([(gene,mode,alpha)]))
    for index,task in enumerate(result): task['run_id']=index
    return result


def totals(y):
    return {'SIC1':y[SIC1_FORMS].sum(axis=0),'CDC6':y[CDC6_FORMS].sum(axis=0),
            'CDC20':y[19],'CDC14':y[23],'NET1':y[25],
            'CLN2':y[1],'CLB2':y[[2,6,8,11,14]].sum(axis=0),'SWI5':y[16]}


def run(task):
    started=time.monotonic(); native_solve=sim.solve_ivp; native_prepare=sim.prepare_initial_conditions
    division_times=[]
    def prepare(override):
        p,y,mad,bub=native_prepare(override)
        if task['initial']=='depleted': y=depleted_initial(y,task['gene'],task['strength'])
        return p,y,mad,bub
    def solve(*args,**kwargs):
        kwargs.update(rtol=task['rtol'],atol=task['atol'])
        solution=native_solve(*args,**kwargs)
        if solution.status==1 and len(solution.t_events[0]):
            division_times.extend(float(t) for t in solution.t_events[0])
        return solution
    p,y0,_,_=prepare(task['overrides'])
    with patch.object(sim,'solve_ivp',solve), patch.object(sim,'prepare_initial_conditions',prepare), \
         patch.object(simulation_guards,'DEFAULT_SIM_TIMEOUT',task['timeout_s']):
        _,y,bio,abc=sim.simulate_mutant('extension_diagnostic',task['overrides'],tf=task['horizon'],ns=task['steps'])
    row=dict(task,success=y is not None,seconds=time.monotonic()-started,bio=bio,abc=abc,
             native_divisions=len(division_times),division_times=division_times)
    if y is not None:
        group=pattern_group_metadata(y)
        row.update(group=str(group['pattern_group_codes'][0]),minimum=float(y.min()),maximum=float(y.max()),
                   mass_peak=float(y[0].max()),clb2_peak=float(y[2].max()),ori_peak=float(y[33].max()))
        target=totals(y)[task['gene']]
        row.update(target_initial=float(target[0]),target_end=float(target[-1]),target_peak=float(target.max()),
            target_tail_mean=float(target[-max(1,task['steps']//4):].mean()))
        # Implicit species must be inspected in addition to explicit states.
        implicit=np.stack([y[23]-y[27]-y[24],y[25]-y[26]-y[23]+y[24]])
        row['implicit_pool_min']=float(implicit.min())
        if task['gene']=='SIC1':
            row['phosphorylated_fraction_end']=float(y[[10,11,12],-1].sum()/max(target[-1],1e-12))
            row['sic1_total_min_increment']=float(np.diff(target).min())
            derivative=eqns(0,y0,p)
            row['sic1_total_derivative_residual']=float(derivative[SIC1_FORMS].sum()-(
                p['ksc1p']+p['ksc1pp']*y0[17]-p['kd3c1']*y0[[10,11,12]].sum()))
        if task['family']=='DAmP' and task['gene'] in ('CDC14','NET1'):
            synthesis,decay,index=('ks14','kd14',23) if task['gene']=='CDC14' else ('ksnet','kdnet',25)
            times=np.linspace(0,task['horizon'],task['steps'])
            exact=p[synthesis]/p[decay]+(y0[index]-p[synthesis]/p[decay])*np.exp(-p[decay]*times)
            row['total_analytic_max_abs_error']=float(np.abs(y[index]-exact).max())
    return row,y


def distance(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def compare(results):
    lookup={(r['family'],r['gene'],r['context'],r['variant'],r['strength'],r['initial'],r['horizon'],r['precision']):(r,y)
            for r,y in results}
    rows=[]
    for r,y in results:
        if y is None: continue
        key=(r['family'],r['gene'],r['context'],r['variant'],r['strength'],r['initial'],r['horizon'],r['precision'])
        references=[]
        if r['precision']=='production': references.append(('numerical',key[:-1]+('tight',)))
        if r['initial']=='depleted': references.append(('initial_state',key[:5]+('canonical',)+key[6:]))
        if r['family']=='SIC1_PM' and r['variant']=='current_PM':
            for variant in ['db_only','all_phosphorylation_off']:
                references.append((f'current_PM_vs_{variant}',key[:3]+(variant,)+key[4:]))
        if r['family']=='DAmP' and r['initial']=='canonical' and r['strength'] not in (0.,1.):
            for alpha in [0.,1.]:
                references.append(('vs_zero' if alpha==0 else 'vs_WT_expression',key[:4]+(alpha,)+key[5:]))
        for kind,ref in references:
            other,z=lookup.get(ref,(None,None))
            if z is None: continue
            rows.append(dict(kind=kind,run_id=r['run_id'],reference_run_id=other['run_id'],
                gene=r['gene'],context=r['context'],variant=r['variant'],strength=r['strength'],
                horizon=r['horizon'],precision=r['precision'],initial=r['initial'],
                rel_l2_all=distance(y,z),log_rel_l2=distance(np.log1p(np.maximum(y,0)),np.log1p(np.maximum(z,0))),
                same_divisions=r['native_divisions']==other['native_divisions'],
                same_group=r['group']==other['group'],same_bio=r['bio']==other['bio']))
    return rows,lookup


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def equation_checks():
    """Verify balance laws and expression scaling at matched physical states."""
    default=get_default_params(); y0=sim.prepare_initial_conditions({})[1]
    rng=np.random.RandomState(42); balance=[]; scale=[]; initials=[]
    for _ in range(100):
        y=y0.copy();y[SIC1_FORMS]*=rng.uniform(.1,3.,len(SIC1_FORMS))
        for definition in PM_DEFINITIONS.values():
            p=dict(default,**definition);dy=eqns(0,y,p)
            expected=p['ksc1p']+p['ksc1pp']*y[17]-p['kd3c1']*y[[10,11,12]].sum()
            balance.append(abs(float(dy[SIC1_FORMS].sum()-expected)))
    for gene in DAMP:
        zero=sim.prepare_initial_conditions(assemble([(gene,'DAmP',0.)]))[0]
        one=sim.prepare_initial_conditions(assemble([(gene,'DAmP',1.)]))[0]
        f0,f1=eqns(0,y0,zero),eqns(0,y0,one)
        for alpha in [.05,.1,.2,.3,.4]:
            p,y,*_=sim.prepare_initial_conditions(assemble([(gene,'DAmP',alpha)]))
            scale.append(float(np.abs(eqns(0,y0,p)-(f0+alpha*(f1-f0))).max()))
            initials.append(bool(np.array_equal(y,y0)))
            depleted=depleted_initial(y0,gene,alpha)
            assert min(depleted[23]-depleted[27]-depleted[24],
                       depleted[25]-depleted[26]-depleted[23]+depleted[24])>=-1e-12
    assert max(balance)<1e-12 and max(scale)<1e-12 and all(initials)
    return dict(sic1_balance_cases=len(balance),sic1_max_balance_error=max(balance),
                fixed_state_DAmP_affine_error=max(scale),DAmP_initial_states_unchanged=all(initials))


def plot(output,lookup,horizon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
    def get(family,gene,context,variant,strength,initial='canonical',span=horizon):
        return lookup.get((family,gene,context,variant,strength,initial,span,'tight'),(None,None))
    def curves(ax,family,gene,context,channel,variants,span=horizon,initial='canonical'):
        for variant,alpha in variants:
            r,y=get(family,gene,context,variant,alpha,initial,span)
            if y is None: continue
            data=totals(y)[gene] if channel=='total' else y[channel]
            ax.plot(np.linspace(0,span,y.shape[-1]),data,label=f'alpha={alpha:g}' if family=='DAmP' else variant,lw=1.4)
        ax.set_xlabel('Time (min)');ax.set_ylabel(f'{gene} total' if channel=='total' else VARIABLE_NAMES[channel])
    with PdfPages(output/'all_mechanism_contexts.pdf') as pdf:
        for gene,contexts in DAMP.items():
            for context,_ in contexts:
                fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
                for ax,ch in zip(axes.flat,['total',{'CDC6':5,'CDC20':20,'CDC14':24,'NET1':26}[gene],0,2,3,33]):
                    curves(ax,'DAmP',gene,context,ch,[('expression_scale',a) for a in [0,.05,.2,.4,1]])
                axes[0,0].legend(fontsize=8);fig.suptitle(f'{gene}-DAmP | {context} | production initial state; tight BDF')
                fig.savefig(output/f'{gene}_DAmP__{context}.png',dpi=140);pdf.savefig(fig);plt.close(fig)
        for context,_ in PM_CONTEXTS:
            fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
            for ax,ch in zip(axes.flat,['total',4,2,3,33,0]):
                curves(ax,'SIC1_PM','SIC1',context,ch,[(v,1) for v in PM_DEFINITIONS])
            axes[0,0].legend(fontsize=6);fig.suptitle(f'SIC1 definitions | {context} | hypothetical controls are not pool recipes')
            fig.savefig(output/f'SIC1_PM__{context}.png',dpi=160);pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(13,13),layout='constrained')
    for (gene,_),row in zip(DAMP.items(),axes):
        for ax,ch in zip(row,['total',2,0]):
            curves(ax,'DAmP',gene,'WT',ch,[('expression_scale',a) for a in [0,.05,.2,.4,1]])
        row[0].set_title(f'{gene}-DAmP',loc='left',fontweight='bold')
    axes[0,0].legend(fontsize=8);fig.suptitle('DAmP strength sweep; alpha=0 and 1 are boundary controls')
    fig.savefig(output/'DAmP_overview.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for (gene,_),row in zip(DAMP.items(),axes):
        for ax,ch in zip(row,['total',2]):
            for initial in ['canonical','depleted']:
                r,y=get('DAmP',gene,'WT','expression_scale',.2,initial,4*horizon)
                if y is None: continue
                ax.plot(np.linspace(0,4*horizon,y.shape[-1]),totals(y)[gene] if ch=='total' else y[ch],label=initial)
            ax.set_ylabel(gene+' total' if ch=='total' else 'CLB2');ax.set_xlabel('Time (min)')
    axes[0,0].legend();fig.suptitle('Initial-state sensitivity at alpha=0.2; depleted is a diagnostic, not a fitted steady state')
    fig.savefig(output/'DAmP_initial_state_sensitivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    styles=[('WT','WT','#777777'),('db_only','Degradation off','#df8b21'),
            ('current_PM','Current PM','#258348'),('all_phosphorylation_off','All phosphorylation off','#6b55af')]
    for variant,label,color in styles:
        r,y=get('SIC1_PM','SIC1','WT',variant,1.)
        if y is None: continue
        total=totals(y)['SIC1'];times=np.linspace(0,horizon,y.shape[-1])
        for ax,data in zip(axes,[total,y[[10,11,12]].sum(axis=0)/np.maximum(total,1e-12),y[33]]):
            ax.plot(times,data,label=label,color=color,lw=1.7,ls='--' if variant=='current_PM' else '-')
    for ax,label in zip(axes,['Total Sic1','Phosphorylated fraction of Sic1','ORI']):
        ax.set_xlabel('Time (min)');ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle('SIC1-PM: similar arrest phenotype does not imply identical molecular mechanism')
    fig.savefig(output/'PM_key_comparison.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--horizon',type=float,default=210.)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--verify-ultra',action='store_true',help='Recheck largest numerical deviations and key mechanisms')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.workers<1 or args.steps<2 or args.horizon<=0:
        parser.error('workers >=1, steps >=2 and horizon >0 are required')
    protected=list((Path(__file__).resolve().parents[2]/'data').rglob('*.py'))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    if args.plot_only or args.verify_ultra:
        rows=json.loads((args.output/'runs.json').read_text())
        with np.load(args.output/'trajectories.npz') as f: values={int(i):f[k] for i,k in zip(f['run_ids'],f['keys'])}
        results=[(r,values.get(r['run_id'])) for r in rows]
        if args.verify_ultra:
            previous,lookup=compare(results)
            ids={int(r['reference_run_id']) for r in previous
                 if r['kind']=='numerical' and r['rel_l2_all']>.01}
            ids.update(r['run_id'] for r,y in results if r['precision']=='tight' and r['horizon']==args.horizon
                       and r['context']=='WT' and ((r['family']=='DAmP' and r['strength']==.2)
                       or (r['family']=='SIC1_PM' and r['variant'] in ('db_only','current_PM','all_phosphorylation_off'))))
            baseline={r['run_id']:(r,y) for r,y in results}
            todo=[dict(baseline[i][0],precision='ultra',rtol=1e-10,atol=1e-12,timeout_s=45.) for i in sorted(ids)]
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
                checked=list(executor.map(run,todo))
            verification=[]
            for r,y in checked:
                old,z=baseline[r['run_id']]
                verification.append(dict(run_id=r['run_id'],gene=r['gene'],context=r['context'],
                    variant=r['variant'],strength=r['strength'],initial=r['initial'],success=y is not None,
                    tight_vs_ultra_rel_l2=distance(z,y) if y is not None else None,
                    tight_divisions=old['native_divisions'],ultra_divisions=r['native_divisions'],
                    tight_group=old['group'],ultra_group=r.get('group')))
            write_csv(args.output/'ultra_verification.csv',verification)
            print(json.dumps(verification,indent=2),flush=True)
            return
    else:
        todo=tasks(args.horizon,args.steps); results=[]
        print(f'{len(todo)} simulations; {args.workers} workers',flush=True)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
            pending=[executor.submit(run,t) for t in todo]
            for future in as_completed(pending):
                results.append(future.result())
                if len(results)%30==0: print(f'Completed {len(results)}/{len(todo)}',flush=True)
        results.sort(key=lambda x:x[0]['run_id'])
        (args.output/'runs.json').write_text(json.dumps([r for r,_ in results],indent=2))
        valid=[(r,y) for r,y in results if y is not None]
        np.savez_compressed(args.output/'trajectories.npz',run_ids=[r['run_id'] for r,y in valid],
            keys=[f'run_{r["run_id"]}' for r,y in valid],**{f'run_{r["run_id"]}':y for r,y in valid})
        write_csv(args.output/'runs.csv',[{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r,_ in results])
    comparisons,lookup=compare(results);write_csv(args.output/'comparisons.csv',comparisons)
    rng=np.random.RandomState(42)
    draws=np.array([sample_scaling_factor('DAmP','CDC20',rng=rng) for _ in range(100000)])
    summary=dict(runs=len(results),successes=sum(r['success'] for r,_ in results),
        families={family:sum(r['family']==family for r,_ in results) for family in ['DAmP','SIC1_PM','other_extensions']},
        DAmP_law=dict(n=len(draws),mean=float(draws.mean()),std=float(draws.std()),
            quantiles=np.quantile(draws,[0,.01,.05,.5,.95,.99,1]).tolist(),
            below_point_one=float((draws<.1).mean())),
        numerical_max_rel_l2=max((r['rel_l2_all'] for r in comparisons if r['kind']=='numerical'),default=None),
        data_source_hashes_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in before.items()),
        equation_checks=equation_checks(),
        source_hashes=before,
        caveats=['Mechanistic consistency does not establish gene-specific wet-lab calibration or training benefit.',
                 'Zero/one strength, alternative PM rules and depleted initial states are diagnostic controls only.',
                 'No real trajectories, prediction errors or test identities used to tune any distribution.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    plot(args.output,lookup,args.horizon)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2),flush=True)


if __name__=='__main__': main()
