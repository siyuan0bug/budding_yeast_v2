"""Paired distribution ablations and source-preserving derivative datasets."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from .contracts import Candidate, SimulationResult
from .io import read_dataset, write_dataset
from .policies import level_schedule, parse_pairs
from .runner import ensure_candidate, run_simulations, _parameter_result
from ..sampling.factory import Sampler
from ..systems.budding.specification import BuddingSystem, SPEC
from ..systems.budding import real_mutants
from ..recipes import get_recipe


def _report(config):
    return Counter({'collisions':0,'rescued':0,'recipe':config.recipe,
                    'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware',
                    'rng_contract':'paired_legacy_algorithms_v1','collision_policy':'rescue'})


def _real_candidates(include):
    return [Candidate(-i-1,n,o,0,True) for i,(n,o) in enumerate(real_mutants.real_tasks())] if include else []


def _save(path,results,config,report,mutation='mutant_aware',distribution='mutant_aware',dtype='float32'):
    report=dict(report,saved_real=sum(r.candidate.is_real for r in results),
                saved_synthetic=sum(not r.candidate.is_real for r in results),
                mutation_profile=mutation,distribution_profile=distribution)
    return write_dataset(path,SPEC,results,replace(config,output=str(path),mutation_profile=mutation,
                          distribution_profile=distribution),report,dtype=dtype)


def _source_profiles(archive,report):
    mutation=str(archive.get('mutation_profile','legacy_unspecified'))
    distribution=str(archive.get('distribution_profile','legacy_unspecified'))
    report['source_mutation_profile']=mutation
    report['source_distribution_profile']=distribution
    return mutation,distribution


def _check_outputs(paths,config):
    resolved=[Path(p).expanduser().resolve() for p in paths]
    if len(set(resolved))!=len(resolved):
        raise ValueError('Derived output paths must differ')
    for path in resolved:
        if path.suffix!='.npz':raise ValueError('Output must end in .npz')
        if config.source and path==Path(config.source).expanduser().resolve():
            raise ValueError('A derived dataset must never overwrite its source')
        if path.exists() and not config.overwrite:raise FileExistsError(path)


def _paired_results(left,right,config):
    if len(left)!=len(right):
        raise ValueError('Mismatched paired candidate counts')
    a=list(run_simulations(left,config,None))
    b=list(run_simulations(right,config,None))
    if len(a)!=len(left) or len(b)!=len(right):
        raise ValueError('Incomplete paired simulation results')
    if any((x.candidate.sample_id,x.candidate.pair_id)!=(y.candidate.sample_id,y.candidate.pair_id)
           for x,y in zip(a,b)):
        raise ValueError('Mismatched paired simulation identities')
    keep=[i for i,(x,y) in enumerate(zip(a,b)) if x.trajectory is not None and y.trajectory is not None
          and x.status==y.status=='success'
          and x.trajectory.shape==y.trajectory.shape==(len(SPEC.variable_names),SPEC.steps)
          and np.isfinite(x.trajectory).all() and np.isfinite(y.trajectory).all()]
    return [a[i] for i in keep],[b[i] for i in keep]


def _twin_rescue(ovr_b,ovr_a,desc,level,number,sampler,report):
    """Change a colliding form once, then rebuild both arms on that shared form."""
    from ..sampling.twins import TARGET_MODES, apply_gene_mutation_A, sample_scaling_factor_A
    backend=sampler.backend
    initial=Candidate(number,f'MASS_{desc}_L{level}_{number}',ovr_b,level,pair_id=number,
                      metadata={'description':desc})
    for _ in range(30):
        if not real_mutants.is_duplicate_real_mutant(ovr_b,backend.defaults()) and not real_mutants.is_duplicate_real_mutant(ovr_a,backend.defaults()):
            name=f'MASS_{desc}_L{level}_{number}'.replace(' ','_')
            return replace(initial,name=name,overrides=ovr_b,metadata={'description':desc}),replace(initial,name=name,overrides=ovr_a,metadata={'description':desc})
        # If only A collides, run the same rescue policy against A's current form.
        source=ovr_b if real_mutants.is_duplicate_real_mutant(ovr_b,backend.defaults()) else ovr_a
        fixed=ensure_candidate(replace(initial,overrides=source,metadata={'description':desc}),sampler,report)
        if parse_pairs(desc)==parse_pairs(fixed.metadata['description']):
            # A carbon-only rescue must not replace B's doses with A's doses,
            # or redraw target doses on either side.
            ovr_b=dict(ovr_b,mdt=fixed.overrides['mdt'])
            ovr_a=dict(ovr_a,mdt=fixed.overrides['mdt'])
            desc=fixed.metadata['description']
            continue
        ovr_b,desc=fixed.overrides,fixed.metadata['description']
        ovr_a={'mdt':ovr_b['mdt']}
        for gene,mode in parse_pairs(desc):
            if mode in TARGET_MODES:
                alpha=sample_scaling_factor_A(mode,gene) if mode in ('ts','db') else 1.
                apply_gene_mutation_A(gene,mode,alpha,backend.defaults(),{},ovr_a)
            else:
                for key in backend.touch_set(gene,mode):
                    if key in ovr_b:ovr_a[key]=ovr_b[key]
    raise RuntimeError('Paired collision rescue exhausted; no mismatched pair was exported')


def generate_twins(config):
    from ..sampling.twins import sample_twin_mutant_pair
    if not config.paired_output:
        output=Path(config.output)
        config=replace(config,paired_output=str(output.with_name(output.stem+'_mass.npz')))
    if config.mutation_profile or config.distribution_profile:
        raise ValueError('Twins explicitly compare the two existing profiles; do not override one arm')
    _check_outputs([config.output,config.paired_output],config)
    recipe=get_recipe('budding_random')
    sampler=Sampler(recipe,config)
    report=_report(config)
    left=_real_candidates(config.include_real)
    right=[replace(c,overrides=dict(c.overrides)) for c in left]
    schedule=level_schedule(config.n_samples,config.level_ratios or (.25,)*4,np.random)
    for number,level in enumerate(schedule,1):
        b,a,desc=sample_twin_mutant_pair(int(level),sampler.backend.defaults(),ma=sampler.backend)
        x,y=_twin_rescue(b,a,desc,int(level),number,sampler,report)
        left.append(x);right.append(y)
    a,b=_paired_results(left,right,config)
    report['paired_failures']=len(left)-len(a)
    return {'mutant_aware':_save(config.output,a,config,report),
            'mass':_save(config.paired_output,b,config,report,'mass','mass')}


def _source_rows(config):
    if not config.source:raise ValueError('This recipe requires --source')
    archive=read_dataset(config.source)
    if list(archive['param_names'])!=list(SPEC.param_names) or archive['data'].shape[1:]!=(39,500):
        raise ValueError('Source must use the Budding 141-parameter / 39-variable / 500-point contract')
    if 'system_id' in archive and str(archive['system_id'])!='budding':raise ValueError('Source is not Budding')
    if 'time_grid' in archive and not np.array_equal(archive['time_grid'],np.linspace(0,210,500)):
        raise ValueError('Source has a different physical time grid')
    n=len(archive['mutant_names'])
    if archive['params'].shape!=(n,141) or archive['data'].shape!=(n,39,500):
        raise ValueError('Source parameter/trajectory row counts do not match')
    for key in ('labels','pattern_labels','sample_id','pair_id','is_real','levels','overrides_json',
                'candidate_metadata_json','simulation_diagnostics_json','simulation_status','y_init'):
        if key in archive and len(archive[key])!=n:
            raise ValueError(f'Source {key} row count does not match')
    if not np.isfinite(archive['params']).all() or not np.isfinite(archive['data']).all():
        raise ValueError('Source contains nonfinite parameter/trajectory values')
    backend=BuddingSystem()
    defaults=backend.defaults()
    rows=[]
    for i,name in enumerate(archive['mutant_names']):
        name=str(name)
        is_real=bool(archive['is_real'][i]) if 'is_real' in archive else name.split('_',1)[0] in real_mutants.REAL_MUTANTS
        if 'overrides_json' in archive:
            overrides=json.loads(str(archive['overrides_json'][i]))
        elif is_real:
            overrides=dict(real_mutants.REAL_MUTANTS[name.split('_',1)[0]][1])
        else:
            overrides=dict(zip(SPEC.param_names,map(float,archive['params'][i])))
        if is_real:
            identity=name.split('_',1)[0]
            if identity not in real_mutants.REAL_MUTANTS or overrides!=real_mutants.REAL_MUTANTS[identity][1]:
                raise ValueError('Source real overrides do not match the canonical real identity')
        expected=[overrides.get(k,defaults[k]) for k in SPEC.param_names]
        if not np.allclose(archive['params'][i],expected,rtol=1e-6,atol=1e-9):
            raise ValueError('Source parameter values conflict with override metadata')
        level=int(archive['levels'][i]) if 'levels' in archive else (0 if is_real else _parse_level(name))
        metadata=json.loads(str(archive['candidate_metadata_json'][i])) if 'candidate_metadata_json' in archive else {}
        candidate=Candidate(int(archive['sample_id'][i]) if 'sample_id' in archive else (-i-1 if is_real else i+1),
                            name,overrides,level,is_real,
                            pair_id=int(archive['pair_id'][i]) if 'pair_id' in archive else -1,metadata=metadata)
        _,y0,_,_=backend.prepare_initial_conditions(overrides)
        if not np.allclose(archive['data'][i,:,0],y0,rtol=1e-6,atol=1e-9):
            raise ValueError('Source initial state does not match canonical MA biology; regenerate it explicitly')
        if 'y_init' in archive and not np.allclose(archive['y_init'][i],y0,rtol=1e-6,atol=1e-9):
            raise ValueError('Source stored initial state conflicts with canonical MA biology')
        status=str(archive['simulation_status'][i]) if 'simulation_status' in archive else 'success'
        if status!='success':raise ValueError('Source contains failed simulation status')
        rows.append(SimulationResult(candidate,archive['data'][i].copy(),archive['params'][i].copy(),
                    np.array(archive['y_init'][i],copy=True) if 'y_init' in archive else y0,
                    str(archive['labels'][i]),str(archive['pattern_labels'][i]),status,
                    json.loads(str(archive['simulation_diagnostics_json'][i])) if 'simulation_diagnostics_json' in archive else {}))
    return archive,rows


def _parse_level(name):
    import re
    match=re.search(r'_L(\d+)_',name)
    return int(match.group(1)) if match else -1


def generate_ko_mirror(config):
    _check_outputs([config.output],config)
    archive,rows=_source_rows(config)
    if not config.include_real:
        rows=[row for row in rows if not row.candidate.is_real]
    backend=BuddingSystem()
    rng=np.random.RandomState(config.seed)
    tasks=[]
    indices=[]
    for i,row in enumerate(rows):
        if row.candidate.is_real or '[KO]' not in row.candidate.name:continue
        candidate=row.candidate
        overrides=dict(candidate.overrides)
        # Identical key iteration/draw order to the original KO mirror.
        for gene,mode in parse_pairs(candidate.name):
            if mode!='KO':continue
            keys={}
            backend.apply_gene_mutation(gene,'KO',0.,backend.defaults(),{},keys)
            for key in keys:
                if key in ('init_CDH1','init_Tem1','init_Cdc15'):continue
                overrides[key]=float(rng.uniform(0.,config.ko_interval_max))
            for total,active in [('init_CDH1T','init_CDH1'),('Tem1T','init_Tem1'),('Cdc15T','init_Cdc15')]:
                if total in keys:overrides[active]=float(rng.uniform(0.,overrides[total]))
        tasks.append(replace(candidate,overrides=overrides));indices.append(i)
    results=list(run_simulations(tasks,config,None))
    if any(r.trajectory is None for r in results):
        raise RuntimeError('KO mirror re-simulation failed; refusing new parameters paired with old trajectories')
    from ..sampling.factory import candidate_name
    sampler=Sampler(get_recipe('budding_random'),config)
    report=_report(config)
    for i,result in zip(indices,results):
        candidate=replace(result.candidate,metadata=dict(result.candidate.metadata,
                          description='+'.join(f'{g}[{m}]' for g,m in parse_pairs(result.candidate.name))))
        fixed=ensure_candidate(candidate,sampler,report)
        if fixed.overrides!=candidate.overrides:
            result=next(run_simulations([fixed],config,None))
            if result.trajectory is None:raise RuntimeError('KO mirror collision rescue failed to simulate')
        rows[i]=result
    report['recomputed']=len(results)
    digest=hashlib.sha256()
    with Path(config.source).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    report['source_sha256']=digest.hexdigest()
    profiles=_source_profiles(archive,report)
    return _save(config.output,rows,config,report,*profiles,dtype=str(archive['data'].dtype))


def generate_bams_ablation(config):
    from ..sampling.targeted_ablation import targeted_pair, TARGETED_SKELETONS
    output=Path(config.output)
    hard=output.with_name(output.stem+'_hard.npz')
    hypo=output.with_name(output.stem+'_hypo.npz')
    _check_outputs([output,hard,hypo],config)
    archive,rows=_source_rows(config)
    real=[r for r in rows if r.candidate.is_real] if config.include_real else []
    synthetic=[r for r in rows if not r.candidate.is_real]
    if len(synthetic)<config.n_samples:raise ValueError('Source has fewer synthetic samples than the requested control budget')
    count=max(1,config.n_samples//5) if config.n_samples else 0
    base=real+synthetic[:config.n_samples-count]
    sampler=Sampler(get_recipe('budding_random'),config)
    report=_report(config)
    left,right=[],[]
    next_sample_id=max(0,max((r.candidate.sample_id for r in rows),default=0))+1
    next_pair_id=max((r.candidate.pair_id for r in rows),default=-1)+1
    for i in range(count):
        a,b,desc=targeted_pair(i)
        # A shared carbon change is a valid rescue for this diagnostic pair and
        # preserves the hard-KO versus hypomorphic treatment.
        if real_mutants.is_duplicate_real_mutant(a,sampler.backend.defaults()) or real_mutants.is_duplicate_real_mutant(b,sampler.backend.defaults()):
            report['collisions']+=1
            rescued=False
            for mdt in (150.,160.,90.):
                a,b=dict(a,mdt=mdt),dict(b,mdt=mdt)
                if not real_mutants.is_duplicate_real_mutant(a,sampler.backend.defaults()) and not real_mutants.is_duplicate_real_mutant(b,sampler.backend.defaults()):
                    rescued=True;report['rescued']+=1;break
            if not rescued:raise RuntimeError('Targeted paired rescue failed without altering the ablation treatment')
        skeleton=TARGETED_SKELETONS[i%len(TARGETED_SKELETONS)]
        genes=[item[0] if isinstance(item,tuple) else item for item in skeleton['lofs']]+[g for g,_ in skeleton['gofs']]
        level=len(set(genes))
        left.append(Candidate(next_sample_id+i,f'BAMS_TARGET_HARD_{i}_{desc}',a,level,pair_id=next_pair_id+i))
        right.append(Candidate(next_sample_id+i,f'BAMS_TARGET_HYPO_{i}_{desc}',b,level,pair_id=next_pair_id+i))
    a,b=_paired_results(left,right,config)
    report['paired_failures']=len(left)-len(a)
    report['diagnostic_targeted']=True
    profiles=_source_profiles(archive,report)
    if not a:
        mixed=profiles
    elif not any(not result.candidate.is_real for result in base):
        mixed=('mutant_aware','mutant_aware')
    else:
        mixed=tuple(p if p=='mutant_aware' else 'mixed' for p in profiles)
    return {'control':_save(output,real+synthetic[:config.n_samples],config,report,*profiles),
            'hard':_save(hard,base+a,config,report,*mixed),'hypo':_save(hypo,base+b,config,report,*mixed)}


def generate_derived(config):
    state=np.random.get_state();py_state=random.getstate()
    np.random.seed(config.seed);random.seed(config.seed)
    try:
        kind=get_recipe(config.recipe).sampler
        return {'twins':generate_twins,'ko_mirror':generate_ko_mirror,'bams_ablation':generate_bams_ablation}[kind](config)
    finally:
        np.random.set_state(state);random.setstate(py_state)
