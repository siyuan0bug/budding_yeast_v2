"""Paired distribution ablations and source-preserving derivative datasets."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from ..contracts import Candidate, SimulationResult
from ..sampling_policy import VERSION, STATIC_RATIOS, valid_candidate
from ..dataset_io import read_dataset, write_dataset
from ..generation import level_schedule, ensure_candidate, run_simulations
from ..registry import make_sampler, candidate_name, get_recipe
from .specification import BuddingSystem, SPEC
from . import real_mutants
from .samplers.rescue import parse_pairs


def _report(config):
    return Counter({'collisions':0,'rescued':0,'recipe':config.recipe,
                    'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware',
                    'rng_contract':'paired_legacy_algorithms_v1','sampling_policy':VERSION,
                    'collision_policy':'fixed_modules_environment_v2'})


def _real_candidates(include):
    return [Candidate(-i-1,n,o,0,True) for i,(n,o) in enumerate(real_mutants.real_tasks())] if include else []


def _save(path,results,config,report,mutation='mutant_aware',distribution='mutant_aware',dtype='float32'):
    if config.include_real and not config.source and sum(r.candidate.is_real for r in results)!=126:
        raise RuntimeError('Incomplete paired/source real benchmark; refusing export')
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
    """Rescue one allele in a pair; preserve both environments and other doses."""
    from .samplers.twins import TARGET_MODES, apply_gene_mutation_A, sample_scaling_factor_A
    backend=sampler.backend
    initial=Candidate(number,f'MASS_{desc}_L{level}_{number}',ovr_b,level,pair_id=number,
                      metadata={'description':desc})
    for _ in range(30):
        if not real_mutants.is_duplicate_real_mutant(ovr_b,backend.defaults()) and not real_mutants.is_duplicate_real_mutant(ovr_a,backend.defaults()):
            name=f'MASS_{desc}_L{level}_{number}'.replace(' ','_')
            return replace(initial,name=name,overrides=ovr_b,metadata={'description':desc}),replace(initial,name=name,overrides=ovr_a,metadata={'description':desc})
        # If only A collides, run the same rescue policy against A's current form.
        from copy import copy
        from .specification import BuddingSystem
        source_is_b=real_mutants.is_duplicate_real_mutant(ovr_b,backend.defaults())
        source=ovr_b if source_is_b else ovr_a
        local=copy(sampler)
        if not source_is_b:local.backend=BuddingSystem('mass','mass')
        fixed=ensure_candidate(replace(initial,overrides=source,metadata={'description':desc}),local,report)
        before,after=parse_pairs(desc),parse_pairs(fixed.metadata['description'])
        if len(before)!=len(after) or any(g!=h for (g,_),(h,_) in zip(before,after)):
            raise RuntimeError('Paired rescue changed the module skeleton')
        changed=[i for i,((g,m),(_,new_mode)) in enumerate(zip(before,after))
                 if m!=new_mode or any(source.get(key)!=fixed.overrides.get(key)
                    for key in backend.touch_set(g,m)|backend.touch_set(g,new_mode))]
        if len(changed)!=1 or source.get('mdt',90.)!=fixed.overrides.get('mdt',90.):
            raise RuntimeError('Paired rescue must change one mutation and preserve mdt')
        index=changed[0];gene,old_mode=before[index];_,mode=after[index]
        other=dict(ovr_a if source_is_b else ovr_b)
        for key in backend.touch_set(gene,old_mode):other.pop(key,None)
        if mode not in TARGET_MODES:
            for key in backend.touch_set(gene,mode):
                if key in fixed.overrides:other[key]=fixed.overrides[key]
        elif source_is_b:
            alpha=sample_scaling_factor_A(mode,gene) if mode in ('ts','db') else 1.
            apply_gene_mutation_A(gene,mode,alpha,backend.defaults(),{},other)
        else:
            alpha=backend.sample_scaling_factor(mode,gene) if mode in ('ts','db') else 1.
            backend.apply_gene_mutation(gene,mode,alpha,backend.defaults(),{},other)
        ovr_b,ovr_a=(fixed.overrides,other) if source_is_b else (other,fixed.overrides)
        desc=fixed.metadata['description']
    raise RuntimeError('Paired collision rescue exhausted; no mismatched pair was exported')


def generate_twins(config):
    from .samplers.twins import sample_twin_mutant_pair
    if not config.paired_output:
        output=Path(config.output)
        config=replace(config,paired_output=str(output.with_name(output.stem+'_mass.npz')))
    if config.mutation_profile or config.distribution_profile:
        raise ValueError('Twins explicitly compare the two existing profiles; do not override one arm')
    _check_outputs([config.output,config.paired_output],config)
    recipe=get_recipe('budding_random')
    sampler=make_sampler(recipe,config)
    report=_report(config)
    left=_real_candidates(config.include_real)
    right=[replace(c,overrides=dict(c.overrides)) for c in left]
    schedule=level_schedule(config.n_samples,config.level_ratios or STATIC_RATIOS,np.random)
    # Simulate in parallel BATCHES; refill failures at the same module count.
    target=Counter(schedule.tolist());accepted_counts=Counter()
    pending_levels=list(schedule);number=0;attempted=0
    base_left,base_right=left,right
    left,right=[],[]
    while True:
        pending_left,pending_right=list(base_left),list(base_right)
        base_left,base_right=[],[]
        for level in pending_levels:
            for draw in range(100):
                number+=1
                b,a,desc=sample_twin_mutant_pair(int(level),sampler.backend.defaults(),ma=sampler.backend)
                try:
                    x,y=_twin_rescue(b,a,desc,int(level),number,sampler,report)
                except RuntimeError as exc:
                    if 'rescue exhausted' not in str(exc).lower():raise
                    report['collision_exhausted']+=1
                    continue
                x,y=valid_candidate(x,'budding'),valid_candidate(y,'budding')
                if x is not None and y is not None:break
            else:raise RuntimeError('Cannot draw a twin pair with the requested module count')
            pending_left.append(x);pending_right.append(y)
        a,b=_paired_results(pending_left,pending_right,config)
        attempted+=len(pending_levels)
        report['paired_failures']+=len(pending_left)-len(a)
        left.extend(a);right.extend(b)
        for row in a:
            if not row.candidate.is_real:accepted_counts[row.candidate.level]+=1
        missing={level:target[level]-accepted_counts[level] for level in target if accepted_counts[level]<target[level]}
        if not missing:break
        if attempted>=config.max_attempts_mult*max(1,config.n_samples):
            raise RuntimeError(f'Twin valid quota exhausted: {missing}')
        pending_levels=[level for level,number_missing in missing.items() for _ in range(number_missing)]
    a,b=left,right
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
        checked=valid_candidate(candidate,'budding')
        if checked is None:raise ValueError('Source contains unsupported/unknown or >3-module synthetic rows; regenerate source')
        candidate=checked
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
    sampler=make_sampler(get_recipe('budding_random'),config)
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
    from .samplers.targeted_ablation import targeted_pair, TARGETED_SKELETONS
    output=Path(config.output)
    hard=output.with_name(output.stem+'_hard.npz')
    hypo=output.with_name(output.stem+'_hypo.npz')
    _check_outputs([output,hard,hypo],config)
    archive,rows=_source_rows(config)
    real=[r for r in rows if r.candidate.is_real] if config.include_real else []
    synthetic=[r for r in rows if not r.candidate.is_real]
    if len(synthetic)<config.n_samples:raise ValueError('Source has fewer synthetic samples than the requested control budget')
    target=Counter(level_schedule(config.n_samples,config.level_ratios or STATIC_RATIOS,np.random).tolist())
    grouped={level:[r for r in synthetic if r.candidate.level==level] for level in (1,2,3)}
    if any(len(grouped[level])<target[level] for level in target):
        raise ValueError('Source cannot supply the requested per-level control quotas')
    control=[row for level in (1,2,3) for row in grouped[level][:target[level]]]
    count=max(1,config.n_samples//5) if config.n_samples else 0
    replace_schedule=level_schedule(count,(0,target[2],target[3]),np.random) if count else []
    replacement_counts=Counter(replace_schedule)
    base=real+[row for level in (1,2,3) for row in grouped[level][:target[level]-replacement_counts[level]]]
    sampler=make_sampler(get_recipe('budding_random'),config)
    report=_report(config)
    left,right=[],[]
    next_sample_id=max(0,max((r.candidate.sample_id for r in rows),default=0))+1
    next_pair_id=max((r.candidate.pair_id for r in rows),default=-1)+1
    skeleton_levels=[]
    for skeleton in TARGETED_SKELETONS:
        genes=[g[0] if isinstance(g,tuple) else g for g in skeleton['lofs']]+[g for g,_ in skeleton['gofs']]
        skeleton_levels.append(len(set(genes)))
    pending_levels=list(replace_schedule);number=0;attempted=0;got=Counter()
    while pending_levels:
        pending_left,pending_right=[],[]
        for level in pending_levels:
            eligible=[j for j,k in enumerate(skeleton_levels) if k==level]
            for attempt in range(config.max_attempts_mult):
                index=eligible[(number+attempt)%len(eligible)]
                a,b,desc=targeted_pair(index)
                if real_mutants.is_duplicate_real_mutant(a,sampler.backend.defaults()) or real_mutants.is_duplicate_real_mutant(b,sampler.backend.defaults()):
                    report['collisions']+=1
                    # This control requires a hard-KO/hypomorphic paired assay.
                    # Changing its KO modes would invalidate the contrast. Redraw
                    # a complete pair at this L level instead of changing mdt.
                    report['collision_exhausted']+=1
                    continue
                skeleton=TARGETED_SKELETONS[index]
                pairs=[g if isinstance(g,tuple) else (g,'KO') for g in skeleton['lofs']]+list(skeleton['gofs'])
                metadata={'pairs':pairs,'description':desc}
                x=Candidate(next_sample_id+number,f'BAMS_TARGET_HARD_{number}_{desc}',a,int(level),pair_id=next_pair_id+number,metadata=metadata)
                y=replace(x,name=f'BAMS_TARGET_HYPO_{number}_{desc}',overrides=b)
                pending_left.append(x);pending_right.append(y);number+=1
                break
            else:raise RuntimeError(f'Targeted pair collision rescue exhausted at L{level}')
        a_rows,b_rows=_paired_results(pending_left,pending_right,config)
        left.extend(a_rows);right.extend(b_rows)
        report['paired_failures']+=len(pending_left)-len(a_rows)
        attempted+=len(pending_left)
        for row in a_rows:got[row.candidate.level]+=1
        pending_levels=[level for level,needed in replacement_counts.items() for _ in range(needed-got[level])]
        if pending_levels and attempted>=config.max_attempts_mult*max(1,count):
            raise RuntimeError('Targeted paired valid quotas exhausted')
    report['diagnostic_targeted']=True
    profiles=_source_profiles(archive,report)
    mixed=tuple(p if p=='mutant_aware' else 'mixed' for p in profiles) if left else profiles
    return {'control':_save(output,real+control,config,report,*profiles),
            'hard':_save(hard,base+left,config,report,*mixed),'hypo':_save(hypo,base+right,config,report,*mixed)}


def generate_derived(config):
    state=np.random.get_state();py_state=random.getstate()
    np.random.seed(config.seed);random.seed(config.seed)
    try:
        kind=get_recipe(config.recipe).sampler
        return {'twins':generate_twins,'ko_mirror':generate_ko_mirror,'bams_ablation':generate_bams_ablation}[kind](config)
    finally:
        np.random.set_state(state);random.setstate(py_state)
