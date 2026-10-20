"""Shared candidate, collision-rescue, bounded simulation and export pipeline."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import random

import numpy as np

from .contracts import Candidate, SimulationResult
from .io import write_dataset
from .policies import level_schedule, parse_pairs, rescue_budding, rescue_fission
from ..recipes import get_recipe,resolve_config
from ..simulation_guards import bounded_ordered_map, simulation_deadline


def _same_fission(vector, references):
    return any(np.allclose(vector,v,rtol=1e-5,atol=1e-9) for v in references)


def ensure_candidate(candidate, sampler, report, references=None, *, rescue_rng=None):
    """Rescue a colliding candidate, never silently accept a collision."""
    if sampler.recipe.system=='budding':
        from ..systems.budding.real_mutants import is_duplicate_real_mutant
        backend=sampler.backend
        def collides(overrides):
            return is_duplicate_real_mutant(overrides,backend.defaults())
    else:
        from ..systems.fission.parameters import WT_parameters, param_vector
        def collides(overrides):
            return _same_fission(param_vector({**WT_parameters(),**overrides}),references)
    if not collides(candidate.overrides):
        return candidate
    if rescue_rng is None:
        # Pure Random's repaired contract explicitly threads its local RNG.
        # The legacy MASS, k-DPP and BAMS drivers call rescue on global numpy.
        rescue_rng=sampler.rng if sampler.recipe.system=='fission' or sampler.recipe.sampler in ('random','random_ko_interval') else np.random
    report['collisions']+=1
    original=dict(candidate.overrides)
    description=candidate.metadata['description']
    if candidate.metadata.get('pairs'):
        description='+'.join(f'{g}[{m}]' for g,m in candidate.metadata['pairs'])
    rescue_limit=sampler.config.max_attempts or 30
    for attempt in range(rescue_limit):
        report['rescue_attempts']+=1
        if sampler.recipe.system=='budding':
            overrides,new_desc=rescue_budding(original,description,backend,rescue_rng)
        else:
            overrides,new_desc=rescue_fission(original,description,rescue_rng)
        if not collides(overrides):
            from ..sampling.factory import candidate_name
            report['rescued']+=1
            metadata=dict(candidate.metadata,description=new_desc,rescued=True,rescue_attempts=attempt+1)
            if candidate.metadata.get('pairs'):
                metadata['pairs']=parse_pairs(new_desc)
            # Original BAMS also disables bisection after rescuing a collision.
            return replace(candidate,name=candidate_name(sampler.recipe.sampler,new_desc,candidate.level,candidate.sample_id),
                           overrides=overrides,refinement=None,metadata=metadata)
    raise RuntimeError(f'Collision rescue exhausted for {candidate.name}; no colliding sample was exported')


def prepare_candidates(recipe, config, sampler, report):
    if recipe.system=='budding':
        from ..systems.budding.real_mutants import real_tasks
        references=None
    else:
        from ..systems.fission.real_mutants import real_tasks
        from ..systems.fission.parameters import WT_parameters, param_vector
        references=[param_vector({**WT_parameters(),**ovr}) for _,ovr in real_tasks()]
    candidates=[]
    if config.include_real and not config.parameter_only:
        for index,(name,overrides) in enumerate(real_tasks()):
            candidates.append(Candidate(-index-1,name,overrides,0,True))
    count=int(np.ceil(config.n_samples*recipe.queue_multiplier))
    schedule_rng=np.random if recipe.sampler in ('fission_random','mass','mass_motifs','mass_random','bams') else sampler.rng
    schedule=level_schedule(count,config.level_ratios or recipe.ratios,schedule_rng,
                            rounding=recipe.quota_rounding,shuffle=recipe.shuffle_levels)
    level_counts=Counter()
    for number,level in enumerate(schedule,1):
        from ..sampling.factory import candidate_name
        overrides,description,refinement,pairs=sampler.sample(int(level))
        display_number=level_counts[int(level)] if recipe.sampler=='motif' else (level_counts[int(level)]+1 if recipe.sampler=='mass_random_dist' else number)
        candidate=Candidate(number,candidate_name(recipe.sampler,description,int(level),display_number),overrides,int(level),
                            refinement=refinement,metadata={'description':description,'pairs':pairs})
        candidate=ensure_candidate(candidate,sampler,report,references)
        candidates.append(candidate)
        level_counts[int(level)]+=1
        if references is not None:
            references.append(param_vector({**WT_parameters(),**candidate.overrides}))
    report['sampled_synthetic']=len(schedule)
    return candidates


def _parameter_result(candidate, system):
    from ..systems.budding.specification import BuddingSystem
    backend=system if isinstance(system,BuddingSystem) else BuddingSystem()
    defaults=backend.defaults()
    _,y0,_,_=backend.prepare_initial_conditions(candidate.overrides)
    params=np.array([candidate.overrides.get(k,defaults[k]) for k in backend.PARAM_KEYS],dtype=float)
    return SimulationResult(candidate,None,params,y0,'Unlabelled',None,'parameter_only')


def simulate_task(payload):
    """Top-level worker function; payload contains no module objects or closures."""
    candidate,config,context=payload
    config=resolve_config(config)
    recipe=get_recipe(config.recipe)
    if config.mode_freq is None:
        config=replace(config,mode_freq=recipe.mode_freq)
    if recipe.system=='budding':
        from ..systems.budding.specification import BuddingSystem
        backend=BuddingSystem(config.mutation_profile or recipe.mutation,config.distribution_profile or recipe.distribution)
        if candidate.refinement is not None:
            from ..sampling.bams import execute_simulation_task
            # Stable per-candidate worker randomness, independent of completion order.
            state=np.random.get_state()
            np.random.seed(np.random.SeedSequence([config.seed,candidate.sample_id]).generate_state(1)[0])
            try:
                result,overrides=execute_simulation_task((candidate.name,dict(candidate.overrides),candidate.refinement,210,500),ma=backend)
            finally:
                np.random.set_state(state)
            candidate=replace(candidate,overrides=overrides)
            from ..systems.budding.real_mutants import is_duplicate_real_mutant
            if not candidate.is_real and is_duplicate_real_mutant(overrides,backend.defaults()):
                from ..sampling.factory import Sampler
                local=Sampler(recipe,config)
                local.rng=np.random.RandomState(config.seed+candidate.sample_id)
                candidate=ensure_candidate(candidate,local,Counter(),rescue_rng=local.rng)
                result=backend.simulate_mutant(candidate.name,candidate.overrides)
        else:
            result=backend.simulate_mutant(candidate.name,candidate.overrides)
        row=_parameter_result(candidate,backend)
        _,trajectory,label,pattern=result
        row.trajectory=trajectory
        row.label=label
        row.pattern=pattern
        row.status='success' if trajectory is not None else ('discarded_mass' if label=='Discarded_Mass' else 'numerical_failure')
        return row
    from ..systems.fission.simulation import simulate_mutant
    from ..systems.fission.parameters import WT_parameters, param_vector
    from ..systems.fission.constants import TF_MIN
    params=param_vector({**WT_parameters(),**candidate.overrides})
    row=SimulationResult(candidate,None,params,np.array(context,copy=True),'Inviable: Numerical Failure',None,'numerical_failure')
    try:
        with simulation_deadline(5.):
            result=simulate_mutant(candidate.name,candidate.overrides,context,level=candidate.level)
        if result['y'] is not None and result['status']=='End' and result.get('completed_time',0)>=TF_MIN:
            row.trajectory=result['y']
            row.params=result['params_vec']
            # Exported y_init is in the same units as exported trajectories.
            row.y_init=result['y_init'].copy()
            from ..systems.fission.constants import MASS_UNIT
            row.y_init[0]/=MASS_UNIT
            row.label=result['label']
            row.pattern=result['pattern']
            row.status='success'
            row.diagnostics={'n_div':result['n_div'],'birth_rel':result['birth_rel']}
        elif result['label']=='Discarded_Mass':
            row.label=result['label']
            row.status='discarded_mass'
    except (Exception,) as exc:
        row.diagnostics={'error':repr(exc)}
    return row


def run_simulations(candidates,config,context):
    # Sequential mode is useful for debugging; each solver still has its deadline.
    if config.workers==1:
        for candidate in candidates:
            yield simulate_task((candidate,config,context))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            yield from bounded_ordered_map(executor,simulate_task,
                [(candidate,config,context) for candidate in candidates],stall_timeout=35.)


def generate_dataset(config):
    """Create one dataset from a recipe. Old scripts and old datasets are untouched."""
    config=resolve_config(config)
    recipe=get_recipe(config.recipe)
    if config.mode_freq is None:
        config=replace(config,mode_freq=recipe.mode_freq)
    if config.max_attempts is not None and config.max_attempts<1:
        raise ValueError('max_attempts must be positive')
    if config.n_samples<0 or config.seed<0 or config.workers<1 or not 0<=config.flat_prob<=1:
        raise ValueError('Invalid sample count, seed, workers or gating probability')
    if not np.isfinite(config.curv_thresh) or config.curv_thresh<0:
        raise ValueError('Curvature threshold must be finite and nonnegative')
    if not 0<=config.ko_interval_max<=1e-4:
        raise ValueError('KO interval must be in [0, 1e-4]')
    if config.level_ratios is not None and len(config.level_ratios)!=len(recipe.ratios):
        raise ValueError(f'{config.recipe} requires {len(recipe.ratios)} level ratios')
    if not np.isfinite(config.temperature) or config.temperature<=0:
        raise ValueError('Temperature must be finite and positive')
    if recipe.sampler=='wt_blind' and config.mode_freq not in ('flat','s5'):
        raise ValueError('WT-Blind supports flat/s5 mode frequencies')
    if config.pilot_samples<1 or config.wt_fold_dist not in ('flat','tier2'):
        raise ValueError('Invalid WT-Blind pilot or fold policy')
    if not 0<=config.nan_cap_frac<=1 or not 0<=config.backlog_thresh<=1 or config.max_attempts_mult<1:
        raise ValueError('Invalid WT-Blind ledger configuration')
    if recipe.system=='fission' and (config.mutation_profile is not None or config.distribution_profile is not None or config.mode_freq not in ('class','flat')):
        raise ValueError('Fission uses its own profile and supports class/flat mode frequencies')
    if config.parameter_only and recipe.system!='budding':
        raise ValueError('Fission initial states require per-mutant warmup; parameter-only pools are Budding-only')
    if not config.output:
        raise ValueError('Specify a new .npz output path')
    output=Path(config.output).expanduser().resolve()
    if output.suffix!='.npz':
        raise ValueError('Dataset output must end in .npz')
    config=replace(config,output=str(output))
    if output.exists() and not config.overwrite:
        raise FileExistsError(config.output)
    if recipe.sampler in ('twins','ko_mirror','bams_ablation'):
        if config.parameter_only:
            raise ValueError('Paired and source-derived datasets require trajectories')
        if config.mutation_profile or config.distribution_profile:
            raise ValueError('Derived datasets fix their profiles to preserve their comparison/source contract')
        from .derived import generate_derived
        return generate_derived(config)
    if recipe.sampler=='pool':
        config=replace(config,parameter_only=True,include_real=False)
    from ..sampling.factory import Sampler
    state=np.random.get_state()
    python_state=random.getstate()
    np.random.seed(config.seed)
    random.seed(config.seed)
    report=Counter(collisions=0,rescued=0)
    dict.update(report,recipe=config.recipe,mutation_profile=config.mutation_profile or recipe.mutation,
                  distribution_profile=config.distribution_profile or recipe.distribution,
                  rng_contract='pure_random_single_stream_v1' if recipe.sampler in ('random','random_ko_interval') else 'legacy_algorithms_scoped_v1')
    try:
        if recipe.system=='budding':
            from ..systems.budding.specification import SPEC
            context=None
        else:
            from ..systems.fission.specification import SPEC
            from ..systems.fission.states import derive_wt_birth_state
            with simulation_deadline(5.):
                context=derive_wt_birth_state()
        sampler=Sampler(recipe,config)
        if recipe.sampler=='wt_blind':
            from ..sampling.wt_pipeline import prepare_wt_candidates
            candidates=prepare_wt_candidates(sampler,config,report)
        else:
            candidates=prepare_candidates(recipe,config,sampler,report)
        if config.parameter_only:
            results=[_parameter_result(c,sampler.backend) for c in candidates]
        else:
            results=[]
            level_counts=Counter()
            schedule=level_schedule(config.n_samples,config.level_ratios or recipe.ratios,np.random.RandomState(config.seed),
                                    rounding=recipe.quota_rounding,shuffle=False)
            target=Counter(schedule.tolist())
            for result in run_simulations(candidates,config,context):
                report[result.status]+=1
                if result.trajectory is None:
                    continue
                if not np.isfinite(result.trajectory).all() or result.trajectory.shape!=(len(SPEC.variable_names),SPEC.steps):
                    report['invalid_trajectory']+=1
                    continue
                if recipe.sampler=='wt_blind' and not result.candidate.is_real and np.max(np.abs(result.trajectory))>1000.:
                    report['trajectory_limit']+=1
                    continue
                if recipe.sampler=='kdpp_curvature' and not result.candidate.is_real:
                    from ..sampling.kdpp import compute_trajectory_curvature
                    level=result.candidate.level
                    accepted=sum(level_counts.values())
                    if accepted>=config.n_samples:
                        continue
                    if level_counts[level]>=target[level] and accepted<config.n_samples-100:
                        continue
                    if compute_trajectory_curvature(result.trajectory)<config.curv_thresh and sampler.rng.rand()>config.flat_prob:
                        report['flatline_filtered']+=1
                        continue
                    level_counts[level]+=1
                results.append(result)
        if recipe.sampler=='wt_blind':
            results=[r for r in results if r.candidate.is_real]+sorted((r for r in results if not r.candidate.is_real),key=lambda r:r.candidate.name)
        report['saved_real']=sum(r.candidate.is_real for r in results)
        if config.include_real and not config.parameter_only:
            expected_real=126 if recipe.system=='budding' else 20
            if report['saved_real']!=expected_real:
                raise RuntimeError(f'Incomplete real-mutant benchmark: {report["saved_real"]}/{expected_real}; refusing to export a partial real test set')
        report['saved_synthetic']=len(results)-report['saved_real']
        report['requested_synthetic']=config.n_samples
        report['collision_policy']='rescue'
        path=write_dataset(config.output,SPEC,results,config,dict(report),parameter_only=config.parameter_only,dtype=recipe.dtype)
        print(f'Saved {path}: {report["saved_real"]} real + {report["saved_synthetic"]} synthetic; {report["rescued"]} rescued')
        return path
    finally:
        np.random.set_state(state)
        random.setstate(python_state)
