"""Shared candidate, collision-rescue, bounded simulation and export pipeline.

Biological systems are selected through an adapter (systems/registry.py); this module
contains no `system == ...` branching.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import random

import numpy as np

from .contracts import Candidate
from .sampling_policy import VERSION, draw_candidate, valid_candidate
from .dataset_io import write_dataset
from .registry import get_recipe, resolve_config, get_adapter, make_sampler
from ..simulation_guards import bounded_ordered_map


def level_schedule(n, ratios, rng, *, rounding='floor', shuffle=True):
    r = np.asarray(ratios, dtype=float)
    if n < 0 or r.ndim != 1 or len(r) < 1 or not np.isfinite(r).all() or (r < 0).any() or r.sum() <= 0:
        raise ValueError('Sample count and level ratios must be finite, nonnegative and nonempty')
    r = r / r.sum()
    if len(r) == 3:
        exact = n*r
        counts = np.floor(exact).astype(int)
        order = np.argsort(-(exact-counts),kind='stable')
        counts[order[:n-int(counts.sum())]] += 1
        counts = counts.tolist()
    elif rounding == 'motif':
        if len(r) != 5:
            raise ValueError('Motif requires five effective level ratios')
        counts = [round(n*x) for x in r[:3]]
        rest = n - sum(counts)
        # Original last 30% was split 70/30 after rounding the first three tiers.
        fourth = round(rest * r[3] / (r[3] + r[4])) if r[3]+r[4] else 0
        counts += [fourth, rest-fourth]
    elif rounding in ('first_remainder','second_remainder'):
        counts = [round(n*x) for x in r]
        counts[0 if rounding=='first_remainder' else min(1,len(counts)-1)] += n-sum(counts)
    else:
        counts = [int(round(n*x) if rounding == 'round' else np.floor(n*x)) for x in r[:-1]]
        counts.append(n-sum(counts))
    if min(counts) < 0:
        raise ValueError('Rounded quotas are negative for this small budget; increase samples or change ratios')
    schedule = np.repeat(np.arange(1,len(counts)+1), counts)
    if shuffle:
        rng.shuffle(schedule)
    return schedule

def ensure_candidate(candidate, sampler=None, report=None, references=None, *, adapter=None, rescue_rng=None):
    """Rescue a colliding candidate; never silently accept a collision."""
    if adapter is None:
        adapter = get_adapter(sampler.recipe.system)
    if report is None:
        report = Counter(collisions=0, rescued=0, rescue_attempts=0)
    if not adapter.is_collision(candidate.overrides, references, sampler):
        return candidate
    report['collisions'] += 1
    description = candidate.metadata.get('description', '')
    if candidate.metadata.get('pairs'):
        description = '+'.join(f'{g}:{m}' if adapter.system_id == 'fission' else f'{g}[{m}]' for g,m in candidate.metadata['pairs'])
    rescue_rng = rescue_rng or adapter.rescue_rng(sampler)
    rescue_limit = sampler.config.max_attempts or 30
    for attempt in range(rescue_limit):
        report['rescue_attempts'] += 1
        overrides, new_description = adapter.rescue(candidate.overrides, description, sampler, rescue_rng)
        if not adapter.is_collision(overrides, references, sampler):
            report['rescued'] += 1
            metadata = dict(candidate.metadata, description=new_description, rescued=True,
                            rescue_attempts=attempt + 1,
                            collision_policy=getattr(adapter,'collision_policy','rescue'))
            pairs = adapter.pairs_from_description(new_description)
            if pairs is not None:
                metadata['pairs'] = pairs
            return replace(candidate, name=sampler.candidate_name(new_description, candidate.level, candidate.sample_id),
                           overrides=overrides, refinement=None, metadata=metadata)
    raise RuntimeError(f'Collision rescue exhausted for {candidate.name}; no colliding sample was exported')


def prepare_candidates(recipe, config, sampler, report, start_id=1):
    adapter = get_adapter(recipe.system)
    candidates = adapter.real_candidates(config.include_real and not config.parameter_only)
    references = getattr(sampler,'_references',None)
    if references is None:references = adapter.references()
    sampler._references = references
    count = config.n_samples if config.parameter_only else int(np.ceil(config.n_samples * recipe.queue_multiplier))
    schedule = level_schedule(count, config.level_ratios or recipe.ratios, sampler.schedule_rng(),
                              rounding=recipe.quota_rounding, shuffle=recipe.shuffle_levels)
    for number, level in enumerate(schedule, start_id):
        for _ in range(config.max_attempts or 100):
            candidate = draw_candidate(sampler, int(level), number, report, references)
            if not config.parameter_only or recipe.system != 'budding':break
            # Exact input identity, including IC overrides; no undisclosed 1% cutoff.
            import hashlib
            row = adapter.parameter_result(candidate, None)
            signature = hashlib.sha256(np.r_[row.params,row.y_init].astype(np.float64).tobytes()).digest()
            seen = getattr(sampler,'_pool_seen',None)
            if seen is None:sampler._pool_seen = seen = set()
            if signature not in seen:
                seen.add(signature);break
            report['pool_duplicates'] += 1
        else:raise RuntimeError(f'Pool L{level} capacity exhausted; no repeated candidates exported')
        candidates.append(candidate)
        adapter.add_reference(references, candidate.overrides)
    report['sampled_synthetic'] += len(schedule)
    return candidates


def simulate_task(payload):
    """Top-level worker function; payload contains no module objects or closures."""
    candidate, config, context = payload
    config = resolve_config(config)
    recipe = get_recipe(config.recipe)
    if config.mode_freq is None:
        config = replace(config, mode_freq=recipe.mode_freq)
    return get_adapter(recipe.system).simulate(candidate, config, context)


def run_simulations(candidates, config, context=None):
    # Sequential mode is useful for debugging; each solver still has its deadline.
    if config.workers == 1:
        for candidate in candidates:
            yield simulate_task((candidate, config, context))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            yield from bounded_ordered_map(executor, simulate_task,
                                           [(candidate, config, context) for candidate in candidates],
                                           stall_timeout=35.)


def generate_dataset(config):
    """Create one dataset from a recipe. Old scripts and old datasets are untouched."""
    config = resolve_config(config)
    recipe = get_recipe(config.recipe)
    if config.mode_freq is None:
        config = replace(config, mode_freq=recipe.mode_freq)
    if config.max_attempts is not None and config.max_attempts < 1:
        raise ValueError('max_attempts must be positive')
    if config.n_samples < 0 or config.seed < 0 or config.workers < 1 or not 0 <= config.flat_prob <= 1:
        raise ValueError('Invalid sample count, seed, workers or gating probability')
    if config.level_ratios is not None and len(config.level_ratios) != len(recipe.ratios):
        raise ValueError(f'{config.recipe} requires {len(recipe.ratios)} level ratios')
    if not np.isfinite(config.temperature) or config.temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    adapter = get_adapter(recipe.system)
    adapter.validate_config(config, recipe)
    if not config.output:
        raise ValueError('Specify a new .npz output path')
    output = Path(config.output).expanduser().resolve()
    if output.suffix != '.npz':
        raise ValueError('Dataset output must end in .npz')
    config = replace(config, output=str(output))
    if output.exists() and not config.overwrite:
        raise FileExistsError(config.output)
    if adapter.handles_derived(recipe):
        if config.parameter_only:
            raise ValueError('Paired and source-derived datasets require trajectories')
        if config.mutation_profile or config.distribution_profile:
            raise ValueError('Derived datasets fix their profiles to preserve their comparison/source contract')
        return adapter.generate_derived(config)
    if recipe.sampler == 'pool':
        config = replace(config, parameter_only=True, include_real=False)
    state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(config.seed)
    random.seed(config.seed)
    report = Counter(collisions=0, rescued=0)
    dict.update(report, recipe=config.recipe,
                collision_policy=getattr(adapter,'collision_policy','rescue'),
                mutation_profile=config.mutation_profile or recipe.mutation,
                distribution_profile=config.distribution_profile or recipe.distribution,
                sampling_policy=VERSION,
                rng_contract='pure_random_single_stream_v1' if recipe.sampler in ('random', 'random_ko_interval')
                else 'legacy_algorithms_scoped_v1')
    try:
        context = adapter.prepare_context(config)
        sampler = make_sampler(recipe, config)
        candidates = adapter.special_candidates(recipe, config, sampler, report)
        if candidates is None:
            if config.parameter_only and recipe.system == 'budding' and recipe.sampler == 'pool':
                from .budding.al_pool import prepare_quota_pool
                candidates = prepare_quota_pool(recipe, config, report)
            elif config.parameter_only and config.workers > 1:
                from .parallel_pool import prepare_parallel_pool
                candidates = prepare_parallel_pool(recipe, config, report)
            else:
                candidates = prepare_candidates(recipe, config, sampler, report)
        else:
            checked = [valid_candidate(c, recipe.system) for c in candidates]
            candidates = [c for c in checked if c is not None]
        if config.parameter_only:
            if report.get('pool_generation_contract'):
                actual = Counter(candidate.level for candidate in candidates)
                config = replace(config,level_ratios=tuple(actual[level]/len(candidates) for level in (1,2,3)))
            results = [adapter.parameter_result(c, context) for c in candidates]
        else:
            results = []
            level_counts = Counter()
            schedule = level_schedule(config.n_samples, config.level_ratios or recipe.ratios,
                                      np.random.RandomState(config.seed),
                                      rounding=recipe.quota_rounding, shuffle=False)
            target = Counter(schedule.tolist())
            state_data = {'level_counts': level_counts, 'target': target,
                          'sampler': sampler, 'report': report, 'recipe': recipe}
            budget = config.max_attempts_mult * max(1, config.n_samples)
            attempted = 0
            next_id = max((c.sample_id for c in candidates), default=0) + 1
            while True:
                remaining_budget = max(0,budget-attempted)
                real_candidates = [c for c in candidates if c.is_real]
                candidates = real_candidates + [c for c in candidates if not c.is_real][:remaining_budget]
                for result in run_simulations(candidates, config, context):
                    report[result.status] += 1
                    report['simulated'] += 1
                    real = result.candidate.is_real
                    if not real:attempted += 1
                    if result.trajectory is None:continue
                    if (not np.isfinite(result.trajectory).all()
                            or result.trajectory.shape != (len(adapter.spec.variable_names), adapter.spec.steps)):
                        report['invalid_trajectory'] += 1
                        continue
                    checked = valid_candidate(result.candidate, recipe.system)
                    if checked is None:
                        report['module_count_rejected_after_simulation'] += 1
                        continue
                    result.candidate = checked
                    level = result.candidate.level
                    if not real and level_counts[level] >= target[level]:continue
                    if not adapter.accept_result(result, state_data, config):continue
                    results.append(result)
                    if not real:level_counts[level] += 1
                missing = {level: target[level]-level_counts[level] for level in target
                           if level_counts[level]<target[level]}
                if not missing:break
                if attempted >= budget:
                    raise RuntimeError(f'Valid-trajectory quota not filled; missing by level: {missing}; '
                                       f'attempted {attempted}/{budget}. No partial dataset exported')
                deficit = sum(missing.values())
                refill = replace(config,n_samples=min(deficit,budget-attempted),include_real=False,
                                 level_ratios=tuple(missing.get(level,0)/deficit for level in (1,2,3)))
                report['refill_rounds'] += 1
                # Preserve WT ledger construction rather than silently bypassing it.
                candidates = adapter.special_candidates(recipe, refill, sampler, report)
                if candidates is None:
                    candidates = prepare_candidates(recipe,refill,sampler,report,start_id=next_id)
                else:
                    candidates = [replace(c,sample_id=next_id+i,
                        name=f'{c.name}_refill{report["refill_rounds"]}') for i,c in enumerate(candidates)]
                next_id += len(candidates)
                if not candidates:
                    raise RuntimeError(f'No refill candidates for missing quotas {missing}')
            dict.update(report, requested_level_counts=dict(target), saved_level_counts=dict(level_counts),
                        simulation_attempt_budget=budget)
        results = adapter.finalize(results, config, recipe)
        report['saved_real'] = sum(r.candidate.is_real for r in results)
        if config.include_real and not config.parameter_only:
            expected_real = adapter.expected_real_count()
            if report['saved_real'] != expected_real:
                raise RuntimeError(f'Incomplete real-mutant benchmark: {report["saved_real"]}/{expected_real}; '
                                   'refusing to export a partial real test set')
        report['saved_synthetic'] = len(results) - report['saved_real']
        report['requested_synthetic'] = config.n_samples
        report['collision_policy'] = 'rescue'
        path = write_dataset(config.output, adapter.spec, results, config, dict(report),
                             parameter_only=config.parameter_only, dtype=recipe.dtype)
        print(f'Saved {path}: {report["saved_real"]} real + {report["saved_synthetic"]} synthetic; '
              f'{report["rescued"]} rescued')
        return path
    finally:
        np.random.set_state(state)
        random.setstate(python_state)


def main(argv=None):
    """Parse the stable dataset CLI and invoke the shared generation workflow."""
    import argparse
    from dataclasses import asdict
    import json
    import os
    from .contracts import GenerationConfig
    from .registry import RECIPES, output_filename

    parser=argparse.ArgumentParser(description='模块化 Budding / Fission 数据集生成；保留全部旧代码')
    parser.add_argument('--list',action='store_true',help='列出可用数据集配置')
    parser.add_argument('--recipe',choices=tuple(RECIPES))
    parser.add_argument('--samples',type=int,help='合成样本请求数；默认继承对应旧 CLI')
    parser.add_argument('--seed',type=int,help='默认继承对应旧 CLI')
    parser.add_argument('--workers',type=int,default=min(60,os.cpu_count() or 1))
    parser.add_argument('--output',help='新的 NPZ 输出路径')
    parser.add_argument('--paired-output',default='',help='Twins 的另一套输出路径（mass 方案）')
    parser.add_argument('--source',default='',help='KO mirror / BAMS ablation 的源 NPZ')
    parser.add_argument('--data_dir',help='输出目录，可与自动文件名配合使用')
    parser.add_argument('--mutation-profile',choices=('mass','mutant_aware'))
    parser.add_argument('--distribution-profile',choices=('mass','mutant_aware'))
    parser.add_argument('--level-ratios',type=float,nargs='+')
    parser.add_argument('--mode-freq',choices=('class','flat','role','s5'))
    parser.add_argument('--temperature',type=float,default=.8)
    parser.add_argument('--curv-thresh',type=float,default=.01)
    parser.add_argument('--flat-prob',type=float,default=.20)
    parser.add_argument('--ko-interval-max',type=float,default=1e-4)
    parser.add_argument('--pilot-samples',type=int)
    parser.add_argument('--no-bisection','--no_bisection',action='store_true',help='关闭 BAMS 二分细化')
    parser.add_argument('--nan-cap-frac',type=float,default=.05)
    parser.add_argument('--backlog-thresh',type=float,default=.20)
    parser.add_argument('--max-attempts-mult',type=int,default=30)
    parser.add_argument('--wt-fold-dist',choices=('flat','tier2'),default='tier2')
    parser.add_argument('--max-attempts',type=int,help='每个碰撞候选的最大救援次数（默认 30）；WT-Blind 也用作候选总尝试上限')
    parser.add_argument('--no-real',action='store_true')
    parser.add_argument('--parameter-only',action='store_true')
    parser.add_argument('--overwrite',action='store_true',help='显式允许覆盖输出，默认拒绝')
    parser.add_argument('--show-config',action='store_true',help='只显示配置，不仿真或写文件')
    args=parser.parse_args(argv)
    if args.list:
        for name,recipe in RECIPES.items():
            filename=output_filename(GenerationConfig(name))
            print(f'{name:30s} samples={recipe.default_samples:<7d} seed={recipe.default_seed:<9d} file={filename}')
        return
    if args.recipe is None:
        parser.error('--recipe is required unless --list is used')
    output=args.output
    if output is None and not args.data_dir and not args.show_config:
        parser.error('Specify --output or --data_dir')
    config=GenerationConfig(recipe=args.recipe,n_samples=args.samples,seed=args.seed,workers=args.workers,
        output=output or '',mutation_profile=args.mutation_profile,distribution_profile=args.distribution_profile,
        level_ratios=tuple(args.level_ratios) if args.level_ratios else None,include_real=not args.no_real,
        parameter_only=args.parameter_only,mode_freq=args.mode_freq or RECIPES[args.recipe].mode_freq,temperature=args.temperature,
        curv_thresh=args.curv_thresh,flat_prob=args.flat_prob,ko_interval_max=args.ko_interval_max,
        pilot_samples=args.pilot_samples,max_attempts=args.max_attempts,overwrite=args.overwrite,
        paired_output=args.paired_output,source=args.source,wt_fold_dist=args.wt_fold_dist,
        enable_bisection=not args.no_bisection,nan_cap_frac=args.nan_cap_frac,
        backlog_thresh=args.backlog_thresh,max_attempts_mult=args.max_attempts_mult)
    config=resolve_config(config)
    if output is None and args.data_dir:
        config=replace(config,output=str(Path(args.data_dir)/output_filename(config)))
    if args.show_config:
        print(json.dumps({'recipe':asdict(RECIPES[args.recipe]),'run':asdict(config)},ensure_ascii=False,indent=2))
        return config
    return generate_dataset(config)

