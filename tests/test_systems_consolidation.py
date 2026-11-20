"""Regression coverage for the directory consolidation and four review findings."""
from copy import deepcopy
import importlib
from pathlib import Path
import pickle

import numpy as np
import pytest

from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig
from budding_yeast_v2.data.systems.registry import RECIPES, make_sampler, get_adapter, get_system_spec
from budding_yeast_v2.data.systems.fission.samplers.references import FissionReferenceIndex
from budding_yeast_v2.scripts.analysis.analyze_experiments import aggregate_group_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_only_one_framework_implementation_is_present():
    for old in ('sampling', 'recipes', 'generation'):
        assert not (ROOT/'data'/old).exists()
    for name in ('registry.py', 'contracts.py', 'generation.py', 'dataset_io.py'):
        assert (ROOT/'data/systems'/name).is_file()
    assert len(RECIPES) == 18
    assert RECIPES['fission_random'].default_seed == 20000
    assert RECIPES['budding_pool'].default_samples == 1000000


@pytest.mark.parametrize('system', ['budding', 'fission'])
def test_one_registry_resolves_specs_adapters_and_samplers(system):
    assert get_adapter(system).spec is get_system_spec(system)
    recipe = RECIPES[f'{system}_random']
    sampler = make_sampler(recipe, GenerationConfig(f'{system}_random'))
    assert sampler.recipe.system == system
    assert hasattr(sampler, 'kernel')


@pytest.mark.parametrize('old,new', [
    ('sampling.common', 'systems.budding.samplers.common'),
    ('sampling.fission_random', 'systems.fission.samplers.random'),
    ('sampling.factory', 'systems.registry'),
    ('sampling.wt_blind', 'systems.budding.samplers.wt_blind'),
    ('generation.contracts', 'systems.contracts'),
    ('generation.runner', 'systems.generation'),
    ('generation.io', 'systems.dataset_io'),
    ('generation.derived', 'systems.budding.derived'),
    ('recipes', 'systems.registry'),
    ('systems.factory', 'systems.registry'),
    ('systems.base', 'systems.registry'),
])
def test_retired_imports_alias_the_actual_module_without_copying_globals(old,new):
    source = importlib.import_module('budding_yeast_v2.data.'+old)
    target = importlib.import_module('budding_yeast_v2.data.'+new)
    assert source is target
    assert source.__spec__.name == 'budding_yeast_v2.data.'+new


@pytest.mark.parametrize('suffix', ['common', 'fission_random'])
def test_legacy_mode_assignment_reaches_the_function_global_namespace(suffix):
    module = importlib.import_module('budding_yeast_v2.data.sampling.'+suffix)
    original = module.MODE_FREQ
    try:
        module.MODE_FREQ = 'flat' if original != 'flat' else 'class'
        assert module._sample_mode.__globals__['MODE_FREQ'] == module.MODE_FREQ
    finally:
        module.MODE_FREQ = original


def test_legacy_sampler_and_pickle_names_remain_usable():
    from budding_yeast_v2.data.sampling.factory import Sampler
    sampler = Sampler(RECIPES['fission_random'], GenerationConfig('fission_random'))
    assert sampler.recipe.system == 'fission' and sampler.kernel is None
    assert pickle.loads(b'cbudding_yeast_v2.data.generation.contracts\nCandidate\n.') is Candidate
    from budding_yeast_v2.data.generation.policies import rescue_budding
    from budding_yeast_v2.data.systems.budding.samplers.rescue import rescue_budding as actual
    assert rescue_budding is actual


@pytest.mark.parametrize('rtol,atol', [(1e-5, 1e-9), (0., 0.), (.01, .01)])
def test_dedup_matches_brute_force_before_and_after_incremental_and_main_rebuilds(rtol, atol):
    rng = np.random.default_rng(31)
    references = [np.zeros(58), np.ones(58)]
    index = FissionReferenceIndex(references, rtol=rtol, atol=atol, rebuild_every=11, recent_every=4)
    for i in range(35):
        vector = rng.uniform(-1, 1, 58) * 10.**(i % 7)
        index.add(vector)
        references.append(vector.copy())
        probes = [vector, vector*(1+rtol*.9)+atol*.1, rng.normal(size=58)*10.**(i % 7)]
        for probe in probes:
            expected = any(np.allclose(probe, ref, rtol=rtol, atol=atol) for ref in references)
            assert index.collides(probe) == expected


def test_dedup_checks_unindexed_tail_after_recent_tree_exists_and_resets():
    index = FissionReferenceIndex([np.zeros(58)], rebuild_every=35, recent_every=32)
    for i in range(1, 75):
        vector = np.full(58, i, dtype=float)
        index.add(vector)
        assert index.collides(vector), i
    assert not index.collides(np.full(58, 1000.))


def test_dedup_acceleration_has_no_hardcoded_50_fold_limit():
    index = FissionReferenceIndex([np.ones(58)], rebuild_every=4, recent_every=1)
    vector = np.full(58, 1e8)
    index.add(vector)
    assert index.collides(vector * (1 + .9e-5))


def metrics(value):
    return {'MAE': value, 'MSE': value, 'Correlation': value, 'Relative L2': value}


def group(*blocks):
    return dict(type='probe', dataset_tag='same_dataset', model_name='fno1d',
        loss_strategy='hybrid_mse_rel_none', hparam_sig='same',
        seeds=[dict(seed=str(i), metrics=dict(real_mutants=block)) for i, block in enumerate(blocks)])


@pytest.mark.parametrize('canonical', ['all_16_vars', 'all_39_vars'])
def test_aggregation_uses_both_alias_only_and_explicit_seeds(canonical):
    data = group({'all_variables': metrics(1.), 'count': 20}, {canonical: metrics(9.), 'count': 20})
    original = deepcopy(data)
    summary = aggregate_group_metrics(data)
    assert summary['n_seeds'] == 2 and summary['real_mae_mean'] == 5.
    stats = summary['detailed_metrics']['real_mutants'][canonical]['MAE']
    assert stats == dict(mean=5., std=4., n_seeds=2)
    assert data == original


def test_aggregation_does_not_double_count_metrics_with_both_keys():
    summary = aggregate_group_metrics(group(
        {'all_variables': metrics(1.), 'all_16_vars': metrics(1.)}, {'all_variables': metrics(9.)}))
    assert summary['real_mae_mean'] == 5.
    assert summary['detailed_metrics']['real_mutants']['all_16_vars']['MAE']['n_seeds'] == 2


def test_aggregation_rejects_mixed_systems_and_conflicting_aliases():
    with pytest.raises(ValueError, match='different variable systems'):
        aggregate_group_metrics(group({'all_16_vars': metrics(1.)}, {'all_39_vars': metrics(9.)}))
    with pytest.raises(ValueError, match='disagree'):
        aggregate_group_metrics(group({'all_16_vars': metrics(1.), 'all_variables': metrics(9.)}))


def test_aggregation_reports_actual_available_seed_count():
    summary = aggregate_group_metrics(group({'all_variables': metrics(1.)}, {}))
    assert summary['n_seeds'] == 2
    assert summary['detailed_metrics']['real_mutants']['all_variables']['MAE']['n_seeds'] == 1


def test_inventory_follows_the_forwarded_cli_and_new_module_locations():
    from budding_yeast_v2.scripts.analysis.audit_rule_inventory import inventory, TARGETS
    report = inventory()
    flags = {flag for entry in report['new_cli'] for flag in entry['flags']}
    assert {'--recipe', '--samples', '--show-config'} <= flags
    for targets in TARGETS.values():
        assert all((ROOT/'data'/target).is_file() for target in targets)
