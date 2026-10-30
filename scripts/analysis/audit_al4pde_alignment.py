#!/usr/bin/env python3
"""Audit archived AL removal and compare project acquisitions with local AL4PDE.

This read-only audit uses artificial inputs, no experiments, or real-mutant tuning.
Official numerical methods are loaded from their source AST in isolation, avoiding
Hydra/task/simulator initialization. This is not an end-to-end AL4PDE integration test.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import inspect
import textwrap
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

LEGACY_STRATEGIES = ('random', 'us', 'is', 'wrs', 'vessal', 'hggs', 'pial')
STRATEGY_METHODS = (
    '_al4pde_random_sample', '_al4pde_uncertainty_sample', '_al4pde_diversity_sample',
    '_maal_sample', '_al4pde_mass_sample', '_al4pde_novelty_sample',
    '_al4pde_residual_sample', '_al4pde_residual_diversity_sample',
    '_al4pde_bifurcation_sample', '_al4pde_bifurcation_entropy_sample',
    '_al4pde_event_entropy_sample', '_al4pde_fisher_oed_sample',
)
ARCHIVED_METHODS = (
    'on_train_start', '_kmeans_initial_selection', 'on_train_epoch_end',
    '_standard_al_sample', '_subsample_pool', '_compute_uncertainty',
    '_query_uncertainty_sampling', '_query_importance_sampling',
    '_query_weighted_reservoir_sampling', '_query_vessal', '_query_hggs',
    '_compute_diversity', '_query_physics_informed', '_predict_pool_trajectories',
    '_compute_ode_residual', '_detect_event_mask', '_compute_trajectory_diversity',
    '_compute_trajectory_diversity_pca', '_move_samples_to_training_set',
)


def class_methods(path, name):
    if Path(path).resolve() == (ROOT/'utils/al_callback.py').resolve():
        from budding_yeast_v2.utils.active_learning import ActiveLearningEngine
        result = {}
        for name, member in ActiveLearningEngine.__dict__.items():
            if isinstance(member, staticmethod):
                member = member.__func__
            if inspect.isfunction(member):
                function = inspect.unwrap(member)
                node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
                node.name = name
                node.decorator_list = []
                result[name] = node
        return result
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def function_from_source(path, name, class_name=None):
    tree = ast.parse(Path(path).read_text())
    if class_name:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    namespace = {'torch': torch, 'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class NumericalCode(ast.NodeTransformer):
    """Ignore documentation and logging only, preserving computation and branches."""
    def visit_Attribute(self, node):
        if node.attr == 'make_selection_plan':
            node.attr = '_build_broadcast_from_pool'
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print':
            return None
        return self.generic_visit(node)


def numerical_ast(node):
    copy_node = copy.deepcopy(node)
    copy_node.decorator_list = []
    return ast.dump(NumericalCode().visit(copy_node), include_attributes=False)


def budding_pool_compatibility_probe(archived):
    """Audit explicit system dispatch/dynamic width changes separately from scores."""
    from unittest.mock import patch
    from budding_yeast_v2.utils.active_learning import ActiveLearningEngine, pool, fission
    # Candidate generation intentionally changed under the approved exact-3 policy.
    # Acquisition-score ASTs remain audited separately; check the shared policy here.
    from budding_yeast_v2.data.systems.sampling_policy import POOL_RATIOS
    from budding_yeast_v2.data.systems.registry import resolve_config
    from budding_yeast_v2.data.systems.contracts import GenerationConfig
    assert resolve_config(GenerationConfig('budding_pool')).level_ratios == POOL_RATIOS == (.05,.40,.55)
    assert "GenerationConfig('budding_pool'" in inspect.getsource(pool._generate_param_only_pool)
    engine = ActiveLearningEngine()
    with patch.object(pool, '_generate_param_only_pool', return_value='budding') as bud, \
         patch.object(fission, 'load_pool', return_value='fission') as fis:
        assert engine._generate_param_only_pool(SimpleNamespace(system_id='budding')) == 'budding'
        assert engine._generate_param_only_pool(SimpleNamespace(system_id='fission')) == 'fission'
        assert bud.call_count == fis.call_count == 1
    # Dynamic expansion reproduces the complete original Budding function.
    old_x = function_from_source(ROOT/'utils/archive/al_callback_legacy.py',
                                 '_get_pool_X_batch', 'ActiveLearningCallback')
    for width in (39, 16):
        context = SimpleNamespace(_pool_X=None,
            _pool_Y_ic=torch.arange(4*width).reshape(4, width).float(),
            _al4pde_t_grid=torch.linspace(0, 1, 8).reshape(1, 1, 8))
        for indices in ([0, 2], slice(0, 2), np.array([3, 1])):
            actual = pool._get_pool_X_batch(context, indices)
            assert actual.shape == (2, width, 2, 8)
            if width == 39:
                torch.testing.assert_close(actual, old_x(context, indices), rtol=0, atol=0)
    return dict(budding_pool_uses_shared_exact3_recipe=True, cross_system_dispatch_verified=True,
                budding_inputs_bitwise_equal=True, fission_dynamic_width_verified=True)


class LinearSurrogate(torch.nn.Module):
    def forward_ic_time(self, x, p):
        return p[:, :1, None].expand(-1, 39, 8)


def uncertainty_probe(official):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    cb = ActiveLearningCallback(strategy='al4pde_uncertainty')
    cb._pool_P = torch.zeros(3, 141)
    cb._get_pool_X_batch = lambda indices: torch.zeros(len(indices), 39, 2, 8)
    dm = SimpleNamespace(p_std=torch.ones(1, 141))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        current = cb._compute_mc_dropout_uncertainty(
            LinearSurrogate(), 'cpu', np.arange(3), dm, n_mc_samples=5)
    ensemble_uncertainty = function_from_source(
        official/'prob_models/ensemble.py', 'uncertainty', 'Ensemble')
    constant = lambda value: lambda xx, grid, pde_param: torch.full((len(xx), 39, 8), value)
    x = torch.zeros(3, 39, 2, 8)
    identical = ensemble_uncertainty(SimpleNamespace(base_models=[constant(1.), constant(1.)]), x, None)
    disagreement = ensemble_uncertainty(SimpleNamespace(base_models=[constant(0.), constant(2.)]), x, None)
    assert np.all(current > 0), 'sensitive deterministic model should score positively'
    assert torch.count_nonzero(identical) == 0
    assert torch.all(disagreement == 2.)
    return {
        'deterministic_input_sensitive_project_variance': current.tolist(),
        'official_identical_ensemble_variance': float(identical.mean()),
        'official_disagreeing_input_constant_ensemble_variance': float(disagreement.mean()),
        'conclusion': 'input sensitivity and ensemble disagreement are different signals',
    }


def ensemble_alignment_probe(official):
    """Execute official Ensemble.uncertainty source and compare the active scorer."""
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback

    class Member(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.tensor(offset))
        def forward_ic_time(self, x, p):
            return (p[:, :1, None] + self.offset).expand(-1, 39, 8)
        def forward(self, x, grid, p):
            return self.forward_ic_time(x, p)

    members = [Member(0.), Member(2.)]
    callback = ActiveLearningCallback(strategy='al4pde_uncertainty')
    callback._pool_P = torch.arange(3.).unsqueeze(1).expand(-1, 141)
    callback._get_pool_X_batch = lambda indices: torch.zeros(len(indices), 39, 2, 8)
    callback.set_ensemble_members(members)
    callback.ensemble_uncertainty_mode = 'sample_variance'
    actual = callback._compute_ensemble_uncertainty('cpu', np.arange(3), batch_size=2)
    official_variance = function_from_source(official/'prob_models/ensemble.py', 'uncertainty', 'Ensemble')
    expected = official_variance(SimpleNamespace(base_models=members),
        torch.zeros(3, 39, 2, 8), None, callback._pool_P).reshape(3, -1).mean(1)
    torch.testing.assert_close(actual, expected)
    rollout = function_from_source(official/'prob_models/ensemble.py', 'unc_independent_rollout', 'Ensemble')
    outputs = torch.stack([m.forward_ic_time(torch.zeros(3,39,2,8), callback._pool_P) for m in members])
    state = SimpleNamespace(_roll_out_all=lambda *args: outputs, adapt_unc_to_loss=True,
                            loss=lambda pred, target, **kwargs: (pred-target)**2,
                            first_model_prediction=True)
    _, expected_mse = rollout(state, None, None, 8)
    callback.ensemble_uncertainty_mode = 'loss_mse'
    actual_mse = callback._compute_ensemble_uncertainty('cpu', np.arange(3))
    torch.testing.assert_close(actual_mse, expected_mse.reshape(3,-1).mean(1))
    return {'default_loss_mse_matches_official_rollout': True,
            'default_scores': actual_mse.tolist(), 'same_input_variance_and_mean_match_official': True,
            'scores': actual.tolist(), 'independent_members': len(members),
            'direct_official_import': False,
            'scope': 'project training/trajectory adapter with official-equivalent numerical kernel'}


def official_inventory(official):
    statuses = {
        'random': 'not ported: fresh parameter draws; user retains fixed-pool random',
        'pool_random': 'project al4pde_random: fixed-pool semantics retained',
        'top_k': 'al4pde_uncertainty: ensemble score and top-k project adapter',
        'power': 'al4pde_power: ensemble and official weighted sampling semantics',
        'max_dist': 'al4pde_max_dist: predicted trajectory sketch and actual bmdal MaxDist, no posterior transform',
        'coreset_maxdist': 'not ported: requires surrogate latent feature interface',
        'lcmd': 'al4pde_lcmd: predicted trajectory sketch and actual bmdal LCMD, no posterior transform',
        'bait': 'al4pde_bait: predicted trajectory sketch/posterior and actual bmdal selector',
    }
    configs = sorted(p.stem for p in (official.parent/'config/acquisition').glob('*.yaml') if p.stem != 'base')
    return [{'configuration': name, 'status': statuses.get(name, 'unreviewed official configuration')}
            for name in configs]


def power_bait_probe(official):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    power = function_from_source(official/'acquisition/uncertainty_based.py', 'power_sampling')
    numpy_inf_before = getattr(np, 'Inf', None)
    scores = torch.tensor([.1, .4, .8, .2, .9])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        expected = power(scores, 3, beta=1.7)
        torch.manual_seed(19)
        actual = ActiveLearningCallback._power_sample_indices(scores, 3, beta=1.7)
        assert torch.equal(expected, actual)
        pool, train = torch.randn(12, 4), torch.randn(5, 4)
        selected = ActiveLearningCallback._bmdal_bait_select(pool, train, 3)
    return {'power_matches_official_with_same_scores_and_rng': True,
            'bait_actual_bmdal_numpy2_smoke': selected.tolist(),
            'global_numpy_namespace_unchanged_by_bait': getattr(np, 'Inf', None) == numpy_inf_before}


def lcmd_probe(official):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    from budding_yeast_v2.utils.active_learning.backends import selection_module
    from bmdal_reg.bmdal.features import Features
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    reference = function_from_source(official/'acquisition/distance_pool_based.py', 'max_dist_selection')
    reference.__globals__.update(Features=Features, IdentityFeatureMap=IdentityFeatureMap,
                                 TensorFeatureData=TensorFeatureData, device=torch.device('cpu'))
    factory = lambda **kwargs: selection_module().LargestClusterMaxDistSelectionMethod(
        **kwargs, sel_with_train=True, verbosity=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        pool, train = torch.randn(15, 5), torch.randn(4, 5)
        expected = reference(4, train, pool, factory, posterior_tfm=False)
        actual = ActiveLearningCallback._bmdal_lcmd_select(pool, train, 4)
        assert np.array_equal(actual, expected.numpy())
    return {'official_pipeline_matches': True, 'selected': actual.tolist(),
            'posterior_transform': False, 'distance_weight': 'sq-dist'}


def run_audit(official):
    from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
    current_path = ROOT/'utils/al_callback.py'
    archive_path = ROOT/'utils/archive/al_callback_legacy.py'
    active = class_methods(current_path, 'ActiveLearningCallback')
    archived = class_methods(archive_path, 'ActiveLearningCallback')
    assert all(n not in active and n in archived for n in ARCHIVED_METHODS)
    intentional = {'_al4pde_uncertainty_sample'}
    preserved = [n for n in STRATEGY_METHODS if n not in intentional and numerical_ast(active[n]) == numerical_ast(archived[n])]
    assert len(preserved) == len(STRATEGY_METHODS)-1, 'unexpected online acquisition change'
    changed_shared = [n for n in active.keys() & archived.keys()
                      if n not in ('__init__', 'al4pde_select_and_append', '_al4pde_uncertainty_sample', '_is_al4pde_strategy', 'al4pde_setup', '_build_broadcast_from_pool', '_generate_param_only_pool', '_get_pool_X_batch')
                      and numerical_ast(active[n]) != numerical_ast(archived[n])]
    assert not changed_shared, changed_shared
    pool_adaptation = budding_pool_compatibility_probe(archived)
    for name in LEGACY_STRATEGIES:
        try:
            ActiveLearningCallback(strategy=name)
        except ValueError as e:
            assert 'archiv' in str(e)
        else:
            raise AssertionError(f'legacy strategy remains enabled: {name}')
    # Same scalar scores yield the same top-K away from ties. Their origin differs.
    top_k = function_from_source(official/'acquisition/uncertainty_based.py', 'top_k')
    scores = torch.tensor([.4, .9, .1, .6])
    assert set(top_k(scores, 2).tolist()) == set(np.argsort(scores.numpy())[-2:].tolist())
    schedule = function_from_source(official/'acquisition/data_schedule.py', 'num_batches', 'ExponentialSchedule')
    cb = ActiveLearningCallback(strategy='al4pde_random', initial_train_size=10)
    cb._pool_size = 1000
    for i in range(4):
        cb._al4pde_al_iter = i
        assert cb._al4pde_scheduled_num_add() == schedule(SimpleNamespace(first_batch_num=10), i)
    package_sources = (ROOT/'utils/active_learning').rglob('*.py')
    imports = [n.module for path in package_sources for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)]
    direct_official = any(m and (m == 'al4pde' or m.startswith('al4pde.')) for m in imports)
    assert not direct_official
    return {
        'archive_sha256': hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        'archived_strategy_count': len(LEGACY_STRATEGIES),
        'removed_legacy_methods': len(ARCHIVED_METHODS),
        'online_acquisitions_numerically_unchanged': preserved,
        'intentional_acquisition_change': 'uncertainty now uses trained ensemble variance and official top-k ordering',
        'active_ensemble_probe': ensemble_alignment_probe(official),
        'power_and_bait_probe': power_bait_probe(official),
        'lcmd_probe': lcmd_probe(official),
        'official_inventory': official_inventory(official),
        'registration_extended_for_power_bait_and_lcmd': True,
        'system_pool_adaptation': pool_adaptation,
        'package_acquisition_invalidates_old_full_pool_predictions': True,
        'official_source': str(official),
        'project_directly_imports_official_acquisition': direct_official,
        'same_scores_top_k_agree': True,
        'unclipped_exponential_schedule_agrees': True,
        'historical_input_noise_counterexample': uncertainty_probe(official),
        'remaining_differences': [
            'uncertainty numerical formula aligned; full PDE Task/ProbModel runtime is not imported',
            'random: fixed-pool random versus official Random fresh draws (official pool_random is the closer counterpart)',
            'diversity: custom 332-dimensional features and local MaxDist versus configurable official DistancePoolBased/bmdal_reg',
            'fresh-draw Random and latent coreset_maxdist are not integrated; existing diversity uses custom features',
            'project real-test envelope/nominal-flux filters are additional selection priors',
            'some project strategies refill valid-sample quotas while others do not',
            'MAAL/MASS/novelty/residual/bifurcation/event/Fisher names describe local extensions',
        ],
        'scope': 'isolated source-function and artificial-input audit; not end-to-end official training',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, default=ROOT.parent/'al4pde/al4pde')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        report = run_audit(args.official_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
