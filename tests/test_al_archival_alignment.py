"""Regression checks for legacy AL archival and documented official differences."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
from budding_yeast_v2.utils.archive.al_callback_legacy import ActiveLearningCallback as ArchivedCallback
from budding_yeast_v2.scripts.analysis.audit_al4pde_alignment import run_audit, LEGACY_STRATEGIES

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ('al4pde_random','al4pde_uncertainty','al4pde_power','al4pde_bait','al4pde_lcmd','al4pde_diversity','maal','al4pde_mass',
          'al4pde_novelty','al4pde_residual','al4pde_residual_diversity','al4pde_res_div',
          'al4pde_bifurcation','al4pde_bifurcation_entropy','al4pde_bif_entropy',
          'al4pde_event_entropy','al4pde_event','al4pde_fisher_oed','al4pde_foed')


@pytest.mark.parametrize('name', LEGACY_STRATEGIES)
def test_retired_strategies_are_explicitly_rejected_and_preserved(name):
    with pytest.raises(ValueError, match='archived'):
        ActiveLearningCallback(strategy=name)
    assert ArchivedCallback(strategy=name).strategy == name


@pytest.mark.parametrize('name', ONLINE)
def test_online_names_and_aliases_still_dispatch(name):
    callback = ActiveLearningCallback(strategy=name)
    called = []
    methods = {'maal':'_maal_sample'}
    aliases = {'al4pde_res_div':'al4pde_residual_diversity',
               'al4pde_bif_entropy':'al4pde_bifurcation_entropy',
               'al4pde_event':'al4pde_event_entropy','al4pde_foed':'al4pde_fisher_oed'}
    method = methods.get(name, '_'+aliases.get(name,name)+'_sample')
    assert callback.acquisition.spec.function == method
    callback.acquisition.select = lambda *args: called.append(name)
    assert callback.al4pde_select_and_append(None, None, device='cpu') == 0
    assert called == [name]


def test_train_parser_only_exposes_online_strategies_and_none():
    tree = ast.parse((ROOT/'train.py').read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
                and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--al_strategy')
    from budding_yeast_v2.utils.sampling.registry import CLI_STRATEGIES
    assert next(k.value for k in call.keywords if k.arg=='choices').id == 'CLI_STRATEGIES'
    choices = CLI_STRATEGIES
    assert set(choices) == {'none', *ONLINE}
    active = ast.parse((ROOT/'utils/sampling/engine.py').read_text())
    cls = next(n for n in active.body if isinstance(n, ast.ClassDef))
    assert not {'on_train_start','on_train_epoch_end','_standard_al_sample'} & {n.name for n in cls.body if isinstance(n,ast.FunctionDef)}


@pytest.mark.parametrize('strategy', ['al4pde_random', 'al4pde_diversity'])
def test_existing_core_selections_match_archive_under_same_random_state(strategy):
    selected = []
    for cls in (ArchivedCallback, ActiveLearningCallback):
        cb=cls(strategy=strategy, initial_train_size=3)
        cb._pool_size=7
        cb._build_broadcast_from_pool=lambda indices,*args,**kwargs: np.asarray(indices)
        cb._pool_P=torch.zeros(7,141)
        cb._get_pool_X_batch=lambda indices:torch.zeros(len(indices),39,2,8)
        cb._compute_pool_features_shape_aware=lambda *args:torch.arange(21.).reshape(7,3)
        cb._compute_train_features_shape_aware=lambda *args:torch.zeros(3,3)
        cb._standardize_shape_features=lambda a,b:(a,b)
        model=SimpleNamespace(eval=lambda:None, train=lambda:None,
              forward_ic_time=lambda x,p:p[:,:1,None].expand(-1,39,8))
        with torch.random.fork_rng(devices=[]), patch('torch.cuda.is_available',return_value=False):
            torch.manual_seed(17)
            rng_state=np.random.get_state()
            try:
                np.random.seed(17)
                method=getattr(cb,'_'+strategy+'_sample')
                selected.append(method(None,model,SimpleNamespace(p_std=torch.ones(1,141)),'cpu',0))
            finally:
                np.random.set_state(rng_state)
    np.testing.assert_array_equal(*selected)


def test_official_source_alignment_audit_and_uncertainty_counterexample():
    report=run_audit(ROOT.parent/'al4pde/al4pde')
    assert len(report['online_acquisitions_numerically_unchanged'])==11
    assert report['removed_legacy_methods']==19
    assert report['same_scores_top_k_agree']
    assert report['unclipped_exponential_schedule_agrees']
    assert not report['project_directly_imports_official_acquisition']
    assert report['active_ensemble_probe']['same_input_variance_and_mean_match_official']
    probe=report['historical_input_noise_counterexample']
    assert min(probe['deterministic_input_sensitive_project_variance'])>0
    assert probe['official_identical_ensemble_variance']==0
    assert probe['official_disagreeing_input_constant_ensemble_variance']==2
