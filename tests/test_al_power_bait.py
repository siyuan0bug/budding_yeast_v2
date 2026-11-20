"""Official Power and BAIT numerical pipeline and oracle-after-selection checks."""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
from budding_yeast_v2.utils.sampling.backends import selection_module
from budding_yeast_v2.scripts.analysis.audit_al4pde_alignment import function_from_source, official_inventory

ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT.parent/'al4pde/al4pde'


@pytest.mark.parametrize('beta',[.5,1.,2.])
def test_power_matches_official_with_positive_scores_and_identical_rng(beta):
    official=function_from_source(OFFICIAL/'acquisition/uncertainty_based.py','power_sampling')
    scores=torch.tensor([.1,.5,.2,.8,1.])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);expected=official(scores,3,beta)
        torch.manual_seed(42);actual=ActiveLearningCallback._power_sample_indices(scores,3,beta)
    torch.testing.assert_close(actual,expected)


@pytest.mark.parametrize('scores',[[0.,0.,0.],[0.,0.,1.],[1e-35,2e-35,3e-35],[1e35,2e35,3e35]])
def test_power_degenerate_scores_still_produce_unique_valid_indices(scores):
    selected=ActiveLearningCallback._power_sample_indices(torch.tensor(scores),3,beta=5.)
    assert set(selected.tolist())=={0,1,2}


@pytest.mark.parametrize('scores',[[-1.,2.],[float('nan'),2.],[float('inf'),2.]])
def test_power_rejects_invalid_scores(scores):
    with pytest.raises(ValueError,match='finite nonnegative'):
        ActiveLearningCallback._power_sample_indices(torch.tensor(scores),1)


def test_power_uses_ensemble_and_only_calls_oracle_after_selection():
    cb=ActiveLearningCallback(strategy='al4pde_power',initial_train_size=2)
    cb._pool_size=5
    cb._compute_ensemble_uncertainty=lambda device,idx:torch.tensor([1.,2.,3.,4.,5.])
    seen=[]
    cb._build_broadcast_from_pool=lambda idx,dm:seen.append(np.asarray(idx))
    cb._al4pde_power_sample(None,None,None,'cpu',0)
    assert len(seen)==1 and len(seen[0])==2 and len(set(seen[0]))==2


def test_bmdal_numpy2_compat_is_local_and_uses_real_library_code():
    original=importlib.import_module('bmdal_reg.bmdal.selection')
    had_inf=hasattr(np,'Inf')
    adapted=selection_module()
    assert adapted is not original and original.np is np
    assert adapted.np.Inf==np.inf
    assert hasattr(np,'Inf')==had_inf
    assert adapted.__file__==original.__file__


def official_selection_function():
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.features import Features
    tree=ast.parse((OFFICIAL/'acquisition/distance_pool_based.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='max_dist_selection')
    ns=dict(torch=torch,device=torch.device('cpu'),Features=Features,
            TensorFeatureData=TensorFeatureData,IdentityFeatureMap=IdentityFeatureMap)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'official_max_dist_selection','exec'),ns)
    return ns['max_dist_selection']


@pytest.mark.parametrize('budget',[1,3,8])
def test_bait_exactly_matches_official_posterior_selection_pipeline(budget):
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        torch.manual_seed(6)
        pool,train=torch.randn(12,5),torch.randn(6,5)
        factory=lambda **kwargs:selection_module().BaitFeatureSpaceSelectionMethod(
            **kwargs,sel_with_train=True,noise_sigma=1e-2,verbosity=0)
        reference=official_selection_function()(budget,train,pool,factory,posterior_tfm=True)
        actual=ActiveLearningCallback._bmdal_bait_select(pool,train,budget)
        np.testing.assert_array_equal(actual,reference.numpy())


class Surrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.inputs=[]
    def forward_ic_time(self,x,p):
        self.inputs.append(p.clone())
        time=torch.arange(4.,device=p.device)[None,None,:]
        return self.weight*(p[:,:3,None]+time)


def test_bait_features_predict_train_and_pool_without_reading_ground_truth():
    cb=ActiveLearningCallback(strategy='al4pde_bait',bait_sketch_dim=5)
    cb._pool_size=4
    cb._pool_P=torch.arange(12.).reshape(4,3)
    cb._get_pool_X_batch=lambda idx:torch.zeros(len(idx),3,2,4)
    tx,tp=torch.zeros(3,3,2,4),torch.arange(9.).reshape(3,3)+.3
    # Labels intentionally NaN: feature generation must never use them.
    dm=SimpleNamespace(train_dataset_synth=TensorDataset(tx,tp,torch.full((3,3,4),float('nan'))))
    model=Surrogate()
    with patch('budding_yeast_v2.utils.sampling.oracle.simulate_mutant_mass',side_effect=AssertionError('oracle before selection')):
        pool,train=cb._compute_bait_features(model,dm,'cpu',batch_size=2)
    assert pool.shape==(4,5) and train.shape==(3,5)
    assert torch.isfinite(pool).all() and torch.isfinite(train).all()
    assert model.training
    first=model.inputs[0]
    torch.testing.assert_close(first,tp[:2])
    old=pool.clone()
    model.weight.data.mul_(2.)
    pool2,_=cb._compute_bait_features(model,dm,'cpu',batch_size=2)
    torch.testing.assert_close(pool2,2*old)


def test_bait_ode_spatial_mean_preserves_time_and_matches_official_shared_sketch():
    cb=ActiveLearningCallback(strategy='al4pde_bait',bait_sketch_dim=5)
    predictions=torch.arange(24.).reshape(2,3,4)
    actual=cb._bait_feature_batch(predictions)
    # Reuse actual sketch with the exact official spatial_mean transformation.
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.features import Features
    ode_as_pde=predictions.transpose(1,2).unsqueeze(1)
    raw=ode_as_pde.flatten(1,-3).mean(1).flatten(1)
    features=Features(IdentityFeatureMap(raw.shape[1]),TensorFeatureData(raw))
    expected=cb._bait_sketch(features).get_feature_matrix()
    torch.testing.assert_close(actual,expected)


def test_bait_oracle_called_once_with_selected_indices_only():
    cb=ActiveLearningCallback(strategy='al4pde_bait',initial_train_size=2)
    cb._pool_size=7
    cb._compute_bait_features=lambda *args:(torch.ones(7,2),torch.ones(3,2))
    cb._bmdal_bait_select=lambda *args:np.array([1,5])
    cb._build_broadcast_from_pool=lambda indices,dm:indices
    np.testing.assert_array_equal(cb._al4pde_bait_sample(None,None,None,'cpu',0),[1,5])


def test_inventory_is_eight_actual_configs_and_marks_unported_variants():
    rows={row['configuration']:row['status'] for row in official_inventory(OFFICIAL)}
    assert set(rows)=={'random','pool_random','top_k','power','max_dist','coreset_maxdist','lcmd','bait'}
    assert rows['coreset_maxdist'].startswith('not ported')
    assert rows['max_dist'].startswith('al4pde_max_dist')


def test_private_bait_works_without_global_numpy_inf_in_isolated_process():
    import os
    import subprocess
    import sys
    code = """
import numpy as np
import torch
from budding_yeast_v2.utils.sampling.backends import selection_module
from bmdal_reg.bmdal.features import Features
from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
from bmdal_reg.bmdal.feature_data import TensorFeatureData
assert not hasattr(np, 'Inf')
torch.manual_seed(3)
pool=Features(IdentityFeatureMap(3),TensorFeatureData(torch.randn(8,3)))
train=Features(IdentityFeatureMap(3),TensorFeatureData(torch.randn(4,3)))
selector=selection_module().BaitFeatureSpaceSelectionMethod(pool_features=pool,
    train_features=train,sel_with_train=True,noise_sigma=1e-2,verbosity=0)
selected=selector.select(2)
assert len(set(selected.tolist())) == 2
assert not hasattr(np, 'Inf')
"""
    result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,
        PYTHONPATH=str(ROOT.parent)),capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
