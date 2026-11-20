"""LCMD official selection parity and predicted-feature integration."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
from budding_yeast_v2.utils.sampling.backends import selection_module
from budding_yeast_v2.scripts.analysis.audit_al4pde_alignment import function_from_source

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('budget',[0,1,5,12,20])
def test_lcmd_matches_official_pipeline_with_pool_clipping(budget):
    from bmdal_reg.bmdal.features import Features
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    reference=function_from_source(ROOT.parent/'al4pde/al4pde/acquisition/distance_pool_based.py','max_dist_selection')
    reference.__globals__.update(Features=Features,IdentityFeatureMap=IdentityFeatureMap,
        TensorFeatureData=TensorFeatureData,device=torch.device('cpu'))
    factory=lambda **kwargs:selection_module().LargestClusterMaxDistSelectionMethod(
        **kwargs,sel_with_train=True,verbosity=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        pool,train=torch.randn(12,5),torch.randn(4,5)
        actual=ActiveLearningCallback._bmdal_lcmd_select(pool,train,budget)
        if budget==0:
            assert len(actual)==0
        else:
            expected=reference(min(budget,12),train,pool,factory,posterior_tfm=False)
            np.testing.assert_array_equal(actual,expected.numpy())


def test_lcmd_degenerate_features_still_select_unique_candidates():
    result=ActiveLearningCallback._bmdal_lcmd_select(torch.ones(7,3),torch.ones(2,3),5)
    assert len(result)==len(set(result))==5


def test_lcmd_invalid_features_are_rejected():
    with pytest.raises(ValueError,match='finite'):
        ActiveLearningCallback._bmdal_lcmd_select(torch.full((4,2),float('nan')),torch.ones(2,2),2)


class Surrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward_ic_time(self,x,p):
        return self.weight*(p[:,:3,None]+torch.arange(4.)[None,None,:])


def test_lcmd_predicts_before_oracle_and_does_not_read_training_targets():
    cb=ActiveLearningCallback(strategy='al4pde_lcmd',initial_train_size=2,bait_sketch_dim=5)
    cb._pool_size=6
    cb._pool_P=torch.arange(18.).reshape(6,3)
    cb._get_pool_X_batch=lambda idx:torch.zeros(len(idx),3,2,4)
    dm=SimpleNamespace(train_dataset_synth=TensorDataset(torch.zeros(3,3,2,4),
        torch.arange(9.).reshape(3,3)+.1,torch.full((3,3,4),float('nan'))))
    model=Surrogate()
    events=[]
    real_compute=cb._compute_predicted_distance_features
    cb._compute_predicted_distance_features=lambda *args:(events.append('predictions') or real_compute(*args))
    real_select=cb._bmdal_lcmd_select
    cb._bmdal_lcmd_select=lambda *args:(events.append('selection') or real_select(*args))
    cb._build_broadcast_from_pool=lambda selected,*args:(events.append('oracle') or selected)
    with patch('budding_yeast_v2.utils.sampling.oracle.simulate_mutant_mass',side_effect=AssertionError('early oracle')):
        selected=cb._al4pde_lcmd_sample(None,model,dm,'cpu',0)
    assert events==['predictions','selection','oracle']
    assert len(selected)==len(set(selected))==2
    assert model.training


def test_lcmd_empty_pool_never_requests_prediction_or_oracle():
    cb=ActiveLearningCallback(strategy='al4pde_lcmd')
    cb._pool_size=0
    assert cb._al4pde_lcmd_sample(None,None,None,'cpu',0) is None


def test_lcmd_numpy2_compat_works_in_clean_process_without_global_patching():
    code="""
import numpy as np
import torch
from budding_yeast_v2.utils.sampling.backends import selection_module
from bmdal_reg.bmdal.features import Features
from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
from bmdal_reg.bmdal.feature_data import TensorFeatureData
assert not hasattr(np,'Inf')
torch.manual_seed(7)
pool=Features(IdentityFeatureMap(3),TensorFeatureData(torch.randn(10,3)))
train=Features(IdentityFeatureMap(3),TensorFeatureData(torch.randn(3,3)))
selected=selection_module().LargestClusterMaxDistSelectionMethod(pool_features=pool,
    train_features=train,sel_with_train=True,verbosity=0).select(3)
assert len(set(selected.tolist()))==3
assert not hasattr(np,'Inf')
"""
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,PYTHONPATH=str(ROOT.parent)),timeout=60)
    assert result.returncode==0,result.stderr
