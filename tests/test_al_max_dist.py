from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.backends import selection_module
from budding_yeast_v2.scripts.analysis.audit_al4pde_alignment import function_from_source


@pytest.mark.parametrize('budget',[0,1,5,12,20])
def test_predicted_maxdist_matches_official_selection(budget):
    from bmdal_reg.bmdal.features import Features
    from bmdal_reg.bmdal.feature_maps import IdentityFeatureMap
    from bmdal_reg.bmdal.feature_data import TensorFeatureData
    root=Path(__file__).resolve().parents[2]
    reference=function_from_source(root/'al4pde/al4pde/acquisition/distance_pool_based.py','max_dist_selection')
    reference.__globals__.update(Features=Features,IdentityFeatureMap=IdentityFeatureMap,
        TensorFeatureData=TensorFeatureData,device=torch.device('cpu'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);pool,train=torch.randn(12,5),torch.randn(4,5)
        actual=ActiveLearningEngine._bmdal_predicted_max_dist_select(pool,train,budget)
        if budget==0:assert len(actual)==0
        else:
            expected=reference(min(budget,12),train,pool,
                lambda **kwargs:selection_module().MaxDistSelectionMethod(**kwargs,sel_with_train=True),posterior_tfm=False)
            np.testing.assert_array_equal(actual,expected.numpy())


def test_maxdist_rejects_nonfinite_features_and_selects_unique_degenerate_rows():
    select=ActiveLearningEngine._bmdal_predicted_max_dist_select
    assert len(set(select(torch.ones(7,3),torch.ones(2,3),5)))==5
    with pytest.raises(ValueError,match='finite'):
        select(torch.full((4,2),float('nan')),torch.ones(2,2),2)


def test_maxdist_uses_predictions_and_never_ground_truth_before_selection():
    class Model(torch.nn.Module):
        def forward_ic_time(self,x,p):return p[:,:3,None]+torch.arange(4.)[None,None,:]
    engine=ActiveLearningEngine(strategy='al4pde_max_dist',initial_train_size=2,bait_sketch_dim=5)
    engine._pool_size=6;engine._pool_P=torch.arange(18.).reshape(6,3)
    engine._get_pool_X_batch=lambda idx:torch.zeros(len(idx),3,2,4)
    dm=SimpleNamespace(system_id='budding',train_dataset_synth=TensorDataset(
        torch.zeros(3,3,2,4),torch.arange(9.).reshape(3,3)+.1,torch.full((3,3,4),float('nan'))))
    model=Model();events=[]
    compute=engine._compute_predicted_distance_features
    engine._compute_predicted_distance_features=lambda *args:(events.append('predict') or compute(*args))
    select=engine._bmdal_predicted_max_dist_select
    engine._bmdal_predicted_max_dist_select=lambda *args:(events.append('select') or select(*args))
    with patch('budding_yeast_v2.utils.sampling.oracle.simulate_mutant_mass',side_effect=AssertionError('early oracle')):
        plan=engine.select(model,dm,'cpu')
    assert events==['predict','select'] and len(plan.candidate_ids)==2 and model.training
