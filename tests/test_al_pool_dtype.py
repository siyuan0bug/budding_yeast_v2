"""Float64 input provenance must never leak into float32 AL model inference."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from budding_yeast_v2.data.normalization import normalize_p,normalize_y
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.utils.sampling import ActiveLearningEngine

STRATEGIES=('al4pde_random','al4pde_uncertainty','al4pde_power','al4pde_max_dist',
            'al4pde_lcmd','al4pde_bait','ours_response')


class Float32Surrogate(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv1d(39*2+141,39,1)
    def forward_ic_time(self,x,p):
        assert x.dtype==p.dtype==self.conv.weight.dtype==torch.float32
        return self.conv(torch.cat((x.flatten(1,2),p[:,:,None].expand(-1,-1,x.shape[-1])),1))


def stats(vars=39,params=141,norm='log1p',shift=0.):
    return SimpleNamespace(system_id='budding',norm_type=norm,
        mean_y=torch.full((1,vars,1),shift),std_y=torch.full((1,vars,1),1.+shift),
        p_mean=torch.full((1,params),shift),p_std=torch.full((1,params),1.+shift),
        s_c=torch.full((1,vars,1),.2))


@pytest.mark.parametrize('norm',('linear','log1p','asinh_z','asinh_scale'))
@pytest.mark.parametrize('dimensions',((39,141),(16,58)))
def test_refit_keeps_physical_precision_and_encodes_float32_for_both_system_shapes(norm,dimensions):
    v,d=dimensions
    engine=ActiveLearningEngine()
    engine._pool_size=3
    engine._pool_P_real=torch.linspace(.01,2,3*d,dtype=torch.float64).reshape(3,d)
    engine._pool_initial_physical=torch.linspace(.123456789012345,2,3*v,dtype=torch.float64).reshape(3,v)
    original_p=engine._pool_P_real.clone();original_y=engine._pool_initial_physical.clone()
    engine._al4pde_t_grid=torch.linspace(0,1,8,dtype=torch.float64).reshape(1,1,-1)
    for shift in (0.,.1,2.):
        dm=stats(v,d,norm,shift)
        engine._pool_shape_features_cached='old model'
        engine._pool_X=torch.full((3,v,2,8),float('nan'),dtype=torch.float64)
        engine.refresh_model_inputs(dm)
        assert engine._pool_X is None
        assert engine._pool_P.dtype==engine._pool_Y_ic.dtype==torch.float32
        assert engine._get_pool_X_batch([0,2]).dtype==torch.float32
        torch.testing.assert_close(engine._pool_P,normalize_p(original_p.float(),dm.p_mean,dm.p_std,norm))
        torch.testing.assert_close(engine._pool_Y_ic,normalize_y(original_y.float(),dm.mean_y,dm.std_y,norm,dm.s_c))
        torch.testing.assert_close(engine._pool_P_real,original_p,rtol=0,atol=0)
        torch.testing.assert_close(engine._pool_initial_physical,original_y,rtol=0,atol=0)
        assert engine._pool_initial_physical.dtype==torch.float64
        assert engine._pool_shape_features_cached is None


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_each_current_strategy_predicts_after_repeated_refits_and_pool_pruning(strategy):
    engine=ActiveLearningEngine(strategy=strategy,initial_train_size=2,bait_sketch_dim=5)
    backend=BuddingSystem();defaults=backend.defaults()
    params=[];initial=[];overrides=[]
    for copies in range(2,10):
        ovr={'ksn2pp':defaults['ksn2pp']*copies,'mdt':90.}
        params.append([ovr.get(k,defaults[k]) for k in backend.PARAM_KEYS])
        initial.append(backend.prepare_initial_conditions(ovr)[1]);overrides.append(ovr)
    engine._pool_size=8
    engine._pool_P_real=torch.tensor(np.asarray(params),dtype=torch.float64)
    engine._pool_initial_physical=torch.tensor(np.asarray(initial),dtype=torch.float64)
    engine._pool_ovr=overrides
    engine._pool_mass_meta=[{'pairs':[('CLN2','MC')],'level':1} for _ in range(8)]
    engine._pool_mutant_names=np.array([f'MASS_RAND_CLN2[MC]_L1_{i}' for i in range(8)])
    engine._al4pde_t_grid=torch.linspace(0,1,16,dtype=torch.float64).reshape(1,1,-1)
    model=Float32Surrogate();engine.set_ensemble_members([model,Float32Surrogate()])
    for shift in (0.,.1,1.):
        dm=stats(shift=shift)
        engine.refresh_model_inputs(dm)
        dm.train_dataset_synth=TensorDataset(engine._get_pool_X_batch([0,1]),engine._pool_P[:2],torch.full((2,39,16),float('nan')))
        with patch.object(engine,'_build_broadcast_from_pool',side_effect=AssertionError('early oracle')):
            plan=engine.select(model,dm,'cpu')
        assert len(plan.candidate_ids)==2 and len(set(plan.candidate_ids))==2
        assert engine._pool_initial_physical.dtype==torch.float64
        keep=np.ones(engine._pool_size,dtype=bool);keep[0]=False
        engine.pool.keep(keep)
