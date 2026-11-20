"""Production selectors must work with author packages and repositories unavailable."""
import builtins
from unittest.mock import patch
import numpy as np
import torch


def test_all_local_kernels_and_projection_run_without_author_imports():
    from budding_yeast_v2.utils.sampling.backends import _bmdal_distance_select,_bmdal_bait_select
    from budding_yeast_v2.utils.sampling.baseline.authors import vessal_penultimate,leverage
    from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(('bmdal_reg','al4pde','baselines')):
            raise AssertionError(f'Runtime author dependency: {name}')
        return original(name,*args,**kwargs)
    torch.manual_seed(12)
    pool=torch.randn(25,5);train=torch.randn(4,5)
    with patch('builtins.__import__',side_effect=guarded):
        outputs=[_bmdal_distance_select(pool,train,7,method=kind) for kind in ('max_dist','lcmd')]
        outputs.append(_bmdal_bait_select(pool,train,7))
        outputs.append(vessal_penultimate(pool.numpy(),7,42)[0])
        outputs.append(leverage(pool.numpy(),7,42,block_size=12)[0])
        predictor=Predictor([],None,10,'cpu',sketch_seed=42)
        predictor.batches=lambda inputs,latent=False:iter([(0,torch.arange(120.).reshape(3,4,10))])
        inputs=type('Inputs',(),{'ids':np.arange(3)})()
        features=predictor.features(inputs,7,True)
    assert all(len(x)==len(set(x))==7 for x in outputs)
    assert features.shape==(3,7) and np.isfinite(features).all()


def test_bait_rank_one_selection_matches_direct_inverse_trace_objective():
    from budding_yeast_v2.utils.sampling.baseline.bait import bait_select
    torch.manual_seed(14)
    pool=torch.randn(20,4,dtype=torch.float64);train=torch.randn(6,4,dtype=torch.float64)
    sigma=.2;eye=torch.eye(4,dtype=pool.dtype)
    covariance=pool.T@pool+train.T@train
    information=eye+train.T@train/sigma**2
    expected=[]
    for _ in range(7):
        inverse=torch.linalg.inv(information)
        transformed=pool@inverse
        numerator=((transformed@covariance)*transformed).sum(1)
        scores=numerator/((transformed*pool).sum(1)+sigma**2+1e-8)
        scores[expected]=-torch.inf
        index=int(scores.argmax());expected.append(index)
        information+=torch.outer(pool[index],pool[index])/sigma**2
    np.testing.assert_array_equal(bait_select(pool,train,7,noise=sigma,posterior=False),expected)
