import numpy as np
import torch
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_al_random_extremes import moments,fit,energy,error_rows
from budding_yeast_v2.utils.training.losses import HybridMSERelLoss


def test_transformed_moments_reproduce_direct_unbiased_stats_and_energy():
    y=np.random.RandomState(42).lognormal(0,3,(12,4,17)).astype('float32')
    y[0,0,0]=-1e-6
    maxima,minima,rawsum,rawsquare,total,square=moments(y)
    rows=np.array([1,3,4,6,9])
    direct=np.log1p(np.maximum(y.astype('float64'),0))
    mean,std=fit(total,square,rows,17)
    np.testing.assert_allclose(mean,direct[rows].mean((0,2)),rtol=1e-12)
    np.testing.assert_allclose(std,direct[rows].std((0,2),ddof=1)+1e-6,rtol=1e-12)
    np.testing.assert_allclose(energy(total,square,mean,std,17),((direct-mean[None,:,None])/std[None,:,None]) .__pow__(2).sum(2),rtol=1e-10,atol=1e-10)
    assert minima[0,0]<0 and maxima.shape==(12,4)


def test_per_sample_hybrid_matches_training_loss_including_relative_clamp():
    rng=np.random.RandomState(5)
    target=rng.normal(size=(5,4,17)).astype('float32');target[0]=0
    pred=rng.normal(size=(5,4,17)).astype('float32')
    physical_target=np.expm1(target);physical_pred=np.expm1(pred)
    rows=error_rows(physical_pred,physical_target,pred,target)
    actual=HybridMSERelLoss()(torch.tensor(pred),torch.tensor(target)).item()
    np.testing.assert_allclose(rows['hybrid'].mean(),actual,rtol=1e-6)
