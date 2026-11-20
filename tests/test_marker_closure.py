"""Numerical/gradient contracts for the experimental marker readout."""
import io

import pytest
import torch

from budding_yeast_v2.data.normalization import normalize_y, normalize_p
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS, get_default_params
from budding_yeast_v2.models.ours.bio_refiner_marker_closure import (
    affine_scan, linear_forcing_step, downward_crossings, marker_closure,
    BioRefinerMarkerClosure,
)


def parameters(batch=1):
    p=get_default_params()
    p.update(init_BUB2=.2,init_CDH1=.9304992,init_CDH1T=1.)
    return torch.tensor([[p[k] for k in PARAM_KEYS]]*batch,dtype=torch.float64)


def test_scan_matches_recurrence_and_gradients_including_exact_reset():
    torch.manual_seed(13)
    a=torch.rand(2,3,37,dtype=torch.float64,requires_grad=True)
    mask=torch.ones_like(a);mask[...,8]=0;mask[...,24]=0
    b=torch.rand_like(a,requires_grad=True);initial=torch.rand(2,3,dtype=a.dtype,requires_grad=True)
    scaled=a*mask
    y=initial;sequence=[]
    for k in range(a.shape[-1]):
        y=scaled[...,k]*y+b[...,k];sequence.append(y)
    expected=torch.stack(sequence,-1);actual=affine_scan(scaled,b,initial)
    torch.testing.assert_close(actual,expected)
    g1=torch.autograd.grad(expected.square().sum(),(a,b,initial),retain_graph=True)
    g2=torch.autograd.grad(actual.square().sum(),(a,b,initial))
    for u,v in zip(g1,g2):torch.testing.assert_close(u,v)


@pytest.mark.parametrize('rate',[0.,1e-8,.06,100.])
def test_linear_forcing_matches_independent_quadrature(rate):
    from scipy.integrate import quad
    a,b=linear_forcing_step(torch.tensor(.2,dtype=torch.float64),torch.tensor(1.3,dtype=torch.float64),
                            torch.tensor(rate,dtype=torch.float64),torch.tensor(.42,dtype=torch.float64))
    integral=quad(lambda t: __import__('math').exp(-rate*(.42-t))*(.2+1.1*t/.42),0,.42)[0]
    assert b.item()==pytest.approx(integral,rel=1e-9,abs=1e-11)
    assert a.item()==pytest.approx(__import__('math').exp(-rate*.42))


def test_direction_and_repeated_crossings():
    mask,fraction=downward_crossings(torch.tensor([[.1,.5,.1,.6,.1]]),.3)
    assert mask.tolist()==[[False,True,False,True]]
    torch.testing.assert_close(fraction[mask],torch.tensor([.5,.6]))


def test_marker_resets_recur_and_are_not_globally_latched():
    y=torch.zeros(1,39,5,dtype=torch.float64)
    y[:,0]=1.;y[:,2]=torch.tensor([.5,.1,.5,.1,.5])
    initial=y[:,:,0].clone();initial[:,33:36]=1
    times=torch.arange(5,dtype=y.dtype)[None]
    result=marker_closure(y,parameters(),initial,times)
    assert result.shape==(1,3,5)
    assert (result[:,:,1] < .3).all()
    assert (result[:,:,2] > result[:,:,1]).all()
    assert (result[:,:,3] < result[:,:,2]).all()
    torch.testing.assert_close(result[:,:,0],initial[:,33:36])
    # Interval crossings are piecewise differentiable, including their time.
    y.requires_grad_()
    marker_closure(y,parameters(),initial,times).sum().backward()
    assert torch.isfinite(y.grad).all()
    assert y.grad[:,2].abs().sum()>0


def test_no_false_reset_when_initially_below_guard():
    y=torch.ones(1,39,2,dtype=torch.float64)*.01
    initial=y[:,:,0].clone();initial[:,33:36]=2
    result=marker_closure(y,parameters(),initial,torch.tensor([[0.,.1]],dtype=y.dtype))
    assert (result[:,:,-1]>1.9).all()


@pytest.mark.parametrize('norm_type',['log1p','linear','asinh_z','asinh_scale'])
def test_model_normalization_backward_and_state_roundtrip(norm_type):
    torch.set_num_threads(2)
    model=BioRefinerMarkerClosure(width=8,modes=3,n_blocks=1,coarse_n_blocks=1).float()
    mean=torch.zeros(1,39,1);std=torch.ones_like(mean)
    pm=torch.zeros(1,141);ps=torch.ones_like(pm);sc=torch.ones_like(mean)*.5
    model.set_stats(mean,std,pm,ps,norm_type=norm_type,s_c=sc)
    physical_ic=torch.ones(2,39,1)*.1
    ic=normalize_y(physical_ic,mean,std,norm_type,sc).expand(-1,-1,17)
    time=torch.linspace(0,1,17)[None,None].expand(2,39,-1)
    x=torch.stack((ic,time),2)
    p=normalize_p(parameters(2).float(),pm,ps,norm_type)
    result=model(x,p)
    assert result.shape==(2,39,17) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert model.stage1_coarse is not None
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
    clone=BioRefinerMarkerClosure(width=8,modes=3,n_blocks=1,coarse_n_blocks=1).float()
    buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
    clone.load_state_dict(torch.load(buffer,weights_only=True),strict=True)
    torch.testing.assert_close(result,clone(x,p))


def test_registry_binding_and_unconfigured_stats():
    from budding_yeast_v2.models.yeast_lit_module import YeastLitModule, MODEL_REGISTRY
    assert MODEL_REGISTRY['biom_refiner_marker_closure'] is BioRefinerMarkerClosure
    model=YeastLitModule(model_name='biom_refiner_marker_closure',width=8,modes=3,n_blocks=1,
                         loss_type='hybrid_mse_rel')
    with pytest.raises(RuntimeError,match='statistics'):
        model(torch.zeros(1,39,2,8),torch.zeros(1,141))


def test_invalid_time_grid_is_rejected():
    with pytest.raises(ValueError,match='increasing'):
        marker_closure(torch.zeros(1,39,2),parameters(),torch.zeros(1,39),torch.zeros(1,2))
