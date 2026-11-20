"""Task information boundaries and direct source parity for added baselines."""
import ast
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from budding_yeast_v2.models.baselines import build_baseline
from budding_yeast_v2.models.baselines._upstream.mamba_core import Mamba,selective_scan_ref

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(device='cpu',length=32):
    t=torch.linspace(0,1,length,device=device).expand(2,3,-1)
    initial=torch.randn(2,3,device=device,requires_grad=True)
    return torch.stack((initial[...,None].expand_as(t),t),2),torch.randn(2,5,device=device,requires_grad=True)


@pytest.mark.parametrize('name',['kan','transformer','mamba'])
def test_only_initial_state_and_parameters_are_used(name):
    model=build_baseline(name,num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4).eval()
    x,p=inputs()
    corrupt=x.clone();corrupt[:,:,0,1:]=torch.randn_like(corrupt[:,:,0,1:])*100
    torch.testing.assert_close(model(x,p),model(corrupt,p))
    changed=x.clone();changed[:,:,0,0]+=0.3
    assert not torch.allclose(model(x,p),model(changed,p))
    assert not torch.allclose(model(x,p),model(x,p+0.3))
    shorter=x[...,:17]
    assert model(shorter,p).shape==(2,3,17)


@pytest.mark.parametrize('name',['kan','transformer','mamba'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_mixed_precision_and_nonuniform_queries(name,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model=build_baseline(name,num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4).to(device)
    x,p=inputs(device)
    x=torch.stack((x[:,:,0,:],x[:,:,1,:].square()),2)
    with torch.autocast(device,dtype=torch.bfloat16):out=model(x,p)
    out.float().square().mean().backward()
    assert out.shape==(2,3,32) and torch.isfinite(out).all()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


def source_nodes(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if getattr(n,'name',None) in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def test_kan_direct_efficient_kan_source_parity():
    path=ROOT/'scratch/probes/operatorfree_sources/efficient-kan/kan.py'
    ref=source_nodes(path,{'KANLinear','KAN'},dict(torch=torch,F=F,math=math))['KAN']([9,8,8,3])
    actual=build_baseline('kan',num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4)
    ref.load_state_dict(actual.net.state_dict(),strict=True)
    x,p=inputs();features=torch.cat((x[:,:,0,0][:,None,:].expand(-1,32,-1),p[:,None,:].expand(-1,32,-1),x[:,:1,1,:].transpose(1,2)),dim=-1)
    torch.testing.assert_close(actual(x,p),ref(features).transpose(1,2))


def rearrange(t,pattern,**kwargs):
    if pattern=='b l d -> d (b l)':return t.reshape(-1,t.shape[-1]).t()
    if pattern=='d (b l) -> b d l':return t.reshape(t.shape[0],-1,kwargs['l']).permute(1,0,2)
    if pattern=='d -> d 1':return t[:,None]
    if pattern=='b d l -> (b l) d':return t.transpose(1,2).reshape(-1,t.shape[1])
    if pattern=='(b l) dstate -> b dstate l':return t.reshape(-1,kwargs['l'],t.shape[-1]).transpose(1,2)
    if pattern=='b d l -> b l d':return t.transpose(1,2)
    raise AssertionError(pattern)


def repeat(t,pattern,**kwargs):
    assert pattern=='n -> d n'
    return t[None,:].expand(kwargs['d'],-1)


def test_full_mamba_official_unfused_source_forward_and_parameter_gradients():
    folder=ROOT/'scratch/probes/operatorfree_sources/mamba'
    ns=dict(torch=torch,nn=nn,F=F,math=math,rearrange=rearrange,repeat=repeat,
            causal_conv1d_fn=None,selective_state_update=None)
    source_nodes(folder/'selective_scan_interface.py',{'selective_scan_ref'},ns)
    ns['selective_scan_fn']=ns['selective_scan_ref']
    source_nodes(folder/'mamba_simple.py',{'Mamba'},ns)
    actual=Mamba(d_model=8,d_state=4,d_conv=4,expand=2)
    ref=ns['Mamba'](d_model=8,d_state=4,d_conv=4,expand=2,use_fast_path=False)
    ref.load_state_dict(actual.state_dict(),strict=True)
    x=torch.randn(2,19,8,requires_grad=True)
    a,b=actual(x),ref(x)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    ga=torch.autograd.grad(a.square().sum(),(x,*actual.parameters()))
    gb=torch.autograd.grad(b.square().sum(),(x,*ref.parameters()))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=3e-4,atol=3e-6)


def test_mamba_input_dependent_scan_and_causal_prefix():
    core=Mamba(8,d_state=4)
    x=torch.randn(2,17,8)
    a=core(x);b=core(x[:,:9])
    torch.testing.assert_close(a[:,:9],b)
    projected=[]
    hook=core.x_proj.register_forward_hook(lambda _m,_i,o:projected.append(o.detach()))
    core(x);core(x+1);hook.remove()
    assert not torch.equal(projected[0],projected[1])


def test_scan_matches_independent_dense_green_function_and_gradients():
    u=torch.randn(2,3,11,requires_grad=True);dt=torch.randn_like(u,requires_grad=True)
    A=-torch.rand(3,4);B=torch.randn(2,4,11);C=torch.randn(2,4,11)
    step=F.softplus(dt)
    exponent=torch.einsum('bdt,dn->bdtn',step,A).cumsum(2)
    decay=torch.exp(exponent[:,:,:,None,:]-exponent[:,:,None,:,:])
    decay=decay*torch.ones(11,11).tril()[None,None,:,:,None]
    state=torch.einsum('bdtsn,bds,bns,bds->bdtn',decay,step,B,u)
    expected=torch.einsum('bdtn,bnt->bdt',state,C)
    actual=selective_scan_ref(u,dt,A,B,C,delta_softplus=True)
    torch.testing.assert_close(actual,expected,atol=3e-6,rtol=2e-5)
    ga=torch.autograd.grad(actual.square().sum(),(u,dt),retain_graph=True)
    gb=torch.autograd.grad(expected.square().sum(),(u,dt))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)


def test_transformer_direct_tau_bno_source_parity():
    import typing
    import numpy as np
    source=ROOT.parent/'baselines/Tau_BNO-main/models/OperatorFree'
    ns=dict(torch=torch,nn=nn,np=np,**{name:getattr(typing,name) for name in ('List','Optional','Union','Sequence','Callable','Dict','Literal')})
    source_nodes(source/'layers.py',{'RMSNorm','get_layer_norm'},ns)
    source_nodes(source/'NN.py',{'get_activation','MLP'},ns)
    source_nodes(source/'TransformerBackbone.py',{'get_emb','PositionalEncoding','TransformerBlock','TransformerPerTokenRegressor'},ns)
    ref=ns['TransformerPerTokenRegressor'](input_dim=9,d_model=8,nhead=4,num_layers=2,dim_feedforward=16,dropout=0.,num_outputs_per_token=3)
    actual=build_baseline('transformer',num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4)
    ref.load_state_dict(actual.net.state_dict(),strict=True)
    x,p=inputs()
    features=torch.cat((x[:,:,0,0][:,None,:].expand(-1,32,-1),p[:,None,:].expand(-1,32,-1),x[:,:1,1,:].transpose(1,2)),dim=-1)
    a,b=actual(x,p),ref(features).transpose(1,2)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(torch.autograd.grad(a.square().mean(),p,retain_graph=True)[0],torch.autograd.grad(b.square().mean(),p)[0])
