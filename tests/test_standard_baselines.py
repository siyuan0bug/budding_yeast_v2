"""Selected baseline contracts, official-core parity, and SC-FNO supervision."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from budding_yeast_v2.models.baselines import BASELINE_REGISTRY, build_baseline
from budding_yeast_v2.models.yeast_lit_module import MODEL_REGISTRY, YeastLitModule
from budding_yeast_v2.models.baselines._upstream.ssd import scan
from budding_yeast_v2.utils.training.losses import SensitivityLoss
from budding_yeast_v2.data.sensitivity import attach_sensitivities, preprocessing_fingerprint


@pytest.fixture(autouse=True)
def cpu_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        yield
    torch.set_num_threads(threads)


def inputs(length=32):
    initial = torch.randn(2,3,requires_grad=True)
    times = torch.linspace(0,1,length).expand(2,3,-1)
    x = torch.stack((initial.unsqueeze(-1).expand_as(times),times),dim=2)
    return x,torch.randn(2,5,requires_grad=True),initial


@pytest.mark.parametrize('name',list(BASELINE_REGISTRY))
def test_standard_forward_backward_all_inputs_and_checkpoint(name):
    model = build_baseline(name,num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4,seq_len=32)
    x,p,y0 = inputs()
    out = model(x,p)
    assert out.shape == (2,3,32) and torch.isfinite(out).all()
    loss = out[:,0].square().mean()
    gi,gp = torch.autograd.grad(loss,(y0,p),retain_graph=True)
    assert torch.all(gi.abs().sum(0)>0),name
    assert torch.all(gp.abs().sum(0)>0),name
    loss.backward()
    assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
    clone = copy.deepcopy(model)
    clone.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(x,p),clone(x,p))


def test_selected_directory_and_registry_have_no_modified_baseline_aliases():
    expected={'unet','mlp','resnet','kan','transformer','mamba','deeponet','fno','wno','mwt','gnot','transolver','lno',
              'neural_ode','sc_fno','bregman_fno','lamo','konode'}
    assert set(BASELINE_REGISTRY)==expected
    for old in ('pure_fno','pure_wno','fno1d','fno_ref','wno_ref','mwt_ref','resnet_film',
                'gcgnet','noble','ncflow','icon','itransformer','dlinear','bregman_wno'):
        assert old not in MODEL_REGISTRY
        with pytest.raises(ValueError,match='removed'):
            YeastLitModule(model_name=old,loss_type='mse_only')
    assert 'cross_wno' in MODEL_REGISTRY and 'cross_lno' in MODEL_REGISTRY
    for cls in BASELINE_REGISTRY.values():
        model = build_baseline(next(k for k,v in BASELINE_REGISTRY.items() if v is cls),
                               num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4,seq_len=32)
        assert all('film' not in name.lower() for name,_ in model.named_modules())


def dense_scan(x,dt,A,B,C,initial_states=None,**kwargs):
    b,l,h,p=x.shape
    B=B.repeat_interleave(h//B.shape[2],2)
    C=C.repeat_interleave(h//C.shape[2],2)
    cumulative=(dt*A).cumsum(1)
    decay=torch.exp(cumulative[:,:,None,:]-cumulative[:,None,:,:])
    decay=decay*torch.ones(l,l,device=x.device).tril()[None,:,:,None]
    coefficients=torch.einsum('bthn,bshn->btsh',C,B)*decay*dt[:,None,:,:]
    out=torch.einsum('btsh,bshp->bthp',coefficients,x)
    if initial_states is not None:
        out=out+torch.einsum('bhpn,bthn->bthp',initial_states,C)*cumulative.exp().unsqueeze(-1)
    return out


def test_hydra_ssd_matches_dense_state_space_equations():
    x=torch.randn(2,7,4,3,requires_grad=True)
    dt=torch.rand(2,7,4)*.1
    A=-torch.rand(4)
    B,C=torch.randn(2,7,2,5),torch.randn(2,7,2,5)
    init=torch.randn(2,4,3,5)
    actual=scan(x,dt,A,B,C,initial_states=init)
    expected=dense_scan(x,dt,A,B,C,initial_states=init)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0],
                               torch.autograd.grad(expected.square().sum(),x)[0],atol=1e-5,rtol=2e-5)


def load_source_classes(path,names,namespace):
    if not path.exists():
        pytest.skip('Upstream checkout unavailable')
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def rearrange_output(x,pattern):
    assert pattern=='b h n d -> b n (h d)'
    return x.transpose(1,2).flatten(2)


def test_transolver_attention_matches_actual_official_source():
    from budding_yeast_v2.models.baselines._upstream.transolver import Physics_Attention_Irregular_Mesh
    root=Path(__file__).resolve().parents[2]
    ns=load_source_classes(root/'baselines/transolver/Physics_Attention.py',
                           {'Physics_Attention_Irregular_Mesh'},dict(torch=torch,nn=nn,rearrange=rearrange_output))
    original=ns['Physics_Attention_Irregular_Mesh'](8,heads=2,dim_head=4,slice_num=3)
    adapted=Physics_Attention_Irregular_Mesh(8,heads=2,dim_head=4,slice_num=3)
    adapted.load_state_dict(original.state_dict())
    x=torch.randn(2,13,8)
    torch.testing.assert_close(original(x),adapted(x))


def test_gnot_expert_attention_matches_actual_official_source():
    from budding_yeast_v2.models.baselines._upstream.gnot import MoEGPTConfig,MIOECrossAttentionBlock
    root=Path(__file__).resolve().parents[2]
    ns=load_source_classes(root/'baselines/GNOT/models/mmgpt.py',
        {'LinearAttention','LinearCrossAttention','MIOECrossAttentionBlock'},
        dict(torch=torch,nn=nn,F=F,rearrange=rearrange_output,MultipleTensors=list,
             GELU=nn.GELU,ReLU=nn.ReLU,Tanh=nn.Tanh,Sigmoid=nn.Sigmoid))
    config=MoEGPTConfig(n_embd=8,n_head=2,n_inputs=2,space_dim=1,n_inner=2)
    original=ns['MIOECrossAttentionBlock'](config)
    adapted=MIOECrossAttentionBlock(config)
    adapted.load_state_dict(original.state_dict())
    x=torch.randn(2,13,8); sources=[torch.randn(2,3,8),torch.randn(2,1,8)];pos=torch.rand(2,13,1)
    torch.testing.assert_close(original(x,sources,pos),adapted(x,sources,pos))


def test_sc_fno_architecture_is_exactly_fno():
    args=dict(num_vars=3,param_dim=5,width=8,n_blocks=2,modes=4)
    fno=BASELINE_REGISTRY['fno'](**args)
    sc=BASELINE_REGISTRY['sc_fno'](**args)
    sc.load_state_dict(fno.state_dict(),strict=True)
    x,p,_=inputs()
    torch.testing.assert_close(fno(x,p),sc(x,p))


def test_sc_sensitivity_loss_has_correct_jacobian_and_parameter_gradient():
    class Analytic(nn.Module):
        def __init__(self):
            super().__init__(); self.scale=nn.Parameter(torch.tensor(2.))
        def forward(self,x,p):
            t=x[:,:,1,:]
            return self.scale*p[:,0,None,None]*t + p[:,1,None,None]
    model=Analytic();x,p,_=inputs(8)
    targets=torch.stack((3*x[:,:,1,:],torch.ones_like(x[:,:,1,:])),dim=-1)
    criterion=SensitivityLoss()
    loss=criterion.sensitivity_term(model,x,p,targets,torch.tensor([0,1]))
    expected=(model.scale-3).square()*x[:,:,1,:].square().mean()/2
    torch.testing.assert_close(loss,expected)
    torch.testing.assert_close(torch.autograd.grad(loss,model.scale)[0],
                               torch.autograd.grad(expected,model.scale)[0])


def test_sc_training_rejects_missing_targets_and_trains_with_explicit_targets():
    model=YeastLitModule(model_name='sc_fno',num_vars=3,param_dim=5,width=8,n_blocks=2,
                          modes=4,seq_len=8,loss_type='sc_sensitivity')
    model.log=lambda *a,**k: None
    x,p,_=inputs(8);y=torch.zeros(2,3,8)
    with pytest.raises(ValueError,match='sensitivity targets'):
        model.training_step((x,p,y),0)
    batch=dict(x=x,p=p,y=y,sensitivity_targets=torch.zeros(2,3,8,2),
               sensitivity_indices=torch.tensor([[0,1],[0,1]]))
    loss=model.training_step(batch,0);loss.backward()
    assert torch.isfinite(loss)
    assert model.model.lift.weight.grad.abs().sum()>0


def test_sensitivity_sidecar_is_bound_to_data_and_normalization(tmp_path):
    x,p,_=inputs(8); y=torch.zeros(2,3,8)
    dm=SimpleNamespace(norm_type='linear',mean_y=torch.zeros(1,3,1),std_y=torch.ones(1,3,1),
        p_mean=torch.zeros(1,5),p_std=torch.ones(1,5),s_c=None,num_params=5,
        _raw_data=np.zeros((2,3,8)),_train_idx_synth=[1,0],
        _dataset_contract={'data_sha256':'data','params_sha256':'params'})
    dm.train_dataset_synth=TensorDataset(x,p,y,torch.zeros(2,dtype=torch.long))
    path=tmp_path/'sens.npz'
    values=dict(dy_dp=np.ones((2,3,8,2)),parameter_indices=np.array([0,2]),
        data_sha256='data',params_sha256='params',preprocessing_sha256=preprocessing_fingerprint(dm),
        initial_conditions_held_fixed=True)
    np.savez(path,**values);attach_sensitivities(dm,path)
    assert dm.train_dataset_synth[0]['sensitivity_targets'].shape==(3,8,2)
    values['preprocessing_sha256']='different'
    np.savez(path,**values)
    with pytest.raises(ValueError,match='normalization'):
        attach_sensitivities(dm,path)


def test_konode_field_and_koopman_dynamics_match_official_equations():
    from budding_yeast_v2.models.baselines.konode import KoNODE
    root=Path(__file__).resolve().parents[2]
    ns=load_source_classes(root/'baselines/KoNODE/main.py',{'ODEFunc'},dict(torch=torch,nn=nn,F=F))
    original=ns['ODEFunc'](w_size=4,num_layers=4,hidden_dim=8,input_dim=8,output_dim=3)
    adapted=KoNODE(num_vars=3,param_dim=5,width=8,n_blocks=4,koopman_dim=4)
    adapted.encoder.load_state_dict(original.encoder.state_dict())
    for head,name in zip(adapted.weight_heads,('encoder_0','encoder_1','encoder_2','encoder_5')):
        head.load_state_dict(getattr(original,name).state_dict())
    y,p,w=torch.randn(2,3),torch.randn(2,5),torch.randn(2,4)
    a=torch.randn(2,1,4)
    dy,dw,_=original(0.,(torch.cat((y,p),-1).unsqueeze(1),w.unsqueeze(1),a))
    torch.testing.assert_close(adapted.vector_field(y,w,p),dy.squeeze(1))
    with torch.no_grad():adapted.spectrum.copy_(a[0,0])
    torch.testing.assert_close(adapted.mode_derivative(w),original.update_w(w.unsqueeze(1),a[:1].expand(2,-1,-1)).squeeze(1))


@pytest.mark.parametrize('name',list(BASELINE_REGISTRY))
def test_actual_budding_500_query_contract(name):
    model=YeastLitModule(model_name=name,width=8,n_blocks=2,modes=4,loss_type='mse_only',seq_len=500)
    x=torch.zeros(1,39,2,500);x[:,:,0,:]=.1;x[:,:,1,:]=torch.linspace(0,1,500)
    with torch.no_grad():out=model(x,torch.ones(1,141)*.1)
    assert out.shape==(1,39,500) and torch.isfinite(out).all()


def test_lno_phca_matches_actual_official_source():
    from budding_yeast_v2.models.baselines._upstream.lno import LNO as Adapted
    root=Path(__file__).resolve().parents[2]
    ns=load_source_classes(root/'baselines/LatentNeuralOperator/LNO-PyTorch/ForwardProblem/module/model.py',
                           {'Attention_Vanilla','MLP','LNO'},dict(torch=torch,math=math))
    ns['ACTIVATION']={'GELU':nn.GELU()};ns['ATTENTION']={'Attention_Vanilla':ns['Attention_Vanilla']}
    args=(2,4,8,2,1,1,9,3,'Attention_Vanilla','GELU',{'time':False})
    original=ns['LNO'](*args);adapted=Adapted(*args)
    adapted.load_state_dict(original.state_dict())
    time=torch.rand(2,13,1);values=torch.randn(2,13,9)
    torch.testing.assert_close(original(time,values),adapted(time,values))


def test_hydra_unfused_path_matches_official_class_with_dense_ssd_backend():
    from budding_yeast_v2.models.baselines._upstream.hydra import Hydra
    from budding_yeast_v2.models.baselines._upstream.ssd import RMSNormGated
    def repeat(x,pattern,**kwargs):
        if pattern=='... -> b ...':return x.unsqueeze(0).expand(kwargs['b'],*x.shape)
        assert pattern=='b l h -> b l (h p)'
        return x.repeat_interleave(kwargs['p'],dim=-1)
    def rearrange(x,pattern,**kwargs):
        if pattern=='b l h p -> b l (h p)':return x.flatten(2)
        if pattern=='b l (h p) -> b l h p':return x.reshape(*x.shape[:2],-1,kwargs['p'])
        assert pattern=='b l (g n) -> b l g n'
        return x.reshape(*x.shape[:2],kwargs['g'],-1)
    root=Path(__file__).resolve().parents[2]
    ns=load_source_classes(root/'baselines/LaMO/model/mamba_2D_v2.py',{'Hydra'},
        dict(torch=torch,nn=nn,F=F,math=math,repeat=repeat,rearrange=rearrange,
             RMSNormGated=RMSNormGated,mamba_chunk_scan_combined=dense_scan))
    args=dict(d_model=8,d_state=4,d_conv=3,expand=1,headdim=2,ngroups=2,use_mem_eff_path=False)
    original=ns['Hydra'](**args);adapted=Hydra(**args)
    adapted.load_state_dict(original.state_dict())
    x=torch.randn(2,7,8)
    torch.testing.assert_close(original(x),adapted(x),atol=2e-6,rtol=2e-5)
