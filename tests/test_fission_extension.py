"""System-specific Fission physics, graphs, checkpoint and evaluation contracts."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import pytorch_lightning as pl

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from budding_yeast_v2.data.systems.fission import ode,simulation
from budding_yeast_v2.data.systems.fission.parameters import WT_parameters,param_vector
from budding_yeast_v2.data.systems.fission.constants import MASS_UNIT
from budding_yeast_v2.data.systems.fission.real_mutants import real_tasks
from budding_yeast_v2.data.yeast_graph import build_yeast_graph
from budding_yeast_v2.data.yeast_relations import build_yeast_relations
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule,FISSION_MODELS
from budding_yeast_v2.utils import fission_physics as physics

torch.set_num_threads(1)


@pytest.mark.parametrize('name,overrides',real_tasks())
def test_torch_rhs_matches_all_canonical_real_parameter_sets(name,overrides):
    rng=np.random.RandomState(17)
    states=rng.uniform(.01,.95,(6,16));states[:,0]+=1
    states[::2,1]=1.0
    for phase in ('kSmin','kSmax'):
        p={**WT_parameters(),**overrides};p['kS']=p[phase]
        expected=np.array([ode.eqns(0.,s,p) for s in states])
        tensor=torch.tensor(states,requires_grad=True)
        actual=physics.eqns_torch(tensor,p,exported=False)
        np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=1e-12,atol=1e-12)
        actual.square().mean().backward()
        assert torch.isfinite(tensor.grad).all()
        exported=states.copy();exported[:,0]/=MASS_UNIT
        expected[:,0]/=MASS_UNIT
        np.testing.assert_allclose(physics.eqns_torch(torch.tensor(exported),p).numpy(),expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('event',['START','SFINISH','DIV'])
def test_event_reset_matches_canonical_solver_branch_and_is_differentiable(event):
    tree=ast.parse(inspect.getsource(simulation.run_event_sim))
    dispatch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="name == 'DIV'")
    code=compile(ast.fix_missing_locations(ast.Module(body=[dispatch],type_ignores=[])),'canonical reset','exec')
    states=np.random.RandomState(3).uniform(.1,.8,(10,16));states[:,0]+=1
    states[:5,1]=1.
    states[::2,15]=.1
    states[::2,2]=.65*.9-.05*states[::2,3]
    expected=[]
    for state in states:
        p=WT_parameters();p['kS']=p['kSmin']
        local=dict(vars(simulation),ye=state.copy(),p=p,name=event,te=1.,
                   rdna_at_div=[],birth_masses=[],div_times=[],div_masses=[],post_div_states=[],start_times=[],sfinish_times=[])
        exec(code,local)
        expected.append(local['ye'])
    value=torch.tensor(states,requires_grad=True)
    actual,_=physics.apply_event_torch(value,WT_parameters(),event)
    np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=1e-12,atol=1e-12)
    actual.sum().backward();assert torch.isfinite(value.grad).all()


def test_latent_phase_and_event_masks_do_not_use_budding_indices():
    states=torch.ones(1,16,15)*.1;states[:,1,:7]=1.;states[:,1,7:]=.2
    p=physics.params_tensor_to_dict(torch.tensor(param_vector(WT_parameters()))[None,:])
    phases=physics.phase_history(states,p)
    assert torch.all(phases[:,:7]==p['kSmin']) and torch.all(phases[:,7:]==p['kSmax'])
    mask=physics.detect_event_mask(states,event_window=1,parameters=p)
    assert not mask[0,5:9].any() and mask[0,:4].all()


def test_cut_strain_initial_dna_does_not_falsely_imply_high_ks():
    states=torch.ones(1,16,8)*.1
    states[:,1,:]=torch.tensor([.2,.4,.7,1.,1.,.01,.3,1.])
    p=physics.params_tensor_to_dict(torch.tensor(param_vector(WT_parameters()))[None,:])
    switches=physics.phase_history(states,p)
    np.testing.assert_allclose(switches.numpy()[0],[.1,.1,.1,.1,.1,5.,5.,.1])
    with pytest.raises(ValueError,match='latent kS'):
        physics.eqns_torch(states[:,:,0],p)


def test_independent_fission_graph_contains_numerical_dependencies_and_latent_switches():
    graph=build_yeast_graph(system_id='fission')
    assert graph.summary()['nonself_edges']['hybrid']==75
    assert graph.variable_names[1:4]==('RDNA','MPF','preMPF')
    states=torch.tensor(np.random.RandomState(12).uniform(.1,.7,(16,)),requires_grad=True)
    jac=torch.autograd.functional.jacobian(lambda x:physics.eqns_torch(x,WT_parameters(),exported=False),states)
    support=graph.adjacency('continuous')
    assert not (jac.detach().abs().numpy()>1e-9)[~support].any()
    # kS changes Mik1/Cdc25 dynamics at START and SFINISH, separately from resets.
    assert {tuple(row['affected_derivatives']) for row in graph.summary()['latent_parameter_events']}=={('Mik1','Cdc25')}
    rel,untyped=build_yeast_relations(system_id='fission').matrices('hybrid',self_loops='none')
    np.testing.assert_array_equal(rel.any(0)|untyped,graph.adjacency('hybrid',self_loops='none'))
    assert build_yeast_graph().summary()['nonself_edges']['hybrid']==226


@pytest.mark.parametrize('loss',['pinn_residual','pino','hybrid_pino_colloc','hybrid_all'])
@pytest.mark.parametrize('mode',['linear','log1p','asinh_z','asinh_scale'])
def test_active_fission_physics_loss_backward(loss,mode):
    model=YeastLitModule(model_name='fno',num_vars=16,param_dim=58,width=8,n_blocks=1,modes=4,
                         loss_type=loss,norm_type=mode)
    criterion=model.criterion
    criterion.set_epoch(100)
    if mode=='asinh_scale':
        criterion.set_stats(torch.zeros(1,16,1),torch.ones(1,16,1),torch.zeros(1,58),torch.ones(1,58),300.,norm_type=mode,s_c=torch.ones(1,16,1))
    p=torch.tensor(np.stack([param_vector(WT_parameters())]*2),dtype=torch.float32)
    # Independent normalization gives valid physical parameters in every mode.
    criterion.p_mean.copy_(torch.log10(p[:1].clamp_min(1e-8)) if mode=='asinh_scale' else p[:1])
    target=torch.ones(2,16,21)*.15
    prediction=(target+.01*torch.randn_like(target)).requires_grad_()
    result=criterion(prediction,target,torch.zeros_like(p),torch.ones(2),1.,0.)
    assert torch.isfinite(result)
    result.backward();assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize('model_name',['fno','cross_fno','graph_resnet_film','mlp','resnet','ultimate_fno','deeponet','neural_ode'])
def test_shared_models_use_fission_dimensions_and_graphs(model_name):
    model=YeastLitModule(model_name=model_name,num_vars=16,param_dim=58,width=8,n_blocks=1,modes=4,loss_type='mse_only')
    x=torch.zeros(2,16,2,32)
    x[:,:,1,:]=torch.linspace(0,1,32)
    result=model(x,torch.zeros(2,58))
    assert result.shape==(2,16,32)
    result.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.system_id=='fission'


def test_budding_specific_architecture_cannot_silently_run_fission():
    with pytest.raises(ValueError,match='Budding-specific'):
        YeastLitModule(model_name='hano',num_vars=16,param_dim=58)


def test_actual_fission_dataset_checkpoint_training_and_empty_split_evaluation(tmp_path):
    from budding_yeast_v2.eval import load_model_from_checkpoint,evaluate_model_on_datamodule
    source=ROOT/'scratch/probes/compat_fission_random_8.npz'
    if not source.exists():pytest.skip('Actual generated integration artifact unavailable')
    dm=YeastDataModule(source,num_workers=0,batch_size_synth=4)
    dm.setup('fit')
    # The current seeded split can contain synthetic tests; explicitly exercise
    # the empty-evaluation branch instead of depending on an obsolete fixture split.
    from torch.utils.data import TensorDataset
    dm.test_dataset_synth=TensorDataset(*(x[:0] for x in dm.test_dataset_synth.tensors))
    dm._test_idx_synth=[]
    assert dm.adj_matrix.shape==(16,16) and dm.t_max==300
    assert len(dm.test_dataset_real)==20
    model=YeastLitModule(model_name='cross_fno',num_vars=16,param_dim=58,width=8,n_blocks=1,modes=4,loss_type='pinn_residual',warmup_epochs=0,max_epochs=1)
    trainer=pl.Trainer(accelerator='cpu',devices=1,fast_dev_run=True,logger=False,enable_checkpointing=False,enable_progress_bar=False)
    trainer.fit(model,datamodule=dm)
    path=tmp_path/'fission.ckpt';trainer.save_checkpoint(path)
    loaded,hp=load_model_from_checkpoint(path)
    assert hp['system_id']=='fission'
    batch=next(iter(dm.train_dataloader()))
    model.eval()
    torch.testing.assert_close(model(batch[0],batch[1]),loaded(batch[0],batch[1]))
    report=evaluate_model_on_datamodule(loaded,dm,device='cpu',save_dir=str(tmp_path/'eval'),max_plots=0)
    assert report['system_id']=='fission' and 'core_9_vars' not in report['all_test']
    assert report['real_mutants']['count']==20 and report['synth_test']['count']==0
    assert report['synth_test']['all_variables'] is None
