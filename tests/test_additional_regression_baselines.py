"""Paper equations, exact regression derivatives and executable CPU workflows."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.sampling.baseline.regression import relative_head_gradients
from budding_yeast_v2.utils.sampling.baseline.low_rank import sensitivity_scores
from budding_yeast_v2.utils.sampling.baseline.peaks import PEAKSStream
from budding_yeast_v2.utils.sampling.baseline.eracs import gradient_snr,weighted_omp
from budding_yeast_v2.utils.sampling.baseline.hggs import gradient_degree,MultigridGenetic,HGGSConfig
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
from budding_yeast_v2.utils.sampling.dynamic.runner import run_dynamic
from budding_yeast_v2.utils.sampling.generation.runner import run_generation
from test_sampling_benchmark import make_pool,config


@pytest.fixture(autouse=True)
def isolated_cpu_resource_slots(tmp_path,monkeypatch):
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')


def test_exact_output_head_derivative_matches_actual_uncapped_relative_loss():
    from budding_yeast_v2.utils.training.losses import RelativeL2Loss
    torch.manual_seed(42)
    hidden=torch.randn(3,4,7,dtype=torch.float64)
    weight=torch.randn(2,4,dtype=torch.float64,requires_grad=True)
    bias=torch.randn(2,dtype=torch.float64,requires_grad=True)
    prediction=torch.einsum('ch,bht->bct',weight,hidden)+bias[None,:,None]
    target=torch.randn(3,2,7,dtype=torch.float64);target[0]=0
    actual=[]
    for i in range(3):
        loss=RelativeL2Loss(eps=.03)(prediction[i:i+1],target[i:i+1])
        a,b=torch.autograd.grad(loss,(weight,bias),retain_graph=True)
        actual.append(torch.cat((a.flatten(),b)))
    expected=relative_head_gradients(prediction,target,hidden,eps=.03)
    torch.testing.assert_close(expected,torch.stack(actual),rtol=1e-6,atol=1e-8)
    # Kernel influence equals actual first-order change of reference loss.
    g=expected.detach().numpy();score=g[1]@g[2]
    eta=1e-7
    before=RelativeL2Loss(eps=.03)(prediction[1:2],target[1:2]).item()
    with torch.no_grad():
        dw=g[2,:8].reshape(2,4);db=g[2,8:]
        after_pred=torch.einsum('ch,bht->bct',weight-eta*torch.tensor(dw),hidden[1:2])+ (bias-eta*torch.tensor(db))[None,:,None]
        after=RelativeL2Loss(eps=.03)(after_pred,target[1:2]).item()
    assert (before-after)/eta==pytest.approx(score,rel=1e-5)


def test_low_rank_formula_has_defined_basis_loss_and_native_weighted_estimator():
    x=np.array([[3.,0.,0.],[0.,2.,0.],[0.,0.,1.],[1.,1.,1.]])
    scores,details=sensitivity_scores(x,rank=2,gamma=5,lam=1,xi=.01)
    _,_,basis=np.linalg.svd(x,full_matrices=False);coefficient=x@basis[:2].T
    residual=np.sum((x-coefficient@basis[:2])**2,axis=1)
    np.testing.assert_allclose(scores,6*np.sum(coefficient**2,1)+5*2*.01+residual)
    assert details['basis_queries_to_ode']==0
    p=scores/scores.sum();loss=np.arange(1,5.)
    assert np.sum(p*loss/(len(loss)*p))==pytest.approx(loss.mean())


def test_peaks_percentile_cache_refresh_and_signed_kernel_reference():
    from scipy.stats import percentileofscore
    stream=PEAKSStream(25,refresh=2,cache_capacity=3);stream.set_reference([[1.,0.]],0)
    history=[]
    for score in (2.,2.,-1.,5.,0.):
        history=(history+[score])[-3:]
        accepted,report=stream.decide([score,0.])
        assert report['percentile']==pytest.approx(percentileofscore(history,score))
        assert accepted==(report['percentile']>=75)
    other=PEAKSStream();other.load_state_dict(deepcopy(stream.state_dict()))
    assert other.decide([3.,0.])==stream.decide([3.,0.])
    stream.set_reference([[0.,2.]],2);assert not stream.cache and stream.reference_step==2


def test_eracs_spectral_covariance_and_weighted_omp_gradient_objective():
    g=np.array([[1.,0.],[3.,0.],[0.,2.],[0.,4.],[2.,2.],[1.,1.]])
    covariance=np.cov(g,rowvar=False)
    assert gradient_snr(g)==pytest.approx(np.linalg.norm(g.mean(0))/np.sqrt(np.linalg.eigvalsh(covariance)[-1]))
    indices,weights,report=weighted_omp(g,4,group_size=2,seed=42)
    assert len(indices)==len(set(indices))==4 and (weights>=0).all()
    actual=np.linalg.norm((g[indices]*weights[:,None]).mean(0)-g.mean(0))
    assert actual==pytest.approx(report['gradient_residual']) and actual<1e-3
    zero_indices,zero_weights,zero=weighted_omp(np.zeros((6,2)),4,group_size=2)
    assert len(zero_indices)==4 and not zero_weights.any() and zero['zero_gradient_estimator']


def test_hggs_gf_equation_and_multigrid_boxes_restore_rng():
    x=np.arange(6.)[:,None];y=x*x
    expected=[]
    for i in range(6):
        ids=np.argsort(np.abs(x[:,0]-x[i,0]));ids=ids[ids!=i][:2]
        expected.append(np.mean(((y[ids]-y[i])**2/(x[ids]-x[i])**2)))
    # Use all neighbors to avoid any nearest-neighbor tie ordering ambiguity.
    all_score=gradient_degree(x,y,neighbors=5)
    for i in range(6):
        ids=np.setdiff1d(np.arange(6),i)
        assert all_score[i]==pytest.approx(np.mean(((y[ids]-y[i])**2/(x[ids]-x[i])**2)))
    sampler=MultigridGenetic(42,HGGSConfig(mixture=False));sampler.refresh(np.c_[x,y],np.arange(6.))
    state=deepcopy(sampler.state_dict());targets,report=sampler.propose(8)
    for target,(a,b) in zip(targets,report['parent_indices']):
        assert np.all(target>=np.minimum(sampler.inputs[a],sampler.inputs[b]))
        assert np.all(target<=np.maximum(sampler.inputs[a],sampler.inputs[b]))
    other=MultigridGenetic(2,HGGSConfig(mixture=False));other.load_state_dict(state)
    np.testing.assert_array_equal(targets,other.propose(8)[0])
    assert report['crossover_count']==report['mutation_count']==4


def protocol(tmp_path,initial=4,target=8):
    pool,_=make_pool(tmp_path/'pool',count=32)
    cfg=replace(config(),budgets=(initial,target),epochs=(1,2),warmup=(0,0),validation_metric='normalized_loss',loss='rel_l2')
    root=prepare_protocol({'mutation':pool,'lhs':pool},tmp_path/'protocol',cfg)
    return root


def test_low_rank_full_weighted_pilot_selection_training_eval_and_cache(tmp_path):
    root=protocol(tmp_path)
    output=run_experiment(root,'mutation','low_rank',tmp_path/'runs',diagnostic_only=True)
    state=json.loads((output/'_state/state.json').read_text())
    assert state['completion']['status']=='complete' and state['completion']['unique_labels_accessed']>8
    assert state['completion']['pilot_training_inputs']==4
    assert state['completion']['historical_training_inputs']>=state['completion']['unique_training_inputs']
    selection=json.loads((output/'_state/one_shot.json').read_text())
    assert selection['diagnostics']['basis_queries_to_ode']==0 and selection['diagnostics']['draw_slots']==8
    assert (output/'eval/global_metrics_denorm.json').exists()
    assert not list(output.rglob('*.npz'))
    assert run_experiment(root,'mutation','low_rank',tmp_path/'runs',diagnostic_only=True)==output


@pytest.mark.parametrize('method',['peaks_regression','eracs_regression','random_stream','random_dynamic'])
@pytest.mark.parametrize('pool',['mutation','lhs'])
def test_dynamic_complete_chain_and_resume_after_eval_interruption(tmp_path,method,pool):
    root=protocol(tmp_path);options=dict(epochs=3,warmup=0,bootstrap_epochs=1,acquisition_epochs=2,steps_per_epoch=2,
        delta=2,stream_batch=4,reference_rows=2,sketch_dim=4,check_every=1,snr_threshold=1e9,omp_group_size=2)
    from budding_yeast_v2.utils.sampling.generation.cohorts import FrozenCohorts
    with patch.object(FrozenCohorts,'evaluate',side_effect=RuntimeError('injected evaluation interruption')):
        with pytest.raises(RuntimeError,match='injected evaluation'):
            run_dynamic(root,pool,method,tmp_path/'runs',dynamic_config=options,diagnostic_only=True)
    output=run_dynamic(root,pool,method,tmp_path/'runs',dynamic_config=options,diagnostic_only=True)
    report=json.loads((output/'_state/completion.json').read_text())
    assert report['resident_training_inputs']==8 and report['optimizer_steps']==6
    assert report['training_draws']==24 and report['unique_labels_accessed']>=8
    assert report['best_checkpoint_exposure'] is not None and report['events']
    if method in ('peaks_regression','random_stream'):
        assert report['events'][0]['step']>=options['bootstrap_epochs']*options['steps_per_epoch']
    definition=json.loads((root/'protocol.json').read_text())
    with np.load(root/'mutation_split.npz') as z:holdout=set(np.r_[z['validation'],z['test']])
    access=json.loads((output/'_state/label_access.json').read_text())
    assert not holdout&set(access['revealed_ids'])
    assert (output/'eval/per_synth_test_metrics.csv').exists() and not list(output.rglob('*.npz'))
    assert run_dynamic(root,pool,method,tmp_path/'runs',dynamic_config=options,diagnostic_only=True)==output


def test_hggs_actual_gf_pilot_mgs_canonical_ode_and_resume(tmp_path):
    from budding_yeast_v2.utils.sampling.generation.cohorts import FrozenCohorts
    pool,_=make_pool(tmp_path/'pool',count=32)
    cfg=replace(config(),budgets=(4,5,6),epochs=(1,1,1),warmup=(0,0,0),
                validation_metric='normalized_loss',loss='rel_l2')
    root=prepare_protocol({'mutation':pool,'lhs':pool},tmp_path/'protocol',cfg)
    options=dict(coarse_multiplier=2,neighbors=2,output_sketch=4,mixture=False)
    original=FrozenCohorts.evaluate
    def interrupted(cohorts,predictor,directory,round_index,*args,**kwargs):
        if round_index==1:raise RuntimeError('injected HGGS round-one evaluation interruption')
        return original(cohorts,predictor,directory,round_index,*args,**kwargs)
    with patch.object(FrozenCohorts,'evaluate',interrupted):
        with pytest.raises(RuntimeError,match='HGGS round-one'):
            run_generation(root,'mutation','hggs_projected',tmp_path/'runs',workers=1,bank_size=4,
                query_batch_size=1,hggs_config=options,diagnostic_only=True)
    output=run_generation(root,'mutation','hggs_projected',tmp_path/'runs',workers=1,bank_size=4,
        query_batch_size=1,hggs_config=options,diagnostic_only=True)
    report=json.loads((output/'_state/completion.json').read_text())
    assert report['accepted']==6 and report['newly_accepted']==2
    assert report['hggs_gf']['coarse_labels']==8 and report['hggs_gf']['pilot_training'] is not None
    gf=json.loads((output/'_state/hggs_gf.json').read_text())
    assert len(gf['boundary_ids'])==len(gf['global_residual_ids'])==2
    with np.load(root/'mutation_split.npz') as z:holdout=set(np.r_[z['validation'],z['test']])
    assert not holdout&set(gf['coarse_source_ids'])
    assert (output/'eval/per_synth_test_metrics.csv').exists()
    assert run_generation(root,'mutation','hggs_projected',tmp_path/'runs',workers=1,bank_size=4,
        query_batch_size=1,hggs_config=options,diagnostic_only=True)==output


@pytest.mark.parametrize('method',['eracs_regression','peaks_regression'])
def test_dynamic_periodic_training_resume_restores_resident_rng_and_matching_best(tmp_path,method):
    import pytorch_lightning as pl
    from budding_yeast_v2.utils.training import runner as training_runner
    root=protocol(tmp_path);options=dict(epochs=8,warmup=0,bootstrap_epochs=1,acquisition_epochs=2,steps_per_epoch=1,
        delta=2,stream_batch=4,reference_rows=2,sketch_dim=4,check_every=1,snr_threshold=1e9,omp_group_size=2)
    original=training_runner.make_trainer
    class Interrupt(pl.Callback):
        def on_train_epoch_start(self,trainer,pl_module):
            if trainer.current_epoch==6:
                # A future best must not survive rollback to periodic epoch5.
                model=list((tmp_path/'runs').rglob('stream/model.ckpt'))[0]
                model.write_bytes(b'uncommitted future checkpoint')
                raise RuntimeError('injected training interruption')
    def factory(**kwargs):
        kwargs['callbacks'].append(Interrupt());return original(**kwargs)
    with patch.object(training_runner,'make_trainer',side_effect=factory):
        with pytest.raises(RuntimeError,match='injected training'):
            run_dynamic(root,'mutation',method,tmp_path/'runs',dynamic_config=options,diagnostic_only=True)
    output=run_dynamic(root,'mutation',method,tmp_path/'runs',dynamic_config=options,diagnostic_only=True)
    report=json.loads((output/'_state/completion.json').read_text())
    assert report['optimizer_steps']==8 and report['training_draws']==32
    assert report['resident_training_inputs']==8 and report['best_checkpoint_exposure'] is not None
    assert (output/'stream/model.ckpt').stat().st_size>1000
