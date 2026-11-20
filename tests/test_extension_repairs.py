"""Regressions for failures observed in the full seed42 extension queue."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.sampling.baseline.authors import leverage
from budding_yeast_v2.utils.sampling.baseline import physics_pre as pre
from budding_yeast_v2.utils.sampling.contracts import PoolInputs
from budding_yeast_v2.utils.sampling.generation.policies import GenerationPolicy
from budding_yeast_v2.utils.sampling.baseline.hggs import HGGSConfig


def test_hggs_refresh_count_survives_multiple_rounds_and_legacy_checkpoint():
    rng=np.random.default_rng(42)
    inputs=PoolInputs(np.arange(8),rng.normal(size=(8,3)),rng.normal(size=(8,2)))
    truth=rng.normal(size=(8,2,10)).astype(np.float32)
    predictor=SimpleNamespace(normalization=SimpleNamespace(targets=torch.as_tensor),
        batches=lambda _:iter([(0,torch.zeros(4,2,10)),(4,torch.zeros(4,2,10))]))
    policy=GenerationPolicy('hggs_projected',inputs,truth,hggs=HGGSConfig(mixture=False))
    policy.refresh(inputs,truth,predictor,target=10)
    assert policy.refreshed_target==10 and policy.controller.cycles==1
    policy.refresh(inputs,truth,predictor,target=10)
    assert policy.controller.cycles==1
    policy.refresh(inputs,truth,predictor,target=12)
    assert policy.refreshed_target==12 and policy.controller.cycles==2
    restored=GenerationPolicy('hggs_projected',inputs,truth,hggs=HGGSConfig(mixture=False))
    restored.load_state_dict(deepcopy(policy.state_dict()))
    restored.refresh(inputs,truth,predictor,target=14)
    assert restored.refreshed_target==14 and restored.controller.cycles==3
    state=deepcopy(restored.state_dict());state['refreshed_target']=torch.zeros(4,2,10)
    with pytest.warns(RuntimeWarning,match='legacy HGGS'):
        restored.load_state_dict(state)
    assert restored.refreshed_target is None
    restored.refresh(inputs,truth,predictor,target=16)
    assert restored.refreshed_target==16


def test_leverage_rank_deficient_blocks_match_direct_ridge_formula():
    x=np.random.default_rng(42).normal(size=(19,28));x[:,4:]*=300
    x[:,0]=0;x[:,1]=1;x[:,2]=x[:,3]
    selected,details=leverage(x,7,42,block_size=19,step_size=2)
    seed=int(np.random.SeedSequence([42,0]).generate_state(1)[0])
    expected=np.random.default_rng(seed).choice(len(x),2,replace=False).tolist()
    while len(expected)<7:
        covariance=x[expected].T@x[expected]/len(expected)+1e-5*np.eye(x.shape[1])
        lower=np.linalg.cholesky(covariance)
        scores=np.square(np.linalg.solve(lower,x.T)).sum(0)
        remaining=np.setdiff1d(np.arange(len(x)),expected)
        expected.extend(remaining[np.argsort(-scores[remaining],kind='stable')[:min(2,7-len(expected))]])
    np.testing.assert_array_equal(selected,expected)
    assert details['linear_algebra_dtype']=='float64'


@pytest.mark.parametrize('budget',[1,7,18,19])
def test_leverage_unique_exact_budget_including_partial_and_singleton_blocks(budget):
    x=np.ones((19,32))
    ids,_=leverage(x,budget,42,block_size=6,step_size=2)
    assert len(ids)==len(set(ids))==budget and (ids<len(x)).all()
    np.testing.assert_array_equal(ids,leverage(x,budget,42,block_size=6,step_size=2)[0])


def test_pre_adaptive_padding_keeps_jumps_excluded_and_records_ids(monkeypatch):
    monkeypatch.setattr(pre,'rhs',lambda y,p,t:torch.zeros_like(y))
    y=np.zeros((1,39,100));y[:,0]=np.arange(100)/100
    # Jump flags every 10 time steps: ±4 plus the stencil covers the horizon.
    y[:,36]=np.arange(100)//10
    scores,details=pre.residual_scores(y,np.ones((1,141)),sample_ids=[96195],return_diagnostics=True)
    expected=.99/210/39
    assert scores[0]==pytest.approx(expected)
    assert details['adaptive_event_padding'][0]['sample_id']==96195
    assert details['adaptive_event_padding'][0]['window']<4
    assert details['minimum_continuous_stencils']>0


def test_pre_distinguishes_no_stencil_and_nonfinite_continuous_rhs(monkeypatch):
    y=np.zeros((1,39,40));y[:,36]=np.arange(40)
    with pytest.raises(ValueError,match='event-free.*sample_ids=\\[17\\]'):
        pre.residual_scores(y,np.ones((1,141)),sample_ids=[17])
    y[:]=0
    monkeypatch.setattr(pre,'rhs',lambda y,p,t:torch.full_like(y,float('inf')))
    with pytest.raises(FloatingPointError,match='nonfinite ODE.*sample_ids=\\[18\\]'):
        pre.residual_scores(y,np.ones((1,141)),sample_ids=[18])


def test_pre_masked_nonfinite_event_is_not_nan_times_zero(monkeypatch):
    y=np.zeros((1,39,100));y[:,0,:50]=2;y[:,0,50:]=1
    def event_rhs(state,p,t):
        result=torch.zeros_like(state);result[:,49,:]=float('nan');return result
    monkeypatch.setattr(pre,'rhs',event_rhs)
    score,details=pre.residual_scores(y,np.ones((1,141)),return_diagnostics=True)
    assert score[0]==0 and details['masked_nonfinite_points']==1


@pytest.mark.parametrize('opposed,triggered',[(False,False),(True,True)])
def test_eracs_default_threshold_executes_both_real_snr_branches(tmp_path,monkeypatch,opposed,triggered):
    from budding_yeast_v2.utils.sampling.dynamic.callback import DynamicConfig,DynamicSelection
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')
    inputs=PoolInputs(np.arange(10,14),np.ones((4,3)),np.ones((4,2)))
    snapshots=[]
    oracle=SimpleNamespace(store=SimpleNamespace(snapshot=lambda *args,**kwargs:snapshots.append(kwargs)))
    config=DynamicConfig(epochs=3,warmup=0,bootstrap_epochs=0,acquisition_epochs=1,check_every=1,omp_group_size=1)
    callback=DynamicSelection(inputs,oracle,[10],2,'eracs_regression',config,42,tmp_path,'cpu',8)
    class Resident:
        rows=[10,11];weights=np.ones(2)
        def set_rows(self,rows,weights):self.rows=list(rows);self.weights=weights
    callback.dataset=Resident()
    full=np.array([[1.,0.],[-1.,0.],[0.,1.],[.5,1.]])
    def gradients(model,rows):
        assert set(rows)<=set(inputs.ids)
        if len(rows)==4:return full
        return full[:2] if opposed else np.array([[1.,0.],[1.01,0.]])
    callback.gradients=gradients
    callback.on_train_epoch_start(SimpleNamespace(current_epoch=1),None)
    event=callback.events[-1];summary=callback.selection_summary()
    assert event['trigger'] is triggered and event['threshold']==.1
    assert summary['actual_reselections']==int(triggered)
    assert summary['omp_mechanism_exercised'] is triggered
    assert len(snapshots)==int(triggered)
    assert len(set(callback.dataset.rows))==2 and np.isfinite(callback.dataset.weights).all()
