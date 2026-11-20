"""Execution changes must preserve acquisition values, order and failures."""
import numpy as np
import pytest
from budding_yeast_v2.utils.sampling.ours.mechanistic.systems import BiologicalODE
from budding_yeast_v2.utils.sampling.ours.mechanistic.transport import TransportConfig
from budding_yeast_v2.utils.sampling.ours.mechanistic.parallel import score_rows, shutdown


def rows_for(system_id):
    system=BiologicalODE(system_id)
    if system_id=='budding':
        from test_portable_ours import budding_inputs
        inputs=budding_inputs([2.,3.,4.,5.])
        params,initial,meta=inputs.params,inputs.initial,inputs.metadata
    else:
        from budding_yeast_v2.data.systems.fission.parameters import WT_parameters,param_vector
        params=np.tile(param_vector(WT_parameters()),(4,1))
        initial=np.tile([2.,.5,.1,.3,.1,.1,.1,.1,.3,.3,.3,.3,.1,.1,.1,.2],(4,1))
        meta=({},)*4
    y=np.repeat(system.initials(initial)[:,:,None],9,axis=2)
    if system_id=='budding':
        y[:,2]=np.linspace(.4,.1,9)
        y[:,33]=np.linspace(.8,1.1,9)
    rows=[]
    for i in range(4):
        reference=y[i]*1.01 if i==1 else y[i] if i==2 else None
        rows.append((y[i],params[i],initial[i],meta[i],reference,
            params[0] if reference is not None else None,
            initial[0] if reference is not None else None,meta[0] if reference is not None else None,
            i==2,np.ones(system.n),np.linspace(0,system.horizon,9),TransportConfig()))
    return system,rows


@pytest.mark.parametrize('system_id',['budding','fission'])
def test_spawn_scores_features_diagnostics_and_order_are_identical(system_id):
    system,rows=rows_for(system_id)
    try:
        serial=score_rows(system,rows,1)
        parallel=score_rows(system,rows,2)
        repeated=score_rows(system,rows,2)
        for a,b,c in zip(serial,parallel,repeated):
            assert a[0]==b[0]==c[0]
            np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(b[1],c[1])
            assert a[2]==b[2]==c[2]
    finally:shutdown()


def test_parallel_failures_are_not_silently_zero_scored():
    system,rows=rows_for('budding')
    broken=list(rows[0]);broken[0]=np.full_like(broken[0],np.nan)
    try:
        with pytest.raises(ValueError,match='finite aligned'):
            score_rows(system,[tuple(broken)],2)
    finally:shutdown()


def test_parallel_worker_bounds():
    system,rows=rows_for('budding')
    for workers in (0,17):
        with pytest.raises(ValueError,match='workers'):score_rows(system,rows,workers)


def test_generation_training_reduction_is_cached_and_refreshed(monkeypatch):
    from test_portable_ours import budding_inputs,PhysicalPredictor
    from budding_yeast_v2.utils.sampling.ours.mechanistic import selectors
    from budding_yeast_v2.utils.sampling.ours.mechanistic.generation import MechanisticGeneration
    monkeypatch.setenv('SAMPLING_MECHANISM_WORKERS','1')
    train=budding_inputs([2.,3.],offset=20)
    pool=budding_inputs([4.,5.])
    truth=np.repeat(train.initial[:,:,None],9,2)
    predictor=PhysicalPredictor('budding')
    scale=selectors._scale;calls=[]
    def measured(value):calls.append(len(value));return scale(value)
    monkeypatch.setattr(selectors,'_scale',measured)
    policy=MechanisticGeneration('ours_dynamics_transport',train,truth,42)
    policy.refresh(train,truth,predictor)
    first=policy.select(pool,1)
    assert len(calls)==1
    policy.select(pool,1)
    assert len(calls)==1
    policy.refresh(train,truth.copy(),predictor)
    policy.select(pool,1)
    assert len(calls)==2
    from budding_yeast_v2.utils.sampling.contracts import SamplingContext,QueriedTraining
    wrong=SamplingContext(pool,QueriedTraining(train,truth),predictor,42,'cpu')
    with pytest.raises(ValueError,match='different queried cohort'):
        selectors.mechanism_scores(wrong,'ours_dynamics_transport',training_preparation=policy._prepared)
