import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.sampling.ours import query_design as design
from budding_yeast_v2.utils.sampling.ours import hybrid,hybrid_feedback as calibration
from budding_yeast_v2.utils.sampling.ours.hybrid_features import PredictionObserver,stratified_tail
from budding_yeast_v2.utils.sampling.ours.experimental import event_disagreement
from budding_yeast_v2.utils.sampling.ours.response import prediction_features
from budding_yeast_v2.utils.sampling import oracle
from budding_yeast_v2.utils.sampling.contracts import QueryBatch
from tests.test_al_query_variants import fixture


@pytest.mark.parametrize('strategy',list(hybrid.VARIANTS))
def test_registered_hybrid_pool_only_budget_audit_and_rng(strategy):
    engine,dm,members=fixture(strategy)
    dm._test_idx_real=property(lambda:(_ for _ in ()).throw(AssertionError('real label')))
    np_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    with patch.object(engine,'_build_broadcast_from_pool',side_effect=AssertionError('ODE during selection')):
        plan=engine.select(members[0],dm,'cpu')
    assert len(plan.candidate_ids)==5 and len(set(plan.candidate_ids))==5
    audit=engine._experimental_audits[0][0]
    assert sum(r['new_queries'] for r in audit['choices'])==5
    assert all(e['kind']=='pool' and 0<=e['candidate_id']<12 for r in audit['choices'] for e in r['endpoints'])
    assert len(json.dumps(audit))>0
    assert all(m.training for m in members)
    assert np.array_equal(np_state[1],np.random.get_state()[1])
    assert torch.equal(torch_state,torch.get_rng_state())


def test_preselected_anchors_and_global_fallback_are_charged_once():
    features=np.zeros((1,10,6),dtype=np.float32);point=np.ones(10)
    calls=[]
    def fallback(count,selected):
        calls.append((count,selected.tolist()))
        return np.setdiff1d(np.arange(10),selected)[:count]
    selected,records=design.marginal_actions(point,features,np.array([[0,1],[2,9]]),np.zeros(2),
        ['same']*10,9,8,42,preselected=[0,2,3,4],point_fraction=0.,random_fraction=0.,fallback=fallback)
    assert len(selected)==len(set(selected))==8 and selected[:4].tolist()==[0,2,3,4]
    assert sum(r['new_queries'] for r in records)==4
    assert all(r['kind']=='global_fallback' for r in records)
    assert calls==[(4,[0,2,3,4])]


def test_tail_round_robin_protects_each_variable_without_global_scale_dominance():
    observer=PredictionObserver(100,42)
    observer.peak_log[:]=0;observer.physical_variance[:]=0
    observer.peak_log[:,0]=np.arange(100);observer.peak_log[:,1]=np.arange(100)[::-1]
    observer.physical_variance[:,0]=1e20;observer.physical_variance[:,1]=1.
    rows,records=stratified_tail(observer,100,6,42)
    assert len(set(rows))==6 and {r['variable'] for r in records}=={0,1}
    assert max(sum(r['variable']==j for r in records) for j in (0,1))<=4


def test_two_member_contrast_compression_preserves_response_variance():
    engine,dm,_=fixture('ours_global_response')
    point,full=prediction_features(engine,dm,'cpu',0)
    compressed_point,compressed=prediction_features(engine,dm,'cpu',0,compact_two_members=True)
    assert compressed.shape[0]==1 and full.shape[0]==2
    edges=np.array([[0,1],[1,5],[5,10]])
    np.testing.assert_array_equal(point,compressed_point)
    np.testing.assert_allclose(design.action_variance(full,edges),design.action_variance(compressed,edges),rtol=1e-5,atol=1e-8)


def test_event_chain_debounce_matches_canonical_labels():
    y=np.ones((2,1,39,40),dtype=np.float32);y[:,:,0]=2.;y[:,:,2]=0.;y[:,:,33]=0.
    for t in (5,11,17):y[0,0,0,t:]-=.2
    y[1,0,0,5:] -= .2
    np.testing.assert_array_equal(event_disagreement(torch.from_numpy(y)),[0.])


@pytest.mark.parametrize('drop_times',[(),(5,),(5,20),(5,20,35)])
def test_event_descriptor_handles_zero_one_and_multiple_divisions(drop_times):
    values=torch.ones(2,3,39,50)
    for t in drop_times:values[:,:,0,t:]-=.2
    # Shared trajectories have exactly zero disagreement at every event count.
    scores=event_disagreement(values)
    np.testing.assert_array_equal(scores,np.zeros(3,dtype=np.float32))
    # A member that never divides must disagree with the dividing member.
    values[1,0,0]=1.
    scores=event_disagreement(values)
    assert np.isfinite(scores).all()
    assert bool(scores[0]>0)==bool(drop_times)
    np.testing.assert_array_equal(scores[1:],np.zeros(2,dtype=np.float32))


def test_complete_edge_feedback_stable_ids_failed_endpoint_and_atomic_output(tmp_path):
    engine,_,_=fixture('ours_global_calibrated')
    engine._pending_hybrid_edges=dict(acquisition=0,rows={(1,2):(np.ones(17),.2),(3,-1):(np.zeros(17),.1)})
    engine._query_feedback_outputs={0:[dict(action_errors=[
        dict(kind='response',endpoints=[1,2],observed_response_error=.3)])]}
    calibration.observe_edges(engine,SimpleNamespace(data={}))
    assert engine._hybrid_edge_history[-1]['features'].shape==(1,17)
    assert engine._hybrid_edge_history[-1]['endpoints'].tolist()==[[1,2]]
    calibration.write_edges(engine,tmp_path,0)
    with np.load(tmp_path/'edge_feedback.npz') as archive:
        assert archive['endpoints'].tolist()==[[1,2]] and archive['observed_error'][0]==pytest.approx(.3)
    assert not (tmp_path/'edge_feedback.npz.tmp').exists()
    assert json.loads((tmp_path/'edge_feedback_schema.json').read_text())['edges']==1
    engine._pending_hybrid_edges=dict(acquisition=1,rows={})
    calibration.observe_edges(engine,QueryBatch(None,(),(),engine.strategy))
    assert not hasattr(engine,'_pending_hybrid_edges')


def test_calibration_uses_only_recent_edges_and_is_bounded():
    rng=np.random.RandomState(3);x=rng.normal(size=(150,17)).astype(np.float32)
    error=np.exp(x[:,0])*.1;variance=np.full(150,.1,dtype=np.float32)
    engine=SimpleNamespace(_hybrid_edge_history=[dict(acquisition=0,features=x[:2]),
        dict(acquisition=1,features=x,log_variance=variance,observed_error=error)])
    multiplier,report=calibration.calibrated_multiplier(engine,x,42)
    assert report['complete_edges']==150 and report['history_acquisition']==1
    assert (multiplier>=1).all() and (multiplier<=2).all()
    assert np.corrcoef(multiplier,x[:,0])[0,1]>.8


def test_oracle_rebuild_retains_completed_order_and_does_not_skip(monkeypatch):
    class Executor:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    calls=[]
    def ordered(executor,function,tasks,**kwargs):
        calls.append(list(tasks))
        for i,task in enumerate(tasks):
            if len(calls)==1 and i==2:raise TimeoutError('controlled pool stall')
            yield task*2
    monkeypatch.setattr(oracle,'ProcessPoolExecutor',Executor)
    monkeypatch.setattr(oracle,'bounded_ordered_map',ordered)
    assert list(oracle._simulation_results([0,1,2,3],2,None))==[0,2,4,6]
    assert calls==[[0,1,2,3],[2,3]]


def test_oracle_worker_drops_unused_input_and_preserves_export_precision(monkeypatch):
    original=np.full((39,500),.123456789,dtype=np.float64);params=np.ones(141,dtype=np.float64)
    monkeypatch.setattr(oracle,'solve_single_mass_bridge',lambda *a,**k:(np.ones((39,2,500)),original,params,'Pattern_C','bio'))
    result=oracle._solve_single_mass_task_wrapper(('name',{}))
    assert result[0] is None and result[1].dtype==np.float32 and result[2].dtype==np.float64
    np.testing.assert_array_equal(result[1],original.astype(np.float32))
    assert result[3:]==('Pattern_C','bio')
