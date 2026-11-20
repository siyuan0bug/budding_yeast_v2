"""Budget, legality, leakage and pre-query feedback regression coverage."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.contracts import QueryBatch
from budding_yeast_v2.utils.sampling.registry import strategy_spec, validate_system
from budding_yeast_v2.utils.sampling.ours import experimental as ex
from budding_yeast_v2.utils.sampling.ours import query_design as design
from budding_yeast_v2.utils.sampling.ours import query_feedback as feedback


def inputs(definitions):
    backend=BuddingSystem();params=[];initials=[];overrides=[];metadata=[]
    for pairs in definitions:
        ovr={'mdt':90.}
        for gene,mode,alpha in pairs:
            backend.apply_gene_mutation(gene,mode,alpha,backend.defaults(),backend.defaults(),ovr,
                                       rng=np.random.RandomState(42))
        p,y,*_=backend.prepare_initial_conditions(ovr)
        params.append([ovr.get(k,backend.defaults()[k]) for k in backend.PARAM_KEYS]);initials.append(y);overrides.append(ovr)
        metadata.append({'pairs':[(g,m) for g,m,_ in pairs]})
    return np.asarray(params,dtype=np.float32),np.asarray(initials),overrides,metadata


class Surrogate(torch.nn.Module):
    def __init__(self,value):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(float(value)))
    def forward_ic_time(self,x,p):
        return self.weight*(p[:,:1,None]+1).expand(-1,39,x.shape[-1])


class DM:
    system_id='budding'
    def __init__(self):
        self.train_dataset_synth=TensorDataset(torch.zeros(1,39,2,16),torch.ones(1,1),torch.zeros(1,39,16))
    def denormalize_y(self,y,device=None):return y.exp().clamp(max=1e5)


def fixture(strategy):
    table=inputs([[('CLN2','GAL-OE',float(a))] for a in np.linspace(.06,.30,12)])
    engine=ActiveLearningEngine(strategy=strategy,initial_train_size=5)
    p,y,o,m=table
    engine._pool_size=len(p);engine._pool_P_real=torch.tensor(p);engine._pool_P=torch.arange(len(p)).float()[:,None]/10
    engine._pool_initial_physical=torch.tensor(y);engine._pool_ovr=o;engine._pool_mass_meta=m
    engine._get_pool_X_batch=lambda ids:torch.zeros(len(ids),39,2,16)
    members=[Surrogate(-.1),Surrogate(.1)];engine.set_ensemble_members(members)
    return engine,DM(),members


def test_no_endpoint_reward_can_promote_zero_response_star():
    point=np.array([100.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
    feature=np.ones((2,10,4),dtype=np.float32)
    edges=np.array([[0,j] for j in range(1,10)])
    selected,records=design.marginal_actions(point,feature,edges,np.zeros(9),['same']*10,10,8,42)
    assert len(set(selected))==8 and selected[0]==0
    assert not any(r['kind']=='response' for r in records)
    assert sum(r['new_queries'] for r in records)==8


def test_shared_endpoints_four_way_actions_and_training_anchor_cost():
    rng=np.random.RandomState(2);feature=rng.normal(size=(2,13,16)).astype(np.float32)
    blocks=np.array([[0,1,2,12],[0,1,3,12],[4,5,6,7]])
    selected,records=design.marginal_actions(np.ones(13),feature,blocks,np.zeros(3),list(range(13)),12,10,42,kind='interaction')
    assert len(set(selected))==10 and selected.max()<12
    assert sum(r['new_queries'] for r in records)==10
    assert all(12 not in r['endpoints'] for r in records if r['kind']!='interaction')


def test_interaction_requires_existing_exact_legal_endpoints():
    definitions=[[('CLN2','MC',2)],[('CLN2','MC',5)],
                 [('CLN2','MC',2),('BCK2','KO',0)],[('CLN2','MC',5),('BCK2','KO',0)]]
    table=inputs(definitions);edges=np.array([[0,1],[2,3]])
    assert design.interaction_blocks(*table,edges).tolist()==[[0,1,2,3]]
    table[0][0,0]+=.001
    assert len(design.interaction_blocks(*table,edges))==0


def test_invariance_exact_condition_not_generic_small_parameter():
    p,y,o,m=inputs([[('TEM1','ts',a)] for a in (.001,.002,.004)])
    edges=np.array([[0,1],[1,2]])
    assert len(design.invariance_edges(p,y,o,edges))==2
    y[0,28]=1e-9
    assert design.invariance_edges(p,y,o,edges).tolist()==[[1,2]]


@pytest.mark.parametrize('strategy',list(ex.VARIANTS))
def test_all_variants_select_without_oracle_or_heldout_labels(strategy,tmp_path):
    engine,dm,members=fixture(strategy)
    # True training labels are not needed for selection when no anchors exist;
    # feedback preparation may read explicitly requested synthetic anchors only.
    dm.train_dataset_synth.tensors[2].fill_(float('nan'))
    members[0].train();members[1].eval()
    with patch.object(engine,'_build_broadcast_from_pool',side_effect=AssertionError('early oracle')):
        plan=engine.select(members[0],dm,'cpu')
    assert len(plan.candidate_ids)==5 and len(set(plan.candidate_ids))==5
    assert members[0].training and not members[1].training
    assert strategy_spec(strategy).requires_ensemble
    with pytest.raises(ValueError):validate_system(strategy,'fission')
    engine._al4pde_al_iter=1
    ex.write_audit(engine,tmp_path,dict(candidate_ids=plan.candidate_ids))
    assert json.loads((tmp_path/'query_design.json').read_text())['strategy']==strategy


def test_feedback_captured_before_oracle_matches_stable_ids_and_failed_endpoint(tmp_path):
    engine,dm,members=fixture('ours_response_feedback')
    p,y,*_=design.input_table(engine);features=feedback.fixed_input_features(engine,p,y)
    records=[dict(kind='response',endpoints=[0,1]),dict(kind='point',endpoints=[2])]
    feedback.prepare_feedback(engine,dm,'cpu',np.array([0,1,2]),records,features)
    ids=engine.pool.ensure_ids().copy();pending=engine._pending_query_feedback
    expected=pending['predictions'][int(ids[2])].copy()
    for member in members:member.weight.data.fill_(100) # Must NOT recompute predictions after querying.
    accepted=np.array([ids[2],ids[0]]) # Reorder and drop the response's other endpoint.
    actual=np.ones((2,39,16),dtype=np.float32)
    data={'X':torch.zeros(2,1),'P':torch.zeros(2,1),'Y':torch.zeros(2,39,16),
          'names':['a','b'],'candidate_ids':accepted,'Y_real':actual}
    feedback.observe_query(engine,QueryBatch(data,tuple(ids[:3]),tuple(accepted),engine.strategy))
    assert not hasattr(engine,'_pending_query_feedback')
    row=engine._query_feedback_history[0]
    np.testing.assert_array_equal(row['candidate_ids'],accepted)
    np.testing.assert_allclose(row['point_error'][0],np.mean((expected-np.log(2))**2),rtol=1e-5)
    assert (row['response_observation_count']==0).all()
    feedback.write_feedback(engine,tmp_path,0)
    with np.load(tmp_path/'prequery_feedback.npz') as archive:
        np.testing.assert_array_equal(archive['candidate_ids'],accepted)
        np.testing.assert_allclose(archive['prediction_log1p'][0],expected)
    assert json.loads((tmp_path/'response_feedback.json').read_text())['actions']==[]


def test_feedback_observed_response_and_feature_stats_survive_refits():
    engine,dm,_=fixture('ours_response_feedback')
    p,y,*_=design.input_table(engine);features=feedback.fixed_input_features(engine,p,y)
    feedback.prepare_feedback(engine,dm,'cpu',np.array([0,1]),[dict(kind='response',endpoints=[0,1])],features)
    ids=engine.pool.ensure_ids()[:2]
    pending=engine._pending_query_feedback
    # Exact log-physical predictions as targets give zero point AND response error.
    targets=np.expm1(np.array([pending['predictions'][int(i)] for i in ids]))
    data={'X':torch.zeros(2,1),'P':torch.zeros(2,1),'Y':torch.zeros(2,39,16),
          'names':['a','b'],'candidate_ids':ids,'Y_real':targets}
    feedback.observe_query(engine,QueryBatch(data,tuple(ids),tuple(ids),engine.strategy))
    row=engine._query_feedback_history[0]
    assert (row['point_error']<1e-10).all() and (row['response_error']<1e-10).all()
    assert (row['response_observation_count']==1).all()
    before=engine._query_feature_stats
    feedback.fixed_input_features(engine,p*100,y*100)
    assert all(np.array_equal(a,b) for a,b in zip(before,engine._query_feature_stats))


def test_event_disagreement_counts_debounced_divisions_and_shared_prediction():
    y=torch.ones(2,2,39,40)
    assert np.array_equal(ex.event_disagreement(y),np.zeros(2))
    y[0,0,0,20:]*=.5
    scores=ex.event_disagreement(y)
    assert scores[0]>0 and scores[1]==0


def test_hierarchical_selection_reduces_within_background_concentration():
    keys=[((('CLN2','MC'),),90.,(0,))]*8+[ ((('CLB2','MC'),),90.,(0,)) ]*2
    selected,records=design.hierarchical_rank(np.ones(10),keys,[],6,42)
    assert len(set(selected))==6 and any(i>=8 for i in selected[:3])


def test_learned_error_uses_only_completed_feedback_rows():
    engine,_,_=fixture('ours_error');rng=np.random.RandomState(1)
    features=rng.normal(size=(100,4)).astype(np.float32)
    scores,report=feedback.predict_error(engine,features)
    assert scores is None and report['status']=='cold_start'
    engine._query_feedback_history=[dict(input_features=features,
        point_error=features[:,0]**2,response_error=features[:,1]**2)]
    scores,report=feedback.predict_error(engine,features)
    assert scores.shape==(100,) and np.isfinite(scores).all() and (scores>=0).all()
    assert report['feedback_rows']==100 and report['target']=='point_error'


def test_research_launcher_preserves_inputs_budgets_and_all_baselines(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'DRY_RUN':'1','AL_SUITE':'research','TRAINING_PROFILE':'screen','EXP_ROOT':str(tmp_path/'experiments')}
    output=subprocess.check_output(['bash',str(root/'run_sampling_budding.sh')],env=env,text=True)
    tasks=[r for r in output.splitlines() if r.startswith('GPU=')]
    assert len(tasks)==14 and len({r.split('strategy=')[1].split()[0] for r in tasks})==14
    assert all('batch=512 lr=1e-3 modes=24 epochs=50/100' in r for r in tasks)
    assert '_refit_gene_ranges_research_v1_screen/' in output and 'seed policy=fixed' in output
    assert 'budding_pool_1000000_seed42_gene_ranges.npz' in output
    assert not (tmp_path/'experiments').exists()


def test_engine_query_observes_feedback_after_prune_and_writes_complete_audit(tmp_path):
    engine,dm,members=fixture('ours_error')
    plan=engine.select(members[0],dm,'cpu')
    def oracle(indices,datamodule,target_valid_num=None):
        ids=engine.pool.ensure_ids()[indices].copy()
        keep=np.ones(engine._pool_size,dtype=bool);keep[indices]=False
        engine.pool.keep(keep)
        return dict(X=torch.zeros(len(ids),1),P=torch.zeros(len(ids),1),Y=torch.zeros(len(ids),39,16),
                    Y_real=np.ones((len(ids),39,16),dtype=np.float32),names=['synthetic']*len(ids),candidate_ids=ids)
    engine._build_broadcast_from_pool=oracle
    batch=engine.query(plan,dm)
    assert batch.n_accepted==5 and engine._pool_size==7
    assert set(engine._query_feedback_history[0]['candidate_ids'])==set(plan.candidate_ids)
    engine._al4pde_al_iter=1
    ex.write_audit(engine,tmp_path,batch.data)
    report=json.loads((tmp_path/'query_design.json').read_text())
    assert all(e['status']=='accepted' for r in report['batches'][0]['choices'] for e in r['endpoints'])
    assert (tmp_path/'prequery_feedback.npz').exists()


def test_no_accepted_query_cannot_create_feedback():
    engine,dm,members=fixture('ours_error');engine.select(members[0],dm,'cpu')
    feedback.observe_query(engine,QueryBatch(None,(),(),engine.strategy))
    assert not hasattr(engine,'_pending_query_feedback')
    assert not getattr(engine,'_query_feedback_history',[])


def test_known_synthetic_anchor_feedback_is_free_and_alignment_safe():
    engine,dm,_=fixture('ours_response_feedback')
    p,y,o,m=design.input_table(engine)
    engine._cum_synth_params=[p[0]]
    engine._cum_synth_provenance=[dict(y_init=y[0],overrides_json=json.dumps(o[0]),
        candidate_metadata_json=json.dumps(m[0]),candidate_id=999)]
    p,y,*_=design.input_table(engine);features=feedback.fixed_input_features(engine,p,y)
    feedback.prepare_feedback(engine,dm,'cpu',np.array([1]),
        [dict(kind='response',endpoints=[1,12])],features)
    pending=engine._pending_query_feedback
    assert -1 in pending['anchors'] and len(pending['candidate_ids'])==1
    ids=engine.pool.ensure_ids()[[1]]
    data=dict(X=torch.zeros(1,1),P=torch.zeros(1,1),Y=torch.zeros(1,39,16),
        Y_real=np.ones((1,39,16)),names=['synthetic'],candidate_ids=ids)
    feedback.observe_query(engine,QueryBatch(data,tuple(ids),tuple(ids),engine.strategy))
    assert engine._query_feedback_history[0]['response_observation_count'].tolist()==[1]
