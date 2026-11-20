"""Response acquisition math, legal contrasts and oracle boundary."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.ours.response import (
    _strength_curve,background_graph,rank_queries,prediction_features,write_response_audit)
from budding_yeast_v2.utils.sampling.registry import strategy_spec,validate_system


def physical_rows(samples):
    backend=BuddingSystem();params=[];initials=[];overrides=[];metadata=[]
    for gene,mode,alpha,mdt in samples:
        ovr={'mdt':mdt}
        backend.apply_gene_mutation(gene,mode,alpha,backend.defaults(),backend.defaults(),ovr,
                                   rng=np.random.RandomState(42))
        p,y,*_=backend.prepare_initial_conditions(ovr)
        params.append([ovr.get(k,backend.defaults()[k]) for k in backend.PARAM_KEYS]);initials.append(y)
        overrides.append(ovr);metadata.append({'pairs':[(gene,mode)]})
    return np.asarray(params,dtype=np.float32),np.asarray(initials),overrides,metadata


def test_exact_mc_probability_coordinates_and_linked_rates():
    keys,values,cdf=_strength_curve('CLB2','MC','')
    assert len(keys)==2
    np.testing.assert_allclose(cdf,np.cumsum((1/np.arange(2,9))/np.sum(1/np.arange(2,9)))-(1/np.arange(2,9))/np.sum(1/np.arange(2,9))/2)
    params,initial,overrides,meta=physical_rows([('CLN2','MC',a,90.) for a in (2,4,7,10)])
    edges,distance=background_graph(params,initial,overrides,meta,4)
    pairs={tuple(e):d for e,d in zip(edges,distance)}
    assert (0,3) in pairs and pairs[0,1]<pairs[0,3]  # Far 2->10 remains legal.


def test_hard_conditions_environment_modes_initials_and_internal_branch():
    rows=physical_rows([('CLN2','MC',4,90),('CLN2','MC',6,90),
                        ('CLN2','MC',7,150),('CLN2','GAL-OE',.2,90)])
    edges,_=background_graph(*rows,4)
    assert edges.tolist()==[[0,1]]
    rows[1][1,0] += .01
    assert len(background_graph(*rows,4)[0])==0
    keys_a,*_=_strength_curve('CDC14','ts','activity')
    keys_b,*_=_strength_curve('CDC14','ts','synthesis')
    assert set(keys_a).isdisjoint(keys_b) and keys_b==('ks14',)


def test_response_variance_and_cost_count_unique_queries():
    # Members agree at x0 but disagree on response to x1. Pair beats point ranking.
    centered=np.array([[[-1.],[1.],[.1]],[[1.],[-1.],[-.1]]],dtype=np.float32)
    selected,records=rank_queries(np.array([1.,1.,.01]),centered,np.array([[0,1]]),np.array([.1]),3,2,42)
    assert set(selected)=={0,1} and records[0]['kind']=='response'
    assert records[0]['response_variance']==4 and records[0]['new_queries']==2
    # A known training anchor costs no query; a shared selected endpoint costs once.
    selected,records=rank_queries(np.array([1.,1.,1.]),centered,np.array([[0,2],[1,2]]),np.zeros(2),2,2,42)
    assert len(set(selected))==2 and all(i<2 for i in selected)
    assert all(r['new_queries']==1 for r in records)
    # Constant member offset has zero response variance, but point variance survives.
    feature=np.array([[[-1.],[-1.]],[[1.],[1.]]])
    selected,records=rank_queries(np.ones(2),feature,np.array([[0,1]]),np.zeros(1),2,2,42)
    assert len(selected)==2


class Surrogate(torch.nn.Module):
    def __init__(self,sign):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(float(sign)))
    def forward_ic_time(self,x,p):return self.weight*p[:,:1,None].expand(-1,39,x.shape[-1])


def engine_fixture():
    physical,initial,overrides,meta=physical_rows([('CLN2','MC',a,90.) for a in (4,5,6,8,10,30)])
    cb=ActiveLearningEngine(strategy='ours_response',initial_train_size=2)
    cb._pool_size=6;cb._pool_P_real=torch.tensor(physical);cb._pool_P=torch.arange(6.).reshape(6,1)
    cb._pool_initial_physical=torch.tensor(initial);cb._pool_ovr=overrides;cb._pool_mass_meta=meta
    cb._get_pool_X_batch=lambda idx:torch.zeros(len(idx),39,2,8)
    models=[Surrogate(-1),Surrogate(1)];cb.set_ensemble_members(models)
    dm=SimpleNamespace(system_id='budding',train_dataset_synth=TensorDataset(torch.zeros(1,39,2,8),torch.ones(1,1),torch.full((1,39,8),float('nan'))))
    return cb,dm,models


def test_registered_selection_never_reads_true_trajectories_or_calls_oracle_and_restores_modes():
    cb,dm,models=engine_fixture();models[0].train();models[1].eval()
    with patch.object(cb,'_build_broadcast_from_pool',side_effect=AssertionError('early oracle')):
        plan=cb.select(models[0],dm,'cpu')
    assert len(plan.candidate_ids)==2 and len(set(plan.candidate_ids))==2
    assert models[0].training and not models[1].training
    assert strategy_spec('ours_response').requires_ensemble
    with pytest.raises(ValueError,match='not supported'):validate_system('ours_response','fission')


def test_training_anchors_use_only_synthetic_training_rows():
    cb,dm,models=engine_fixture()
    cb._cum_synth_params=[cb._pool_P_real[0].numpy()]
    cb._cum_synth_provenance=[dict(y_init=cb._pool_initial_physical[0].numpy(),overrides_json=json.dumps(cb._pool_ovr[0]),candidate_metadata_json=json.dumps(cb._pool_mass_meta[0]),candidate_id=999)]
    plan=cb.select(models[0],dm,'cpu')
    assert len(plan.candidate_ids)==2
    # The anchor is outside the pool and must never appear in a query plan.
    assert 999 not in plan.candidate_ids


def test_prediction_nonfinite_and_missing_ensemble_rejected():
    cb,dm,models=engine_fixture()
    cb._pool_P[0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):prediction_features(cb,dm,'cpu',0)
    assert models[0].training
    cb.set_ensemble_members(None)
    with pytest.raises(RuntimeError,match='independently trained'):prediction_features(cb,dm,'cpu',0)


def test_pair_trace_reports_failed_endpoint_and_known_anchor(tmp_path):
    engine=SimpleNamespace(_al4pde_al_iter=1,_response_acquisitions={0:[dict(kind='response',endpoints=[dict(kind='pool',candidate_id=1),dict(kind='pool',candidate_id=2)]),dict(kind='response',endpoints=[dict(kind='training_anchor',candidate_id=3)])]})
    write_response_audit(engine,tmp_path,{'candidate_ids':[1]})
    report=json.loads((tmp_path/'response_pairs.json').read_text())
    assert [e['status'] for r in report['choices'] for e in r['endpoints']]==['accepted','not_accepted','known_training']


def test_selected_training_anchor_serializes_after_numpy_pool_pruning(tmp_path):
    cb,dm,models=engine_fixture()
    # Pool pruning sets size from a NumPy mask sum in a real acquisition.
    cb._pool_size=np.int64(cb._pool_size)
    cb._cum_synth_params=[cb._pool_P_real[0].numpy()]
    cb._cum_synth_provenance=[dict(y_init=cb._pool_initial_physical[0].numpy(),
        overrides_json=json.dumps(cb._pool_ovr[0]),
        candidate_metadata_json=json.dumps(cb._pool_mass_meta[0]),candidate_id=999)]
    records=[dict(kind='response',endpoints=(0,6),score=.5,new_queries=1)]
    with patch('budding_yeast_v2.utils.sampling.ours.response.rank_queries',
               return_value=(np.array([0]),records)):
        cb.select(models[0],dm,'cpu')
    cb._al4pde_al_iter=1
    write_response_audit(cb,tmp_path,{'candidate_ids':cb.pool.ensure_ids()[:1]})
    report=json.loads((tmp_path/'response_pairs.json').read_text())
    anchor=report['choices'][0]['endpoints'][1]
    assert anchor==dict(kind='training_anchor',training_row=0,candidate_id=999,status='known_training')
    assert report['timings'][0]['pool_size']==6


def test_audit_serialization_failure_preserves_previous_report(tmp_path):
    target=tmp_path/'response_pairs.json';target.write_text('{"previous":true}')
    cb=SimpleNamespace(_al4pde_al_iter=1,_response_acquisitions={0:[dict(
        kind='point',endpoints=[],unsupported=object())]})
    with pytest.raises(TypeError):write_response_audit(cb,tmp_path,{})
    assert json.loads(target.read_text())=={'previous':True}


def test_all_current_mutation_controls_recover_their_probability_coordinates():
    from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME
    from budding_yeast_v2.utils.sampling.ours.response import _CoordinateRNG
    backend=BuddingSystem();defaults=backend.defaults()
    for gene,group in GROUP_BY_NAME.items():
        if gene=='MDT':continue
        for mode in group['modes']:
            branches=('synthesis','activity') if (gene,mode)==('CDC14','ts') else ('',)
            for branch in branches:
                keys,values,cdf=_strength_curve(gene,mode,branch)
                for u in (.2,.8):
                    ovr={}
                    alpha=backend.sample_scaling_factor_sobol(mode,gene,u)
                    backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),ovr,rng=_CoordinateRNG(u,branch))
                    if keys and mode not in ('MC','GAL-MC'):
                        assert abs(np.interp(ovr[keys[0]],values,cdf)-u)<.001,(gene,mode,branch)
