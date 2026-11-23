"""Response eligibility, immediate labels, matched path control and exact task logging."""
from copy import deepcopy
import numpy as np
import pytest
import torch

from test_adaptive_generation import setup
from budding_yeast_v2.utils.sampling.ours.adaptive_generation.policy import AdaptiveGeneration
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.sampling.contracts import PoolInputs
from budding_yeast_v2.utils.sampling.generation.path_bank import path_bank
from budding_yeast_v2.utils.sampling.ours.response_v2.policy import ResponseGeneration
from budding_yeast_v2.utils.sampling.ours.response_v2.metrics import relative
from budding_yeast_v2.utils.sampling.ours.response_v2.paths import path_id
from budding_yeast_v2.utils.training.losses import RelativeL2Loss


class Predictor:
    def __init__(self,train,truth):self.normalization=NormalizationStats.fit(train.params,truth);self.offset=.01
    def batches(self,inputs,physical=False):
        y=np.repeat(inputs.initial[:,:,None],12,axis=-1)+self.offset
        value=self.normalization.targets(y)
        yield 0,self.normalization.decode(value) if physical else value


def view(bank,ids):return PoolInputs(bank.ids[ids],bank.params[ids],bank.initial[ids])


def test_cold_start_is_unscored_and_cannot_win_as_fake_error(tmp_path):
    _,_,train,truth=setup(tmp_path)
    c=ResponseGeneration('ours_response_v2_generate',train,truth,42);c.refresh(train,truth,Predictor(train,truth),target=4)
    bank=PoolInputs(np.arange(8),train.params[:8],train.initial[:8],tuple({} for _ in range(8)))
    ids,d=c.select(bank,4)
    assert d['supported_candidates']==0 and d['selected_scores']==[None]*4
    assert d['action_counts'].get('response',0)==0 and d['capacity_shortfall']>0


@pytest.mark.parametrize('name',['ours_response_v2_generate','path_random_generate'])
def test_queried_support_is_used_immediately_and_budget_counters_restore(tmp_path,name):
    domain,_,train,truth=setup(tmp_path)
    c=ResponseGeneration(name,train,truth,42);pred=Predictor(train,truth);c.refresh(train,truth,pred,target=100)
    records=[]
    for _ in range(5):
        candidates,signatures,bank=path_bank(domain,c,64)
        for candidate,p,meta in zip(candidates,bank.params,bank.metadata):
            if meta.get('adaptive_path_id'):
                assert path_id(p,meta,meta['adaptive_axis'],c.system.param_names)==meta['adaptive_path_id']
        ids,d=c.select(bank,8);records.append(d)
        for i in ids:domain.consume(signatures[i])
        v=view(bank,ids);labels=np.repeat(v.initial[:,:,None],12,axis=-1)
        c.observe(v,labels)
    assert any(r['selected_supported']>=4 for r in records[1:])
    assert len(c.journal)==40 and c.total_selected==40 and c.exploration_selected==8
    assert not c.pending
    assert all(r['normalized_relative_l2']>=0 and r['physical_relative_l2']>=0 for r in c.journal)
    assert all(r['score'] is None for r in c.journal if r['action'] in ('explore','support'))
    state=deepcopy(c.state_dict());other=ResponseGeneration(name,train,truth,42);other.load_state_dict(state)
    assert other.journal==c.journal and other.total_selected==40
    with pytest.raises(PermissionError):c.observe(v,labels)


def test_control_has_identical_proposal_stream_before_feedback(tmp_path):
    d1,_,train,truth=setup(tmp_path/'a');d2,_,_,_=setup(tmp_path/'b')
    a=ResponseGeneration('ours_response_v2_generate',train,truth,42)
    b=ResponseGeneration('path_random_generate',train,truth,42)
    pred=Predictor(train,truth)
    for c in (a,b):c.refresh(train,truth,pred,target=10)
    x,s1,v1=path_bank(d1,a,64);y,s2,v2=path_bank(d2,b,64)
    np.testing.assert_array_equal(v1.params,v2.params);assert s1==s2
    assert a.rng.bit_generator.state==b.rng.bit_generator.state


def test_relative_diagnostic_equals_training_formula_and_saved_stats(tmp_path):
    domain,_,train,truth=setup(tmp_path)
    c=ResponseGeneration('ours_response_v2_generate',train,truth,42);pred=Predictor(train,truth)
    c.refresh(train,truth,pred,target=10);_,_,bank=path_bank(domain,c,32);ids,_=c.select(bank,4)
    v=view(bank,ids);labels=np.repeat(v.initial[:,:,None],12,axis=-1)
    original=[c.pending[AdaptiveGeneration._key(p,h)] for p,h in zip(v.params,v.initial)]
    actual=relative(np.stack([r['prediction'] for r in original]),pred.normalization.targets(labels).numpy())
    same=RelativeL2Loss(eps=.01).per_sample(torch.tensor(np.stack([r['prediction'] for r in original])),pred.normalization.targets(labels)).numpy()
    np.testing.assert_array_equal(actual,same)
    # Changing current stats/model after select cannot change the logged pre-query metric.
    pred.normalization.mean_y.add_(100.)
    c.observe(v,labels)
    np.testing.assert_array_equal(actual,[r['normalized_relative_l2'] for r in c.journal])


def test_failure_refunds_exploration_and_does_not_create_support(tmp_path):
    domain,_,train,truth=setup(tmp_path);c=ResponseGeneration('ours_response_v2_generate',train,truth,42)
    c.refresh(train,truth,Predictor(train,truth),target=10);_,_,bank=path_bank(domain,c,32);ids,_=c.select(bank,8)
    c.discard(view(bank,ids));assert c.total_selected==0 and c.exploration_selected==0 and not c.pending
    assert not c.journal


@pytest.mark.parametrize('name',['ours_response_v2_generate','path_random_generate'])
def test_two_round_real_fno_and_ode_with_reload(tmp_path,monkeypatch,name):
    from test_adaptive_generation import test_real_fno_ode_two_round_cpu_workflow as exercise
    exercise(tmp_path,monkeypatch,name)


def test_recovery_rebuilds_revealed_support_and_reproduces_selection(tmp_path):
    domain,_,train,truth=setup(tmp_path/'original');pred=Predictor(train,truth)
    a=ResponseGeneration('ours_response_v2_generate',train,truth,42);a.refresh(train,truth,pred,target=100)
    _,signatures,bank=path_bank(domain,a,64);ids,_=a.select(bank,8)
    for i in ids:domain.consume(signatures[i])
    v=view(bank,ids);a.observe(v,np.repeat(v.initial[:,:,None],12,axis=-1))
    state=deepcopy(a.state_dict());b=ResponseGeneration(a.name,train,truth,42);b.load_state_dict(state)
    b.refresh(a.inputs,np.repeat(a.inputs.initial[:,:,None],12,axis=-1),pred,target=100)
    from budding_yeast_v2.utils.sampling.generation.domain import FrozenBuddingDomain
    other=FrozenBuddingDomain('mutation',42,tmp_path/'restored',seen=domain.seen,next_index=domain.next_index)
    _,s1,bank1=path_bank(domain,a,64);_,s2,bank2=path_bank(other,b,64)
    assert s1==s2 and a.round==b.round
    i1,_=a.select(bank1,8);i2,_=b.select(bank2,8)
    np.testing.assert_array_equal(i1,i2)


def test_response_v2_suite_has_only_six_distinct_method_seed_pairs(tmp_path,capsys):
    import json
    from test_multiseed_adaptive_queue import protocols
    from budding_yeast_v2.scripts.launch.run_sampling_extensions import main
    root=protocols(tmp_path)
    main(['--suite','response_v2','--seeds','42','123','1024','--protocol',str(root),
          '--directory',str(tmp_path/'not_created'),'--gpus','0','1','2','3',
          '--initial-jobs','6','--max-jobs','6','--jobs-per-gpu','2','--dry-run'])
    plan=json.loads(capsys.readouterr().out)
    assert plan['count']==6 and {r['method'] for r in plan['tasks']}=={'ours_response_v2_generate','path_random_generate'}
    assert len({(r['method'],r['seed']) for r in plan['tasks']})==6
    assert not (tmp_path/'not_created').exists()
