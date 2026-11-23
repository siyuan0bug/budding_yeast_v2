"""Scientific/data isolation, exact legal compilation and feedback recovery."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.sampling.contracts import PoolInputs
from budding_yeast_v2.utils.sampling.generation.domain import FrozenBuddingDomain
from budding_yeast_v2.utils.sampling.generation.interventions import intervention_bank
from budding_yeast_v2.utils.sampling.generation.policies import GenerationPolicy
from budding_yeast_v2.utils.sampling.ours.adaptive_generation import LEGACY_METHODS as ADAPTIVE_METHODS
from budding_yeast_v2.utils.sampling.ours.adaptive_generation.policy import AdaptiveGeneration
from budding_yeast_v2.utils.sampling.ours.adaptive_generation.features import event_features
from budding_yeast_v2.utils.sampling.ours.adaptive_generation.feedback import Feedback
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.budding.catalog import touch_set


class Predictor:
    def __init__(self):self.offset=0.
    def batches(self,inputs,physical=False):
        assert physical
        # Does not read any oracle or labels.
        y=np.repeat(inputs.initial[:,:,None],12,axis=-1).copy()
        y+=self.offset
        yield 0,torch.from_numpy(y)


def setup(tmp_path):
    domain=FrozenBuddingDomain('mutation',42,tmp_path)
    c,s,v=domain.bank(12)
    train=PoolInputs(v.ids,v.params,v.initial,tuple(dict(x.metadata,_effective_overrides=x.overrides) for x in c))
    truth=np.repeat(v.initial[:,:,None],12,axis=-1)
    return domain,c,train,truth


@pytest.mark.parametrize('name',ADAPTIVE_METHODS)
def test_canonical_paths_feedback_recovery_and_no_label_reader(tmp_path,name):
    domain,_,train,truth=setup(tmp_path)
    controller=AdaptiveGeneration(name,train,truth,42);pred=Predictor();controller.refresh(train,truth,pred,target=4)
    candidates,signatures,bank=intervention_bank(domain,controller,8)
    assert len(set(signatures))==8
    assert all(s not in domain.forbidden and s not in domain.seen for s in signatures)
    assert domain.counts['adaptive_interventions']>0
    for c,p,y,s in zip(candidates,bank.params,bank.initial,signatures):
        np.testing.assert_array_equal(domain.proposals.backend.prepare_initial_conditions(c.overrides)[1],y)
        assert s==hashlib.sha256(effective_inputs(c.overrides,domain.proposals.defaults).astype('<f8').tobytes()).hexdigest()
        if 'adaptive_axis' in c.metadata:
            j=int(c.metadata['adaptive_anchor_id']);gene=c.metadata['adaptive_axis']
            assert c.overrides['mdt']==train.metadata[j]['_effective_overrides']['mdt']
            assert c.metadata['pairs']==train.metadata[j]['pairs']
            touched=set().union(*(touch_set(g,m) for g,m in c.metadata['pairs'] if g==gene))
            outside=[i for i,k in enumerate(controller.system.param_names) if k not in touched]
            np.testing.assert_array_equal(p[outside],train.params[j,outside])
    state=deepcopy(controller.state_dict());other=AdaptiveGeneration(name,train,truth,42)
    other.load_state_dict(state);other.refresh(train,truth,pred,target=4)
    assert other.feedback.round==controller.feedback.round
    ids,report=controller.select(bank,3);ids2,_=other.select(bank,3)
    np.testing.assert_array_equal(ids,ids2)
    assert not report['candidate_truth_used'] and not report['pattern_labels_used']
    view=PoolInputs(bank.ids[ids],bank.params[ids],bank.initial[ids])
    # Reveal truth after select; mutate current model to prove saved pre-query prediction is used.
    target=np.repeat(view.initial[:,:,None],12,axis=-1)
    pred.offset=1e6
    controller.observe(view,target)
    assert controller.feedback.rows and max(r['error'] for r in controller.feedback.rows)==0
    assert not controller.pending
    with pytest.raises(PermissionError):controller.observe(view,target)


def test_prequential_progress_changes_anchor_weights_and_state():
    a=Feedback();a.round=1
    a.observe('improving',1,10);a.observe('worsening',1,1)
    a.round=2;a.observe('improving',1,1);a.observe('worsening',1,10)
    assert a.weight('improving')>1>a.weight('worsening')
    b=Feedback();b.load_state_dict(a.state_dict());assert b.weight('improving')==a.weight('improving')
    with pytest.raises(ValueError):a.observe('bad',np.inf,1)


def test_event_features_capture_time_and_order_without_patterns():
    class System:
        def guards(self,y,p):return y, (1,1)
    y=np.array([[[-1,-1,1,1],[-1,1,1,1]]],dtype=float)
    a=event_features(y,np.zeros((1,1)),System());b=event_features(y[:,::-1],np.zeros((1,1)),System())
    assert not np.array_equal(a,b)
    np.testing.assert_array_equal(a,event_features(y,np.zeros((1,1)),System()))
    with pytest.raises(ValueError):event_features(y*np.nan,np.zeros((1,1)),System())


def test_dispatch_observe_is_not_a_noop(tmp_path):
    domain,_,train,truth=setup(tmp_path);policy=GenerationPolicy('ours_response_generate',train,truth)
    policy.refresh(train,truth,Predictor(),target=4)
    _,_,bank=intervention_bank(domain,policy.controller,4)
    ids,_=policy.select(bank,2)
    v=PoolInputs(bank.ids[ids],bank.params[ids],bank.initial[ids])
    policy.observe(v,np.repeat(v.initial[:,:,None],12,axis=-1))
    assert policy.feedback_rows==2 and len(policy.controller.feedback.rows)==2
    restored=GenerationPolicy(policy.name,train,truth);restored.load_state_dict(policy.state_dict())
    assert restored.controller.feedback.rows==policy.controller.feedback.rows


@pytest.mark.parametrize('name',ADAPTIVE_METHODS)
def test_real_fno_ode_two_round_cpu_workflow(tmp_path,monkeypatch,name):
    import json
    from test_sampling_benchmark import config
    from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
    from budding_yeast_v2.utils.sampling.generation.runner import run_generation
    from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
    domain=FrozenBuddingDomain('mutation',42,tmp_path/'source_proposals')
    candidates,_,v=domain.bank(28)
    # Training pipeline fixture only. New queried trajectories below use the REAL ODE.
    y=np.repeat(v.initial[:,:,None],500,axis=-1).astype(np.float32)
    y+=np.linspace(0,.01,500)[None,None,:]
    arrays=dict(params=v.params,y_init=v.initial,data=y,candidate_ids=np.arange(28),
        mutant_names=np.asarray([c.name for c in candidates]),is_real=np.zeros(28,bool),
        param_names=np.asarray(domain.proposals.backend.PARAM_KEYS),
        time_grid=np.linspace(0,210,500),labels=np.asarray(['fixture']*28),
        sampling_paradigm=np.array('mutation_random'),
        pattern_labels=np.asarray(['fixture']*28),
        candidate_metadata_json=np.asarray([json.dumps(c.metadata) for c in candidates]),
        overrides_json=np.asarray([json.dumps(c.overrides) for c in candidates]),**pattern_group_metadata(y))
    from budding_yeast_v2.data.dataset_archive import write_dataset_manifest
    chunk=tmp_path/'source.npz';np.savez_compressed(chunk,**arrays)
    source=tmp_path/'source.json';write_dataset_manifest(source,arrays,[(chunk,None)])
    root=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',
        replace(config(),budgets=(2,3),epochs=(1,1),warmup=(0,0)))
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MAX_LOAD_PER_CPU','100')
    if name=='ours_response_generate':
        from concurrent.futures import Future
        from budding_yeast_v2.utils.sampling.generation import runner
        original=runner.simulate;attempts=[]
        def fail_first(candidate):
            result=original(candidate);attempts.append(candidate)
            return replace(result,status='injected_failure',trajectory=None) if len(attempts)==1 else result
        monkeypatch.setattr(runner,'simulate',fail_first)
        class ImmediateExecutor:
            # Main thread retains the native solver's signal timeout contract.
            def __init__(self,*args,**kw):self._max_workers=1
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def submit(self,fn,*args):
                future=Future()
                try:future.set_result(fn(*args))
                except BaseException as error:future.set_exception(error)
                return future
        monkeypatch.setattr(runner,'ProcessPoolExecutor',ImmediateExecutor)
    directory=run_generation(root,'mutation',name,tmp_path/'results',device='cpu',workers=1,
        bank_size=4,query_batch_size=1,diagnostic_only=True)
    result=json.loads((directory/'_state/completion.json').read_text())
    assert result['accepted']==3 and result['newly_accepted']==1 and result['solver_attempts_exact']
    if name=='ours_response_generate':assert result['failed_inputs']>=1 and result['attempted_inputs']>=2
    assert (directory/'rounds/round_01/model.ckpt').exists()
    assert (directory/'eval/global_metrics_denorm.json').exists()
    states=sorted((directory/'_state').glob('policy_*.pt'))
    state=torch.load(states[-1],map_location='cpu',weights_only=False)
    if 'journal' in state['controller']:assert len(state['controller']['journal'])==1
    else:assert len(state['controller']['feedback']['rows'])==1
    assert not state['controller']['pending']
    assert run_generation(root,'mutation',name,tmp_path/'results',device='cpu',workers=1,
        bank_size=4,query_batch_size=1,diagnostic_only=True)==directory
