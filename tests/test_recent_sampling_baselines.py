"""Algorithm equations, frozen domain, discontinuities and runnable CPU workflows."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from budding_yeast_v2.utils.sampling.baseline.oed import BayesianOED,OEDConfig,temporal_informativeness
from budding_yeast_v2.utils.sampling.baseline.ogas import ConditionalOGAS,OGASConfig,density_ratio
from budding_yeast_v2.utils.sampling.generation.domain import FrozenBuddingDomain,project_legal_targets
from budding_yeast_v2.utils.sampling.generation.storage import GeneratedTrainingStore
from budding_yeast_v2.data.dataset_archive import TrajectoryStore,open_dataset
from budding_yeast_v2.data.systems.budding.offline_pool import Proposals,simulate
from budding_yeast_v2.utils.sampling.benchmark.protocol import prepare_protocol
from test_sampling_benchmark import make_pool,config


def test_oed_sequential_posterior_matches_closed_form_and_resume():
    x=np.random.default_rng(1).normal(size=(12,3));y=np.arange(12)/12
    cfg=OEDConfig(projection_dim=2,noise_variance=.3,prior_variance=2.)
    a=BayesianOED(x,42,cfg);phi=a.features(x);a.observe(x,y)
    expected=np.linalg.inv(np.eye(phi.shape[1])/2+phi.T@phi/.3)
    np.testing.assert_allclose(a.covariance,expected,atol=1e-12)
    np.testing.assert_allclose(a.coefficients,expected@phi.T@y/.3,atol=1e-12)
    b=BayesianOED(x,1,cfg);b.load_state_dict(a.state_dict())
    np.testing.assert_array_equal(a.scores(x),b.scores(x))
    assert a.observations==12


def test_oed_temporal_metric_resolves_spectral_content_without_biological_changes():
    t=np.arange(128)/128
    y=np.stack((np.sin(2*np.pi*t),np.sin(96*np.pi*t)))[:,None,:]
    before=y.copy();score=temporal_informativeness(y)
    assert score[1]>score[0]
    np.testing.assert_array_equal(y,before)
    assert temporal_informativeness(np.ones((1,3,128)))[0]==0
    with pytest.raises(ValueError):temporal_informativeness(y,channel_scale=np.array([0.]))


def test_ogas_density_ratio_direction_and_protection():
    actual=density_ratio(torch.tensor([.2,.5,.8,1.,0.],dtype=torch.float64))
    torch.testing.assert_close(actual,torch.tensor([.25,1.,4.,10.,.1],dtype=torch.float64))


def test_ogas_classifier_free_null_is_distinct_from_zero_signal():
    from budding_yeast_v2.utils.sampling.baseline.ogas import ConditionalDenoiser
    model=ConditionalDenoiser(3,12)
    x=torch.zeros(2,3);t=torch.ones(2)/2;signal=torch.zeros(2)
    conditional=model(x,t,signal)
    unconditional=model(x,t,signal,torch.ones(2,dtype=torch.bool))
    assert not torch.equal(conditional,unconditional)
    assert torch.isfinite(unconditional).all()


def test_ogas_conditional_training_mixture_and_exact_rng_resume():
    x=np.random.default_rng(2).normal(size=(20,4)).astype(np.float32)
    cfg=OGASConfig(width=12,diffusion_steps=4,updates=3,batch_size=8,capacity=12)
    rng_state=torch.get_rng_state().clone()
    a=ConditionalOGAS(x,42,cfg)
    torch.testing.assert_close(torch.get_rng_state(),rng_state)
    a.observe(x,np.linspace(0,2,len(x)));assert len(a.inputs)==12
    report=a.train_cycle(x);assert np.isfinite(report['diffusion_loss'])
    assert cfg.minimum_weight<=report['weight_min']<=report['weight_max']<=cfg.maximum_weight
    state=deepcopy(a.state_dict());b=ConditionalOGAS(x,42,cfg);b.load_state_dict(state)
    p,da=a.sample_targets(8,x);q,db=b.sample_targets(8,x)
    np.testing.assert_array_equal(p,q);assert da==db and np.isfinite(p).all()


def test_legal_domain_matches_unmodified_canonical_proposals_and_exclusions(tmp_path):
    canonical=Proposals('mutation',42,100000,tmp_path,None)
    expected,_=canonical.propose(0)
    a=FrozenBuddingDomain('mutation',42,tmp_path/'a');rows,sigs,view=a.bank(3)
    assert rows[0].overrides==expected[0].overrides
    a.verify();a.consume(sigs[0])
    with pytest.raises(ValueError):a.consume(sigs[0])
    b=FrozenBuddingDomain('mutation',42,tmp_path/'b',forbidden=[sigs[0]])
    _,other,_=b.bank(2);assert sigs[0] not in other and b.counts['holdout_collision']==1
    np.testing.assert_array_equal(view.initial[0],a.proposals.backend.prepare_initial_conditions(rows[0].overrides)[1])


def test_projection_only_returns_unique_existing_legal_points():
    legal=np.array([[0.,0.],[1.,1.],[2.,2.]])
    before=legal.copy()
    selected=project_legal_targets(np.array([[.1,.1],[.2,.2]]),legal,np.zeros(2),np.ones(2))
    assert selected.tolist()==[0,1];np.testing.assert_array_equal(before,legal)


def test_generator_rejects_changed_biological_catalog_and_lhs_specification(tmp_path):
    from types import SimpleNamespace
    from budding_yeast_v2.utils.sampling.generation.cohorts import verify_frozen_rules
    from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    from budding_yeast_v2.data.dataset_archive import file_sha256
    spec=tmp_path/'spec.json';spec.write_text('{"frozen":true}')
    values=dict(catalog_version=CATALOG_VERSION,sampling_range_fingerprint=RANGE_FINGERPRINT,
                generation_config=json.dumps(dict(lhs_spec_sha256=file_sha256(spec))))
    source=SimpleNamespace(fields=values,scalar=lambda key:np.array(values[key]))
    verify_frozen_rules(source,'lhs',spec)
    spec.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='exactly match'):verify_frozen_rules(source,'lhs',spec)
    values['catalog_version']='other'
    with pytest.raises(ValueError,match='rule mismatch'):verify_frozen_rules(source,'mutation',None)


def test_pre_does_not_differentiate_across_true_jump(monkeypatch):
    from budding_yeast_v2.utils.sampling.baseline import physics_pre as pre
    monkeypatch.setattr(pre,'rhs',lambda y,p,t:torch.zeros_like(y))
    y=np.zeros((1,39,100));y[:,0,:50]=2.;y[:,0,50:]=1.
    p=np.ones((1,141))
    assert pre.residual_scores(y,p)[0]==0.
    y[:,1]=np.linspace(0,1,100)
    assert pre.residual_scores(y,p)[0]>0
    # Continuous PRE alone cannot certify a correct jump/reset magnitude.
    y[:,1]=0;y[:,0,50:]=.3
    assert pre.residual_scores(y,p)[0]==0.


@pytest.mark.parametrize('horizon',[0.,-1.,float('nan')])
def test_pre_rejects_invalid_time_without_silently_zeroing(horizon):
    from budding_yeast_v2.utils.sampling.baseline.physics_pre import residual_scores
    with pytest.raises(ValueError):residual_scores(np.zeros((1,39,10)),np.ones((1,141)),horizon)


def test_generated_storage_references_initial_rows_and_rounds_do_not_copy_trajectories(tmp_path):
    source,arrays=make_pool(tmp_path/'pool')
    initial=np.array([27,1,9])
    store=GeneratedTrainingStore(source,initial,tmp_path/'generated','mutation',42)
    manifest=store.publish(0);data=TrajectoryStore(manifest).read(np.arange(3),('params','y_init','data'))
    for key,value in data.items():np.testing.assert_array_equal(value,arrays[key][initial])
    assert not list((tmp_path/'generated/chunks').glob('*.npz'))
    with open_dataset(manifest) as a:
        np.testing.assert_array_equal(a['data'],arrays['data'][initial])
    text=json.loads(manifest.read_text());assert text['provenance']['source_trajectories_copied'] is False


def test_generated_storage_preserves_field_specific_numerical_overlays(tmp_path):
    from budding_yeast_v2.data.dataset_repairs import write_row_patch
    source,arrays=make_pool(tmp_path/'pool')
    ids=np.array([27,0,9]);changed=arrays['data'][ids].copy();changed[:,2]*=2
    patched=write_row_patch(source,tmp_path/'patched/dataset.json',ids,
        dict(data=changed,labels=np.array(['longer patched label']*3)),provenance={'test':True})
    initial=np.array([9,1,27,0]);expected=arrays['data'].copy();expected[ids]=changed
    store=GeneratedTrainingStore(patched,initial,tmp_path/'generated','mutation',42)
    path=store.publish(0)
    with open_dataset(path) as archive:
        np.testing.assert_array_equal(archive['data'],expected[initial])
        np.testing.assert_array_equal(archive['params'],arrays['params'][initial])
        assert archive['labels'][0]=='longer patched label'


def test_generation_replenishes_failed_queries_and_charges_all_attempts(tmp_path,monkeypatch):
    from budding_yeast_v2.utils.sampling.generation import runner
    from budding_yeast_v2.utils.sampling.generation.policies import GenerationPolicy
    from budding_yeast_v2.data.systems.contracts import SimulationResult
    from budding_yeast_v2.utils.sampling.contracts import PoolInputs
    source,_=make_pool(tmp_path/'pool');store=GeneratedTrainingStore(source,[1,2],tmp_path/'data','mutation',42)
    initial=PoolInputs(np.arange(2),store.array('params'),store.array('y_init'))
    policy=GenerationPolicy('random_generate',initial,store.array('data'))
    domain=FrozenBuddingDomain('mutation',42,tmp_path/'proposals');(tmp_path/'_state').mkdir()
    calls=[]
    def simulated(executor,function,tasks,**kwargs):
        for c in tasks:
            p,y0,_,_=domain.proposals.backend.prepare_initial_conditions(c.overrides)
            params=np.array([c.overrides.get(k,domain.proposals.defaults[k]) for k in domain.proposals.backend.spec.param_names])
            failed=not calls;calls.append(c.name)
            yield SimulationResult(c,None if failed else np.repeat(y0[:,None],500,1),params,y0,
                'Inviable: G1 Arrest','Pattern_C: Exponential-Arrest (==0)',
                'numerical_failure_or_timeout' if failed else 'success',{'attempts':[{'success':not failed}]})
    monkeypatch.setattr(runner,'bounded_ordered_map',simulated)
    runner._acquire(4,domain,store,policy,None,None,tmp_path,bank_size=2,query_batch_size=1,max_attempts_mult=5)
    assert store.count==4 and len(calls)==3
    records=[json.loads(x) for x in (tmp_path/'_state/ode_attempts.jsonl').read_text().splitlines()]
    assert sum(x['event']=='start' for x in records)==3
    assert sum(x.get('status')=='numerical_failure_or_timeout' for x in records)==1
    assert len(store.chunks)==1  # flush one source block, not one NPZ per query
    assert len(list((tmp_path/'_state').glob('policy_*.pt')))<=2


@pytest.mark.parametrize('method',['random_generate','oed_temporal','ogas_round_projected'])
def test_actual_generation_fits_evaluates_and_resumes_cpu(tmp_path,method,monkeypatch):
    # Tiny complete workflow, actual canonical ODE, one fresh label per method.
    from budding_yeast_v2.utils.sampling.generation.runner import run_generation
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')
    monkeypatch.setenv('SAMPLING_MAX_LOAD_PER_CPU','100')
    source,_=make_pool(tmp_path/'pool');cfg=replace(config(),budgets=(2,3),epochs=(1,1),warmup=(0,0))
    root=prepare_protocol({'mutation':source,'lhs':source},tmp_path/'protocol',cfg)
    kwargs=dict(workers=1,bank_size=4,query_batch_size=1,diagnostic_only=True,
                oed_config=dict(projection_dim=2),
                ogas_config=dict(paper_version='arxiv_v2',width=12,diffusion_steps=4,updates=2,batch_size=4))
    result=run_generation(root,'mutation',method,tmp_path/'results',**kwargs)
    done=json.loads((result/'_state/completion.json').read_text())
    assert done['accepted']==3 and done['newly_accepted']==1
    assert done['attempted_inputs']>=1 and done['solver_attempts']>=done['attempted_inputs']
    assert done['sources_unchanged'] and not done['native_online_ogas']
    snapshot=json.loads((result/'rounds/round_01/dataset.json').read_text())
    values=TrajectoryStore(snapshot['source']).read(snapshot['rows'],('data','params','y_init'))
    assert values['data'].shape==(3,39,500) and np.isfinite(values['data']).all()
    assert (result/'eval/global_metrics_denorm.json').exists()
    assert '[Epoch ' in (result/'train.log').read_text()
    assert run_generation(root,'mutation',method,tmp_path/'results',**kwargs)==result
    history=result/'rounds/round_01/history.csv';history.write_text('corrupt')
    with pytest.raises(ValueError,match='artifact changed'):
        run_generation(root,'mutation',method,tmp_path/'results',**kwargs)
