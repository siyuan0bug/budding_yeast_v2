"""Native model/ODE contracts, oracle separation and test-data perturbation guards."""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import joblib
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.ours.auxiliary.views import build_view,Inputs,Normalization
from budding_yeast_v2.utils.sampling.ours.auxiliary.pattern_classifier import PatternClassifier
from budding_yeast_v2.utils.sampling.ours.auxiliary.phenotype import describe,training_descriptors
from budding_yeast_v2.utils.sampling.ours.auxiliary.coverage import coverage_select,balanced_select
from budding_yeast_v2.utils.physics.canonical_rhs import rhs
from budding_yeast_v2.utils.sampling.ours.auxiliary.mutation_directions import probes
from budding_yeast_v2.utils.sampling.ours.auxiliary import sensitivity
from budding_yeast_v2.utils.sampling.ours.auxiliary.artifacts import write_artifacts
from tests.test_al_query_variants import inputs


def curves(count):
    y=np.ones((count,39,500),dtype=np.float32);y[:,0]=2.;y[:,2]=0.;y[:,32]=.2;y[:,33]=0.
    for i in range(count):
        c=i%6
        if c==0:y[i,0,100:]-=.4;y[i,0,300:]-=.4
        elif c==1:y[i,0,150:]-=.4
        elif c==3:y[i,33]=1.
        elif c==4:y[i,2]=.5
        elif c==5:y[i,2]=12.
    return y


class Model(torch.nn.Module):
    def __init__(self,weight):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(weight))
    def forward_ic_time(self,x,p):
        return x[:,:,0]+self.weight*x[:,:,1]*(1+p[:,:1,None].square())


class DM(SimpleNamespace):
    def __getattribute__(self,key):
        if key.startswith(('_test','_val','val_','test_','biological','dataset_path','_Y','_mutant_names')):
            raise AssertionError('Forbidden held-out information: '+key)
        return super().__getattribute__(key)


def fixture(strategy='ours_pattern_support'):
    definitions=[[('CLN2','GAL-OE',float(a))] for a in np.linspace(.06,.3,12)]
    definitions += [[('CLB2','GAL-OE',float(a))] for a in np.linspace(.01,.6,12)]
    definitions += [[('CLN3','GAL-OE',float(a))] for a in np.linspace(1.,10.,12)]
    p,y,o,m=inputs(definitions);n=18
    train=np.arange(0,36,2);pool=np.arange(1,36,2)
    engine=ActiveLearningEngine(strategy=strategy,initial_train_size=5)
    engine._pool_size=n;engine._pool_P_real=torch.tensor(p[pool]);engine._pool_P=torch.tensor(p[pool])
    engine._pool_initial_physical=torch.tensor(y[pool]);engine._pool_mass_meta=[m[i] for i in pool]
    engine._pool_ovr=[o[i] for i in pool];engine.pool.candidate_ids=pool.copy()
    engine._cum_synth_data=list(curves(n));engine._cum_synth_params=list(p[train])
    engine._cum_synth_provenance=[dict(y_init=y[i],overrides_json=json.dumps(o[i]),
        candidate_metadata_json=json.dumps(m[i]),candidate_id=int(i)) for i in train]
    dm=DM(train_dataset_synth=TensorDataset(torch.zeros(n,39,2,500),torch.tensor(p[train]),torch.tensor(curves(n))),
          mean_y=torch.zeros(1,39,1),std_y=torch.ones(1,39,1),p_mean=torch.zeros(1,141),p_std=torch.ones(1,141),
          norm_type='log1p',t_max=210.,system_id='budding',fit_scope='train_synthetic')
    members=[Model(.001),Model(.002)];engine.set_ensemble_members(members)
    engine._benchmark_real_data=[np.full((39,500),np.nan)]
    return engine,dm,members


def test_safe_view_never_accesses_full_datamodule_or_real_cache():
    engine,dm,_=fixture();view=build_view(engine,dm)
    assert len(view.train.trajectories)==18 and not hasattr(view,'datamodule')
    assert set(view.pool.ids).isdisjoint(view.train.inputs.ids)
    with pytest.raises(ValueError,match='overlap'):
        engine.pool.candidate_ids=view.train.inputs.ids.copy();build_view(engine,dm)


def test_legacy_all_synthetic_preprocessing_is_rejected():
    engine,dm,_=fixture();dm.fit_scope='all_synthetic_legacy'
    with pytest.raises(ValueError,match='train_synthetic'):build_view(engine,dm)


@pytest.mark.parametrize('strategy',['ours_pattern_balance','ours_pattern_context','ours_pattern_support'])
def test_pattern_plan_before_oracle_reproducible_and_independent_of_forbidden_data(strategy,tmp_path):
    engine,dm,members=fixture(strategy)
    with patch.object(engine,'_build_broadcast_from_pool',side_effect=AssertionError('oracle during selection')):
        state=torch.get_rng_state().clone();first=engine.select(members[0],dm,'cpu')
        engine._benchmark_real_data=None;engine._heldout={'data':'corrupted','labels':'corrupted'}
        second=engine.select(members[0],dm,'cpu')
    assert first.candidate_ids==second.candidate_ids and len(set(first.candidate_ids))==5
    torch.testing.assert_close(state,torch.get_rng_state())
    write_artifacts(engine,tmp_path,0)
    assert (tmp_path/'pattern_classifier.joblib').exists()
    estimator=joblib.load(tmp_path/'pattern_classifier.joblib')
    assert estimator.report['training_rows']==18
    assert estimator.report['calibration']['status']=='train_group_oof'
    with np.load(tmp_path/'mechanism_predictions.npz') as saved:
        assert saved['pattern_probability'].shape==(10,6)
        assert np.isfinite(saved['physical_descriptor']).all()


def test_classifier_labels_and_metadata_alignment_missing_classes_and_reload(tmp_path):
    engine,dm,_=fixture();view=build_view(engine,dm);_,labels=training_descriptors(view)
    model=PatternClassifier(42).fit(view.train.inputs,labels)
    p=model.predict(view.pool);assert p.shape==(18,6) and np.all(p>0)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
    changed=PatternClassifier(42).fit(view.train.inputs,np.zeros(len(labels),dtype=int))
    assert not np.allclose(p,changed.predict(view.pool))
    joblib.dump(model,tmp_path/'model.joblib');np.testing.assert_array_equal(p,joblib.load(tmp_path/'model.joblib').predict(view.pool))


def test_sparse_log_objective_matches_full_greedy():
    from scipy.sparse import csc_matrix
    rng=np.random.RandomState(9);k=rng.random((9,7));support=rng.random(9)+.1
    selected,gains=coverage_select(csc_matrix(k),support,5,42)
    remaining=list(range(7));baseline=[];den=support.copy()
    for _ in range(5):
        candidate=max(remaining,key=lambda i:np.log1p(k[:,i]/den).sum())
        baseline.append(candidate);remaining.remove(candidate);den+=k[:,candidate]
    assert selected.tolist()==baseline and np.all(np.diff(gains)<=1e-10)


def test_balance_microbatches_equal_probabilities_and_last_partial_batch():
    probabilities=np.ones((4000,6),dtype=np.float32)/6
    first,gains=balanced_select(probabilities,np.arange(6),1025,42)
    second,_=balanced_select(probabilities,np.arange(6),1025,42)
    assert len(set(first))==1025 and np.array_equal(first,second)
    assert np.isfinite(gains).all() and np.all(np.diff(gains)<=1e-8)
    empty,_=balanced_select(probabilities,np.arange(6),0,42);assert len(empty)==0


@pytest.mark.parametrize('sbf,special,time',[(False,False,20.),(True,False,20.),(False,True,179.),(True,True,181.)])
def test_canonical_torch_matches_current_numpy_flags_and_jvp(sbf,special,time):
    from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
    from budding_yeast_v2.data.systems.budding.ode import eqns
    backend=BuddingSystem();p=backend.defaults();p['ksc1p']=.07
    flags=dict(is_sbf_ca=sbf,is_mutant_104=special);p.update(flags)
    params=torch.tensor([[p[k] for k in backend.PARAM_KEYS]],dtype=torch.float64)
    y=torch.tensor(np.random.RandomState(3).uniform(.1,.7,(1,4,39)),dtype=torch.float64)
    output=rhs(y,params,time,flags).numpy()[0]
    np.testing.assert_allclose(output,[eqns(time,value,p) for value in y.numpy()[0]],rtol=1e-12,atol=1e-12)
    dy=y*.01;dp=params*.001
    _,jvp=torch.autograd.functional.jvp(lambda y,p:rhs(y,p,time,flags),(y,params),(dy,dp))
    eps=1e-4;finite=(rhs(y+eps*dy,params+eps*dp,time,flags)-rhs(y-eps*dy,params-eps*dp,time,flags))/(2*eps)
    torch.testing.assert_close(jvp,finite,rtol=1e-5,atol=1e-7)


def test_probe_rules_preserve_environment_and_discrete_copy_support():
    p,y,o,m=inputs([[('CLN2','GAL-OE',.12)],[('CLN2','MC',5)],
                  [('CLB2','GAL-OE',.12),('CLB2','db',.034)],[('BCK2','KO',0)]])
    block=Inputs(p,y,tuple(m),tuple(o),np.arange(4));rows=probes(block,range(4))
    assert len(rows)==3 and rows[1]['kind']=='integer_secant'
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    mdt=PARAM_KEYS.index('mdt');degradation=PARAM_KEYS.index('kdb2pp')
    for row in rows:
        assert row['p_minus'][mdt]==row['p_plus'][mdt]==p[row['row'],mdt]
    np.testing.assert_allclose(rows[2]['p_minus'][degradation],p[2,degradation],rtol=1e-6)


def test_sensitivity_residual_detects_wrong_parameter_response(monkeypatch):
    # Analytic smooth system: y'=a*y. True sensitivity solves S'=a*S+y.
    monkeypatch.setattr(sensitivity,'rhs',lambda y,p,t,flags=None:p[:,None,:1]*y)
    t=torch.linspace(0,2,500,dtype=torch.float64);a=.2;h=.001
    center=torch.exp(a*t)[None,None].expand(1,39,-1).clone()
    minus=torch.exp((a-h)*t)[None,None].expand_as(center);plus=torch.exp((a+h)*t)[None,None].expand_as(center)
    p=torch.zeros(1,141,dtype=torch.float64);p[:,0]=a;dp=torch.zeros_like(p);dp[:,0]=1.
    flags={k:torch.zeros(1,dtype=torch.bool) for k in ('is_sbf_ca','is_mutant_104')}
    dy=torch.zeros(1,39,dtype=torch.float64);delta=torch.tensor([2*h],dtype=torch.float64)
    args=(p,dp,dy,delta,torch.zeros(1,dtype=torch.bool),flags,2.,torch.ones(39,dtype=torch.float64))
    good,ratio,_=sensitivity.residual_scores(center,minus,plus,*args)
    # Perturb the sensitivity directly, preserving the nominal trajectory.
    bad,_,_=sensitivity.residual_scores(center,center-(plus-minus),center+(plus-minus),*args)
    assert good.item()<1e-4 and ratio.item()>.9 and bad.item()>100*good.item()


@pytest.mark.parametrize('gene,mode,alpha',[('CDC20','APC-A',.004),('NET1','TAB6-1',.05),
                                         ('CLB2','GAL-MC',8),('TEM1','ts',.002)])
def test_special_alleles_use_legal_rules_and_actual_initial_state(gene,mode,alpha):
    from budding_yeast_v2.data.systems.budding.simulation import prepare_initial_conditions
    p,y,o,m=inputs([[(gene,mode,alpha)]])
    rows=probes(Inputs(p,y,tuple(m),tuple(o),np.array([123])),[0])
    from budding_yeast_v2.data.systems.budding.distributions.ranges import EXACT_ZERO_TARGETS
    if (gene,mode) in EXACT_ZERO_TARGETS:
        # Reviewed fixed alleles have no legal continuous intensity probe.
        assert rows == []
        assert all(o[0][key] == 0. for key in EXACT_ZERO_TARGETS[gene,mode])
        return
    assert len(rows)==1 and rows[0]['gene']==gene
    row=rows[0]
    np.testing.assert_allclose(row['center_y'],prepare_initial_conditions(o[0])[1])
    np.testing.assert_allclose(row['dy'],(row['y_plus']-row['y_minus'])/row['delta'])
    if gene=='TEM1':
        from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
        index=list(VARIABLE_NAMES).index('TEM1')
        assert row['y_minus'][index]==row['y_plus'][index]==0.


def test_integer_response_is_exact_rhs_secant_not_tangent(monkeypatch):
    # Nonlinear parameter dependence: the finite endpoint RHS difference is
    # required for integer copy changes, even with perfect endpoint solutions.
    monkeypatch.setattr(sensitivity,'rhs',lambda y,p,t,flags=None:p[:,None,:1].square()*y)
    t=torch.linspace(0,1,500,dtype=torch.float64)
    pm=torch.zeros(1,141,dtype=torch.float64);pp=pm.clone();pm[:,0]=.2;pp[:,0]=.5
    p=(pm+pp)/2;delta=torch.tensor([.3],dtype=torch.float64)
    center=torch.exp(p[0,0].square()*t)[None,None].expand(1,39,-1)
    minus=torch.exp(pm[0,0].square()*t)[None,None].expand_as(center)
    plus=torch.exp(pp[0,0].square()*t)[None,None].expand_as(center)
    flags={key:torch.zeros(1,dtype=torch.bool) for key in ('is_sbf_ca','is_mutant_104')}
    flags.update(_p_minus=pm,_p_plus=pp)
    score,fraction,_=sensitivity.residual_scores(center,minus,plus,p,(pp-pm)/delta[:,None],
        torch.zeros(1,39,dtype=torch.float64),delta,torch.ones(1,dtype=torch.bool),flags,1.,torch.ones(39,dtype=torch.float64))
    assert score.item()<1e-5 and fraction.item()>.9


def test_initial_condition_parameter_chain_is_not_treated_as_fixed(monkeypatch):
    monkeypatch.setattr(sensitivity,'rhs',lambda y,p,t,flags=None:p[:,None,:1]*y)
    t=torch.linspace(0,2,500,dtype=torch.float64);a=.2;h=.001
    def trajectory(value):return (value*torch.exp(value*t))[None,None].expand(1,39,-1)
    center=trajectory(a);minus=trajectory(a-h);plus=trajectory(a+h)
    p=torch.zeros(1,141,dtype=torch.float64);p[:,0]=a;dp=torch.zeros_like(p);dp[:,0]=1.
    flags={k:torch.zeros(1,dtype=torch.bool) for k in ('is_sbf_ca','is_mutant_104')}
    delta=torch.tensor([2*h],dtype=torch.float64);discrete=torch.zeros(1,dtype=torch.bool)
    one=torch.ones(1,39,dtype=torch.float64);scale=one[0]
    good,_,good_ic=sensitivity.residual_scores(center,minus,plus,p,dp,one,delta,discrete,flags,2.,scale)
    bad,_,bad_ic=sensitivity.residual_scores(center,minus,plus,p,dp,torch.zeros_like(one),delta,discrete,flags,2.,scale)
    assert good.item()<1e-4 and good_ic.item()<1e-12 and bad_ic.item()>90 and bad.item()>9


def test_event_masks_shifted_divisions_and_keeps_positive_growth():
    base=torch.ones(1,39,500,dtype=torch.float64);base[:,0]=2.;base[:,2]=1.;base[:,33]=torch.linspace(2,500,500)
    values=[]
    for time in (100,90,110):
        y=base.clone();y[:,0,time:]-=.4;y[:,2,time:]=.1;values.append(y)
    mask=sensitivity.event_safe_mask(values)
    assert not mask[0,89:112].any() and mask[0,200:450].all()


def test_sensitivity_selection_is_unique_and_no_arbitrary_ko_perturbation():
    engine,dm,members=fixture('ours_sensitivity')
    view=build_view(engine,dm)
    # Shorten pool for a meaningful full RHS+model probe test.
    block=replace(view.pool,**{key:getattr(view.pool,key)[:3] for key in ('params','initial','metadata','overrides','ids')})
    small=replace(view,pool=block)
    scores,details=sensitivity.score_pool(small,members[0],'cpu')
    assert np.isfinite(scores).all() and (details['direction_kind']!='no_legal_direction').all()
    chosen,count=sensitivity.select_scores(scores,3,42);assert len(set(chosen))==3


def test_physics_plan_ignores_destroyed_benchmark_and_preserves_model_rng():
    engine,dm,members=fixture('ours_sensitivity');state=torch.get_rng_state().clone()
    first=engine.select(members[0],dm,'cpu')
    engine._benchmark_real_data='removed';engine._heldout=None
    second=engine.select(members[0],dm,'cpu')
    assert first.candidate_ids==second.candidate_ids and len(set(first.candidate_ids))==5
    torch.testing.assert_close(state,torch.get_rng_state())
