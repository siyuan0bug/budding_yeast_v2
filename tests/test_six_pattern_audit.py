"""Scientific contract tests for the experimental, read-only pattern audit."""
import numpy as np
import pandas as pd
import pytest

from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import (
    CODES, chunks, decide, features, make_splits, run_one)
from budding_yeast_v2.data.systems.budding.phenotypes import analyze_mass_pattern
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_pure_random_evidence import split_indices


def trajectory():
    y = np.zeros((39, 60), dtype=np.float64)
    y[0] = 2.; y[2] = .1; y[32] = .5; y[33] = .1; y[35] = .1
    return y


@pytest.mark.parametrize('code,bmax,bend,eend,omax',[
    ('P3',.1,.1,.5,.49), ('P4',.1,.1,.5,.5),
    ('P5a',.35,.35,.5,.1), ('P5a',.1,.1,.099,.1),
    ('P5a',10.,5.,.5,.1), ('P5b',10.01,.1,.5,.1),
    ('P5b',5.01,5.01,.5,.1),
])
def test_exact_branch_boundaries(code,bmax,bend,eend,omax):
    y=trajectory(); y[2,10]=bmax; y[2,-1]=bend; y[32,-1]=eend; y[33,10]=omax
    f,_=features(y[None]); assert f['pattern_code'][0] == code
    g,_=features(y.T[None])
    for key in f: np.testing.assert_equal(f[key],g[key])


def test_drop_chain_suppression_matches_production_and_attachment():
    y=trajectory()
    for drop in (0,6,12): y[0,drop+1:] -= .2
    f,_=features(y[None])
    assert f['raw_mass_drops'][0]==3
    assert f['divisions'][0]==1  # Anchored-deadtime implementation would count 2.
    assert f['pattern_code'][0]=='P2'
    assert analyze_mass_pattern(y[0]).startswith('Pattern_B')


def test_multi_division_priority_and_hyper_flag_are_separate():
    y=trajectory(); y[0,12:] -= .2; y[0,40:] -= .2; y[2]=12
    f,_=features(y[None])
    assert f['pattern_code'][0]=='P1' and f['hyper_clb2_any_v'][0]


def test_p4_prose_is_not_the_executable_rule():
    y=trajectory(); y[33]=3; y[35]=2
    f,_=features(y[None])
    assert f['pattern_code'][0]=='P4'
    assert f['p4_prose_inconsistent'][0]
    assert f['prose_rule_matches'][0]==0


@pytest.mark.parametrize('index', [0,2,10,32,38])
def test_nonfinite_is_invalid_not_an_arrest(index):
    y=trajectory(); y[index,10]=np.nan
    f,_=features(y[None])
    assert not f['valid'][0] and f['pattern_code'][0]=='INVALID'
    assert not f['is_bifurcation_critical'][0]


def test_float32_boundaries_use_python_float_reference():
    y=trajectory().astype(np.float32); y[2]=.35
    f,_=features(y[None])
    assert float(y[2,-1]) < .35  # float32 representation is slightly below .35.
    assert f['pattern_code'][0]=='P3'


def test_all_finite_feature_combinations_have_one_code():
    import itertools
    grid=np.array(list(itertools.product((0,1,2),(.1,.35,5,10,11),(.1,.35,5,6),(.09,.1,.5),(.49,.5,1))))
    f=dict(zip(('divisions','max_clb2','end_clb2','end_esp1','max_ori'),grid.T))
    f['valid']=np.ones(len(grid),bool)
    code,_,_=decide(f); assert np.isin(code,CODES).all()


def test_split_membership_matches_current_nonlhs_rule():
    names=np.array(['22_real']+[f'RANDOM_L1_{i}' for i in range(33)])
    patterns=np.array(['Pattern_A']*12+['Pattern_B']*12+['Pattern_C']*10)
    real,synth,train,val,test=split_indices(names,patterns,42)
    split=make_splits(names,patterns,np.arange(len(names))==0,42)
    for name,indices in [('test_real',real),('train_synth',train),('val_synth',val),('test_synth',test)]:
        np.testing.assert_equal(np.flatnonzero(split==name),np.sort(indices))


def test_streaming_and_name_join_without_source_mutation(tmp_path):
    y=np.stack([trajectory() for _ in range(4)]).astype(np.float32)
    path=tmp_path/'small.npz'; names=np.array(['42_real','22_real','SYN_L1_0','SYN_L1_1'])
    np.savez_compressed(path,data=y,mutant_names=names,params=np.zeros((4,141)),
        pattern_labels=np.array(['Pattern_C: test']*4),labels=np.array(['test']*4))
    raw=path.read_bytes()
    np.testing.assert_equal(np.concatenate([b for _,b in chunks(path,3)]),y)
    metrics=tmp_path/'metrics.csv'
    pd.DataFrame({'Mutant':['22_real','42_real'],'MAE':[22.,42.]}).to_csv(metrics,index=False)
    run_one((str(path),str(tmp_path/'out'),3,42,'small.npz',str(metrics)))
    joined=pd.read_csv(tmp_path/'out/small/real_evaluation.csv').set_index('mutant_name')
    assert joined.loc['42_real','MAE']==42. and joined.loc['22_real','MAE']==22.
    assert path.read_bytes()==raw


def test_bad_shape_rejected():
    with pytest.raises(ValueError):features(np.zeros((1,39,39)))
    with pytest.raises(ValueError):features(np.zeros((1,38,500)))


def test_badcase_rank_uses_errors_not_correlation_and_is_order_independent():
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.summarize_six_pattern_audit import select_badcases
    d=pd.DataFrame({'Mutant':['15_real','124_real','42_real'],
        'MAE':[.6,3.3,.2],'MSE':[4.8,557,.6], 'Relative_L2':[7.5,.32,.4],
        'Correlation':[.99,.99,-1.]})
    r=select_badcases(d,2);s=select_badcases(d.iloc[::-1],2)
    assert set(r.loc[r.selected_top_k,'Mutant'])=={'15_real','124_real'}
    assert r.Mutant.tolist()==s.Mutant.tolist()
    with pytest.raises(ValueError):select_badcases(d.assign(MAE=np.nan))


def test_frozen_predictions_require_exact_current_target_digest(tmp_path):
    from budding_yeast_v2.scripts.analysis.dataset_diagnosis.summarize_six_pattern_audit import inspect_frozen_predictions
    file=tmp_path/'frozen.npz'
    y=np.zeros((1,39,500),np.float32)
    np.savez_compressed(file,predictions=y,targets=y,names=np.array(['15_real']))
    real=pd.DataFrame({'mutant_name':['15_real'],'trajectory_sha256_float32':['wrong']})
    with pytest.raises(ValueError,match='targets differ'):
        inspect_frozen_predictions(file,real,tmp_path)
