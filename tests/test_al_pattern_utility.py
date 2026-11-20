"""Protect identity holdouts, hierarchy and experiment bookkeeping."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from budding_yeast_v2.scripts.analysis.dataset_diagnosis.compare_al_pattern_labels import (
    SCHEMES,crossfit,expected_topk_hits,group_prediction,mutant_folds,
    read_evaluations,schemes_from_labels)


def label_frame():
    n=30
    return pd.DataFrame(dict(mutant_name=[f'{i}_real' for i in range(n)],
        original_pattern=['Pattern_A: test']*10+['Pattern_B: test']*10+['Pattern_C: test']*10,
        original_label=['Viable']*15+['Inviable']*15,
        pattern_code=['P1']*10+['P2']*10+['P3']*3+['P4']*2+['P5a']*3+['P5b']*2,
        is_bifurcation_critical=np.arange(n)%7==0))


def test_no_heldout_algorithm_or_mutant_outcome_leaks_into_prediction():
    labels=schemes_from_labels(label_frame())
    y=np.random.RandomState(8).rand(3,30,4)
    folds=mutant_folds(labels['ABC'],1)
    before=crossfit(y,labels,folds)
    test=folds[0]==0
    # Change the entire target algorithm plus every test identity in the other
    # algorithms. None of these values may influence that held-out block.
    changed=y.copy();changed[0]=1000.;changed[:,test,:]=-1000.
    after=crossfit(changed,labels,folds)
    for scheme in SCHEMES:
        np.testing.assert_allclose(before[scheme][0,0,test],after[scheme][0,0,test],atol=1e-11)


def test_six_refinement_preserves_unsplit_p1_p2_predictions_exactly():
    labels=schemes_from_labels(label_frame());y=np.random.RandomState(9).rand(3,30,4)
    pred=crossfit(y,labels,mutant_folds(labels['ABC'],2))
    np.testing.assert_array_equal(pred['SIX'][:,:,:20],pred['ABC'][:,:,:20])
    np.testing.assert_array_equal(pred['SIX_BIO'][:,:,:20],pred['ABC_BIO'][:,:,:20])


def test_unseen_child_falls_back_without_nonfinite_predictions():
    child=np.array(['a','b']);parent=np.array(['all','all'])
    source=np.array([[[.2],[.9]]]);prior=np.full_like(source,.4)
    out=group_prediction(child,parent,source,np.array([True,False]),prior,0.)
    assert out[0,1,0]==.4 and out[0,0,0]==.2


def test_child_must_refine_parent():
    with pytest.raises(ValueError,match='refine'):
        group_prediction(np.array(['one','one']),np.array(['a','b']),np.zeros((2,2,1)),
            np.array([True,False]),np.zeros((2,2,1)),5.)


def test_tied_scores_cannot_benefit_from_mutant_order():
    score=np.ones(126);bad=np.arange(126)<20
    assert expected_topk_hits(score,bad)==pytest.approx(400/126)
    assert expected_topk_hits(score,bad[::-1])==pytest.approx(400/126)
    assert expected_topk_hits(np.arange(126),np.arange(126)>=106)==20


def test_folds_do_not_use_error_or_new_label_and_hold_out_every_identity():
    abc=np.array(['A']*10+['B']*10+['C']*10)
    folds=mutant_folds(abc,3,k=5,seed=1)
    np.testing.assert_array_equal(folds,mutant_folds(abc,3,k=5,seed=1))
    for row in folds:
        for group in np.unique(abc):
            np.testing.assert_array_equal(np.bincount(row[abc==group],minlength=5),np.full(5,2))


def test_duplicate_baselines_aliases_and_failed_runs_are_not_independent(tmp_path):
    real=label_frame().iloc[:3].copy()
    base=pd.DataFrame({'Mutant':real.mutant_name,'Pattern':real.original_pattern,
        'MAE':[.1,.2,.3],'MSE':[.1,.3,.5],'Relative_L2':[.1,.2,.4],'Correlation':[.9,.8,.7]})
    for strategy in ('al4pde_random','al4pde_diversity'):
        run=tmp_path/f'hybrid_mse_rel_{strategy}/w_test/seed_42';run.mkdir(parents=True)
        (run/'config.json').write_text(json.dumps({'args':{'al_strategy':strategy,'seed':42,'loss_type':'hybrid_mse_rel'}}))
        for dirname in ('round_0_train7996','round_1_train17996',''):
            out=run/'eval'/dirname;out.mkdir(parents=True,exist_ok=True)
            base.iloc[::-1].to_csv(out/'per_mutant_metrics.csv',index=False)
    failed=tmp_path/'hybrid_mse_rel_al4pde_residual/w_test/seed_42_failed_v1'
    (failed/'eval/round_1_train9633').mkdir(parents=True)
    (failed/'config.json').write_text(json.dumps({'args':{'al_strategy':'al4pde_residual','loss_type':'hybrid_mse_rel'}}))
    base.to_csv(failed/'eval/round_1_train9633/per_mutant_metrics.csv',index=False)
    out=tmp_path/'out';out.mkdir()
    frame,manifest=read_evaluations(tmp_path,real,out)
    assert len(frame)==9  # One common initial model + two actual post-AL models.
    assert sum(r['status']=='duplicate_shared_round0' for r in manifest)==1
    assert sum(r['status']=='endpoint_alias_or_extra' for r in manifest)==2
    assert sum(r['status']=='excluded_failed_run' for r in manifest)==1
