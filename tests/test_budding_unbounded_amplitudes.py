"""Budding finite amplitudes survive static/AL ingestion; invalid solves do not."""
from concurrent.futures import Future
from types import SimpleNamespace
import hashlib,json,time
from pathlib import Path

import numpy as np
import pytest
import torch

from budding_yeast_v2.data.systems.budding import simulation
from budding_yeast_v2.data.systems.budding.adapter import ADAPTER
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.contracts import Candidate,SimulationResult
from budding_yeast_v2.data.systems import registry
from budding_yeast_v2.utils.sampling import ActiveLearningEngine,oracle


def test_completed_mass_above_ten_is_retained():
    # A solver boundary test outside the default MDT catalog, not a change to it.
    overrides={'mdt':30.,'ks20p':0.,'ks20pp':0.}
    _,y,bio,pattern=simulation.simulate_mutant('mass_boundary',overrides)
    assert y is not None and np.isfinite(y).all() and y.shape==(39,500)
    assert y[0].max()>10
    assert bio.startswith('Inviable:') and pattern is not None


def test_supported_high_clb2_case_matches_independent_calibration():
    path=Path(__file__).resolve().parents[1]/'scripts/analysis/outputs/budding_extremes_2026-10-22_targeted'
    if not path.exists():pytest.skip('Local calibration artifact not installed')
    candidate=next(json.loads(s) for s in (path/'candidates.jsonl').read_text().splitlines()
        if '"index": 86,' in s)
    _,y,bio,pattern=simulation.simulate_mutant(candidate['name'],candidate['overrides'])
    assert np.array_equal(y,np.load(path/'trajectories.npy',mmap_mode='r')[86])
    assert y[2].max()>400
    assert bio=='Inviable: M-phase Arrest'


def test_wt_blind_does_not_reject_large_finite_trajectory():
    result=SimulationResult(Candidate(1,'CLB2[MC]',{},1),np.full((39,500),2000.),
        np.ones(141),np.ones(39),'Inviable: M-phase Arrest','Pattern_C','success')
    state={'recipe':registry.RECIPES['budding_wt_blind']}
    assert ADAPTER.accept_result(result,state,None)


@pytest.mark.parametrize('fault',['failed','truncated','nonfinite'])
def test_numerically_invalid_budding_still_rejected(monkeypatch,fault):
    def broken(fun,span,y0,**kwargs):
        y=np.tile(y0[:,None],(1,2));t=np.array([0.,210.])
        if fault=='truncated':t[-1]=105.
        if fault=='nonfinite':y[1,-1]=np.nan
        return SimpleNamespace(y=y,t=t,status=0,success=fault!='failed')
    monkeypatch.setattr(simulation,'solve_ivp',broken)
    assert simulation.simulate_mutant('bad',{})[1] is None


def test_timeout_still_discards_sample(monkeypatch):
    from budding_yeast_v2.data import simulation_guards
    monkeypatch.setattr(simulation_guards,'DEFAULT_SIM_TIMEOUT',.02)
    monkeypatch.setattr(simulation,'solve_ivp',lambda *a,**k:time.sleep(.2))
    started=time.monotonic()
    assert simulation.simulate_mutant('slow',{})[1] is None
    assert time.monotonic()-started<.15


@pytest.mark.parametrize('quota',[False,True])
def test_al_single_and_refill_keep_large_values_and_skip_bad_outputs(monkeypatch,quota):
    count=5;engine=ActiveLearningEngine(strategy='al4pde_random')
    engine._pool_size=count
    engine._pool_mutant_names=np.array([f'row{i}' for i in range(count)])
    engine._pool_P_real=torch.ones(count,141);engine._pool_P=torch.ones(count,141)
    engine._pool_X=torch.zeros(count,39,2,500);engine._pool_ovr=[{} for _ in range(count)]
    engine._pool_mass_meta=[{'level':1} for _ in range(count)]
    # No real dataset is needed for a numeric-validity gate.
    dm=SimpleNamespace(norm_type='log1p',mean_y=torch.zeros(1,39,1),std_y=torch.ones(1,39,1),
        p_mean=torch.zeros(1,141),p_std=torch.ones(1,141),s_c=None)
    large=np.full((39,500),2000.);large[2,0]=-1e-7
    nonfinite=large.copy();nonfinite[1,20]=np.inf
    results=[None,(None,nonfinite,np.ones(141),'Pattern_C'),
        (None,np.ones((39,499)),np.ones(141),'Pattern_C'),
        (None,large,np.full(141,np.nan),'Pattern_C'),
        (None,large,np.ones(141),'Pattern_C','Inviable: M-phase Arrest')]
    class FakeExecutor:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def submit(self,fn,task):
            future=Future();future.set_result(results[int(task[0][3:])]);return future
    monkeypatch.setattr(oracle,'ProcessPoolExecutor',FakeExecutor)
    batch=engine._build_broadcast_from_pool(np.arange(count),dm,target_valid_num=1 if quota else None)
    assert batch['names']==['row4'] and len(batch['Y_real'])==1
    assert batch['Y_real'][0][2,0]==-1e-7
    assert batch['Y_real'][0].max()==2000.
    assert torch.isfinite(batch['Y']).all()
    assert engine._pool_size==0


def test_protected_originals_fission_normalization_and_ours_unchanged():
    root=Path(__file__).resolve().parents[1]
    path=root/'scratch/probes/remove_budding_amplitude_caps_2026-10-22/protected.json'
    if not path.exists():pytest.skip('Local pre-change hash manifest is unavailable')
    for p,h in json.loads(path.read_text()).items():
        # The later collision-policy task intentionally changes only the real
        # deduplication helpers; the table itself is protected in catalog tests.
        if p=='data/systems/budding/real_mutants.py':continue
        assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
