"""Regression checks for the authorized MA closures, excluding CDC14-ts."""
import hashlib
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest

from budding_yeast_v2.data.systems.budding import catalog
from budding_yeast_v2.data.systems.budding.distributions.ranges import EXACT_ZERO_TARGETS
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.simulation import prepare_initial_conditions
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.budding.quotas import enumerate_combinations
from budding_yeast_v2.data.systems.budding.al_pool import finite_options
from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'scratch/probes/exact_zero_update_20261112'


@pytest.mark.parametrize('gene,mode',list(EXACT_ZERO_TARGETS))
def test_random_quantile_and_direct_assignment_all_enforce_zero(gene,mode):
    backend=BuddingSystem();defaults=backend.defaults()
    for seed in (0,7,42):
        rng=np.random.RandomState(seed);before=rng.get_state()
        assert backend.sample_scaling_factor(mode,gene,rng=rng)==0.
        after=rng.get_state()
        assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]
    for u in (0.,.1,.5,.9,1.):
        assert backend.sample_scaling_factor_sobol(mode,gene,u)==0.
    for alpha in (0.,.001,.1,1.,30.):
        overrides={};backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),overrides)
        assert all(overrides[key]==0. for key in EXACT_ZERO_TARGETS[gene,mode])
        expected=set(EXACT_ZERO_TARGETS[gene,mode])
        if (gene,mode)==('APC_Core','ts'):
            expected|={'init_CDH1T','init_CDH1'}
        assert set(overrides)==expected
        if (gene,mode)!=('APC_Core','ts'):
            np.testing.assert_array_equal(prepare_initial_conditions(overrides)[1],prepare_initial_conditions({})[1])
        row=Candidate(1,'x',overrides,1,metadata={'description':f'{gene}[{mode}]'})
        assert valid_candidate(row,'budding') is not None
        assert valid_candidate(Candidate(1,'x',dict(overrides,**{EXACT_ZERO_TARGETS[gene,mode][0]:1e-5}),1,
                                         metadata=row.metadata),'budding') is None


def test_unaffected_modes_mass_quantiles_and_rng_match_saved_baseline():
    if not (SNAPSHOT/'before_observations.pkl').exists():pytest.skip('Before snapshot not available')
    with (SNAPSHOT/'before_observations.pkl').open('rb') as handle:observations,quantiles=pickle.load(handle)
    for (profile,gene,mode,seed),(alpha,overrides,params,state) in observations.items():
        if profile=='mutant_aware' and (gene,mode) in EXACT_ZERO_TARGETS:continue
        backend=BuddingSystem(profile,profile);defaults=backend.defaults();p=defaults.copy();o={};rng=np.random.RandomState(seed)
        a=backend.sample_scaling_factor(mode,gene,rng=rng)
        backend.apply_gene_mutation(gene,mode,a,defaults,p,o,rng=rng)
        current=rng.get_state()
        assert (a,o,p)==(alpha,overrides,params),(profile,gene,mode)
        assert current[0]==state[0] and np.array_equal(current[1],state[1]) and current[2:]==state[2:]
    for (profile,gene,mode),values in quantiles.items():
        if profile=='mutant_aware' and (gene,mode) in EXACT_ZERO_TARGETS:continue
        actual=[BuddingSystem(profile,profile).sample_scaling_factor_sobol(mode,gene,u) for u in (0.,.1,.5,.9,1.)]
        assert actual==values,(profile,gene,mode)


def test_CDC14_both_branches_are_exactly_preserved():
    if not (SNAPSHOT/'before_sources.zip').exists():pytest.skip('Before source snapshot not available')
    namespace={'__package__':'budding_yeast_v2.data.systems.budding.mutations'}
    with zipfile.ZipFile(SNAPSHOT/'before_sources.zip') as archive:
        exec(compile(archive.read('data/systems/budding/mutations/mutant_aware.py'),'<before>','exec'),namespace)
    backend=BuddingSystem();defaults=backend.defaults();branches=set()
    for seed in range(100):
        old_rng=np.random.RandomState(seed);new_rng=np.random.RandomState(seed)
        alpha=backend.sample_scaling_factor('ts','CDC14',rng=new_rng)
        other=backend.sample_scaling_factor('ts','CDC14',rng=old_rng);assert alpha==other
        old={};new={}
        namespace['apply_gene_mutation']('CDC14','ts',alpha,defaults,defaults.copy(),old,rng=old_rng)
        backend.apply_gene_mutation('CDC14','ts',alpha,defaults,defaults.copy(),new,rng=new_rng)
        assert old==new
        assert np.array_equal(old_rng.get_state()[1],new_rng.get_state()[1]) and old_rng.get_state()[2:]==new_rng.get_state()[2:]
        branches.add('synthesis' if new['ks14']<.01 else 'activity')
    assert branches=={'synthesis','activity'}


def test_fixed_modes_have_finite_unique_capacity_and_no_response_coordinate():
    from budding_yeast_v2.utils.sampling.ours.response import _strength_curve
    cells,_=enumerate_combinations()
    backend=BuddingSystem()
    for (gene,mode),keys in EXACT_ZERO_TARGETS.items():
        cell=next(r for r in cells if r['alleles']==f'{gene}[{mode}]')
        assert cell['finite_capacity_upper']==3 and not cell['has_continuous_strength']
        support=finite_options(cell,backend)
        assert support is not None and len(support)<=3
        assert all(all(row[1][key]==0 for key in keys) for row in support)
        assert _strength_curve(gene,mode,'')==((),None,None)
    a={};b={}
    backend.apply_gene_mutation('CDC20','KO',0.,backend.defaults(),{},a)
    backend.apply_gene_mutation('CDC20','ts',.7,backend.defaults(),{},b)
    np.testing.assert_array_equal(effective_inputs(a,backend.defaults()),effective_inputs(b,backend.defaults()))


@pytest.mark.parametrize('gene,mode',list(EXACT_ZERO_TARGETS))
def test_wt_blind_cannot_restore_residual_values(monkeypatch,gene,mode):
    from budding_yeast_v2.data.systems.budding.samplers import wt_blind
    monkeypatch.setattr(wt_blind,'MODE_FREQ','flat')
    monkeypatch.setattr(wt_blind,'sample_skeleton',lambda *args:([gene],'test'))
    monkeypatch.setattr(wt_blind,'next_L1_cell',lambda:(gene,mode))
    candidate=wt_blind.build_candidate(1,{},ma=BuddingSystem())
    assert all(candidate['ovr'][key]==0 for key in EXACT_ZERO_TARGETS[gene,mode])


def test_originals_fission_canonical_biology_and_mass_sources_are_unchanged():
    if not (SNAPSHOT/'protected.json').exists():pytest.skip('Before hash manifest not available')
    for name,digest in json.loads((SNAPSHOT/'protected.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
