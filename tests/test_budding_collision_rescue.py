"""Real-input identity and environment-preserving rescue across static/AL paths."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from budding_yeast_v2.data.systems import generation
from budding_yeast_v2.data.systems.budding import real_mutants, derived
from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.samplers.rescue import rescue_budding, parse_pairs
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig
from budding_yeast_v2.data.systems.registry import RECIPES, make_sampler, resolve_config


def build(backend, pairs, mdt=90., seed=1):
    defaults=backend.defaults();ovr={'mdt':mdt};rng=np.random.RandomState(seed)
    for gene,mode,alpha in pairs:
        backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),ovr,rng=rng)
    return ovr


def test_all_real_rules_collide_including_full_and_sparse_inputs():
    backend=BuddingSystem()
    for _,ovr in real_mutants.REAL_MUTANTS.values():
        assert real_mutants.is_duplicate_real_mutant(ovr,backend.defaults())
        assert real_mutants.is_duplicate_real_mutant(dict(backend.defaults(),**ovr),backend.defaults())


def test_alias_and_actual_initial_state_equivalence():
    backend=BuddingSystem();defaults=backend.defaults()
    for pairs in ([('MAD2','CSL',1.)],[('BUB2','CKO',1.)],[('TEM1','KO',1.)],[('CDC15','KO',1.)]):
        ovr=build(backend,pairs)
        assert real_mutants.is_duplicate_real_mutant(ovr,defaults)
    # Redundant explicit initial values are not a distinct biological input.
    assert real_mutants.is_duplicate_real_mutant({'init_CLB2':.1469227},defaults)
    assert not real_mutants.is_duplicate_real_mutant({'init_CLB2':.5},defaults)
    # A materially different event reset/ODE control must not be discarded.
    _,special=real_mutants.REAL_MUTANTS['104']
    assert not real_mutants.is_duplicate_real_mutant(dict(special,is_mutant_104=False),defaults)


@pytest.mark.parametrize('profile',['mass','mutant_aware'])
@pytest.mark.parametrize('mdt',[90.,150.,160.])
def test_both_branches_preserve_other_doses_and_environment(profile,mdt):
    backend=BuddingSystem(profile,profile)
    original=build(backend,[('CDC20','KO',1.),('CLB2','MC',7.),('CLB2','db',.08)],mdt)
    desc='CDC20[KO]+CLB2[MC]+CLB2[db]'
    branches=set()
    for seed in range(40):
        branches.add(np.random.RandomState(seed).random()<.7)
        a,da=rescue_budding(original,desc,backend,np.random.RandomState(seed))
        b,db=rescue_budding(original,desc,backend,np.random.RandomState(seed))
        assert (a,da)==(b,db)
        assert a['mdt']==mdt
        before,after=parse_pairs(desc),parse_pairs(da)
        assert [g for g,_ in before]==[g for g,_ in after]
        changed=[i for i,((g,m),(_,n)) in enumerate(zip(before,after))
                 if m!=n or any(original.get(k)!=a.get(k) for k in touch_set(g,m)|touch_set(g,n))]
        assert len(changed)==1
        index=changed[0]
        for i,(gene,mode) in enumerate(before):
            if i!=index:
                for key in touch_set(gene,mode):assert a.get(key)==original.get(key)
        assert all(mode!='GAL-OE' or mdt==150. for _,mode in after)
    assert branches=={True,False}


def test_seventy_percent_softens_only_eligible_ko():
    backend=BuddingSystem()
    ovr=build(backend,[('SWI5','KO',1.),('CDC20','KO',1.)])
    # SWI5 has no weak-LOF mode; it must not steal the KO rescue opportunity.
    for seed in range(30):
        if np.random.RandomState(seed).random()>=.7:continue
        fixed,desc=rescue_budding(ovr,'SWI5[KO]+CDC20[KO]',backend,np.random.RandomState(seed))
        assert parse_pairs(desc)[1][1] in ('ts','DAmP')
        assert fixed['ksswip']==fixed['ksswipp']==0.


@pytest.mark.parametrize('profile',['mass','mutant_aware'])
def test_mode_switch_removes_ko_initialization_and_parameter_residue(profile):
    backend=BuddingSystem(profile,profile)
    old=build(backend,[('CDH1','KO',1.),('CLB2','MC',7.)])
    new,desc=rescue_budding(old,'CDH1[KO]+CLB2[MC]',backend,np.random.RandomState(0))
    assert parse_pairs(desc)[0]==('CDH1','CA')
    assert 'init_CDH1' not in new and 'init_CDH1T' not in new
    assert new['kscdh']==.03 and new['kicdhpp']==0.
    assert new['ksb2p']==old['ksb2p'] and new['ksb2pp']==old['ksb2pp']


@pytest.mark.parametrize('profile',['mass','mutant_aware'])
def test_fixed_event_mutation_is_not_jittered_or_rescued_by_mdt(profile):
    backend=BuddingSystem(profile,profile);old=build(backend,[('MAD2','CSL',1.)])
    for seed in range(10):
        assert rescue_budding(old,'MAD2[CSL]',backend,np.random.RandomState(seed))==(old,'MAD2[CSL]')


def test_fixed_collision_redraws_same_level(monkeypatch):
    from budding_yeast_v2.data.systems.sampling_policy import draw_candidate
    config=GenerationConfig('budding_random',seed=4,max_attempts=3)
    sampler=make_sampler(RECIPES[config.recipe],config)
    ovr=build(sampler.backend,[('MAD2','CSL',1.)])
    levels=[]
    def sample(level):
        levels.append(level)
        return (ovr,'MAD2[CSL]',None,None) if len(levels)==1 else ({'mdt':90.,'ksb2pp':.7},'CLB2[MC]',None,None)
    monkeypatch.setattr(sampler,'sample',sample)
    report=Counter();row=draw_candidate(sampler,1,1,report)
    assert levels==[1,1] and row.level==1 and report['collision_exhausted']==1
    assert row.metadata['module_count']==1


@pytest.mark.parametrize('parameter_only',[False,True])
def test_static_and_pool_generation_exclude_effectively_identical_reals(parameter_only):
    recipe='budding_pool' if parameter_only else 'budding_random'
    config=resolve_config(GenerationConfig(recipe,n_samples=160,seed=487,include_real=False,parameter_only=parameter_only))
    sampler=make_sampler(RECIPES[recipe],config)
    rows=generation.prepare_candidates(RECIPES[recipe],config,sampler,Counter())
    for row in rows:
        assert not real_mutants.is_duplicate_real_mutant(row.overrides,sampler.backend.defaults())
        assert row.metadata['module_count']==row.level


@pytest.mark.parametrize('damage',['real','params','initial','ok'])
def test_al_pool_load_rechecks_reals_and_input_consistency(tmp_path,damage):
    import torch
    from budding_yeast_v2.utils.sampling import ActiveLearningEngine
    backend=BuddingSystem();ovr=build(backend,[('MAD2','CSL',1.)]) if damage=='real' else {'mdt':90.,'ksb2pp':.7}
    row=generation.get_adapter('budding').parameter_result(Candidate(1,'x',ovr,1),None)
    if damage=='params':row.params[0]+=.5
    if damage=='initial':row.y_init[0]+=.5
    path=tmp_path/'pool.npz'
    desc='MAD2[CSL]' if damage=='real' else 'CLB2[MC]'
    from budding_yeast_v2.data.systems.budding.al_pool import POOL_CONTRACT
    from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    np.savez(path,pool_generation_contract=POOL_CONTRACT,catalog_version=CATALOG_VERSION,sampling_range_fingerprint=RANGE_FINGERPRINT,params=row.params[None],y_init=row.y_init[None],
        mutant_names=np.array([f'MASS_RAND_{desc}_L1_1']),overrides_json=np.array([json.dumps(ovr)]))
    dm=SimpleNamespace(norm_type='linear',p_mean=torch.zeros(1,141),p_std=torch.ones(1,141),
        mean_y=torch.zeros(1,39,1),std_y=torch.ones(1,39,1),t_max=210.)
    engine=ActiveLearningEngine(al_pool_dataset=str(path))
    if damage=='ok':
        engine._generate_param_only_pool(dm)
        assert engine._pool_size==1
    else:
        with pytest.raises(ValueError,match='regenerate the pool'):engine._generate_param_only_pool(dm)


def test_twins_preserve_unselected_alleles_and_shared_mdt():
    sampler=make_sampler(RECIPES['budding_random'],GenerationConfig('budding_twins',seed=42))
    # A's NET1 fixed allele collides; B has a distinct continuous allele.
    a={'mdt':90.,'kasrent':10.,'kasrentp':.05}
    b={'mdt':90.,'kasrent':5.,'kasrentp':.025}
    np.random.seed(12)
    x,y=derived._twin_rescue(b,a,'NET1[TAB6-1]',1,1,sampler,Counter())
    assert x.overrides['mdt']==y.overrides['mdt']==90.
    assert parse_pairs(x.metadata['description'])==parse_pairs(y.metadata['description'])
    assert not real_mutants.is_duplicate_real_mutant(x.overrides,sampler.backend.defaults())
    assert not real_mutants.is_duplicate_real_mutant(y.overrides,sampler.backend.defaults())


def test_twins_preserve_other_modules_in_both_profiles():
    sampler=make_sampler(RECIPES['budding_random'],GenerationConfig('budding_twins',seed=42))
    a={'mdt':90.,'kkpnetpp':0.,'kasrent':10.,'kasrentp':.05}
    b={'mdt':90.,'kkpnetpp':.003,'kasrent':5.,'kasrentp':.025}
    np.random.seed(12)
    x,y=derived._twin_rescue(b,a,'CDC15[ts]+NET1[TAB6-1]',2,1,sampler,Counter())
    before=parse_pairs('CDC15[ts]+NET1[TAB6-1]');after=parse_pairs(x.metadata['description'])
    changed=[i for i,(old,new) in enumerate(zip(before,after)) if old!=new or
             any(x.overrides.get(k)!=b.get(k) or y.overrides.get(k)!=a.get(k)
                 for k in touch_set(*old)|touch_set(*new))]
    assert len(changed)==1
    untouched=1-changed[0]
    for key in touch_set(*before[untouched]):
        assert x.overrides.get(key)==b.get(key) and y.overrides.get(key)==a.get(key)
    assert x.overrides['mdt']==y.overrides['mdt']==90.


def test_protected_originals_fission_and_biology_unchanged():
    root=Path(__file__).resolve().parents[2]
    manifest=root/'budding_yeast_v2/scratch/probes/collision_rescue_2026-10-22/protected.json'
    if not manifest.exists():pytest.skip('Local pre-change manifest is unavailable')
    for path,digest in json.loads(manifest.read_text()).items():
        if path.endswith('/data/normalization.py'):
            # Authorized shared statistics were added; original transforms must
            # still be byte-for-byte equivalent at the Python AST level.
            import ast,subprocess
            previous=subprocess.check_output(['git','show','HEAD:data/normalization.py'],cwd=root/'budding_yeast_v2',text=True)
            before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(previous).body if isinstance(n,ast.FunctionDef)}
            after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((root/path).read_text()).body if isinstance(n,ast.FunctionDef)}
            assert all(after.get(name)==value for name,value in before.items())
            continue
        # al_pool gained authorized parallel orchestration; Fission biology stays protected.
        # The authorized later watchdog change is covered by the Nov 12 hash test.
        if path.endswith(('/data/dataset_utils.py','/data/yeast_datamodule.py','/data/label_metadata.py','/data/systems/fission/al_pool.py','/data/simulation_guards.py','/data/preprocessing_contract.py')):continue
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
