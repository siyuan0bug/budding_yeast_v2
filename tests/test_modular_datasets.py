"""Legacy parity and independent contract checks for the modular generators."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))

from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig, SimulationResult
from budding_yeast_v2.data.systems.generation import level_schedule
from budding_yeast_v2.data.systems.budding.samplers.rescue import rescue_budding
from budding_yeast_v2.data.systems.generation import generate_dataset, simulate_task
from budding_yeast_v2.data.systems.dataset_io import write_dataset, read_dataset
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem, SPEC
from budding_yeast_v2.data.systems.budding import states, parameters, ode, simulation, real_mutants, catalog
from budding_yeast_v2.data.systems.registry import RECIPES
from budding_yeast_v2.data.systems.budding.samplers import common

legacy_ma=importlib.import_module('budding_yeast_v2.data.mutant_aware')
legacy_mass=importlib.import_module('budding_yeast_v2.data.mass_dataset_generator')


@pytest.fixture(autouse=True)
def restore_rng():
    state=np.random.get_state()
    yield
    np.random.set_state(state)


def test_all_legacy_sources_unchanged():
    manifest=json.loads((ROOT/'data/systems/archived_generators.json').read_text())
    assert len(manifest)==31
    for path,digest in manifest.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path


def test_canonical_biology_and_real_mutants_match_ma():
    assert parameters.get_default_params()==legacy_ma.get_default_params()
    assert parameters.PARAM_KEYS==legacy_ma.PARAM_KEYS
    assert len(parameters.PARAM_KEYS)==141
    assert np.array_equal(states.initial_conds(),legacy_ma.initial_conds())
    assert real_mutants.REAL_MUTANTS==legacy_ma.REAL_MUTANTS
    assert len(real_mutants.real_tasks())==126
    # Synthetic catalogs are intentionally pruned independently of biology.
    assert set(catalog.GENE_NAMES) < {g['name'] for g in legacy_ma.BINDING_GROUPS.values()}
    rng=np.random.RandomState(91)
    for _ in range(20):
        y=states.initial_conds()*(.8+.4*rng.random(39))
        p=parameters.get_default_params()
        assert np.array_equal(ode.eqns(0,y,p),legacy_ma.eqns(0,y,p))
    for i in range(1,5):
        a,b=getattr(ode,f'event{i}'),getattr(legacy_ma,f'event{i}')
        assert a.direction==b.direction and a.terminal==b.terminal
        assert a(0,y,p)==b(0,y,p)


@pytest.mark.parametrize('profile,reference',[('mass',legacy_mass),('mutant_aware',legacy_ma)])
def test_retained_mutation_cells_and_distribution_draws_match_original(profile,reference):
    backend=BuddingSystem(profile,profile)
    defaults=backend.defaults()
    cells=[(g,m) for g in catalog.GENE_NAMES for m in catalog.GROUP_BY_NAME[g]['modes']]
    assert len(cells)==65
    for seed in (0,7,42):
        for gene,mode in cells:
            np.random.seed(seed)
            alpha_old=reference.sample_scaling_factor(mode,gene)
            old={}
            reference.apply_gene_mutation(gene,mode,alpha_old,defaults,defaults.copy(),old)
            state_old=np.random.get_state()
            np.random.seed(seed)
            alpha_new=backend.sample_scaling_factor(mode,gene)
            new={}
            backend.apply_gene_mutation(gene,mode,alpha_new,defaults,defaults.copy(),new)
            state_new=np.random.get_state()
            assert alpha_old==alpha_new,(profile,gene,mode)
            assert old==new,(profile,gene,mode)
            assert np.array_equal(state_old[1],state_new[1]) and state_old[2:]==state_new[2:]


@pytest.mark.parametrize('interval',[False,True])
def test_pure_random_candidate_stream_is_exact(interval):
    from budding_yeast_v2.data import generate_pure_random_dataset as old
    from budding_yeast_v2.data.systems.budding.samplers.random import sample_pure_random_mutant
    common.MODE_FREQ=old.ma.MODE_FREQ
    a,b=np.random.RandomState(42),np.random.RandomState(42)
    backend=BuddingSystem()
    for i in range(160):
        assert old.sample_pure_random_mutant(1+i%4,a,enable_ko_interval=interval)==sample_pure_random_mutant(1+i%4,b,enable_ko_interval=interval,ma=backend)


def test_rescue_is_repeatable_and_keeps_environment():
    for seed in range(12):
        backend=BuddingSystem()
        args=({'ks20p':0.,'ks20pp':0.,'mdt':90.},'CDC20[KO]+CDC14[ts]',backend.defaults())
        a=rescue_budding(args[0],args[1],backend,np.random.RandomState(seed))
        b=rescue_budding(args[0],args[1],backend,np.random.RandomState(seed))
        assert a==b and a[0]['mdt']==90.


@pytest.mark.parametrize('recipe_name',[n for n,r in RECIPES.items() if r.sampler not in ('wt_blind','twins','ko_mirror','bams_ablation')])
def test_all_algorithm_adapters_construct_candidates(recipe_name):
    from budding_yeast_v2.data.systems.registry import make_sampler
    config=GenerationConfig(recipe_name,n_samples=4,seed=17)
    sampler=make_sampler(RECIPES[recipe_name],config)
    overrides,desc,_,_=sampler.sample(2)
    assert overrides and desc
    assert all(np.isfinite(float(v)) for v in overrides.values())


@pytest.mark.parametrize('kind,oldmodule,classname',[('gibbs','gibbs_boltzmann_sampler','GibbsBoltzmannGraphSampler'),
    ('kdpp','generate_kdpp_gated_dataset','GraphDiffusionKDPP'),
    ('kdpp','generate_kdpp_curv_gated_dataset','SignedDirectedGraphKDPP'),
    ('motif','generate_motif_topology_dataset','MotifTopologySampler')])
def test_graph_kernel_selection_matches_original(kind,oldmodule,classname):
    old=importlib.import_module('budding_yeast_v2.data.'+oldmodule)
    new=importlib.import_module('budding_yeast_v2.data.systems.budding.samplers.'+kind)
    a=getattr(old,classname)(seed=13)
    b=getattr(new,classname)(seed=13,ma=BuddingSystem())
    method={'GibbsBoltzmannGraphSampler':'sample_genes','GraphDiffusionKDPP':'sample',
            'SignedDirectedGraphKDPP':'sample_genes','MotifTopologySampler':'sample_motif'}[classname]
    for k in [2,3,4]*10:
        assert getattr(a,method)(k)==getattr(b,method)(k)
    for attr in ('L','S','W','eigvals'):
        if hasattr(a,attr):assert np.array_equal(getattr(a,attr),getattr(b,attr))


def test_sobol_quantiles_match_both_original_copies():
    from budding_yeast_v2.data import gibbs_sobol_sampler as old
    from budding_yeast_v2.data import generate_1m_pure_random_pool as pool
    backend=BuddingSystem()
    for gene in catalog.GENE_NAMES:
        for mode in catalog.GROUP_BY_NAME[gene]['modes']:
            for u in (0.,.001,.1,.5,.9,.999,1.):
                assert backend.sample_scaling_factor_sobol(mode,gene,u)==old.sample_scaling_factor_sobol(mode,gene,u)
                assert backend.sample_scaling_factor_sobol(mode,gene,u)==pool.sample_scaling_factor_sobol(mode,gene,u)
    assert BuddingSystem('mass','mass').sample_scaling_factor_sobol('db','CLB5',.5)==0


@pytest.mark.parametrize('source,newname,fn,profile',[
    ('mass_random_sampler','random','sample_mass_random','mass'),
    ('mass_dataset_generator','mass_motifs','sample_mass_mutant','mass'),
    ('mutant_aware','mass','sample_mass_mutant','mutant_aware'),
    ('bams_dataset_generator','bams','sample_bams_mutant_task','mutant_aware'),
])
def test_selection_and_assignment_streams_match_legacy(source,newname,fn,profile):
    old=importlib.import_module('budding_yeast_v2.data.'+source)
    new=importlib.import_module('budding_yeast_v2.data.systems.budding.samplers.'+newname)
    def reset(module):
        if hasattr(module,'_L1_IDX'):module._L1_IDX['i']=0
        if hasattr(module,'MODE_FREQ'):module.MODE_FREQ='class'
        for attr,quotas in (('_L2_PICKER','STRUCT_L2'),('_L34_PICKER','STRUCT_L34')):
            if hasattr(module,attr):
                q=getattr(module,quotas)
                setattr(module,attr,module.QuotaPicker(list(q),list(q.values())))
    reset(old)
    np.random.seed(31)
    expected=[getattr(old,fn)(level=1+i%4) for i in range(32)]
    reset(new);reset(common)
    np.random.seed(31)
    backend=BuddingSystem(profile,profile)
    actual=[getattr(new,fn)(level=1+i%4,ma=backend) for i in range(32)]
    assert actual==expected


def test_wt_blind_qss_and_candidate_match_legacy():
    old=importlib.import_module('budding_yeast_v2.data.mass_wt_blind_sampler')
    from budding_yeast_v2.data.systems.budding.samplers import wt_blind as new
    backend=BuddingSystem('mass','mass')
    for module in (old,new):
        module.MODE_FREQ='flat'
        module.FOLD_DIST='tier2'
        module._L1_IDX['i']=0
        module._L2_PICKER=module.QuotaPicker(list(module.STRUCT_L2),list(module.STRUCT_L2.values()))
        module._L34_PICKER=module.QuotaPicker(list(module.STRUCT_L34),list(module.STRUCT_L34.values()))
    p=backend.defaults()
    assert old.qss_g1(p,1.5)==new.qss_g1(p,1.5)
    assert old.qss_M(p,2.4)==new.qss_M(p,2.4)
    np.random.seed(11)
    bands=old.make_samplers()
    expected=[old.build_candidate(1+i%4,bands) for i in range(16)]
    np.random.seed(11)
    bands=new.make_samplers()
    actual=[new.build_candidate(1+i%4,bands,ma=backend) for i in range(16)]
    assert actual==expected


def test_fission_candidates_match_original_first_run():
    old=importlib.import_module('budding_yeast_v2.data.fission_random_sampler')
    from budding_yeast_v2.data.systems.registry import make_sampler
    initial=[(m,mo) for m in old.SAMPLING_MODULES for mo in old.SAMPLING_MODULES[m]['modes']]
    old._L1_GRID[:]=initial
    np.random.RandomState(42).shuffle(old._L1_GRID)
    old._L1_IDX['i']=0;old.MODE_FREQ='class'
    np.random.seed(42)
    rng=np.random.default_rng(42)
    expected=[old.sample_random_mutant(1+i%3,rng) for i in range(60)]
    np.random.seed(42)
    sampler=make_sampler(RECIPES['fission_random'],GenerationConfig('fission_random',seed=42))
    actual=[sampler.sample(1+i%3)[:2] for i in range(60)]
    assert actual==expected


@pytest.mark.parametrize('overrides',[{}, {'C0':.5},{'mdt':150.}, {'Tem1T':0.,'init_Tem1':0.}, {'Cdc15T':0.,'init_Cdc15':0.}])
def test_budding_trajectories_and_labels_match_original(overrides):
    a=legacy_ma.simulate_mutant('control',overrides)
    b=simulation.simulate_mutant('control',overrides)
    assert a[1] is not None and b[1] is not None
    assert a[0]==b[0] and a[2:]==b[2:]
    assert np.array_equal(a[1],b[1])


def test_fission_rhs_warmup_and_successful_trajectory_match_original():
    old=importlib.import_module('budding_yeast_v2.data.fission_random_sampler')
    from budding_yeast_v2.data.systems.fission import ode as fode, states as fs, simulation as fsim, parameters as fp
    assert old.WT_parameters()==fp.WT_parameters()
    assert len(fp.EXPORT_KEYS)==58
    y=old._default_birth_state()
    assert np.array_equal(old.eqns(0,y,old.WT_parameters()),fode.eqns(0,y,fp.WT_parameters()))
    a=old.derive_wt_birth_state(verbose=False)
    b=fs.derive_wt_birth_state(verbose=False)
    assert np.array_equal(a,b)
    for overrides in ({},{'k5':0.}):
        x=old.simulate_mutant('control',overrides,a)
        z=fsim.simulate_mutant('control',overrides,b)
        assert x['y'] is not None and z['y'] is not None
        for key in ('label','pattern','n_div','status','birth_rel'):
            assert x[key]==z[key],key
        assert np.array_equal(x['y'],z['y'])
        assert np.array_equal(x['params_vec'],z['params_vec'])


def test_pipeline_exports_identical_old_pure_random_inputs_and_labels(tmp_path):
    from budding_yeast_v2.data import generate_pure_random_dataset as old
    path=generate_dataset(GenerationConfig('budding_random',n_samples=8,seed=42,workers=1,
                         include_real=False,output=str(tmp_path/'random.npz')))
    rng=np.random.RandomState(42)
    expected=[]
    defaults=BuddingSystem().defaults()
    for level in [1,1,2,2,3,3,4,4]:
        ovr,desc=old.sample_pure_random_mutant(level,rng)
        if old.ma.is_duplicate_real_mutant(ovr,defaults):
            ovr,desc=old.ma.rescue_collision_candidate(ovr,desc,defaults,rng=rng)
        expected.append(ovr)
    data=read_dataset(path)
    assert data['params'].shape==(8,141)
    assert data['data'].shape==(8,39,500)
    for i,ovr in enumerate(expected):
        assert np.array_equal(data['params'][i],[ovr.get(k,defaults[k]) for k in SPEC.param_names])
        _,y,label,pattern=old.ma.simulate_mutant('reference',ovr)
        assert np.array_equal(data['data'][i],y.astype(np.float32))
        assert data['labels'][i]==label and data['pattern_labels'][i]==pattern


def test_metadata_pool_and_atomic_no_overwrite(tmp_path):
    config=GenerationConfig('budding_pool',n_samples=8,output=str(tmp_path/'pool.npz'),workers=1)
    generate_dataset(config)
    d=read_dataset(config.output,require_trajectories=False)
    assert 'data' not in d and str(d['artifact_kind'])=='parameter_pool'
    assert d['y_init'].shape==(8,39)
    with pytest.raises(ValueError,match='no trajectory'):
        read_dataset(config.output)
    before=Path(config.output).read_bytes()
    with pytest.raises(FileExistsError):generate_dataset(config)
    assert Path(config.output).read_bytes()==before


def test_fission_identity_is_explicit_and_repeated_runs_replay(tmp_path):
    for i in range(2):
        generate_dataset(GenerationConfig('fission_random',n_samples=4,seed=42,workers=1,
                         include_real=False,output=str(tmp_path/f'fission{i}.npz')))
    a,b=[read_dataset(tmp_path/f'fission{i}.npz') for i in range(2)]
    for key in ('params','y_init','data','labels','pattern_labels','levels','mutant_names'):
        assert np.array_equal(a[key],b[key]),key
    assert str(a['system_id'])=='fission'
    assert a['data'].shape[1:]==(16,500)
    assert not a['is_real'].any()
    assert a['params'].shape[1]==58


def test_failed_simulations_cannot_be_serialized_as_valid_labels(tmp_path):
    c=Candidate(1,'bad',{},1)
    bad=SimulationResult(c,None,np.ones(141),np.ones(39),'Failure',None,'numerical_failure')
    config=GenerationConfig('budding_random',output=str(tmp_path/'bad.npz'))
    with pytest.raises(ValueError):
        write_dataset(config.output,SPEC,[bad],config,{'mutation_profile':'mutant_aware','distribution_profile':'mutant_aware','rng_contract':'test'})
    assert not Path(config.output).exists()


def test_small_quota_rules_are_preserved():
    assert level_schedule(7,(.25,.25,.25,.25),np.random.RandomState(1),rounding='round',shuffle=False).tolist()==[1,1,2,2,3,3,4]
    assert len(level_schedule(100,(.1,.3,.3,.21,.09),np.random.RandomState(1),rounding='motif'))==100


def test_modular_graph_matches_legacy_source_kernel():
    from budding_yeast_v2.data.systems.budding.graph import build_budding_graph
    from budding_yeast_v2.data.yeast_graph import build_yeast_graph
    a,b=build_budding_graph(),build_yeast_graph()
    assert a.source_sha256==b.source_sha256
    for view in ('continuous','event','hybrid'):
        assert np.array_equal(a.adjacency(view),b.adjacency(view))


def test_twins_preserve_paired_inputs_and_common_valid_rows(tmp_path):
    from budding_yeast_v2.data.systems.generation import generate_dataset
    cfg=GenerationConfig('budding_twins',n_samples=4,seed=42,include_real=False,workers=2,
                         output=str(tmp_path/'b.npz'),paired_output=str(tmp_path/'a.npz'))
    paths=generate_dataset(cfg)
    a,b=read_dataset(paths['mass']),read_dataset(paths['mutant_aware'])
    assert np.array_equal(a['sample_id'],b['sample_id'])
    assert np.array_equal(a['pair_id'],b['pair_id'])
    assert np.array_equal(a['mutant_names'],b['mutant_names'])
    assert str(a['mutation_profile'])=='mass'
    assert str(b['mutation_profile'])=='mutant_aware'


def test_ko_mirror_recomputes_and_never_pairs_new_inputs_with_old_labels(tmp_path):
    from budding_yeast_v2.data.systems.generation import generate_dataset
    from budding_yeast_v2.data.systems.budding.samplers.rescue import parse_pairs
    path=generate_dataset(GenerationConfig('budding_random',n_samples=8,seed=42,include_real=False,workers=1,output=str(tmp_path/'base.npz')))
    output=generate_dataset(GenerationConfig('budding_ko_mirror',source=path,output=str(tmp_path/'mirror.npz'),workers=2))
    base,mirror=read_dataset(path),read_dataset(output)
    assert any(mode=='KO' for name in base['mutant_names'] for _,mode in parse_pairs(str(name)))
    for i,name in enumerate(base['mutant_names']):
        if '[KO]' not in name:
            assert np.array_equal(base['params'][i],mirror['params'][i])
            assert np.array_equal(base['data'][i],mirror['data'][i])
        else:
            overrides=json.loads(str(mirror['overrides_json'][i]))
            _,y,_,_=legacy_ma.simulate_mutant('mirror_reference',overrides)
            assert y is not None and np.array_equal(mirror['data'][i],y.astype(base['data'].dtype))


def test_targeted_ablation_preserves_shared_base_and_pair_identity(tmp_path):
    source=generate_dataset(GenerationConfig('budding_random',n_samples=8,seed=42,include_real=False,workers=1,output=str(tmp_path/'base.npz')))
    paths=generate_dataset(GenerationConfig('budding_bams_ablation',n_samples=5,seed=42,include_real=False,workers=2,source=source,output=str(tmp_path/'control.npz')))
    a,b=read_dataset(paths['hard']),read_dataset(paths['hypo'])
    assert np.array_equal(a['data'][:4],b['data'][:4])
    assert np.array_equal(a['params'][:4],b['params'][:4])
    assert np.array_equal(a['pair_id'],b['pair_id'])


def test_rescue_carbon_marker_is_not_a_gene():
    from budding_yeast_v2.data.systems.budding.samplers.rescue import parse_pairs
    assert parse_pairs('MASS_CLB2[KO]+CLB5[ts]_MDT[150]_L2_1')==[('CLB2','KO'),('CLB5','ts')]
    assert parse_pairs('MASS_MDT[MDT]_L1_1')==[('MDT','MDT')]


def test_paired_failures_drop_both_arms(monkeypatch):
    from budding_yeast_v2.data.systems.budding import derived
    candidate=Candidate(1,'pair',{},1,pair_id=1)
    good=SimulationResult(candidate,np.ones((39,500)),np.ones(141),np.ones(39),'Viable','A','success')
    bad=SimulationResult(candidate,None,np.ones(141),np.ones(39),'failure',None,'numerical_failure')
    calls=iter([[good],[bad]])
    monkeypatch.setattr(derived,'run_simulations',lambda *args:iter(next(calls)))
    assert derived._paired_results([candidate],[candidate],GenerationConfig('budding_twins'))==([],[])


def test_ko_failure_aborts_without_writing_new_dataset(tmp_path,monkeypatch):
    from budding_yeast_v2.data.systems.budding import derived
    candidate=Candidate(1,'RANDOM_CLB2[KO]_L1_1',{'ksb2p':0.,'ksb2pp':0.},1)
    source=SimulationResult(candidate,np.ones((39,500)),np.ones(141),np.ones(39),'Viable','A','success')
    bad=SimulationResult(candidate,None,np.ones(141),np.ones(39),'failure',None,'numerical_failure')
    monkeypatch.setattr(derived,'_source_rows',lambda config:({},[source]))
    monkeypatch.setattr(derived,'run_simulations',lambda *args:iter([bad]))
    output=tmp_path/'bad_mirror.npz'
    with pytest.raises(RuntimeError,match='re-simulation failed'):
        generate_dataset(GenerationConfig('budding_ko_mirror',source='unused',output=str(output)))
    assert not output.exists()


def test_wt_blind_default_recipe_and_repeatability(tmp_path):
    from dataclasses import replace
    cfg=GenerationConfig('budding_wt_blind',n_samples=4,seed=42,include_real=False,
                         parameter_only=True,pilot_samples=16,output=str(tmp_path/'wt1.npz'))
    a=read_dataset(generate_dataset(cfg),require_trajectories=False)
    b=read_dataset(generate_dataset(replace(cfg,output=str(tmp_path/'wt2.npz'))),require_trajectories=False)
    assert np.array_equal(a['params'],b['params'])
    assert json.loads(str(a['generation_config']))['mode_freq']=='flat'


@pytest.mark.parametrize('name',['budding_mass_random','budding_gibbs_sobol'])
@pytest.mark.parametrize('seed',[0,7,42])
def test_actual_legacy_driver_order_and_rng_match(name,seed):
    from budding_yeast_v2.scripts.analysis.audit_sampler_refactor import old_driver,new_driver
    assert old_driver(name,seed,64)==new_driver(name,seed,64)


@pytest.mark.parametrize('seed',[0,7,42])
def test_twins_full_candidate_stream_matches_original(seed):
    from budding_yeast_v2.scripts.analysis.audit_sampler_refactor import special_stream
    assert special_stream('twins',seed,256,False)==special_stream('twins',seed,256,True)


def test_gibbs_sobol_uses_original_remainder_and_names():
    from budding_yeast_v2.data.systems.registry import candidate_name
    recipe=RECIPES['budding_gibbs_sobol']
    schedule=level_schedule(64,recipe.ratios,np.random.RandomState(42),rounding=recipe.quota_rounding,shuffle=recipe.shuffle_levels)
    counts=[round(r*64) for r in recipe.ratios]
    counts[0]+=64-sum(counts)
    assert np.array_equal(schedule,np.repeat(np.arange(1,6),counts))
    assert candidate_name('gibbs_sobol','CLB2[ts]',1,1)=='SOBOL_CLB2[ts]_L1_1'


@pytest.mark.parametrize('name',['budding_random','budding_mutant_aware','budding_kdpp','budding_bams'])
def test_collision_uses_original_callers_rng_stream(name,monkeypatch):
    from collections import Counter
    from budding_yeast_v2.data.systems.registry import make_sampler
    from budding_yeast_v2.data.systems.generation import ensure_candidate
    for seed in range(12):
        sampler=make_sampler(RECIPES[name],GenerationConfig(name,seed=seed))
        sampler.rng.rand(17)
        local_state=sampler.rng.get_state()
        overrides={'ks20p':0.,'ks20pp':0.,'mdt':90.}
        np.random.seed(seed+100)
        expected,description=rescue_budding(overrides,'CDC20[KO]',sampler.backend,
                         sampler.rng if name=='budding_random' else np.random)
        expected_global=np.random.get_state();expected_local=sampler.rng.get_state()
        np.random.seed(seed+100);sampler.rng.set_state(local_state)
        calls=iter([True,False])
        monkeypatch.setattr(real_mutants,'is_duplicate_real_mutant',lambda *args:next(calls))
        candidate=Candidate(1,'candidate',overrides,1,metadata={'description':'CDC20[KO]'})
        actual=ensure_candidate(candidate,sampler,Counter())
        assert actual.overrides==expected
        assert actual.metadata['description']==description
        assert np.array_equal(expected_global[1],np.random.get_state()[1]) and expected_global[2:]==np.random.get_state()[2:]
        assert np.array_equal(expected_local[1],sampler.rng.get_state()[1]) and expected_local[2:]==sampler.rng.get_state()[2:]
