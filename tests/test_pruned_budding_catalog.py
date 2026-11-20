"""Pruned synthetic controls cannot re-enter through profiles, graphs or pools."""
from collections import Counter
from dataclasses import replace
import importlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from budding_yeast_v2.data.systems.budding import catalog, topology
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.samplers import common, wt_blind, motif, bams, twins
from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig
from budding_yeast_v2.data.systems.registry import RECIPES, resolve_config, make_sampler
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate, draw_candidate

REMOVED = {'CDC5_Proxy','SCC1','CDC28','SBF_MBF','PPX','Mcm1','ORI_BUD_SPN'}
BANNED = {(g,m) for g in REMOVED|{'CDC5'} for m in ('KO','MC','GAL-OE','ts')} | {('TEM1','DAmP'),('CDC15','DAmP')}
CELLS = [(g,m) for g in catalog.GENE_NAMES for m in catalog.GROUP_BY_NAME[g]['modes']]
ROOT = Path(__file__).resolve().parents[1]

def approved_since_october_snapshot(gene,mode,profile):
    # Explicitly authorized range/GAL-MC changes and Nov 12 exact closures.
    # The historical snapshot remains unchanged and still checks other modes.
    from budding_yeast_v2.data.systems.budding.distributions.ranges import EXACT_ZERO_TARGETS
    return (mode in ('MC','GAL-MC') or (gene,mode) in {
        ('CLB2','GAL-OE'),('CLB2','db'),('TEM1','ts'),('ESP1','ts'),
        ('NET1','ts'),('NET1','TAB6-1')} or
        (profile=='mutant_aware' and (gene,mode) in EXACT_ZERO_TARGETS))


def test_catalog_graphs_and_specialized_tables_pruned():
    assert len(catalog.GENE_NAMES)==20 and len(CELLS)==65
    assert len(set(catalog.GENE_NAMES)-{'MDT'})==19
    assert not REMOVED & set(catalog.GENE_NAMES)
    assert sum(g['P_bio_G'] for g in catalog.GROUP_BY_NAME.values())==pytest.approx(1.)
    for name in ('TEM1','CDC15'):
        assert set(catalog.GROUP_BY_NAME[name]['modes'])=={'KO','MC','GAL-OE','ts'}
        assert sum(catalog.GROUP_BY_NAME[name]['modes'].values())==pytest.approx(1.)
    for module in (catalog,wt_blind):
        assert not BANNED & set(module.TOUCH)
        assert not REMOVED & {g for g,m in module.TOUCH}
    assert not REMOVED & set(catalog.GENE_ROLE)
    for module in (topology,wt_blind,bams):
        assert not REMOVED & {g for edge in module.MODULE_EDGES for g in edge[:2]}
        assert not REMOVED & set(module.ADJ)
        assert not REMOVED & {g for neighbours in module.ADJ.values() for g in neighbours}
    assert not REMOVED & set(wt_blind.STAR_CENTERS)
    assert not REMOVED & {g for gs in wt_blind.STAR_LEAVES.values() for g in gs}
    assert not REMOVED & {g for gs in motif.MASTER_SWITCH_CLUSTERS for g in gs}
    for grid in (common._L1_GRID,bams._L1_GRID,wt_blind._L1_GRID):
        assert not BANNED & set(grid)
        assert set(grid)<=set(CELLS)
    for name in ('ALLELE_RESIDUAL','_NATIVE_SCALE_PARAMS','GAL_ALPHA_BIO','PRIMARY_COORDS','SIGN'):
        keys=getattr(wt_blind,name)
        assert not REMOVED & {k[0] if isinstance(k,tuple) else k for k in keys}
    assert not {'scc1-RR','cdc24Δ','cdc45Δ','spindle_defect'} & (twins.TARGET_MODES|bams.LOF_MODES)


@pytest.mark.parametrize('mutation',['mass','mutant_aware'])
@pytest.mark.parametrize('distribution',['mass','mutant_aware'])
def test_removed_calls_fail_before_rng_or_parameter_edits(mutation,distribution):
    backend=BuddingSystem(mutation,distribution)
    defaults=backend.defaults()
    for gene,mode in BANNED:
        p=defaults.copy();overrides={};rng=np.random.RandomState(31);before=rng.get_state()
        with pytest.raises(ValueError,match='Unsupported Budding'):
            backend.sample_scaling_factor(mode,gene,rng=rng)
        with pytest.raises(ValueError,match='Unsupported Budding'):
            backend.apply_gene_mutation(gene,mode,.2,defaults,p,overrides,rng=rng)
        with pytest.raises(ValueError,match='Unsupported Budding'):
            backend.sample_scaling_factor_sobol(mode,gene,.7)
        assert p==defaults and overrides=={}
        assert np.array_equal(before[1],rng.get_state()[1]) and before[2:]==rng.get_state()[2:]


@pytest.mark.parametrize('profile',['mass','mutant_aware'])
def test_remaining_cells_keep_pre_removal_parameters_distributions_and_rng(profile):
    # Compare to the exact start-of-task snapshot, including current v5.2 rules.
    archive=ROOT/'scratch/probes/catalog_prune_2026-10-22/before.zip'
    if not archive.exists():pytest.skip('Local pre-removal snapshot is unavailable')
    with zipfile.ZipFile(archive) as z:
        old={}
        for folder in ('mutations','distributions'):
            ns={'__name__':'baseline'}
            exec(compile(z.read(f'data/systems/budding/{folder}/{profile}.py'),'<baseline>','exec'),ns)
            old[folder]=ns
    backend=BuddingSystem(profile,profile);defaults=backend.defaults()
    for gene,mode in CELLS:
        if approved_since_october_snapshot(gene,mode,profile):continue
        for seed in (0,7,42):
            a=np.random.RandomState(seed);b=np.random.RandomState(seed)
            alpha_a=old['distributions']['sample_scaling_factor'](mode,gene,rng=a)
            alpha_b=backend.sample_scaling_factor(mode,gene,rng=b)
            assert alpha_a==alpha_b,(gene,mode)
            for alpha in (alpha_a,0.,1e-4,.2,1.,30.):
                oa={};ob={};pa=defaults.copy();pb=defaults.copy()
                old['mutations']['apply_gene_mutation'](gene,mode,alpha,defaults,pa,oa,rng=a)
                backend.apply_gene_mutation(gene,mode,alpha,defaults,pb,ob,rng=b)
                assert (oa,pa)==(ob,pb),(profile,gene,mode,alpha)
            assert np.array_equal(a.get_state()[1],b.get_state()[1])
            assert a.get_state()[2:]==b.get_state()[2:]
    with zipfile.ZipFile(archive) as z:
        ns={'__name__':'baseline'}
        exec(compile(z.read('data/systems/budding/distributions/quantiles.py'),'<baseline>','exec'),ns)
    if profile=='mutant_aware':
        for gene,mode in CELLS:
            if approved_since_october_snapshot(gene,mode,profile):continue
            for u in (0.,.01,.5,.99,1.):
                assert backend.sample_scaling_factor_sobol(mode,gene,u)==ns['sample_scaling_factor_sobol'](mode,gene,u)


@pytest.mark.parametrize('description',[
    'CLN2[KO]+PPX[MC]', 'CLN2[KO]+TEM1[DAmP]', 'CLN2[KO]+CDC15[DAmP]',
    *[f'CLN2[KO]+{gene}[KO]' for gene in REMOVED],
])
def test_removed_mutations_cannot_hide_behind_valid_module_counts(description):
    assert valid_candidate(Candidate(1,'x',{},1,metadata={'description':description}),'budding') is None


@pytest.mark.parametrize('recipe',[k for k,v in RECIPES.items() if v.system=='budding' and v.sampler not in ('twins','ko_mirror','bams_ablation')])
def test_all_sampler_paths_and_rescue_only_emit_retained_controls(recipe):
    config=resolve_config(GenerationConfig(recipe,n_samples=30,seed=189,include_real=False))
    np.random.seed(config.seed);sampler=make_sampler(RECIPES[recipe],config);report=Counter()
    for level in (1,2,3):
        for index in range(20):
            row=draw_candidate(sampler,level,index,report)
            assert not row.metadata['invalid_mutations']
            assert not REMOVED & set(row.metadata['module_names'])
            assert not BANNED & set(row.metadata['pairs'])
            assert row.metadata['module_count']==level


def test_original_files_and_biological_definitions_unchanged():
    import hashlib
    base=ROOT/'scratch/probes/catalog_prune_2026-10-22'
    if not base.exists():pytest.skip('Local pre-removal snapshot is unavailable')
    manifest=json.loads((base/'original_data_hashes.json').read_text())
    for path,digest in manifest.items():
        # simulation_guards gained the authorized timeout/process-pool watchdog;
        # its newer protected hash is checked by test_canonical_exact_zero_update.
        if path in ('data/dataset_utils.py','data/yeast_datamodule.py','data/label_metadata.py','data/simulation_guards.py','data/preprocessing_contract.py'):continue
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    with zipfile.ZipFile(base/'before.zip') as z:
        for filename in ('states.py','parameters.py','ode.py','phenotypes.py','pattern_groups.py'):
            path='data/systems/budding/'+filename
            assert z.read(path)==(ROOT/path).read_bytes(),path
        import ast
        path='data/systems/budding/real_mutants.py'
        def rules(source):
            return next(ast.literal_eval(node.value) for node in ast.parse(source).body
                        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REAL_MUTANTS' for t in node.targets))
        assert rules(z.read(path).decode())==rules((ROOT/path).read_text())
        path='data/systems/budding/simulation.py'
        old=z.read(path).decode()
        expected=old.replace('    if tf <= 210 and max_m > 10.0:\n        return name, None, "Discarded_Mass", None\n\n','')
        assert (ROOT/path).read_text()==expected


@pytest.mark.parametrize('description,level',[
    ('CLN2[KO]+PPX[MC]',1),
    ('CLN2[KO]+TEM1[DAmP]',2),
    ('CLN2[KO]+CDC15[DAmP]',2),
])
def test_al_pool_loader_rejects_removed_controls(tmp_path,description,level):
    import torch
    from types import SimpleNamespace
    from budding_yeast_v2.utils.sampling import ActiveLearningEngine
    from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS
    from budding_yeast_v2.data.systems.budding.al_pool import POOL_CONTRACT
    from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    defaults=BuddingSystem().defaults()
    path=tmp_path/'old_pool.npz'
    np.savez(path,params=[[defaults[k] for k in PARAM_KEYS]],
        mutant_names=[f'MASS_RAND_{description}_L{level}_0'],
        pool_generation_contract=POOL_CONTRACT,catalog_version=CATALOG_VERSION,
        sampling_range_fingerprint=RANGE_FINGERPRINT)
    engine=ActiveLearningEngine(al_pool_dataset=str(path))
    with pytest.raises(ValueError,match='unsupported mutations'):
        engine._generate_param_only_pool(SimpleNamespace())


def test_collision_rescue_cannot_restore_removed_damp_modes():
    from budding_yeast_v2.data.systems.budding.samplers.rescue import rescue_budding, parse_pairs
    for profile in ('mass','mutant_aware'):
        backend=BuddingSystem(profile,profile)
        for gene in ('TEM1','CDC15'):
            for seed in range(30):
                ovr={};defaults=backend.defaults()
                backend.apply_gene_mutation(gene,'KO',0.,defaults,defaults.copy(),ovr)
                _,desc=rescue_budding(ovr,f'{gene}[KO]',backend,np.random.RandomState(seed))
                assert (gene,'DAmP') not in parse_pairs(desc)
                assert all(g in catalog.GENE_NAMES for g,m in parse_pairs(desc))
