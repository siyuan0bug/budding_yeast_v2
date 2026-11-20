"""Package boundaries, pool alignment, independent bug fixes and real AL smoke."""
from dataclasses import replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.contracts import SelectionPlan, QueryBatch
from budding_yeast_v2.utils.sampling.pool import CandidatePool, POOL_FIELDS
from budding_yeast_v2.utils.sampling.registry import SPECS, ALIASES, strategy_spec
from budding_yeast_v2.utils.sampling import oracle
from budding_yeast_v2.utils.al_callback import ActiveLearningCallback
from budding_yeast_v2.data.systems.budding.api import PARAM_KEYS, prepare_initial_conditions
from budding_yeast_v2.data.systems.budding.pattern_groups import abc_pattern_ids
from budding_yeast_v2.scripts.analysis.audit_al_package_refactor import fixture


def full_pool(count=4):
    pool=CandidatePool();pool.size=count
    for field in set(POOL_FIELDS.values())-{'size'}:
        if field in ('overrides','mutation_metadata'):
            setattr(pool,field,[{'row':i} for i in range(count)])
        elif field=='names':
            setattr(pool,field,np.array([f'row{i}' for i in range(count)]))
        else:
            setattr(pool,field,torch.arange(count).reshape(count,1))
    pool.ensure_ids()
    return pool


def test_old_public_import_and_pickled_worker_paths_resolve():
    assert ActiveLearningCallback is ActiveLearningEngine
    assert pickle.loads(pickle.dumps(oracle._solve_single_mass_task_wrapper)) is oracle._solve_single_mass_task_wrapper
    assert oracle._solve_single_mass_task_wrapper.__module__.endswith('.sampling.online.oracle')
    from budding_yeast_v2.utils.al_callback import _solve_single_mass_task_wrapper
    assert _solve_single_mass_task_wrapper is oracle._solve_single_mass_task_wrapper


def test_pool_prunes_all_arrays_and_preserves_stable_ordered_ids():
    pool=full_pool()
    plan=pool.plan([3,0],'al4pde_random')
    pool.keep(np.array([True,False,True,True]))
    assert type(pool.size) is int
    assert pool.candidate_ids.tolist()==[0,2,3]
    for field in set(POOL_FIELDS.values())-{'size'}:
        value=getattr(pool,field)
        if isinstance(value,list):assert [v['row'] for v in value]==[0,2,3]
        elif field=='names':assert value.tolist()==['row0','row2','row3']
        else:assert value.flatten().tolist()==[0,2,3]
    with pytest.raises(ValueError,match='Stale'):pool.resolve(plan)
    new=pool.plan([2,0],'al4pde_random')
    assert new.candidate_ids==(3,0)
    assert pool.resolve(new).tolist()==[2,0]
    foreign=full_pool(3).plan([0],'al4pde_random')
    with pytest.raises(ValueError,match='Stale'):pool.resolve(foreign)


def test_pool_alignment_error_does_not_partially_mutate_rows():
    pool=full_pool();pool.overrides=[{}]
    old=pool.params_physical.clone()
    with pytest.raises(ValueError,match='alignment'):pool.keep(np.array([True,False,True,True]))
    torch.testing.assert_close(pool.params_physical,old)
    assert pool.revision==0 and pool.size==4


@pytest.mark.parametrize('indices',[[-1],[4],[0,0],[[0,1]]])
def test_invalid_selection_plans_rejected(indices):
    with pytest.raises(ValueError,match='invalid|duplicate'):full_pool().plan(indices,'al4pde_random')


@pytest.mark.parametrize('strategy',list(SPECS))
def test_all_strategies_return_plan_before_oracle_with_original_quota_policy(strategy):
    if strategy in ('ours_pattern_balance','ours_pattern_context','ours_pattern_support','ours_sensitivity'):
        from tests.test_al_mechanism import fixture as mechanism_fixture
        cb,dm,members=mechanism_fixture(strategy);model=members[0]
    else:cb,model,dm=fixture(ActiveLearningEngine,strategy,42)
    original_size=cb._pool_size
    if strategy=='ours_invariance':
        # The historical proof pins an older solver source. Governance must not
        # silently refresh its scientific certificate merely to make a test pass.
        from budding_yeast_v2.utils.sampling.ours.query_design import validate_invariance_rule
        try:validate_invariance_rule()
        except RuntimeError as error:
            assert 're-audit TEM1 invariance' in str(error)
            with patch.object(cb,'_build_broadcast_from_pool',side_effect=AssertionError('simulation during selection')):
                with pytest.raises(RuntimeError,match='re-audit TEM1 invariance'):cb.select(model,dm,'cpu')
            return
    with patch.object(cb,'_build_broadcast_from_pool',side_effect=AssertionError('simulation during selection')):
        plan=cb.select(model,dm,'cpu')
    assert isinstance(plan,SelectionPlan)
    assert plan.fill_policy==SPECS[strategy].fill_policy
    assert plan.pool_revision==0 and cb._pool_size==original_size
    assert len(plan.candidate_ids)==len(set(plan.candidate_ids))


@pytest.mark.parametrize('alias',list(ALIASES))
def test_aliases_keep_original_display_name_and_share_capabilities(alias):
    cb=ActiveLearningEngine(strategy=alias)
    assert cb.strategy==alias
    assert cb.acquisition.spec is strategy_spec(ALIASES[alias])


def test_initial_training_name_pruning_also_prunes_mutation_metadata():
    cb=ActiveLearningEngine(strategy='al4pde_random')
    cb.pool=full_pool(3)
    cb._generate_param_only_pool=lambda dm:None
    dm=SimpleNamespace(train_dataset_synth=TensorDataset(torch.zeros(1,39,2,4),
        torch.zeros(1,141),torch.zeros(1,39,4)),_mutant_names=['row1'])
    cb.al4pde_setup(dm,device='cpu')
    assert cb.pool.candidate_ids.tolist()==[0,2]
    assert cb._pool_mass_meta==[{'row':0},{'row':2}]
    assert cb._pool_ovr==[{'row':0},{'row':2}]


def test_each_acquisition_invalidates_cached_old_model_features():
    cb=ActiveLearningEngine(strategy='al4pde_diversity')
    cb._pool_size=1
    seen=[]
    def select(context,budget):
        seen.append(context.engine._pool_shape_features_cached)
        context.engine._pool_shape_features_cached=torch.ones(1,332)
        return context.engine.make_selection_plan([0])
    cb.acquisition.select=select
    cb._pool_shape_features_cached=torch.zeros(1,332)
    cb.select(None,None,'cpu');cb.select(None,None,'cpu')
    assert seen==[None,None]


def test_rejected_query_consumes_pool_ids_without_advancing_training_round():
    cb=ActiveLearningEngine(strategy='al4pde_random',initial_train_size=1)
    cb.pool=full_pool(3)
    cb.select=lambda *args:cb.make_selection_plan([2,0])
    def reject(indices,dm):
        mask=np.ones(cb._pool_size,dtype=bool);mask[indices]=False
        cb.pool.keep(mask)
        return None
    cb._build_broadcast_from_pool=reject
    assert cb.al4pde_select_and_append(None,None,device='cpu')==0
    assert cb.last_query.consumed_ids==(2,0)
    assert cb.last_query.n_accepted==0 and cb._al4pde_al_iter==0
    assert cb.pool.candidate_ids.tolist()==[1]


def test_query_batch_rejects_unaligned_labels():
    with pytest.raises(ValueError,match='alignment'):
        QueryBatch({'X':torch.zeros(2,1),'P':torch.zeros(1,1)},(),(),'al4pde_random')


def test_fission_al_and_unimplemented_latent_strategy_remain_rejected():
    with pytest.raises(ValueError,match='Fission'):
        ActiveLearningEngine().al4pde_setup(SimpleNamespace(system_id='fission'))
    with pytest.raises(ValueError,match='archived'):
        ActiveLearningEngine(strategy='coreset_maxdist')


class TinySurrogate(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.scale=torch.nn.Parameter(torch.tensor(.9))
    def forward_ic_time(self,x,p):return x[:,:,0,:]*self.scale
    def training_step(self,batch,batch_idx):
        x,p,y,*_=batch
        return torch.nn.functional.mse_loss(self.forward_ic_time(x,p),y)
    def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.01)


class TinyData(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(self.train_dataset_synth,batch_size=2)


def test_real_spawn_ode_train_append_snapshot_and_next_round(tmp_path,monkeypatch):
    # Controlled synthetic fixtures; no historical experiment or benchmark writes.
    monkeypatch.setenv('AL_SIM_WORKERS','2')
    pl.seed_everything(42,workers=True)
    solved=oracle.solve_single_mass_bridge('initial',{},return_biological_label=True)
    assert solved is not None
    _,y,p,pattern,bio=solved
    base=np.asarray(p)
    overrides=[{'mdt':float(100+10*i),'ksb2p':float(.002+.001*i),'init_Tem1':.123456789012345} for i in range(3)]
    params=np.array([[o.get(k,oracle._MASS_DEFAULTS[k]) for k in PARAM_KEYS] for o in overrides])
    states=np.array([prepare_initial_conditions(o)[1] for o in overrides])
    pool_path=tmp_path/'pool.npz'
    from budding_yeast_v2.data.systems.budding.al_pool import POOL_CONTRACT
    from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    np.savez(pool_path,pool_generation_contract=POOL_CONTRACT,catalog_version=CATALOG_VERSION,
        sampling_range_fingerprint=RANGE_FINGERPRINT,params=params,y_init=states,overrides_json=np.array([json.dumps(o) for o in overrides]),
        mutant_names=np.array([f'MASS_RAND_CLB2[GAL-OE]_L1_{i}' for i in range(3)]))
    source=tmp_path/'source.npz'
    np.savez(source,data=np.array([y,y,y],dtype=np.float32),params=np.array([p,p,p]),
        mutant_names=np.array(['1_fixture','initial_a','initial_b']),
        pattern_labels=np.array([pattern]*3),labels=np.array([bio]*3),param_names=np.array(PARAM_KEYS),
        levels=np.array([0,1,1]),module_counts=np.array([-1,1,1]))
    dm=TinyData()
    dm.system_id='budding';dm.dataset_path=str(source);dm.norm_type='linear';dm.t_max=210.
    dm.variable_names=oracle.VARIABLE_NAMES
    dm.mean_y=torch.zeros(1,39,1);dm.std_y=torch.ones(1,39,1)
    dm.p_mean=torch.zeros(1,141);dm.p_std=torch.ones(1,141)
    dm._Y=torch.full((1,39,500),100.);dm._test_idx_real=np.array([0])
    dm._train_idx_synth=np.array([1,2]);dm._mutant_names=['1_fixture','initial_a','initial_b']
    dm.pattern_labels=np.array([pattern]*3);dm.biological_labels=np.array([bio]*3)
    dm.biological_labels_available=np.ones(3,dtype=bool)
    x=torch.zeros(2,39,2,500);x[:,:,0,:]=torch.tensor(y[:,0])[None,:,None]
    x[:,:,1,:]=torch.linspace(0,1,500)
    dm.train_dataset_synth=TensorDataset(x,torch.tensor(np.tile(base,(2,1)),dtype=torch.float32),
        torch.tensor(np.tile(y,(2,1,1)),dtype=torch.float32),
        torch.tensor(abc_pattern_ids([pattern]*2),dtype=torch.long))
    engine=ActiveLearningEngine(strategy='al4pde_random',initial_train_size=1,al_pool_dataset=str(pool_path))
    engine.al4pde_setup(dm,device='cpu',exp_dir=str(tmp_path))
    counts=[]
    for round_index in range(3):
        model=TinySurrogate()
        trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=1,logger=False,
            enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
            limit_val_batches=0,num_sanity_val_steps=0)
        trainer.fit(model,datamodule=dm)
        assert model.scale.detach().item()!=pytest.approx(.9)
        if round_index<2:
            count=engine.al4pde_select_and_append(model,dm,device='cpu',exp_dir=str(tmp_path))
            counts.append(count)
    assert counts==[1,2]
    assert engine._pool_size==0 and len(dm.train_dataset_synth)==5
    path=tmp_path/'rounds/round_02/dataset.json'
    with open_dataset(path) as saved:
        assert saved['data'].shape==(6,39,500)
        assert saved['params'].shape==(6,141)
        assert saved['is_real'].tolist()==[True,False,False,False,False,False]
        assert len(saved['labels'])==len(saved['pattern_labels'])==len(saved['pattern_group_ids'])==6
        assert np.isfinite(saved['data']).all()
        assert set(saved['candidate_ids'][3:].tolist())=={0,1,2}
        for i in range(3,6):
            candidate_id=int(saved['candidate_ids'][i])
            assert json.loads(str(saved['overrides_json'][i]))==overrides[candidate_id]
            np.testing.assert_array_equal(saved['y_init'][i],states[candidate_id])
            assert saved['initial_state_source'][i]=='source_pool'
            assert json.loads(str(saved['mutation_pairs_json'][i]))==[['CLB2','GAL-OE']]
            assert saved['overrides_available'][i]
    with np.load(tmp_path/'acquisitions/after_round_01/added_samples.npz') as added:
        assert len(added['candidate_metadata_json'])==len(added['data'])==2
        for i,candidate_id in enumerate(added['candidate_ids']):
            assert json.loads(str(added['overrides_json'][i]))==overrides[int(candidate_id)]
            np.testing.assert_array_equal(added['y_init'][i],states[int(candidate_id)])
    assert (tmp_path/'acquisitions/after_round_01/selected.csv').exists()
    from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
    consumer=YeastDataModule(dataset_path=str(path),use_adj_matrix=False,num_workers=0,norm_type='linear')
    consumer.setup('fit')
    assert consumer.num_vars==39 and consumer.num_params==141
    assert len(consumer.test_dataset_real)==1


def test_changed_model_recomputes_full_pool_shape_features():
    cb,model,dm=fixture(ActiveLearningEngine,'al4pde_diversity',7)
    cb.select(model,dm,'cpu')
    first=cb._pool_shape_features_cached.clone()
    with torch.no_grad():model.offset.add_(.5)
    cb.select(model,dm,'cpu')
    assert not torch.equal(first,cb._pool_shape_features_cached)


def test_public_aliases_and_query_reject_wrong_strategy_plan():
    cb=ActiveLearningEngine(strategy='al4pde_random')
    cb.pool=full_pool(2)
    assert cb.setup.__func__ is cb.al4pde_setup.__func__
    assert cb.select_and_append.__func__ is cb.al4pde_select_and_append.__func__
    plan=cb.make_selection_plan([0])
    with pytest.raises(ValueError,match='different strategy'):
        cb.execute_plan(replace(plan,strategy='al4pde_bait'),None)
    with pytest.raises(ValueError,match='duplicate'):
        cb.pool.resolve(replace(plan,candidate_ids=(0,0)))


def test_pre_package_archive_is_retained_without_production_imports():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    assert (root/'utils/archive/al_callback_before_package.py').exists()
    for path in [root/'train.py',root/'utils/_compat.py',*(root/'utils/sampling').rglob('*.py')]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom):
                assert not (node.module or '').startswith('budding_yeast_v2.utils.archive')
