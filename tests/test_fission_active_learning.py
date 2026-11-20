"""Fission native warmup, generic acquisitions and actual AL train/query/reload."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset
import pytest
import torch
import pytorch_lightning as pl

from budding_yeast_v2.data.systems.fission.al_pool import generate_pool, INITIAL_STATE_PROTOCOL
from budding_yeast_v2.data.systems.fission.adapter import ADAPTER
from budding_yeast_v2.data.systems.fission.constants import MASS_UNIT
from budding_yeast_v2.data.systems.fission.parameters import WT_parameters, param_vector
from budding_yeast_v2.data.systems.fission.states import mutant_birth_state
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.sampling import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.dataset import pattern_target_ids
from budding_yeast_v2.utils.sampling.registry import SPECS, validate_system
from budding_yeast_v2.utils.sampling import fission

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = ('al4pde_random','al4pde_uncertainty','al4pde_power','al4pde_lcmd','al4pde_bait')


@pytest.fixture(scope='module')
def native_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp('fission_al')
    pool = directory/'pool.npz'
    generate_pool(pool, samples=9, seed=919191, workers=2)
    source = ROOT/'scratch/probes/compat_fission_random_8.npz'
    assert source.exists(), 'Native Fission integration trajectory fixture is required'
    return directory, pool, source


def make_dm(source, norm='linear'):
    dm = YeastDataModule(str(source), num_workers=0, batch_size_synth=2, norm_type=norm)
    dm.setup('fit')
    return dm


class Prediction(torch.nn.Module):
    def __init__(self, offset=0):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(float(offset)))
    def forward_ic_time(self, x, p):
        return x[:,:,0,:] + self.offset + p.square().mean(1)[:,None,None]*.01*x[:,:,1,:]


def test_warmed_pool_has_native_initial_states_and_no_labels(native_inputs):
    _, path, _ = native_inputs
    with np.load(path) as saved:
        assert 'data' not in saved and 'labels' not in saved and 'pattern_labels' not in saved
        assert saved['params'].shape == (9,58) and saved['y_init'].shape == (9,16)
        assert saved['time_grid'][-1] == 300 and len(saved['time_grid']) == 500
        assert json.loads(str(saved['generation_report']))['initial_state_protocol'] == INITIAL_STATE_PROTOCOL
        context = ADAPTER.prepare_context(None)
        for index in (0,4,8):
            overrides = json.loads(str(saved['overrides_json'][index]))
            expected = mutant_birth_state(overrides,context); expected[0] /= MASS_UNIT
            np.testing.assert_allclose(saved['y_init'][index],expected,rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(saved['params'][index],param_vector({**WT_parameters(),**overrides}))


@pytest.mark.parametrize('strategy', SUPPORTED)
def test_generic_selection_does_not_query_ode_and_preserves_fission_dimensions(native_inputs,strategy,monkeypatch):
    _, path, source = native_inputs
    dm=make_dm(source)
    engine=ActiveLearningEngine(strategy=strategy,initial_train_size=2,al_pool_dataset=str(path),bait_sketch_dim=4)
    engine.setup(dm,'cpu')
    engine.set_ensemble_members([Prediction(.1),Prediction(.3)])
    def forbidden(*args,**kwargs):raise AssertionError('ODE called while selecting')
    monkeypatch.setattr(fission,'simulate_task',forbidden)
    plan=engine.select(Prediction(),dm,'cpu')
    assert len(plan.candidate_ids)==2 and len(set(plan.candidate_ids))==2
    assert engine._get_pool_X_batch([0,1]).shape==(2,16,2,500)
    assert engine._pool_P.shape==(9,58)


@pytest.mark.parametrize('strategy',[name for name in SPECS if name not in SUPPORTED])
def test_budding_specific_acquisitions_rejected_before_pool_or_oracle(strategy):
    with pytest.raises(ValueError,match='not supported for fission'):
        ActiveLearningEngine(strategy=strategy).setup(SimpleNamespace(system_id='fission'),'cpu')
    assert validate_system(strategy,'budding') is SPECS[strategy]


def test_native_missing_pattern_is_not_mapped_to_budding_default():
    np.testing.assert_array_equal(pattern_target_ids(['Pattern_A','Pattern_B','Pattern_C','None',None],'fission'),[0,1,2,-1,-1])
    with pytest.raises(ValueError,match='Unknown original Fission pattern'):
        pattern_target_ids(['P5b'],'fission')


@pytest.mark.parametrize('bad_key,bad_value',[
    ('system_id',np.array('budding')),
    ('artifact_kind',np.array('trajectory_dataset')),
    ('param_names',np.array(['wrong']*58)),
    ('time_grid',np.linspace(0,210,500)),
    ('generation_report',np.array('{}')),
    ('overrides_json',np.array(['{}']*9)),
    ('y_init',np.zeros((9,39))),
])
def test_cross_system_or_malformed_pool_rejected(native_inputs,tmp_path,bad_key,bad_value):
    _, path, source=native_inputs
    with np.load(path) as archive:fields={key:archive[key] for key in archive.files}
    fields[bad_key]=bad_value
    bad=tmp_path/'bad.npz';np.savez(bad,**fields)
    engine=ActiveLearningEngine(al_pool_dataset=str(bad))
    with pytest.raises(ValueError):engine.setup(make_dm(source),'cpu')


def test_failed_and_inviable_queries_have_different_consumption_and_label_behavior(native_inputs,monkeypatch,tmp_path):
    _,path,source=native_inputs
    dm=make_dm(source);engine=ActiveLearningEngine(al_pool_dataset=str(path),initial_train_size=2)
    engine.setup(dm,'cpu')
    # The worker contract is separately exercised with real spawn below. This
    # controlled map forces solver failure and native inviable-without-pattern.
    class Executor:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(fission,'ProcessPoolExecutor',Executor)
    def rows(executor,worker,tasks,**kwargs):
        yield None
        name,overrides,context,horizon,steps=tasks[1]
        y=np.tile(engine._fission_initial_by_id[1][:,None],(1,steps))
        yield y,param_vector({**WT_parameters(),**overrides}),'None','Inviable: G2 Arrest',y[:,0]
    monkeypatch.setattr(fission,'bounded_ordered_map',rows)
    original=len(dm.train_dataset_synth)
    engine.acquisition.select=lambda context,budget:engine.make_selection_plan([0,1],dm)
    assert engine.select_and_append(Prediction(),dm,'cpu',str(tmp_path))==1
    assert engine.last_query.consumed_ids==(0,1) and engine.last_query.accepted_ids==(1,)
    assert engine._pool_size==7 and len(dm.train_dataset_synth)==original+1
    assert dm.train_dataset_synth.tensors[3][-1].item()==-1
    assert dm.pattern_labels[-1]=='None' and dm.biological_labels[-1]=='Inviable: G2 Arrest'
    assert dm.pattern_group_codes is None


def test_real_spawn_query_cold_restart_training_snapshot_and_reload(native_inputs,tmp_path,monkeypatch):
    _,path,source=native_inputs
    monkeypatch.setenv('AL_SIM_WORKERS','2')
    dm=make_dm(source,'log1p')
    engine=ActiveLearningEngine(al_pool_dataset=str(path),initial_train_size=1)
    engine.setup(dm,'cpu',str(tmp_path))
    frozen=dm.mean_y.clone(),dm.std_y.clone(),dm.p_mean.clone(),dm.p_std.clone()
    sizes=[len(dm.train_dataset_synth)]
    for round_index in range(2):
        model=YeastLitModule(model_name='cross_fno',num_vars=16,param_dim=58,system_id='fission',width=4,n_blocks=1,modes=2,loss_type='mse_only',warmup_epochs=0,max_epochs=1)
        trainer=pl.Trainer(accelerator='cpu',devices=1,fast_dev_run=True,logger=False,enable_checkpointing=False,enable_progress_bar=False)
        trainer.fit(model,datamodule=dm)
        if round_index==0:
            assert engine.select_and_append(model,dm,'cpu',str(tmp_path))>0
            sizes.append(len(dm.train_dataset_synth))
    assert sizes[1]>sizes[0]
    for expected,actual in zip(frozen,(dm.mean_y,dm.std_y,dm.p_mean,dm.p_std)):
        torch.testing.assert_close(actual,expected)
    snapshot=tmp_path/'rounds/round_01/dataset.npz'
    with open_dataset(snapshot) as saved:
        assert str(saved['system_id'])=='fission'
        assert saved['data'].shape[1:]==(16,500) and saved['params'].shape[1]==58
        assert saved['is_real'].sum()==20
        assert 'pattern_group_codes' not in saved and 'pattern_group_ids' not in saved
    reloaded=make_dm(snapshot)
    assert len(reloaded.test_dataset_real)==20 and reloaded.pattern_group_codes is None
    assert all(len(values)==len(dm._mutant_names) for values in (dm.biological_labels,dm.pattern_labels))


def test_empty_evaluation_splits_are_null_in_al_curve():
    from budding_yeast_v2.utils.sampling.io import round_record, metric_text
    report=round_record({'real_mutants':{'count':0,'all_variables':None},
                         'synth_test':{'count':0,'all_variables':None}},0,7)
    assert report['real_mutants_mae'] is None and report['synth_test_mae'] is None
    assert report['real_mutants_count']==report['synth_test_count']==0
    assert metric_text(report['synth_test_mae'])=='N/A'


def test_pool_excludes_existing_parameters_even_when_mutant_names_differ(native_inputs,tmp_path):
    _,pool,source=native_inputs
    dm=make_dm(source)
    with np.load(pool) as archive:
        existing=archive['params'][0].copy()
    held_out=tmp_path/'held_out.npz';np.savez(held_out,params=existing[None,:])
    dm.dataset_path=str(held_out)
    engine=ActiveLearningEngine(al_pool_dataset=str(pool))
    fission.load_pool(engine,dm)
    assert engine._pool_size==8
    assert not np.isclose(engine._pool_P_real.numpy(),existing,rtol=1e-5,atol=1e-9).all(axis=1).any()
