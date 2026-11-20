"""AL round data, deterministic splits and cumulative normalization contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset
import pytest
import torch

from budding_yeast_v2.data.systems.al_initial import random_split,select_representatives
from budding_yeast_v2.data.systems.al_initial import validate_existing_initial, SPLIT_POLICY
from budding_yeast_v2.data.systems.budding.specification import SPEC
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.utils.sampling.engine import ActiveLearningEngine
from budding_yeast_v2.utils.sampling.io import cache_source_metadata


def test_random_split_exact_sizes_seed_and_no_global_rng_changes():
    state=np.random.get_state()
    for n,expected in ((12500,(10000,1250,1250)),(6250,(5000,625,625))):
        split=random_split(np.arange(126,n+126),73)
        assert tuple(map(len,split))==expected
        np.testing.assert_array_equal(np.concatenate(split),np.random.RandomState(73).permutation(np.arange(126,n+126)))
        assert len(set(np.concatenate(split)))==n
    after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);assert state[2:]==after[2:]


def test_kmeans_selects_real_unique_rows_and_is_repeatable():
    params=np.random.RandomState(9).normal(size=(90,12))
    a=select_representatives(params,12,5);b=select_representatives(params,12,5)
    np.testing.assert_array_equal(a,b)
    assert len(set(a))==12 and a.min()>=0 and a.max()<len(params)


def test_kmeans_uses_supplied_pool_zscore_without_tail_compression(monkeypatch):
    # A layer subset must retain the full pool's scaling, including tail values.
    params=np.array([[1.,7.],[3.,7.],[101.,7.]])
    mean=np.array([1.,7.]);std=np.array([2.,0.])
    class CaptureKMeans:
        def __init__(self,**kwargs):pass
        def fit_predict(self,features):
            np.testing.assert_array_equal(features,[[0.,0.],[1.,0.],[50.,0.]])
            self.cluster_centers_=np.array([[0.,0.],[50.,0.]])
            return np.array([0,0,1])
    monkeypatch.setattr('budding_yeast_v2.data.systems.al_initial.MiniBatchKMeans',CaptureKMeans)
    np.testing.assert_array_equal(select_representatives(params,2,5,mean,std),[0,2])


def test_existing_initial_rejects_stale_rules_and_wrong_run(tmp_path):
    pool=tmp_path/'pool.npz';path=tmp_path/'initial.npz'
    report=dict(initial_parameter_transform='zscore',initial_parameter_fit_scope='full_source_pool',
                split_policy=SPLIT_POLICY,split_seed=42,source_pool=str(pool))
    def write():
        np.savez(path,generation_report=json.dumps(report),split_train=np.arange(8),split_seed=42)
    write();assert validate_existing_initial(path,pool,8,42)==str(path)
    with pytest.raises(ValueError,match='train size'):validate_existing_initial(path,pool,12,42)
    with pytest.raises(ValueError,match='different source pool'):validate_existing_initial(path,tmp_path/'other.npz',8,42)
    with pytest.raises(ValueError,match='different initialization rules'):validate_existing_initial(path,pool,8,43)
    report.pop('initial_parameter_transform');write()
    with pytest.raises(ValueError,match='new INIT_DATASET'):validate_existing_initial(path,pool,8,42)


def source(path, extra=0., labels=None):
    rng=np.random.RandomState(9);n=21
    data=rng.uniform(.01,1,(n,39,20)).astype('float32')
    params=np.tile(np.arange(141)/100.,(n,1))+rng.uniform(0,.1,(n,141))
    data[1:]+=extra;params[1:]+=extra
    np.savez_compressed(path,data=data,params=params,mutant_names=np.array(['1_WT']+[f'RANDOM_CLB2[MC]_L1_{i}' for i in range(20)]),
        is_real=np.r_[True,np.zeros(20,dtype=bool)],system_id=np.array('budding'),param_names=np.array(SPEC.param_names),
        variable_names=np.array(SPEC.variable_names),time_grid=np.linspace(0,210,20),
        pattern_labels=np.array(labels or ['Pattern_A']*n),labels=np.array(['Viable']*n),
        levels=np.r_[0,np.ones(20,dtype=int)],module_counts=np.r_[-1,np.ones(20,dtype=int)])


def dm(path,seed=9):
    module=YeastDataModule(str(path),seed=seed,num_workers=0,use_adj_matrix=False,norm_type='log1p')
    module.setup('fit');return module


def test_labels_do_not_influence_random_split_and_train_only_stats(tmp_path):
    a=tmp_path/'a.npz';b=tmp_path/'b.npz';source(a);source(b,labels=['Pattern_C']*21)
    state=np.random.get_state()
    left,right=dm(a),dm(b)
    after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);assert state[2:]==after[2:]
    assert left._train_idx_synth==right._train_idx_synth
    assert left._val_idx_synth==right._val_idx_synth
    assert left._test_idx_synth==right._test_idx_synth
    assert len(left.train_dataset_synth)==16
    torch.testing.assert_close(left.mean_y,right.mean_y,rtol=0,atol=0)


def test_snapshot_preserves_holdouts_L_and_refits_on_cumulative_train(tmp_path):
    initial=tmp_path/'initial.npz';source(initial);module=dm(initial)
    engine=ActiveLearningEngine(initial_train_size=16,random_seed=9)
    with np.load(initial) as archive:
        train=np.asarray(module._train_idx_synth)
        real=np.asarray(module._test_idx_real)
        engine._cum_synth_data=list(archive['data'][train]);engine._cum_synth_params=list(archive['params'][train])
        engine._cum_synth_names=list(archive['mutant_names'][train]);engine._cum_synth_labels=list(archive['pattern_labels'][train])
        engine._cum_synth_biological_labels=list(archive['labels'][train])
        engine._benchmark_real_data=list(archive['data'][real]);engine._benchmark_real_params=list(archive['params'][real])
        engine._benchmark_real_names=list(archive['mutant_names'][real]);engine._benchmark_real_labels=list(archive['pattern_labels'][real])
        engine._benchmark_real_biological_labels=list(archive['labels'][real]);engine._param_names_cached=archive['param_names']
        cache_source_metadata(engine,module,archive)
    # Add a finite extreme; it belongs only to train, never a held-out split.
    engine._cum_synth_data.append(np.full((39,20),1000.,dtype='float32'))
    engine._cum_synth_params.append(np.full(141,100.))
    engine._cum_synth_names.append('RANDOM_CLB2[MC]+CDH1[KO]_L2_new')
    engine._cum_synth_labels.append('Pattern_C');engine._cum_synth_biological_labels.append('Inviable: G1 Arrest')
    engine._cum_synth_levels.append(2);engine._cum_synth_requested_levels.append(2);engine._cum_synth_module_counts.append(2)
    snapshot=tmp_path/'round.npz';engine._save_al_snapshot(snapshot,module)
    changed=dm(snapshot)
    assert len(changed.train_dataset_synth)==17 and len(changed.val_dataset_synth)==2 and len(changed.test_dataset_synth)==2
    for split in ('val','test'):
        old=np.asarray(getattr(module,'_'+split+'_idx_synth'));new=np.asarray(getattr(changed,'_'+split+'_idx_synth'))
        np.testing.assert_array_equal(module._raw_data[old],changed._raw_data[new])
        np.testing.assert_array_equal(np.asarray(module._mutant_names)[old],np.asarray(changed._mutant_names)[new])
    assert not torch.equal(module.mean_y,changed.mean_y)
    expected=torch.log1p(torch.tensor(changed._raw_data[changed._train_idx_synth]).clamp(min=0)).mean((0,2),keepdim=True)
    torch.testing.assert_close(changed.mean_y,expected)
    with open_dataset(snapshot) as saved:
        assert saved['levels'][saved['split_train']][-1]==2
        assert saved['requested_levels'][saved['split_train']][-1]==2
        assert saved['module_counts'][saved['split_real']].tolist()==[-1]
    replay=YeastDataModule(str(snapshot),num_workers=0,use_adj_matrix=False,norm_type='log1p',preprocessing_state=changed.preprocessing_state())
    replay.setup('fit');torch.testing.assert_close(replay.mean_y,changed.mean_y,rtol=0,atol=0)


def test_linear_acquisition_multipliers_and_pool_limit():
    engine=ActiveLearningEngine(initial_train_size=10000);engine._pool_size=100000
    for i,n in enumerate((10000,20000,30000)):
        engine._al4pde_al_iter=i;assert engine._al4pde_scheduled_num_add()==n
    engine.fill_acquisition_budget=True;engine._pool_size=100
    with pytest.raises(RuntimeError,match='Insufficient AL'):engine._al4pde_scheduled_num_add()


def test_query_level_metadata_survives_pool_pruning(monkeypatch):
    # The oracle's alignment across failures is exercised by the real spawn
    # integration suite; ensure physical initial arrays follow generic pruning.
    engine=ActiveLearningEngine();engine._pool_size=3
    engine._pool_initial_physical=torch.arange(9).reshape(3,3)
    engine.pool.keep(np.array([False,True,True]))
    assert engine._pool_initial_physical.tolist()==[[3,4,5],[6,7,8]]


def test_failed_queries_refill_valid_budget_without_advancing_round(monkeypatch):
    from torch.utils.data import TensorDataset
    from budding_yeast_v2.utils.sampling.contracts import QueryBatch
    engine=ActiveLearningEngine(initial_train_size=3);engine._pool_size=8
    engine.fill_acquisition_budget=True
    engine._train_size_history=[]
    module=SimpleNamespace(system_id='budding',train_dataset_synth=TensorDataset(
        torch.empty(0,39,2,5),torch.empty(0,141),torch.empty(0,39,5)))
    requested=[];call=[0]
    def select(*args):
        requested.append(engine._al4pde_scheduled_num_add());return None
    def query(*args):
        n=1 if call[0]==0 else 2;attempted=3 if call[0]==0 else 2
        ids=engine.pool.ensure_ids().copy();accepted=ids[:n];consumed=ids[:attempted]
        engine.pool.keep(np.arange(engine._pool_size)>=attempted)
        batch=dict(X=torch.zeros(n,39,2,5),P=torch.zeros(n,141),Y=torch.zeros(n,39,5),
            names=[f'row{x}' for x in accepted],labels=['Pattern_C']*n,biological_labels=['Viable']*n,
            Y_real=[np.zeros((39,5))]*n,P_real=[np.zeros(141)]*n,candidate_ids=accepted,consumed_ids=consumed,
            levels=[1]*n,requested_levels=[1]*n,module_counts=[1]*n)
        call[0]+=1
        return QueryBatch(batch,tuple(consumed),tuple(accepted),engine.strategy)
    monkeypatch.setattr(engine,'select',select);monkeypatch.setattr(engine,'query',query)
    assert engine.select_and_append(None,module,'cpu')==3
    assert requested==[3,2] and engine._al4pde_al_iter==1
    assert len(module.train_dataset_synth)==3 and engine._pool_size==3


def test_experiment_analysis_reads_final_summary(tmp_path):
    from budding_yeast_v2.scripts.analysis.analyze_experiments import load_metrics_from_folder
    (tmp_path/'summary').mkdir();(tmp_path/'eval').mkdir()
    (tmp_path/'summary/global_metrics_denorm.json').write_text(json.dumps({'real_mutants':{'count':20}}))
    (tmp_path/'eval/global_metrics_denorm.json').write_text(json.dumps({'real_mutants':{'count':126}}))
    assert load_metrics_from_folder(str(tmp_path))['real_mutants']['count']==20


def test_loss_history_contains_first_epoch_train_reduction(tmp_path):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader,TensorDataset
    from budding_yeast_v2.utils.experiments.curves import TrainingCurveCallback
    class Tiny(pl.LightningModule):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
        def training_step(self,batch,index):
            loss=(self.weight-batch[0]).square().mean()
            self.log('train_loss',loss,on_step=False,on_epoch=True);return loss
        def validation_step(self,batch,index):self.log('val_loss_synth',self.weight.square()+1.)
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=0.)
    callback=TrainingCurveCallback(str(tmp_path),round_output=True)
    loader=DataLoader(TensorDataset(torch.tensor([1.,3.])),batch_size=1)
    trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=2,logger=False,
        callbacks=[callback],enable_checkpointing=False,enable_model_summary=False,enable_progress_bar=False,num_sanity_val_steps=0)
    trainer.fit(Tiny(),train_dataloaders=loader,val_dataloaders=loader)
    assert callback.curve_logger.epochs==[1,2]
    assert callback.curve_logger.train_loss==[5.,5.]
    assert (tmp_path/'loss_curves.png').is_file() and (tmp_path/'training_history.csv').is_file()
