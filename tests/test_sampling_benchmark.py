import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from budding_yeast_v2.data.dataset_archive import write_dataset_manifest
from budding_yeast_v2.data.systems.budding.parameters import PARAM_KEYS, _defaults
from budding_yeast_v2.data.systems.budding.simulation import prepare_initial_conditions
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.models.baselines.fno import FNO
from budding_yeast_v2.utils.sampling.benchmark.storage import TrajectoryStore, read_selection
from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig, child_seed, effective_signatures, prepare_protocol
from budding_yeast_v2.utils.sampling.benchmark.contracts import PoolInputs, QueriedTraining, SamplingContext, LabelOracle
from budding_yeast_v2.utils.sampling.benchmark.methods import select
from budding_yeast_v2.utils.training.losses import HybridMSERelLoss


def make_pool(root, offset=0, count=28, steps=500, mode='KO'):
    root.mkdir(parents=True)
    params=np.asarray([[float(_defaults[k]) for k in PARAM_KEYS] for _ in range(count)])
    params[:,PARAM_KEYS.index('C0')]*=1+(.01*np.arange(count)+offset)
    if mode == 'MC':
        params[:,PARAM_KEYS.index('C0')]*=2
    initial=np.stack([prepare_initial_conditions(dict(zip(PARAM_KEYS,row)))[1] for row in params])
    t=np.linspace(0,1,steps)
    y=np.maximum(initial[:,:,None]*(1+.05*t[None,None,:])+ .01*t[None,None,:],0).astype(np.float32)
    for i in range(count):y[i]*=1+.01*i
    y[:,:,0]=initial
    groups=pattern_group_metadata(y)
    a=dict(params=params,y_init=initial,data=y,candidate_ids=np.arange(count)*3,
           mutant_names=np.array([f'sample-{i}' for i in range(count)]),is_real=np.zeros(count,dtype=bool),
           param_names=np.asarray(PARAM_KEYS),time_grid=np.linspace(0,210,steps),
           labels=np.asarray(['Inviable: G1 Arrest']*count),
           sampling_paradigm=np.array('mutation_random'),
           pattern_labels=np.asarray(['Pattern_C: Exponential-Arrest (==0)']*count),
           candidate_metadata_json=np.asarray([json.dumps(dict(pairs=[['CLN3',mode]]) if mode else {})]*count),
           **groups)
    blocks=[]
    for i in range(0,count,9):
        chunk={k:v[i:i+9] if v.ndim and len(v)==count else v for k,v in a.items()}
        path=root/f'chunk{i}.npz';np.savez_compressed(path,**chunk);blocks.append((path,None))
    write_dataset_manifest(root/'dataset.json',a,blocks)
    return root/'dataset.json',a


def config():
    return BenchmarkConfig(budgets=(2,4,7,10),epochs=(1,1,1,2),warmup=(0,0,0,0),holdout_fraction=.1,
                           width=4,modes=2,n_blocks=1,batch_size=4,prediction_batch_size=4,sketch_dim=3,lr=1e-4)


def test_lazy_alias_preserves_old_pickle_module_and_patch_identity():
    for old,new in [('active_learning.oracle','sampling.oracle'),('losses','training.losses'),
                    ('active_learning.strategies.random','sampling.baseline.random')]:
        a=importlib.import_module('budding_yeast_v2.utils.'+old)
        b=importlib.import_module('budding_yeast_v2.utils.'+new)
        assert a is b and a.__spec__.name==b.__name__
    import pickle
    name=b'budding_yeast_v2.utils.active_learning.oracle\n_solve_single_mass_task_wrapper\n'
    restored=pickle.loads(b'c'+name+b'.')
    from budding_yeast_v2.utils.sampling.oracle import _solve_single_mass_task_wrapper
    assert restored is _solve_single_mass_task_wrapper


def test_index_reads_keep_arbitrary_order_duplicates_and_exact_initial_dtype(tmp_path):
    path,a=make_pool(tmp_path/'pool')
    store=TrajectoryStore(path)
    rows=np.array([27,1,9,1,0])
    for k,v in store.read(rows,('params','y_init','data','candidate_ids')).items():
        np.testing.assert_array_equal(v,a[k][rows])
    store.snapshot(tmp_path/'selection.json',[27,1,9])
    np.testing.assert_array_equal(read_selection(tmp_path/'selection.json')['data'],a['data'][[27,1,9]])
    assert len(store._cache)<=2


def test_input_signature_equals_canonical_effective_input(tmp_path):
    import hashlib
    _,a=make_pool(tmp_path/'pool')
    actual=effective_signatures(a['params'],a['y_init'])
    for p,h in zip(a['params'],actual):
        vector=effective_inputs(dict(zip(PARAM_KEYS,p)),_defaults)
        assert h==hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()


def test_protocol_has_shared_grouped_splits_initial_and_seed_changes(tmp_path):
    p,_=make_pool(tmp_path/'pool')
    # Identical cross-pool identities MUST receive the same split assignment.
    root=prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',config())
    m=np.load(root/'mutation_split.npz');l=np.load(root/'lhs_split.npz')
    for role in ('candidates','validation','test'):np.testing.assert_array_equal(m[role],l[role])
    assert len(m['initial'])==2 and set(m['initial'])<=set(m['candidates'])
    assert not set(m['validation'])&set(m['candidates'])
    assert prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',config())==root
    from dataclasses import replace
    other=prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',replace(config(),seed=2026))
    assert other!=root
    assert not np.array_equal(np.load(other/'mutation_split.npz')['initial'],m['initial'])
    assert child_seed(42,'split')!=child_seed(42,'selection')


def test_oracle_forbids_holdout_and_al_candidate_truth_and_counts_unique(tmp_path):
    path,_=make_pool(tmp_path/'pool');store=TrajectoryStore(path)
    a=LabelOracle(store,[0,1,2],tmp_path/'a.json','A')
    with pytest.raises(PermissionError):a.scoring_reader([1])
    with pytest.raises(PermissionError):a.reveal([3])
    a.reveal([0,1]);a.reveal([1])
    assert a.revealed=={0,1} and a.accesses==3
    b=LabelOracle(store,[0,1,2],tmp_path/'b.json','B')
    b.scoring_reader([0,1,2]);assert len(b.revealed)==3


def test_fno_feature_interface_keeps_predictions_and_state_dict():
    torch.manual_seed(42);model=FNO(width=4,modes=2,n_blocks=1)
    x=torch.randn(2,39,2,12);x[:,:,1,:]=torch.linspace(0,1,12);p=torch.randn(2,141)
    expected=model(x,p);actual,latent=model(x,p,return_features=True)
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    assert latent.shape==(2,128,12)
    assert list(model.state_dict())==list(FNO(width=4,modes=2,n_blocks=1).state_dict())


def test_weighted_loss_is_linear_and_duplicate_aggregation_is_equivalent():
    torch.manual_seed(1);p=torch.randn(3,39,12,requires_grad=True);y=torch.randn_like(p)
    loss=HybridMSERelLoss();per=loss.per_sample(p,y)
    torch.testing.assert_close(loss(p,y,batch_weights=torch.tensor([1.,2.,0.])),(per*torch.tensor([1.,2.,0.])).mean())
    torch.testing.assert_close(loss(p,y,batch_weights=torch.ones(3)),loss(p,y))
    # Three equally weighted draw slots [0,0,1] aggregated into two unique rows.
    left=loss(p[[0,0,1]],y[[0,0,1]])
    right=loss(p[[0,1]],y[[0,1]],batch_weights=torch.tensor([4/3,2/3]))
    torch.testing.assert_close(left,right)
    right.backward();assert torch.isfinite(p.grad).all()


class FakePredictor:
    members=(object(),object())
    def uncertainty(self,inputs):return np.square(inputs.params[:,0])+np.arange(len(inputs.ids))*.01
    def features(self,inputs,dimension,latent=False):return np.column_stack((inputs.params[:,:2],inputs.initial[:,0])).astype(np.float32)


@pytest.mark.parametrize('name',['random','topk','power','max_dist_pred','coreset_ll','lcmd_ll','bait_ll','vessal_penultimate','lev_input'])
def test_author_and_geometric_selectors_use_input_views_and_repeat_seed(name):
    rng=np.random.default_rng(4);p=rng.normal(size=(20,141));ic=rng.normal(size=(20,39))
    pool=PoolInputs(np.arange(20),p,ic)
    train=QueriedTraining(PoolInputs(np.array([40,41]),p[:2],ic[:2]),np.zeros((2,39,12)))
    context=SamplingContext(pool,train,FakePredictor(),42,'cpu',3)
    a=select(name,context,4);b=select(name,context,4)
    assert len(a.ids)==len(set(a.ids))==4
    np.testing.assert_array_equal(a.ids,b.ids)
    assert not hasattr(context.pool,'trajectories') and context.label_reader is None


def test_unsupported_papers_fail_explicitly_instead_of_silent_random():
    with pytest.raises(NotImplementedError,match='hggs_projected'):select('hggs',None,10)


@pytest.mark.parametrize('name,budgets',[
    ('random',(2,4,7,10)),
    ('random',(2,4,6,8,10)),
    ('topk',(2,4,6,8,10)),
])
def test_configured_rounds_offline_and_resume_without_ode(tmp_path,name,budgets):
    from dataclasses import replace
    p,_=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=budgets,epochs=(1,)*(len(budgets)-1)+(2,),warmup=(0,)*len(budgets))
    root=prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',cfg)
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    with patch('budding_yeast_v2.data.systems.budding.simulation.simulate_mutant',side_effect=AssertionError('offline called ODE')):
        run=run_experiment(root,'mutation',name,tmp_path/'runs',diagnostic_only=True)
        assert run_experiment(root,'mutation',name,tmp_path/'runs',diagnostic_only=True)==run
    from budding_yeast_v2.utils.sampling.layout import experiment_directory
    assert run==experiment_directory(tmp_path/'runs',cfg,'mutation',name)
    complete=json.loads((run/'_state/state.json').read_text())['completion']
    assert complete['unique_training_inputs']==complete['unique_labels_accessed']==10
    assert complete['new_ode_calls']==0 and complete['diagnostic_only']
    assert set(complete['shared_holdout_rows']['validation'])=={'mutation'}
    assert set(complete['shared_holdout_rows']['test'])=={'mutation','lhs'}
    snapshots=sorted(run.glob('rounds/round_*/dataset.json'))
    rows=[json.loads(x.read_text())['rows'] for x in snapshots]
    assert [len(x) for x in rows]==list(budgets)
    assert all(set(left)<set(right) for left,right in zip(rows,rows[1:]))
    additions=sorted(run.glob('rounds/round_*/selection.json'))
    assert [len(json.loads(x.read_text())['selected_ids']) for x in additions]==list(np.diff(budgets))
    assert all(json.loads(x.read_text())['scoring_wait_seconds']>=0 for x in additions)
    if name=='topk':
        assert all((run/f'_state/round_{i:02d}/ensemble_member_1/model.ckpt').is_file()
                   for i in range(len(budgets)-1))
        assert not (run/f'_state/round_{len(budgets)-1:02d}/ensemble_member_1').exists()
        from budding_yeast_v2.utils.training.artifacts import read_history
        auxiliary=read_history(run/'_state/round_00/ensemble_member_1/history.csv')
        assert not any(key.startswith('test_') for row in auxiliary for key in row)
        primary=read_history(run/'rounds/round_00/history.csv')
        assert all('test_loss_synth' in row for row in primary)
    assert all(x.stat().st_size<10000 for x in snapshots)
    assert not list(run.rglob('*.npz'))
    assert (run/'loss.png').is_file()
    assert (run/'performance.csv').is_file()


def test_batch_and_worker_share_flat_result_path_for_both_access_modes(tmp_path,monkeypatch):
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    from argparse import Namespace
    from dataclasses import asdict
    from types import SimpleNamespace
    from budding_yeast_v2.utils.sampling.benchmark import cli,runner
    root=tmp_path/'protocol';root.mkdir()
    (root/'protocol.json').write_text(json.dumps(dict(protocol_id='example',config=asdict(config()))))
    output=tmp_path/'runs'
    args=Namespace(methods=['random','true_pattern_random'],gpus=['0'],jobs_per_gpu=1,
                   pools=['mutation'],dry_run=False,output=output,seed=42,plots=False,wandb=False)
    visited=[]
    def child(cmd,**kwargs):
        method=cmd[cmd.index('--method')+1]
        destination=runner.run_experiment(root,'mutation',method,output,diagnostic_only=True)
        from budding_yeast_v2.utils.sampling.layout import experiment_directory
        assert destination==experiment_directory(output,config(),'mutation',method)
        assert Path(kwargs['stdout'].name).parent==destination
        visited.append(destination)
        return SimpleNamespace(pid=99999999,returncode=0,poll=lambda:0,wait=lambda:0)
    with patch.object(cli,'_prepare',return_value=(root,tmp_path/'reference.npz')), \
         patch.object(cli.subprocess,'Popen',side_effect=child), \
         patch.object(runner,'_run',side_effect=lambda *a:a[5]):
        cli._batch(args)
    tasks=json.loads((output/'batches/seed_42/tasks.json').read_text())
    assert {row['directory'] for row in tasks}=={str(path) for path in visited}
    assert len(visited)==2
    assert not (output/'example/A').exists() and not (output/'example/B').exists()


@pytest.mark.parametrize('pool,scope',[('lhs','source'),('mutation','both')])
def test_validation_scope_keeps_source_selection_and_cross_pool_test(tmp_path,pool,scope):
    from dataclasses import replace
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    path,_=make_pool(tmp_path/'pool')
    cfg=replace(config(),budgets=(2,),epochs=(1,),warmup=(0,),validation_scope=scope)
    root=prepare_protocol({'mutation':path,'lhs':path},tmp_path/'protocol',cfg)
    result=run_experiment(root,pool,'random',tmp_path/'runs',diagnostic_only=True)
    counts=json.loads((result/'_state/state.json').read_text())['completion']['shared_holdout_rows']
    assert set(counts['validation'])==({pool} if scope=='source' else {'mutation','lhs'})
    assert set(counts['test'])=={'mutation','lhs'}


@pytest.mark.parametrize('name',['topk','power','lcmd_ll','bait_ll','vessal_penultimate','lev_input',
                                  'residual_topk','physics_pre_topk','physics_pre_power','true_pattern_random','ours_pattern_balance',
                                  'ours_pattern_context','ours_pattern_support'])
def test_offline_methods_complete_small_cold_start_chain(tmp_path,name):
    p,_=make_pool(tmp_path/'pool')
    root=prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',config())
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    with patch('budding_yeast_v2.data.systems.budding.simulation.simulate_mutant',side_effect=AssertionError('offline ODE')):
        result=run_experiment(root,'mutation',name,tmp_path/'runs',diagnostic_only=True)
    complete=json.loads((result/'_state/state.json').read_text())['completion']
    assert complete['unique_training_inputs']==10
    assert complete['unique_labels_accessed']==(len(np.load(root/'mutation_split.npz')['candidates']) if name in ('residual_topk','true_pattern_random') else 10)
    if name in ('topk','power','ours_pattern_balance','ours_pattern_context','ours_pattern_support'):
        assert (result/'_state/round_00/ensemble_member_1/model.ckpt').exists()
        assert not (result/'_state/round_03/ensemble_member_1').exists()


def test_production_manifest_without_optional_segment_count(tmp_path):
    path,a=make_pool(tmp_path/'pool')
    manifest=json.loads(path.read_text())
    for segment in manifest['segments']:segment.pop('count',None)
    path.write_text(json.dumps(manifest))
    store=TrajectoryStore(path)
    np.testing.assert_array_equal(store.read([27,9,0],('params',))['params'],a['params'][[27,9,0]])


def test_selection_manifest_reads_via_public_dataset_api_and_keeps_all_rows_train(tmp_path):
    path,a=make_pool(tmp_path/'pool');store=TrajectoryStore(path)
    target=tmp_path/'selected.json';store.snapshot(target,[27,1,9,2],provenance={'seed':2026})
    from budding_yeast_v2.data.dataset_archive import open_dataset,read_dataset_rows
    with open_dataset(target) as archive:
        np.testing.assert_array_equal(archive['data'],a['data'][[27,1,9,2]])
        assert archive['split_train'].tolist()==[0,1,2,3]
        assert not len(archive['split_val']) and int(archive['split_seed'])==2026
    np.testing.assert_array_equal(read_dataset_rows(target,[2,0]),a['data'][[9,27]])
    from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
    dm=YeastDataModule(dataset_path=str(target),use_adj_matrix=False,num_workers=0,seed=2026)
    dm.setup('fit')
    assert len(dm.train_dataset_synth)==4
    assert len(dm.val_dataset_synth)==len(dm.test_dataset_real)==0


def test_physics_event_mask_does_not_reject_large_continuous_growth():
    from budding_yeast_v2.utils.sampling.baseline.physics_pre import continuous_mask
    y=torch.zeros(1,39,500,dtype=torch.float64)
    y[:,0]=torch.exp(torch.linspace(0,5,500))*100
    y[:,2]=.3001
    y[:,33]=torch.exp(-torch.linspace(0,4,500))*10000
    assert continuous_mask(y).any() and continuous_mask(y).float().mean()>.95


@pytest.mark.parametrize('name',['clustering_sensitivity','clustering_sensitivity_wor'])
def test_clustering_sensitivity_native_weights_and_unique_adapter(tmp_path,name):
    p,_=make_pool(tmp_path/'pool');cfg=config()
    root=prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',cfg)
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    result=run_experiment(root,'mutation',name,tmp_path/'runs',diagnostic_only=True)
    complete=json.loads((result/'_state/state.json').read_text())['completion']
    if name=='clustering_sensitivity':
        assert 2<=complete['unique_training_inputs']<=10
        assert complete['training_draw_slots']==10
        selection=json.loads((result/'_state/one_shot.json').read_text())
        assert selection['diagnostics']['replacement'] and selection['weights'] is not None
        p_i=np.asarray(selection['diagnostics']['draw_probability'])
        counts=np.asarray(selection['diagnostics']['multiplicity'])
        universe=len(np.load(root/'mutation_split.npz')['candidates'])
        k=len(selection['selected_ids'])
        np.testing.assert_allclose(selection['weights'],k*counts/(10*universe*p_i))
        assert (result/'_state/pilot/primary/model.ckpt').exists()
    else:assert complete['unique_training_inputs']==10


def test_weighted_selection_is_not_silently_dropped_by_legacy_training(tmp_path):
    path,_=make_pool(tmp_path/'pool');store=TrajectoryStore(path)
    target=tmp_path/'selected.json';store.snapshot(target,[27,1,9,2],weights=[1.,2.,0.,3.])
    from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
    from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
    dm=YeastDataModule(dataset_path=str(target),use_adj_matrix=False,num_workers=0)
    dm.setup('fit');assert len(dm.train_dataset_synth.tensors)==5
    model=YeastLitModule(model_name='fno',width=4,modes=2,n_blocks=1,loss_type='hybrid_mse_rel')
    model.log=lambda *args,**kwargs:None
    x,p,y,pat,w=dm.train_dataset_synth.tensors
    expected=model.criterion(model.forward_ic_time(x,p),y,batch_weights=w)
    torch.testing.assert_close(model.training_step((x,p,y,pat,w),0),expected)


@pytest.mark.parametrize('pool_name,method',[('lhs','ours_pattern_balance'),('mutation','ours_sensitivity')])
def test_ours_lhs_metadata_free_and_mutation_response_paths(tmp_path,pool_name,method):
    p,_=make_pool(tmp_path/'pool',mode=None if pool_name=='lhs' else 'MC')
    root=prepare_protocol({'mutation':p,'lhs':p},tmp_path/'protocol',config())
    from budding_yeast_v2.utils.sampling.benchmark.runner import run_experiment
    result=run_experiment(root,pool_name,method,tmp_path/'runs',diagnostic_only=True)
    complete=json.loads((result/'_state/state.json').read_text())['completion']
    assert complete['unique_training_inputs']==complete['unique_labels_accessed']==10


def test_response_probes_preserve_final_effective_initial_override():
    from budding_yeast_v2.utils.sampling.ours.offline import _inputs
    from budding_yeast_v2.utils.sampling.ours.auxiliary.mutation_directions import probes
    override={'C0':3*_defaults['C0'],'init_CLB2':9.}
    p=np.asarray([override.get(k,_defaults[k]) for k in PARAM_KEYS])
    y=prepare_initial_conditions(override)[1]
    view=PoolInputs(np.array([7]),p[None],y[None],({'pairs':[['CLN3','MC']],
                    '_effective_overrides':override,'original_overrides':{'init_CLB2':1.}},))
    points=probes(_inputs(view),[0])
    assert len(points)==1
    np.testing.assert_array_equal(points[0]['center_y'],y)
    assert points[0]['y_minus'][2]==points[0]['y_plus'][2]==9.
