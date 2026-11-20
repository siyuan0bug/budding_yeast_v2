"""Background extension tasks must remain bounded, immutable and input-audited."""
import json
import numpy as np
import pytest

from budding_yeast_v2.scripts.launch.run_sampling_extensions import tasks,command,main,validate_result,failed_tasks
from budding_yeast_v2.utils.sampling.query import LabelOracle
from budding_yeast_v2.data.dataset_archive import TrajectoryStore
from test_sampling_benchmark import make_pool


def test_queue_has_only_unfinished_paper_methods_and_required_controls():
    rows=tasks();assert len(rows)==30
    assert len({t['task_id'] for t in rows})==30
    completed={'random','topk','power','max_dist_pred','coreset_ll','lcmd_ll','bait_ll','vessal_penultimate'}
    assert not {t['method'] for t in rows}&completed
    assert {t['pool'] for t in rows}=={'mutation','lhs'}
    assert {t['method'] for t in rows if t['kind']=='dynamic'}=={'peaks_regression','eracs_regression','random_stream','random_dynamic'}
    for row in rows:
        row['output']='/tmp/results'
        cmd=command(row,'/tmp/protocol','/tmp/spec.json')
        assert cmd[2] in ('run','generate','dynamic') and 'extensions' not in cmd
        assert '--wandb' in cmd and '--device' in cmd
        if row['kind']=='generate':assert cmd[cmd.index('--workers')+1]=='2'
        if row['kind']=='generate' and row['pool']=='lhs':assert '--lhs-config' in cmd


def test_ours_suite_has_current_methods_on_supported_pools_and_both_query_sources():
    from budding_yeast_v2.utils.sampling.registry import METHODS
    rows=tasks('ours')
    assert len(rows)==len({r['task_id'] for r in rows})==17
    assert sum(r['kind']=='run' for r in rows)==12
    assert sum(r['kind']=='generate' for r in rows)==5
    assert all(not METHODS[r['method']].mutation_only or r['pool']=='mutation' for r in rows)
    for row in rows:
        row['output']='/tmp/ours'
        cmd=command(row,'/tmp/protocol','/tmp/spec.json')
        if row['kind']=='generate':
            assert cmd[cmd.index('--bank-size')+1]=='64'
            assert cmd[cmd.index('--query-batch-size')+1]=='8'


def test_ours_queue_dry_run_is_budding_fno_seed42_and_uses_only_requested_gpus(tmp_path,capsys):
    from dataclasses import asdict
    from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
    root=tmp_path/'protocol';root.mkdir()
    (root/'protocol.json').write_text(json.dumps(dict(config=asdict(BenchmarkConfig()),protocol_id='fixture')))
    main(['--suite','ours','--protocol',str(root),'--directory',str(tmp_path/'absent'),
          '--gpus','0','1','--max-jobs','4','--initial-jobs','4','--dry-run'])
    report=json.loads(capsys.readouterr().out)
    assert report['count']==17 and report['gpus']==['0','1']
    assert not (tmp_path/'absent').exists()


def test_queue_rejects_exclusive_cards_before_any_files(tmp_path):
    with pytest.raises(ValueError,match='shared-GPU'):
        main(['--directory',str(tmp_path/'absent'),'--gpus','4'])
    assert not (tmp_path/'absent').exists()


def test_queue_can_rerun_only_ogas_on_both_pools_without_marking_old_runs_failed(tmp_path,capsys):
    from dataclasses import asdict
    from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
    protocol=tmp_path/'protocol';protocol.mkdir()
    (protocol/'protocol.json').write_text(json.dumps(dict(config=asdict(BenchmarkConfig()),protocol_id='fixture')))
    main(['--protocol',str(protocol),'--directory',str(tmp_path/'queue'),'--results-root',str(tmp_path/'new'),
          '--methods','ogas_async_projected','--initial-jobs','2','--max-jobs','2','--dry-run'])
    report=json.loads(capsys.readouterr().out)
    assert report['count']==2
    assert {r['method'] for r in report['tasks']}=={'ogas_async_projected'}
    assert {r['pool'] for r in report['tasks']}=={'mutation','lhs'}
    assert all(r['status']=='pending' and r['output']==str(tmp_path/'new') for r in report['tasks'])
    assert not (tmp_path/'queue').exists()


def test_failed_only_rerun_preserves_identity_but_not_previous_state_or_output():
    rows=tasks();previous=[dict(row,status='complete') for row in rows]
    previous[0].update(status='failed',output='/old',pid=123,result_directory='/old/failed',exit_code=1)
    new=failed_tasks(rows,previous)
    assert len(new)==1 and new[0]['status']=='pending'
    assert new[0]['rerun_from_result_directory']=='/old/failed'
    assert 'pid' not in new[0] and 'output' not in new[0]
    previous[0]['method']='changed'
    with pytest.raises(ValueError,match='Unknown/changed'):failed_tasks(rows,previous)
    with pytest.raises(ValueError,match='no failed'):failed_tasks(rows,rows)


def test_success_requires_real_metrics_csvs_and_figures(tmp_path):
    task=dict(kind='dynamic',result_directory=str(tmp_path))
    (tmp_path/'_state').mkdir();(tmp_path/'eval/real_mutants').mkdir(parents=True)
    (tmp_path/'_state/completion.json').write_text('{"status":"complete"}')
    (tmp_path/'eval/global_metrics_denorm.json').write_text(json.dumps(dict(real_mutants=dict(count=126),synth_test=dict(count=2))))
    (tmp_path/'eval/per_mutant_metrics.csv').write_text('id\n'+'\n'.join(map(str,range(126)))+'\n')
    (tmp_path/'eval/per_synth_test_metrics.csv').write_text('id\n0\n1\n')
    with pytest.raises(ValueError,match='figures'):validate_result(task)
    for i in range(126):(tmp_path/f'eval/real_mutants/{i}.png').write_bytes(b'fixture')
    assert validate_result(task)==dict(real_mutants=126,synth_test=2,real_figures=126)


def test_oracle_ram_cache_changes_no_rows_or_authorized_label_counts(tmp_path,monkeypatch):
    pool,_=make_pool(tmp_path/'pool')
    small=TrajectoryStore(pool,cache_chunks=2);large=TrajectoryStore(pool,cache_chunks=2)
    reference=LabelOracle(small,np.arange(20),tmp_path/'small.json',track='B')
    monkeypatch.setenv('SAMPLING_ORACLE_CACHE_CHUNKS','16')
    cached=LabelOracle(large,np.arange(20),tmp_path/'large.json',track='B')
    monkeypatch.setattr('budding_yeast_v2.utils.experiments.resources.resource_state',lambda:(False,{}))
    for rows in ([0,10,19],[19,0],[1,11,18]):
        np.testing.assert_array_equal(reference.reveal(rows)['data'],cached.reveal(rows)['data'])
    assert small.cache_chunks==2 and large.cache_chunks==16
    assert reference.revealed==cached.revealed and reference.accesses==cached.accesses
    assert len(large._cache)<=16
    with pytest.raises(PermissionError):cached.reveal([20])
    monkeypatch.setenv('SAMPLING_ORACLE_CACHE_CHUNKS','0')
    with pytest.raises(ValueError,match='RAM cache'):LabelOracle(large,[],tmp_path/'bad.json')


def test_dynamic_adapter_does_not_hold_a_preprocessing_io_slot(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from test_shared_training_cleanup import selected_fixture
    from test_sampling_benchmark import config
    from budding_yeast_v2.utils.sampling.benchmark import rounds
    _,_,snapshot,training,validation=selected_fixture(tmp_path)
    inside=[False]
    @contextmanager
    def slot(kind,phase):
        inside[0]=True
        try:yield
        finally:inside[0]=False
    def adapter(dm,norm):
        assert not inside[0], 'Resident label query would need a nested data slot'
        assert len(dm.train_dataset_synth)==len(training.inputs.ids)
    monkeypatch.setattr(rounds,'resource_slot',slot)
    rounds.fit_round(config(),0,training,validation,tmp_path/'fit',42,
                     dataset_path=snapshot,data_adapter=adapter)


def test_dynamic_initial_and_resident_queries_never_nest_data_admission(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from test_additional_regression_baselines import protocol
    from budding_yeast_v2.utils.sampling.dynamic import runner,callback
    from budding_yeast_v2.utils.sampling.benchmark import rounds
    from budding_yeast_v2.utils.sampling import query
    depth={}
    @contextmanager
    def slot(kind,phase):
        if kind=='data':assert not depth.get(kind,0),f'Nested data admission: {phase}'
        depth[kind]=depth.get(kind,0)+1
        try:yield
        finally:depth[kind]-=1
    for module in (runner,callback,rounds,query):monkeypatch.setattr(module,'resource_slot',slot)
    root=protocol(tmp_path)
    runner.run_dynamic(root,'mutation','random_stream',tmp_path/'runs',diagnostic_only=True,
        dynamic_config=dict(epochs=3,warmup=0,bootstrap_epochs=1,acquisition_epochs=2,
                            steps_per_epoch=1,delta=2,stream_batch=4,sketch_dim=4))
    assert not any(depth.values())


def test_long_cpu_tasks_do_not_block_pending_short_methods():
    from budding_yeast_v2.scripts.launch.run_sampling_extensions import next_pending
    rows=tasks('ours')
    active=[rows[0],rows[1]]
    waiting=[dict(r) for r in rows[4:]]
    selected=next_pending(waiting,active,2)
    assert selected['method']=='ours_sensitivity'
    assert next_pending(waiting,active[:1],2)['method']=='ours_intervention_closure'
    assert next_pending([waiting[0]],active,2) is None
    assert next_pending([],active,2) is None
