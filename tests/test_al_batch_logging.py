"""Verify launch queues and W&B keys without training or network access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

from budding_yeast_v2.utils.sampling.logging import make_round_logger, log_round_evaluation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def explicit_legacy_seven_method_fixture(monkeypatch):
    # This file tests the historical subset and explicit overrides; the default
    # full fourteen-method launch is covered in test_al_training_budget.py.
    monkeypatch.setenv('AL_SUITE','original')
    monkeypatch.setenv('TRAINING_PROFILE','screen')


def test_round_wandb_keys_match_declared_axes(tmp_path):
    class Run:
        def __init__(self):self.definitions=[];self.rows=[]
        def define_metric(self,*args,**kwargs):self.definitions.append((args,kwargs))
        def log(self,row):self.rows.append(row)
    run=Run()
    for round_index in range(4):
        logger=make_round_logger(run,str(tmp_path),round_index)
        logger.log_metrics({'epoch':0,'train_loss':1.},step=0)
        assert f'round_{round_index:02d}/epoch' in run.rows[-1]
        assert f'round_{round_index:02d}/train_loss' in run.rows[-1]
    log_round_evaluation(run,dict(round=3,train_samples=70000,real_mutants_mae=.1,missing=None))
    assert run.rows[-1]=={'al/round':3,'al/train_samples':70000,'al/real_mutants_mae':.1}
    assert (('al/*',),dict(step_metric='al/train_samples',step_sync=True)) in run.definitions


def test_batch_dry_run_has_six_fno_baselines_plus_ours_and_no_outputs(tmp_path):
    env={**os.environ,'DRY_RUN':'1','EXP_ROOT':str(tmp_path/'results')}
    output=subprocess.check_output(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,text=True)
    lines=[line for line in output.splitlines() if line.startswith('GPU=')]
    assert len(lines)==7
    queues={0:('al4pde_random','al4pde_max_dist'),1:('al4pde_uncertainty','al4pde_lcmd'),2:('al4pde_power','ours_response'),3:('al4pde_bait',)}
    assert all('model=fno ' in line for line in lines)
    for gpu,strategies in queues.items():
        queue=[line for line in lines if line.startswith(f'GPU={gpu} ')]
        assert len(queue)==len(strategies)
        assert all(f'strategy={strategy} ' in line for line,strategy in zip(queue,strategies))
    assert not (tmp_path/'results').exists()


def fake_python(tmp_path):
    fake=tmp_path/'fake_python';calls=tmp_path/'calls'
    fake.write_text(f'#!{sys.executable}\n'+'''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
row=dict(args=args,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid(),wandb_mode=os.environ.get('WANDB_MODE'))
with open(os.environ['TASK_CALLS'],'a') as handle:handle.write(json.dumps(row)+'\\n')
if '--output' in args:Path(args[args.index('--output')+1]).touch()
if '--al_strategy' in args and args[args.index('--al_strategy')+1]==os.environ.get('TEST_FAIL_STRATEGY'):sys.exit(7)
if '-u' in args and os.environ.get('TEST_SLEEP'):__import__('time').sleep(30)
''')
    fake.chmod(0o755)
    return fake,calls


def test_batch_prepares_once_then_runs_each_queue_with_model_settings(tmp_path):
    fake,calls=fake_python(tmp_path)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),'DRY_RUN':'0'}
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True)
    rows=[json.loads(line) for line in calls.read_text().splitlines()]
    training=[row for row in rows if row['args'][0]=='-u']
    assert len(training)==7
    assert all(row['wandb_mode']=='offline' for row in training)
    assert sum('--recipe' in row['args'] for row in rows)==1
    settings={'fno':('192','512','4')}
    initial=[row for row in rows if 'budding_yeast_v2.data.systems.al_initial' in row['args']]
    assert len(initial)==1
    assert len({row['args'][row['args'].index('--dataset_name')+1] for row in training})==1
    assert len({row['args'][row['args'].index('--al_pool_dataset')+1] for row in training})==1
    assert all(row['args'][row['args'].index('--al_pool_dataset')+1].endswith('_gene_ranges.npz') for row in training)
    assert all(row['args'][row['args'].index('--dataset_name')+1].endswith('_gene_ranges.npz') for row in training)
    assert all('_refit_gene_ranges_screen/seed_42' in row['args'][row['args'].index('--exp_dir')+1] for row in training)
    for row in training:
        args=row['args'];model=args[args.index('--model')+1]
        assert tuple(args[args.index(key)+1] for key in ('--width','--batch_size','--n_blocks'))==settings[model]
        assert args[args.index('--lr')+1]=={'fno':'1e-3','deeponet':'1e-3','unet':'1e-3','transformer':'5e-4'}[model]
        schedule=tuple(args[args.index(key)+1] for key in ('--al_round_epochs','--al_final_epochs','--al_round_warmup','--al_final_warmup'))
        assert schedule==('50','100','4','8')
        assert args[args.index('--min_lr')+1]=='1e-5'
        assert row['gpu'] in {'0','1','2','3'}


def test_single_model_uses_one_gpu_and_generic_overrides(tmp_path):
    fake,calls=fake_python(tmp_path)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'MODEL':'fno','GPU_ID':'2','WIDTH':'96','BATCH_SIZE':'32',
         'STRATEGIES':'al4pde_random al4pde_power','WANDB':'0'}
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True)
    rows=[json.loads(line) for line in calls.read_text().splitlines()]
    training=[row for row in rows if row['args'][0]=='-u']
    assert len(training)==2
    for row in training:
        args=row['args']
        assert row['gpu']=='2' and args[args.index('--model')+1]=='fno'
        assert args[args.index('--width')+1]=='96' and args[args.index('--batch_size')+1]=='32'
        assert '--no_wandb' in args


def test_failed_gpu_queue_returns_failure_to_launcher(tmp_path):
    fake,calls=fake_python(tmp_path)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'MODELS':'fno','TEST_FAIL_STRATEGY':'al4pde_power'}
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True)
    assert result.returncode!=0
    import re
    logs=list((tmp_path/'results/al_budding/fno').glob('*/*/seed_*/train.log'))
    assert logs
    for path in logs:
        lines=path.read_text().splitlines()
        assert re.match(r'^\[\d{4}-\d\d-\d\d .*\+08:00\] START ',lines[0])
        assert '] END GPU=' in lines[-1]
        assert ('exit=7' if 'al4pde_power' in str(path) else 'exit=0') in lines[-1]


def test_fission_launcher_keeps_native_pool_and_uses_standard_fno(tmp_path):
    fake,calls=fake_python(tmp_path)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'STRATEGIES':'al4pde_random','GPU_ID':'3'}
    subprocess.run(['bash',str(ROOT/'run_sampling_fission.sh')],env=env,check=True,capture_output=True)
    rows=[json.loads(line) for line in calls.read_text().splitlines()]
    pool=rows[0]['args'];assert 'budding_yeast_v2.data.systems.fission.al_pool' in pool
    assert pool[pool.index('--samples')+1]=='500000'
    row=next(row for row in rows if row['args'][0]=='-u');args=row['args']
    assert row['gpu']=='3' and args[args.index('--model')+1]=='fno'
    assert args[args.index('--lr')+1]=='1e-3' and args[args.index('--modes')+1]=='24'
    assert args[args.index('--batch_size')+1]=='1024' and args[args.index('--width')+1]=='128'
    assert args[args.index('--al_round_epochs')+1]=='60' and args[args.index('--al_final_epochs')+1]=='120'
    assert args[args.index('--al_round_warmup')+1]=='6' and args[args.index('--al_final_warmup')+1]=='12'


def test_editable_rows_multiple_seeds_share_pool_and_keep_gpu_order(tmp_path):
    fake,calls=fake_python(tmp_path)
    rows=('fno hybrid_mse_rel al4pde_random 16 96 3 32 1e-3 40 80 3 6 2\n'
          'unet hybrid_mse_rel al4pde_power 24 24 3 16 5e-4 50 90 4 7 2')
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'EXPERIMENTS_TEXT':rows,'SEED_LIST':'42 1024','MAX_JOBS_PER_GPU':'1'}
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True)
    records=[json.loads(line) for line in calls.read_text().splitlines()]
    assert sum('--recipe' in row['args'] for row in records)==1
    initial=[row['args'] for row in records if 'budding_yeast_v2.data.systems.al_initial' in row['args']]
    assert len(initial)==2 and initial[0][initial[0].index('--output')+1]!=initial[1][initial[1].index('--output')+1]
    training=[row for row in records if row['args'][0]=='-u']
    assert [row['args'][row['args'].index('--seed')+1] for row in training]==['42','42','1024','1024']
    assert [row['args'][row['args'].index('--model')+1] for row in training]==['fno','unet','fno','unet']
    assert all(row['gpu']=='2' for row in training)
    assert training[0]['args'][training[0]['args'].index('--modes')+1]=='16'
    assert training[1]['args'][training[1]['args'].index('--al_final_epochs')+1]=='90'
    status=list((tmp_path/'results/al_budding/batch_logs').glob('*/tasks.tsv'))
    assert len(status)==1 and len(status[0].read_text().splitlines())==5


def test_failure_is_reported_and_next_experiment_on_same_gpu_runs(tmp_path):
    fake,calls=fake_python(tmp_path)
    rows=('fno hybrid_mse_rel al4pde_power 24 128 4 128 1e-3 50 100 4 8 0\n'
          'deeponet hybrid_mse_rel al4pde_random 24 256 3 256 1e-3 50 100 4 8 0')
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'EXPERIMENTS_TEXT':rows,'TEST_FAIL_STRATEGY':'al4pde_power'}
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True)
    assert result.returncode==1
    records=[json.loads(line) for line in calls.read_text().splitlines()]
    assert sum(row['args'][0]=='-u' for row in records)==2
    status=next((tmp_path/'results/al_budding/batch_logs').glob('*/tasks.tsv'))
    assert {line.split('\t')[2]:line.split('\t')[3] for line in status.read_text().splitlines()[1:]}=={'al4pde_power':'7','al4pde_random':'0'}


def test_duplicate_rows_rejected_before_data_preparation(tmp_path):
    row='fno hybrid_mse_rel al4pde_random 24 128 4 128 1e-3 50 100 4 8 0'
    env={**os.environ,'DRY_RUN':'1','EXPERIMENTS_TEXT':row+'\n'+row,'DATA_DIR':str(tmp_path/'data')}
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'Duplicate experiment output' in result.stderr
    assert not (tmp_path/'data').exists()


def test_fission_editable_list_supports_multiple_models_and_seeds(tmp_path):
    rows=('fno hybrid_mse_rel al4pde_random 24 128 4 128 1e-3 30 60 3 6 1\n'
          'deeponet hybrid_mse_rel al4pde_power 24 256 3 128 1e-3 30 60 3 6 1')
    env={**os.environ,'DRY_RUN':'1','EXPERIMENTS_TEXT':rows,'SEED_LIST':'42 1024','MAX_JOBS_PER_GPU':'1','DATA_DIR':str(tmp_path/'data')}
    output=subprocess.check_output(['bash',str(ROOT/'run_sampling_fission.sh')],env=env,text=True)
    tasks=[line for line in output.splitlines() if line.startswith('GPU=')]
    assert len(tasks)==4 and all('GPU=1 ' in line and 'epochs=30/60 ' in line for line in tasks)
    assert not (tmp_path/'data').exists()


def test_sigterm_stops_tracked_training_child(tmp_path):
    fake,calls=fake_python(tmp_path)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'MODEL':'fno','GPU_ID':'0','STRATEGIES':'al4pde_random','TEST_SLEEP':'1'}
    process=subprocess.Popen(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    child=None
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if calls.exists():
                rows=[json.loads(line) for line in calls.read_text().splitlines()]
                training=[row for row in rows if row['args'][0]=='-u']
                if training:child=training[0]['pid'];break
            time.sleep(.05)
        assert child is not None
        process.terminate();assert process.wait(timeout=5)==130
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            stat=Path(f'/proc/{child}/stat')
            if not stat.exists() or stat.read_text().split()[2]=='Z':break
            time.sleep(.05)
        else:raise AssertionError('Training child survived launcher SIGTERM')
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=5)


def test_malformed_width_rejected_before_generation(tmp_path):
    row='fno hybrid_mse_rel al4pde_random 24 typo 4 128 1e-3 50 100 4 8 0'
    env={**os.environ,'EXPERIMENTS_TEXT':row,'DATA_DIR':str(tmp_path/'data')}
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'positive integer' in result.stderr
    assert not (tmp_path/'data').exists()


def test_real_wandb_al_is_offline_even_when_environment_requests_online(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import wandb
    from budding_yeast_v2.utils.experiments.paths import setup_wandb_logger
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setenv('WANDB_SILENT', 'true')
    args=SimpleNamespace(model='fno', al_strategy='al4pde_random', system='budding',
                         seed=42, dataset_tag='al_local_monitor_check', no_wandb=False)
    logger=setup_wandb_logger(args, str(tmp_path), project='al_local_monitor_check')
    run=logger.experiment
    try:
        assert run.settings.mode in ('offline', 'dryrun')
        for round_index in range(4):
            round_logger=make_round_logger(run, str(tmp_path), round_index)
            # Each cold-start Trainer begins at step zero; all rounds must survive.
            round_logger.log_metrics({'epoch':0, 'train_loss':1./(round_index+1), 'val_loss':.5}, step=0)
        log_round_evaluation(run, dict(round=3,train_samples=70000,real_mutants_mae=.1))
        assert run.summary['al/real_mutants_mae']==.1
        assert all(run.summary[f'round_{r:02d}/train_loss']==1./(r+1) for r in range(4))
    finally:
        run.finish()
    assert list((tmp_path/'wandb').glob('offline-run-*/run-*.wandb'))


def test_skip_prepare_refuses_missing_pool_without_generating(tmp_path):
    fake,calls=fake_python(tmp_path)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'SKIP_PREPARE':'1','DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results')}
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==2 and 'requires existing pool' in result.stderr
    assert not calls.exists()


def test_skip_prepare_reuses_inputs_and_only_checks_existing_initial(tmp_path):
    fake,calls=fake_python(tmp_path)
    pool=tmp_path/'pool.npz';initial=tmp_path/'initial.npz'
    pool.touch();initial.touch()
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'SKIP_PREPARE':'1','POOL_FILE':str(pool),'INIT_DATASET':str(initial),
         'EXP_ROOT':str(tmp_path/'results'),'MODEL':'fno','STRATEGIES':'al4pde_random'}
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True)
    records=[json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records)==2
    assert '--check-existing' in records[0]['args']
    assert '--recipe' not in str(records)


def test_wait_for_gpu_delays_training_until_card_is_idle(tmp_path):
    fake,calls=fake_python(tmp_path)
    counter=tmp_path/'gpu_queries'
    query=tmp_path/'nvidia-smi'
    query.write_text('#!/bin/bash\n' + f'echo query >> "{counter}"\n' +
                     f'if [[ $(wc -l < "{counter}") -lt 2 ]]; then echo 9000; else echo 2; fi\n')
    query.chmod(0o755)
    env={**os.environ,'PYTHON_BIN':str(fake),'TASK_CALLS':str(calls),
         'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],'WAIT_FOR_GPU':'1','GPU_WAIT_SECONDS':'.01',
         'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
         'MODEL':'fno','STRATEGIES':'al4pde_random'}
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True)
    assert len(counter.read_text().splitlines())==2
    assert sum(json.loads(line)['args'][0]=='-u' for line in calls.read_text().splitlines())==1


def concurrent_fake_python(tmp_path):
    fake=tmp_path/'concurrent_python'
    fake.write_text(f'#!{sys.executable}\n'+'''import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
if '--output' in args:Path(args[args.index('--output')+1]).touch()
if '--al_strategy' not in args:sys.exit(0)
strategy=args[args.index('--al_strategy')+1]
def event(kind):
    row=json.dumps(dict(kind=kind,time=time.monotonic(),strategy=strategy,pid=os.getpid()))+'\\n'
    with open(os.environ['EVENTS'],'a') as f:f.write(row)
event('start')
time.sleep(float(json.loads(os.environ.get('DELAYS','{}')).get(strategy,.3)))
event('end')
sys.exit(7 if strategy==os.environ.get('TEST_FAIL_STRATEGY') else 0)
''')
    fake.chmod(0o755)
    return fake


def concurrent_environment(tmp_path,limit=2):
    fake=concurrent_fake_python(tmp_path)
    rows='\n'.join(f'fno hybrid_mse_rel {strategy} 24 192 4 512 1e-3 50 100 4 8 0'
                   for strategy in ('al4pde_random','al4pde_uncertainty','al4pde_power','al4pde_bait'))
    return {**os.environ,'PYTHON_BIN':str(fake),'EVENTS':str(tmp_path/'events'),
            'DATA_DIR':str(tmp_path/'data'),'EXP_ROOT':str(tmp_path/'results'),
            'EXPERIMENTS_TEXT':rows,'MAX_JOBS_PER_GPU':str(limit)}


def test_same_gpu_concurrency_is_bounded_and_refills_first_free_slot(tmp_path):
    env=concurrent_environment(tmp_path)
    env['DELAYS']=json.dumps({'al4pde_random':.9,'al4pde_uncertainty':.1,'al4pde_power':.1})
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True,timeout=10)
    events=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    active=peak=0
    for row in events:
        active+=1 if row['kind']=='start' else -1
        peak=max(peak,active)
        assert 0<=active<=2
    assert active==0 and peak==2
    ends={r['strategy']:r['time'] for r in events if r['kind']=='end'}
    starts={r['strategy']:r['time'] for r in events if r['kind']=='start'}
    assert starts['al4pde_power']<ends['al4pde_random']
    assert starts['al4pde_power']>=ends['al4pde_uncertainty']
    status=next((tmp_path/'results/al_budding/batch_logs').glob('*/tasks.tsv'))
    assert len(status.read_text().splitlines())==5


def test_same_gpu_serial_override_never_overlaps(tmp_path):
    env=concurrent_environment(tmp_path,limit=1)
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True,timeout=10)
    events=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    assert [row['kind'] for row in events]==['start','end']*4


def test_parallel_failure_halts_pending_but_drains_already_running_tasks(tmp_path):
    env=concurrent_environment(tmp_path)
    env.update(CONTINUE_ON_ERROR='0',TEST_FAIL_STRATEGY='al4pde_uncertainty',
               DELAYS=json.dumps({'al4pde_random':.5,'al4pde_uncertainty':.1}))
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True,timeout=10)
    assert result.returncode==1
    events=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    assert {row['strategy'] for row in events}=={'al4pde_random','al4pde_uncertainty'}
    assert sum(row['kind']=='end' for row in events)==2


def test_invalid_parallel_limit_rejected_before_creating_data(tmp_path):
    for value in ('0','-1','oops','1.5'):
        env={**os.environ,'MAX_JOBS_PER_GPU':value,'DATA_DIR':str(tmp_path/'data')}
        result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True,text=True)
        assert result.returncode==2 and 'positive integer' in result.stderr
    assert not (tmp_path/'data').exists()


def test_sigterm_stops_both_concurrent_training_children(tmp_path):
    env=concurrent_environment(tmp_path)
    env['DELAYS']=json.dumps({'al4pde_random':30,'al4pde_uncertainty':30})
    process=subprocess.Popen(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    children=[]
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            if (tmp_path/'events').exists():
                events=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
                children=[r['pid'] for r in events if r['kind']=='start']
                if len(children)==2:break
            time.sleep(.05)
        assert len(children)==2
        process.terminate();assert process.wait(timeout=5)==130
        for child in children:
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                stat=Path(f'/proc/{child}/stat')
                if not stat.exists() or stat.read_text().split()[2]=='Z':break
                time.sleep(.05)
            else:raise AssertionError('Concurrent training child survived SIGTERM')
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=5)


def test_parallel_default_continues_queued_work_after_failure(tmp_path):
    env=concurrent_environment(tmp_path)
    env.update(TEST_FAIL_STRATEGY='al4pde_uncertainty',
               DELAYS=json.dumps({'al4pde_random':.5,'al4pde_uncertainty':.01}))
    result=subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,capture_output=True,timeout=10)
    assert result.returncode==1
    events=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    assert len([r for r in events if r['kind']=='start'])==4
    status=next((tmp_path/'results/al_budding/batch_logs').glob('*/tasks.tsv'))
    codes={r.split('\t')[2]:int(r.split('\t')[3]) for r in status.read_text().splitlines()[1:]}
    assert codes==dict(al4pde_random=0,al4pde_uncertainty=7,al4pde_power=0,al4pde_bait=0)


def test_idle_wait_does_not_block_second_owned_parallel_slot(tmp_path):
    env=concurrent_environment(tmp_path)
    env['EXPERIMENTS_TEXT']='\n'.join(env['EXPERIMENTS_TEXT'].splitlines()[:2])
    counter=tmp_path/'gpu_queries';query=tmp_path/'nvidia-smi'
    query.write_text('#!/bin/bash\n'+f'echo query >> "{counter}"\n'+
                     f'if [[ $(wc -l < "{counter}") == 1 ]]; then echo 2; else echo 9000; fi\n')
    query.chmod(0o755)
    env.update(PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],WAIT_FOR_GPU='1',GPU_WAIT_SECONDS='.01')
    subprocess.run(['bash',str(ROOT/'run_sampling_budding.sh')],env=env,check=True,capture_output=True,timeout=5)
    assert len(counter.read_text().splitlines())==1
    events=[json.loads(line) for line in (tmp_path/'events').read_text().splitlines()]
    assert [row['kind'] for row in events[:2]]==['start','start']
