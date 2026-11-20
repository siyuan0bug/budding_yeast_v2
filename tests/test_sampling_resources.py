"""Health admission and file locks, tested without stressing the server."""
import multiprocessing as mp
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from budding_yeast_v2.utils.experiments import resources


def test_disk_counters_preserve_kernel_field_layout(monkeypatch):
    raw='8 48 sdd 10 2 100 5 20 3 200 8 1 13 21 0 0 0 0\n'
    raw+='7 0 loop0 1 0 2 3 4 0 5 6 0 7 8\n'
    monkeypatch.setattr(resources.Path,'read_text',lambda self:raw)
    assert resources.disk_counters()=={'sdd':dict(read_ios=10,read_sectors=100,
        read_ms=5,write_ios=20,write_sectors=200,write_ms=8,
        in_flight=1,busy_ms=13,weighted_ms=21)}


def test_disk_congestion_requires_busy_device_and_backlog():
    moderate=dict(busy_percent=50.,queue=4.,await_ms=40.)
    busy=dict(busy_percent=99.,queue=20.,await_ms=500.)
    assert resources.severe_disk_congestion({'sdd':moderate}) is False
    assert resources.severe_disk_congestion({'sdd':busy}) is True
    assert resources.severe_disk_congestion({}) is None


def test_disk_counter_rates_use_sector_and_millisecond_units():
    before=dict(read_ios=0,write_ios=0,read_sectors=0,write_sectors=0,
        read_ms=0,write_ms=0,busy_ms=0,weighted_ms=0)
    after=dict(read_ios=8,write_ios=2,read_sectors=4096,write_sectors=2048,
        read_ms=800,write_ms=200,busy_ms=1900,weighted_ms=4000)
    rate=resources.disk_activity({'sdd':before},{'sdd':after},2)['sdd']
    assert rate==dict(busy_percent=95.,queue=2.,await_ms=100.,read_mib_s=1.,write_mib_s=.5)


def _loader(directory, events, kind='data'):
    os.environ['SAMPLING_RESOURCE_DIR'] = directory
    with patch.object(resources, 'wait_for_resources'):
        with resources.resource_slot(kind, 'test'):
            events.put(('enter', os.getpid(), time.monotonic()))
            time.sleep(.1)
            events.put(('exit', os.getpid(), time.monotonic()))


@pytest.mark.parametrize('limit',[1,2,3])
@pytest.mark.parametrize('kind,variable',[('data','SAMPLING_IO_CONCURRENCY'),
                                         ('scoring','SAMPLING_SCORING_CONCURRENCY')])
def test_cross_process_loaders_respect_configured_limit(tmp_path,monkeypatch,limit,kind,variable):
    monkeypatch.setenv(variable,str(limit))
    context = mp.get_context('spawn')
    events = context.Queue()
    processes = [context.Process(target=_loader, args=(str(tmp_path), events, kind)) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    rows = sorted([events.get(timeout=2) for _ in range(6)], key=lambda x: x[2])
    active=peak=0
    for event,_,_ in rows:
        active+=1 if event=='enter' else -1
        peak=max(peak,active)
        assert 0<=active<=limit
    assert active==0 and peak>=1


@pytest.mark.parametrize('available,load,memory_psi,io_psi,blocked', [
    (200, 0, 0, 0, False), (10, 0, 0, 0, True),
    (200, 200, 0, 0, True), (200, 0, 2, 0, True), (200, 0, 0, 6, True)])
def test_health_blocks_memory_cpu_or_pressure(monkeypatch,available,load,memory_psi,io_psi,blocked):
    monkeypatch.setenv('SAMPLING_MAX_LOAD_PER_CPU', '1.5')
    monkeypatch.setenv('SAMPLING_MAX_IO_PSI','5')
    monkeypatch.setenv('SAMPLING_MAX_MEMORY_PSI','1')
    monkeypatch.setenv('SAMPLING_MIN_AVAILABLE_GIB','128')
    monkeypatch.setattr(resources, 'memory_status', lambda: (755 * 2**30, available * 2**30))
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda _: set(range(64)))
    monkeypatch.setattr(resources.os, 'getloadavg', lambda: (load, 0, 0))
    monkeypatch.setattr(resources, '_pressure', lambda name, _: memory_psi if name == 'memory' else io_psi)
    assert resources.resource_state()[0] is blocked


@pytest.mark.parametrize('available,load,memory_psi,blocked',[
    (200,0,0,False),(10,0,0,True),(200,200,0,True),(200,0,2,True)])
def test_diagnostic_io_warning_keeps_other_admission_guards(monkeypatch,available,load,memory_psi,blocked):
    monkeypatch.setenv('SAMPLING_IO_ADMISSION','warn')
    monkeypatch.setenv('SAMPLING_MIN_AVAILABLE_GIB','100')
    monkeypatch.setenv('SAMPLING_MAX_LOAD_PER_CPU','1.5')
    monkeypatch.setattr(resources,'memory_status',lambda:(755*2**30,available*2**30))
    monkeypatch.setattr(resources.os,'sched_getaffinity',lambda _:set(range(64)))
    monkeypatch.setattr(resources.os,'getloadavg',lambda:(load,0,0))
    monkeypatch.setattr(resources,'_pressure',lambda name,_:memory_psi if name=='memory' else 30.)
    assert resources.resource_state()[0] is blocked


@pytest.mark.parametrize('available,blocked',[(530.,False),(90.,True)])
def test_available_memory_admission_obeys_reserve_despite_psi(monkeypatch,available,blocked):
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')
    monkeypatch.setenv('SAMPLING_MIN_AVAILABLE_GIB','100')
    monkeypatch.setattr(resources,'memory_status',lambda:(755*2**30,available*2**30))
    monkeypatch.setattr(resources.os,'getloadavg',lambda:(0,0,0))
    monkeypatch.setattr(resources,'_pressure',lambda name,_:3. if name=='memory' else 0.)
    assert resources.resource_state()[0] is blocked


def test_exception_releases_admission_slot(tmp_path,monkeypatch):
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR', str(tmp_path))
    with patch.object(resources, 'wait_for_resources'):
        with pytest.raises(RuntimeError):
            with resources.resource_slot('data', 'interrupted loader'):
                raise RuntimeError('interrupted')
        with resources.resource_slot('data', 'next loader'):
            pass


def test_batch_global_cap_overrides_per_gpu_concurrency(tmp_path,monkeypatch):
    import json
    import threading
    from types import SimpleNamespace
    from budding_yeast_v2.utils.sampling.benchmark import cli
    root=tmp_path/'protocol';root.mkdir()
    (root/'protocol.json').write_text(json.dumps({'protocol_id':'example'}))
    monkeypatch.setenv('SAMPLING_MAX_TOTAL_JOBS','2')
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    args=SimpleNamespace(methods=['random','topk','lcmd_ll','bait_ll'],pools=['mutation','lhs'],
        gpus=['0','1','2','3'],jobs_per_gpu=2,dry_run=False,output=tmp_path/'output',
        seed=42,plots=False,wandb=False)
    active=peak=0
    lock=threading.Lock()
    next_pid=99900000
    def child(*args,**kwargs):
        nonlocal next_pid
        with lock:
            next_pid+=1;pid=next_pid
        class FakeChild:
            returncode=None
            def poll(self):return self.returncode
            def wait(self):
                nonlocal active,peak
                with lock:active+=1;peak=max(peak,active)
                time.sleep(.05)
                with lock:active-=1
                self.returncode=0
                return 0
        value=FakeChild();value.pid=pid
        return value
    with patch.object(cli,'_prepare',return_value=(root,tmp_path/'reference.npz')), \
         patch.object(cli,'wait_for_resources'), patch.object(resources,'wait_for_resources'), \
         patch.object(cli.subprocess,'Popen',side_effect=child):
        cli._batch(args)
    tasks=json.loads((args.output/'example/batch/tasks.json').read_text())
    assert len(tasks)==8 and peak==2


def test_exclusive_gpu_waits_for_existing_process(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','6')
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path))
    with patch.object(resources.subprocess,'check_output',side_effect=['777777\n','']), \
         patch.object(resources.time,'sleep') as sleep:
        with resources.exclusive_gpu_slot('cuda:0'):
            pass
    sleep.assert_called_once_with(5)


def test_exclusive_cards_have_one_slot_even_with_larger_job_setting(monkeypatch,capsys):
    import json
    from types import SimpleNamespace
    from budding_yeast_v2.utils.sampling.benchmark import cli
    monkeypatch.setenv('SAMPLING_MAX_TOTAL_JOBS','8')
    args=SimpleNamespace(methods=['random','topk','lcmd_ll','bait_ll'],pools=['mutation','lhs'],
        gpus=['0','6'],jobs_per_gpu=2,dry_run=True,seed=42)
    with patch.object(cli,'_prepare',side_effect=AssertionError('dry-run accessed data')):
        cli._batch(args)
    rows=[json.loads(row) for row in capsys.readouterr().out.splitlines()]
    assert [row['gpu'] for row in rows[:3]]==['0','6','0']
    assert all(row['jobs_per_gpu']==1 for row in rows if row['gpu']=='6')


@pytest.mark.parametrize('continue_on_error,expected',[('1',2),('0',1)])
def test_ordinary_failed_job_obeys_continue_policy(tmp_path,monkeypatch,continue_on_error,expected):
    import json
    from types import SimpleNamespace
    from budding_yeast_v2.utils.sampling.benchmark import cli
    root=tmp_path/'protocol';root.mkdir()
    (root/'protocol.json').write_text(json.dumps({'protocol_id':'example'}))
    monkeypatch.setenv('CONTINUE_ON_ERROR',continue_on_error)
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    args=SimpleNamespace(methods=['random','topk'],pools=['mutation'],gpus=['0'],jobs_per_gpu=1,
        dry_run=False,output=tmp_path/'output',seed=42,plots=False,wandb=False)
    codes=iter([1,0])
    def child(*args,**kwargs):
        code=next(codes)
        return SimpleNamespace(pid=99999999,returncode=code,poll=lambda:code,wait=lambda:code)
    with patch.object(cli,'_prepare',return_value=(root,tmp_path/'reference.npz')), \
         patch.object(cli,'wait_for_resources'),patch.object(resources,'wait_for_resources'), \
         patch.object(cli.subprocess,'Popen',side_effect=child):
        with pytest.raises(SystemExit) as error:cli._batch(args)
    assert error.value.code==1
    tasks=json.loads((args.output/'example/batch/tasks.json').read_text())
    assert len(tasks)==expected and tasks[0]['exit_code']==1


def test_mechanism_and_ordinary_scoring_have_separate_admission(tmp_path,monkeypatch):
    import fcntl
    from budding_yeast_v2.utils.experiments import resources
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path))
    monkeypatch.setenv('SAMPLING_SCORING_CONCURRENCY','1')
    monkeypatch.setenv('SAMPLING_MECHANISM_CONCURRENCY','2')
    monkeypatch.setenv('SAMPLING_IO_CONCURRENCY','1')
    monkeypatch.setattr(resources,'wait_for_resources',lambda phase:None)
    with (tmp_path/'cpu_scoring_0.lock').open('a') as occupied:
        fcntl.flock(occupied,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with resources.resource_slot('cpu_scoring','test independent long scorer'):
            assert (tmp_path/'cpu_scoring_1.lock').exists()
            with resources.resource_slot('scoring','test short scorer'):
                assert (tmp_path/'scoring_0.lock').exists()
