"""A supervisor change must never restart or accidentally own somebody else's job."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from budding_yeast_v2.scripts.launch import queue_handover as handovers


def test_adopt_live_group_preserves_pid_and_reports_verified_completion():
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
    calls=[]
    try:
        identity=handovers.process_identity(child.pid)
        handle=handovers.AdoptedProcess(dict(pid=child.pid),identity,lambda row:calls.append(row))
        assert handle.poll() is None and not calls
        with pytest.raises(subprocess.TimeoutExpired):handle.wait(.05)
        child.terminate();child.wait(timeout=5)
        assert handle.wait(1)==0 and len(calls)==1
    finally:
        if child.poll() is None:child.kill();child.wait()


def test_reused_or_unowned_process_is_rejected(monkeypatch):
    monkeypatch.setattr(handovers,'process_identity',lambda pid:dict(start_ticks=2,uid=os.getuid(),pgid=999))
    with pytest.raises(RuntimeError,match='reused'):
        handovers.AdoptedProcess(dict(pid=999),dict(start_ticks=1,uid=os.getuid(),pgid=999),lambda _:None)
    with pytest.raises(RuntimeError,match='unowned'):
        handovers.AdoptedProcess(dict(pid=999),dict(start_ticks=2,uid=os.getuid()+1,pgid=999),lambda _:None)


def test_failed_adopted_process_is_not_marked_successful():
    def invalid(_):raise ValueError('Incomplete output')
    handle=handovers.AdoptedProcess(dict(pid=99999999),None,invalid)
    assert handle.poll()==1


def test_real_supervisor_detach_keeps_worker_and_rejects_duplicate_schedulers(tmp_path):
    # The finally block would stop the child during Python unwinding. Detach
    # must use verified OS-default SIGTERM and avoid that unwinding entirely.
    script=tmp_path/'run_sampling_extensions.py'
    script.write_text('''import subprocess,sys,os,json,time
from pathlib import Path
q=Path(sys.argv[-1]);child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'],start_new_session=True,
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',WANDB_MODE='offline'))
try:
    (q/'manifest.json').write_text(json.dumps(dict(sources={},protocol_fingerprint='same')))
    (q/'tasks.json').write_text(json.dumps([dict(task_id='a',pid=child.pid,gpu='0',status='running')]))
    (q/'status.json').write_text(json.dumps(dict(queue_pid=os.getpid())))
    time.sleep(90)
finally:child.terminate()
''')
    supervisor=subprocess.Popen([sys.executable,str(script),'--directory',str(tmp_path)],start_new_session=True)
    worker=None
    try:
        deadline=time.monotonic()+5
        while not (tmp_path/'status.json').exists():
            assert time.monotonic()<deadline;time.sleep(.05)
        worker=json.loads((tmp_path/'tasks.json').read_text())[0]['pid']
        birth=handovers.process_identity(worker)
        with pytest.raises(RuntimeError,match='still runs'):
            # Build a receipt-like fixture; load checks the live scheduler.
            (tmp_path/'handover.json').write_text(json.dumps(dict(supervisor_pid=supervisor.pid,sources={},
                manifest_sha256=handovers.file_sha256(tmp_path/'manifest.json'))))
            handovers.load_handover(tmp_path,{})
        (tmp_path/'handover.json').unlink()
        report=handovers.detach_supervisor(tmp_path)
        assert report['workers_restarted'] is False
        supervisor.wait(timeout=5)
        assert supervisor.returncode==-signal.SIGTERM
        assert handovers.process_identity(worker)==birth
        loaded=handovers.load_handover(tmp_path,{})
        assert loaded['processes']['a']==birth
        with pytest.raises(ValueError,match='source inventory'):handovers.load_handover(tmp_path,{'bad':'hash'})
    finally:
        if supervisor.poll() is None:supervisor.kill();supervisor.wait()
        if worker is not None and handovers.process_identity(worker) is not None:os.killpg(worker,signal.SIGTERM)


def test_ten_job_bounds_and_ramp_do_not_relax_exclusive_gpu_rule(tmp_path,capsys,monkeypatch):
    from dataclasses import asdict
    from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
    from budding_yeast_v2.scripts.launch import run_sampling_extensions as module
    protocol=tmp_path/'protocol';protocol.mkdir()
    (protocol/'protocol.json').write_text(json.dumps(dict(config=asdict(BenchmarkConfig()),protocol_id='fixture')))
    args=['--suite','ours','--protocol',str(protocol),'--directory',str(tmp_path/'queue'),
        '--gpus','0','1','--initial-jobs','6','--max-jobs','10','--jobs-per-gpu','5',
        '--mechanism-workers','4','--mechanism-concurrency','5','--ramp-step','2','--dry-run']
    module.main(args)
    assert json.loads(capsys.readouterr().out)['maximum_jobs']==10
    for flags in (['--gpus','4'],['--jobs-per-gpu','9'],['--io-concurrency','9']):
        with pytest.raises(ValueError,match='shared-GPU'):module.main(args+flags)
    assert not (tmp_path/'queue').exists()


def test_all_fifteen_admission_does_not_raise_cpu_phase_limit(tmp_path,capsys):
    from dataclasses import asdict
    from budding_yeast_v2.utils.sampling.benchmark.protocol import BenchmarkConfig
    from budding_yeast_v2.scripts.launch import run_sampling_extensions as module
    protocol=tmp_path/'protocol';protocol.mkdir()
    (protocol/'protocol.json').write_text(json.dumps(dict(config=asdict(BenchmarkConfig()),protocol_id='fixture')))
    module.main(['--suite','ours','--protocol',str(protocol),'--directory',str(tmp_path/'absent'),
        '--gpus','0','1','--initial-jobs','15','--max-jobs','15','--jobs-per-gpu','8',
        '--mechanism-workers','4','--mechanism-concurrency','5','--admit-all-pending','--dry-run'])
    data=json.loads(capsys.readouterr().out)
    assert data['maximum_jobs']==data['initial_jobs']==15 and data['count']==17
    rows=module.tasks('ours')
    active=[t for t in rows if t['kind']=='run' and t['method'] in
        ('ours_mechanism_witness','ours_dynamics_transport','ours_intervention_closure')]
    waiting=[t for t in rows if t['kind']=='generate']
    assert len(active)==5 and len(waiting)==5
    assert module.next_pending(waiting,active,5) is None
    assert module.next_pending(waiting,active,15)['kind']=='generate'
    assert not (tmp_path/'absent').exists()
