"""Resource probes cannot survive a lost supervisor; fast slots keep working."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

from budding_yeast_v2.utils.experiments import resources
from budding_yeast_v2.utils.sampling.benchmark import cli


def test_probe_cleanup_continues_if_one_child_already_exited(monkeypatch):
    from budding_yeast_v2.scripts.analysis import profile_sampling_concurrency as probe
    first=SimpleNamespace(pid=99999997,poll=lambda:None,wait=lambda timeout:0)
    second=SimpleNamespace(pid=99999998,poll=lambda:None,wait=lambda timeout:0)
    def group(pid):
        if pid==first.pid:raise ProcessLookupError('Already exited')
        return pid
    with patch.object(probe.os,'getpgid',side_effect=group),patch.object(probe.os,'killpg') as kill:
        probe.stop_owned([first,second])
    kill.assert_called_once_with(second.pid,__import__('signal').SIGTERM)


def test_parent_death_stops_registered_probe(tmp_path):
    marker = tmp_path/'terminated'
    child = '''
import os, signal, sys, time
from pathlib import Path
from budding_yeast_v2.scripts.analysis.profile_sampling_resources import arm_resource_supervisor
def stopped(*args):
    Path(sys.argv[1]).write_text('terminated by parent-death signal')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stopped)
arm_resource_supervisor()
print('ready',flush=True)
time.sleep(10)
'''
    parent = '''
import os, subprocess, sys
p = subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]],
    env=dict(os.environ,SAMPLING_PROBE_SUPERVISOR_PID=str(os.getpid())),
    stdout=subprocess.PIPE,text=True)
assert p.stdout.readline().strip()=='ready'
'''
    result = subprocess.run([sys.executable,'-c',parent,child,str(marker)],timeout=8)
    assert result.returncode == 0
    deadline = time.monotonic()+5
    while not marker.exists() and time.monotonic()<deadline:
        time.sleep(.02)
    assert marker.exists()


def test_fast_slot_takes_remaining_work_while_other_slot_is_busy(tmp_path,monkeypatch):
    root = tmp_path/'protocol';root.mkdir()
    (root/'protocol.json').write_text(json.dumps({'protocol_id':'example'}))
    monkeypatch.setenv('SAMPLING_MAX_TOTAL_JOBS','2')
    monkeypatch.setenv('SAMPLING_RESOURCE_DIR',str(tmp_path/'locks'))
    args=SimpleNamespace(methods=['random','topk','lcmd_ll','bait_ll'],pools=['mutation'],
        gpus=['0','1'],jobs_per_gpu=1,dry_run=False,output=tmp_path/'runs',seed=42,plots=False,wandb=False)
    ready=threading.Event();done=[];lock=threading.Lock()
    def child(cmd,**kwargs):
        method=cmd[cmd.index('--method')+1]
        class Child:
            pid=99999999
            returncode=None
            def poll(self):return self.returncode
            def wait(self):
                if method=='random':
                    assert ready.wait(2), 'Fast slot left pending work in the slow queue'
                else:
                    with lock:
                        done.append(method)
                        if len(done)==3:ready.set()
                self.returncode=0
                return 0
        return Child()
    with patch.object(cli,'_prepare',return_value=(root,tmp_path/'reference.npz')), \
         patch.object(cli,'wait_for_resources'),patch.object(resources,'wait_for_resources'), \
         patch.object(cli.subprocess,'Popen',side_effect=child):
        cli._batch(args)
    tasks=json.loads((args.output/'example/batch/tasks.json').read_text())
    assert len(tasks)==4 and len({t['method'] for t in tasks})==4
