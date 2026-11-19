"""Explicit ownership transfer of live isolated jobs; no worker restart or identity relaxation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from budding_yeast_v2.data.dataset_archive import atomic_json, file_sha256


def process_identity(pid):
    try:
        path=Path(f'/proc/{int(pid)}')
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return dict(start_ticks=int(fields[19]),uid=path.stat().st_uid,pgid=os.getpgid(pid))
    except (FileNotFoundError,ProcessLookupError):return None


class AdoptedProcess:
    """Popen-shaped guard handle. Completion is verified, not a guessed OS exit code."""
    def __init__(self,task,identity,validator):
        self.pid=int(task['pid']);self.identity=identity;self.task=task;self.validator=validator
        self.returncode=None
        if identity is not None and (identity['uid']!=os.getuid() or identity['pgid']!=self.pid):
            raise RuntimeError('Refuse an unowned or non-isolated adopted group')
        actual=process_identity(self.pid)
        if identity is not None and actual is not None and actual!=identity:
            raise RuntimeError('PID identity changed; refuse to adopt a reused process')

    def poll(self):
        if self.returncode is not None:return self.returncode
        if self.identity is not None and process_identity(self.pid)==self.identity:return None
        try:self.validator(self.task)
        except (OSError,ValueError,KeyError,RuntimeError):self.returncode=1
        else:self.returncode=0
        return self.returncode

    def wait(self,timeout=None):
        started=time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic()-started>=timeout:
                raise subprocess.TimeoutExpired(['adopted',str(self.pid)],timeout)
            time.sleep(.05)
        return self.returncode


def load_handover(directory,expected_sources):
    directory=Path(directory)
    receipt=directory/'handover.json'
    data=json.loads(receipt.read_text())
    if data['sources']!=expected_sources:raise ValueError('Handover source inventory differs')
    if file_sha256(directory/'manifest.json')!=data['manifest_sha256']:
        raise ValueError('Handover manifest changed')
    if process_identity(data['supervisor_pid']) is not None:
        raise RuntimeError('Old supervisor still runs; refuse duplicate scheduling')
    # Validate live ownership before starting the new memory guard.
    for task in data['tasks']:
        if task['status']=='running':AdoptedProcess(task,data['processes'][task['task_id']],lambda _:None)
    return data


def detach_supervisor(directory):
    directory=Path(directory).resolve()
    status=json.loads((directory/'status.json').read_text());pid=int(status['queue_pid'])
    identity=process_identity(pid)
    if identity is None or identity['uid']!=os.getuid():raise RuntimeError('No owned live supervisor')
    argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if (not any(b'run_sampling_extensions.py' in a for a in argv) or b'--directory' not in argv
            or Path(os.fsdecode(argv[argv.index(b'--directory')+1])).resolve()!=directory):
        raise RuntimeError('Supervisor command does not match the queue')
    signal_fields=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
    if int(signal_fields['SigCgt'].strip(),16)&(1<<(signal.SIGTERM-1)):
        raise RuntimeError('Supervisor catches SIGTERM; refusing an unsafe detach')
    receipt=directory/'handover.json'
    if receipt.exists():raise ValueError('Handover already exists')
    os.kill(pid,signal.SIGSTOP)
    try:
        rows=json.loads((directory/'tasks.json').read_text())
        processes={}
        for task in rows:
            if task['status']!='running':continue
            current=process_identity(task['pid'])
            if current is not None:
                if current['uid']!=os.getuid() or current['pgid']!=task['pid']:
                    raise RuntimeError('Queue contains an unowned/non-isolated job')
                env=dict(x.split(b'=',1) for x in Path(f"/proc/{task['pid']}/environ").read_bytes().split(b'\0') if b'=' in x)
                if env.get(b'CUDA_VISIBLE_DEVICES')!=str(task['gpu']).encode() or env.get(b'WANDB_MODE')!=b'offline':
                    raise RuntimeError('Running job does not match its GPU/offline ledger')
            processes[task['task_id']]=current
        manifest=json.loads((directory/'manifest.json').read_text())
        atomic_json(receipt,dict(supervisor_pid=pid,supervisor_identity=identity,
            manifest_sha256=file_sha256(directory/'manifest.json'),sources=manifest['sources'],
            protocol_fingerprint=manifest['protocol_fingerprint'],tasks=rows,processes=processes,
            transferred_at=time.time(),worker_processes_restarted=False))
    except BaseException:
        os.kill(pid,signal.SIGCONT);raise
    # SIGTERM has its OS default action: no Python context unwinding, so the
    # guard does not terminate workers. Only the verified scheduler PID dies.
    os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
    deadline=time.monotonic()+10
    while process_identity(pid) is not None:
        if time.monotonic()>deadline:raise RuntimeError('Old supervisor did not exit')
        time.sleep(.05)
    for task in rows:
        if task['status']=='running' and processes[task['task_id']] is not None:
            actual=process_identity(task['pid'])
            if actual is not None and actual!=processes[task['task_id']]:raise RuntimeError('Worker identity changed')
    return data_summary(receipt)


def data_summary(receipt):
    data=json.loads(Path(receipt).read_text())
    return dict(receipt=str(receipt),live_jobs=sum(t['status']=='running' for t in data['tasks']),
                workers_restarted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--queue',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(detach_supervisor(args.queue)))
