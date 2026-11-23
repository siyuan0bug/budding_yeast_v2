"""Small dependency scheduler; policies supply commands and result validation.

No model, biological rule or study decision lives here. Only Popen-owned
isolated process groups are registered with the existing batch memory guard.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from budding_yeast_v2.data.dataset_archive import atomic_json
from .memory_guard import BatchMemoryGuard
from .resources import resource_state


def pending_task(tasks, skip):
    statuses = {task['id']: task['status'] for task in tasks}
    for task in tasks:
        if task['status'] != 'pending':
            continue
        if any(statuses[d] in ('failed', 'blocked') for d in task['dependencies']):
            task['status'] = 'blocked'; statuses[task['id']] = 'blocked'
            continue
        if all(statuses[d] in ('complete', 'skipped') for d in task['dependencies']):
            if skip(task):
                task['status'] = 'skipped'; statuses[task['id']] = 'skipped'
                continue
            return task
    return None


def gpu_free():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {k.strip(): float(v)/1024 for k, v in (line.split(',') for line in raw.splitlines())}


def run_queue(tasks, directory, *, command, validate, skip, verify_sources,
              cwd, environment, gpus=('0', '1', '2', '3'), max_jobs=8, jobs_per_gpu=2):
    directory = Path(directory)
    (directory/'logs').mkdir(parents=True, exist_ok=True)
    active = {}; started = time.monotonic(); notice = 0.
    with BatchMemoryGuard(directory/'resources.jsonl') as guard:
        while any(t['status'] in ('pending', 'running') for t in tasks):
            for pid, (child, task, stream, tracking) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                tracking.__exit__(None, None, None); stream.close(); active.pop(pid)
                task.update(status='complete' if code == 0 else 'failed', exit_code=code,
                    seconds=time.time()-task['started_wall'], finished_wall=time.time())
                if code == 0:
                    try: validate(task)
                    except Exception as exc: task.update(status='failed', failure=str(exc))
                print(f"END {task['id']} {task['status']} exit={code}", flush=True)
            if guard.aborted:
                atomic_json(directory/'tasks.json', tasks)
                raise RuntimeError(guard.reason)
            task = pending_task(tasks, skip)
            blocked, health = resource_state()
            counts = {g: sum(t.get('gpu') == g for _, t, _, _ in active.values()) for g in gpus}
            free = gpu_free() if task and task.get('gpu_required', True) else {}
            available = [g for g in gpus if counts[g] < jobs_per_gpu and free.get(g, 0) >= 8.]
            can_start = (task is not None and len(active) < max_jobs and not blocked
                         and health['available_gib'] >= 140
                         and (available or not task.get('gpu_required', True)))
            if can_start:
                verify_sources()
                gpu = min(available, key=lambda g: (counts[g], -free[g])) if task.get('gpu_required', True) else None
                cmd = command(task)
                log = directory/'logs'/f"{task['id']}.log"
                stream = log.open('a')
                child = subprocess.Popen(cmd, cwd=cwd,
                    env=dict(environment, CUDA_VISIBLE_DEVICES='' if gpu is None else gpu),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                tracking = guard.track(child); tracking.__enter__()
                task.update(status='running', pid=child.pid, gpu=gpu, command=cmd,
                            started_wall=time.time(), launcher_log=str(log))
                active[child.pid] = (child, task, stream, tracking)
                print(f"START {task['id']} GPU={gpu} pid={child.pid}", flush=True)
            atomic_json(directory/'tasks.json', tasks)
            status = dict(active=len(active), available_gib=health['available_gib'],
                completed=sum(t['status']=='complete' for t in tasks),
                pending=sum(t['status']=='pending' for t in tasks), skipped=sum(t['status']=='skipped' for t in tasks),
                failed=sum(t['status'] in ('failed', 'blocked') for t in tasks),
                seconds=time.monotonic()-started, queue_pid=os.getpid())
            atomic_json(directory/'status.json', status)
            if time.monotonic()-notice > 60:
                print('QUEUE '+json.dumps(status), flush=True); notice = time.monotonic()
            time.sleep(5)
    failures = [t['id'] for t in tasks if t['status'] in ('failed', 'blocked')]
    atomic_json(directory/'completion.json', dict(status='complete' if not failures else 'completed_with_failures',
        failures=failures, seconds=time.monotonic()-started, finished_wall=time.time()))
    return int(bool(failures))
