"""Cross-process admission for experiments and memory-heavy NPZ preprocessing.

These are cooperative limits for this project's jobs, not OS-enforced cgroups.
Waiting never kills other users' work or changes the source data.
"""
from contextlib import contextmanager
import fcntl
import os
import subprocess
from pathlib import Path
import time

from .logging import log_event


def process_tree(pid):
    """Linux descendants, including children spawned by non-main threads."""
    pending, result = [pid], []
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.append(current)
        try:
            for children in Path(f'/proc/{current}/task').glob('*/children'):
                pending.extend(map(int, children.read_text().split()))
        except (OSError, ValueError):
            pass
    return result


def process_counters(pid):
    """RSS/PSS and physical reads for one explicitly owned process tree."""
    rss = pss = read_bytes = write_bytes = 0
    for current in process_tree(pid):
        try:
            entries = dict(line.split(':', 1) for line in Path(f'/proc/{current}/smaps_rollup').read_text().splitlines()
                           if ':' in line)
            rss += int(entries['Rss'].split()[0]) * 1024
            pss += int(entries['Pss'].split()[0]) * 1024
            io = dict(line.split(':', 1) for line in Path(f'/proc/{current}/io').read_text().splitlines())
            read_bytes += int(io['read_bytes'])
            write_bytes += int(io['write_bytes'])
        except (OSError, ValueError, KeyError):
            pass
    return dict(rss_gib=rss / 2**30, pss_gib=pss / 2**30,
                read_bytes=read_bytes, write_bytes=write_bytes)


def disk_counters():
    """Raw Linux block-device counters; partitions are retained for attribution."""
    result = {}
    try:
        for line in Path('/proc/diskstats').read_text().splitlines():
            fields = line.split()
            if len(fields) < 14 or not fields[2].startswith(('sd', 'nvme', 'dm-')):
                continue
            values = list(map(int, fields[3:14]))
            result[fields[2]] = dict(read_ios=values[0], read_sectors=values[2],
                read_ms=values[3], write_ios=values[4], write_sectors=values[6],
                write_ms=values[7], in_flight=values[8], busy_ms=values[9],
                weighted_ms=values[10])
    except OSError:
        pass
    return result


def disk_activity(previous, current, seconds):
    """Convert counter deltas to per-device rates; None means unavailable."""
    if seconds <= 0:
        return {}
    result = {}
    for name, values in current.items():
        before = previous.get(name)
        if before is None:
            continue
        operations = values['read_ios'] - before['read_ios'] + values['write_ios'] - before['write_ios']
        busy = values['busy_ms'] - before['busy_ms']
        if operations < 0 or busy < 0:
            continue
        result[name] = dict(busy_percent=busy / seconds / 10,
            queue=(values['weighted_ms'] - before['weighted_ms']) / seconds / 1000,
            await_ms=((values['read_ms'] - before['read_ms'] + values['write_ms'] - before['write_ms'])
                      / operations if operations else 0.),
            read_mib_s=(values['read_sectors'] - before['read_sectors']) / seconds / 2048,
            write_mib_s=(values['write_sectors'] - before['write_sectors']) / seconds / 2048)
    return result


def severe_disk_congestion(activity):
    """Diagnostic heuristic; idle/moderate disks do not prove a PSI false alarm."""
    if not activity:
        return None
    return any(v['busy_percent'] >= 90 and (v['queue'] >= 16 or v['await_ms'] >= 250)
               for v in activity.values())


def memory_status():
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            fields[key] = int(value.split()[0]) * 1024
    total, available = fields['MemTotal'], fields['MemAvailable']
    # User/session cgroups may impose a limit smaller than physical RAM.
    try:
        group = next(line.split('::', 1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        directory = Path('/sys/fs/cgroup') / group.lstrip('/')
        while directory != Path('/sys/fs/cgroup'):
            limit = (directory / 'memory.max').read_text().strip()
            if limit != 'max':
                cap = int(limit)
                total = min(total, cap)
                available = min(available, max(0, cap - int((directory / 'memory.current').read_text())))
            directory = directory.parent
    except (OSError, StopIteration, ValueError):
        pass
    return total, available


def _pressure(name, category):
    try:
        line = next(x for x in Path(f'/proc/pressure/{name}').read_text().splitlines()
                    if x.startswith(category + ' '))
        return float(dict(item.split('=') for item in line.split()[1:])['avg10'])
    except (OSError, StopIteration, ValueError, KeyError):
        return 0.


def resource_state():
    total, available = memory_status()
    reserve = float(os.environ.get('SAMPLING_MIN_AVAILABLE_GIB', min(100., total / 2**30 * .2)))
    load_limit = float(os.environ.get('SAMPLING_MAX_LOAD_PER_CPU', '1.5'))
    cpus = len(os.sched_getaffinity(0))
    values = dict(available_gib=available / 2**30, reserve_gib=reserve,
                  load1=os.getloadavg()[0], cpu_count=cpus,
                  memory_stall=_pressure('memory', 'some'), io_stall=_pressure('io', 'full'))
    io_policy = os.environ.get('SAMPLING_IO_ADMISSION', 'strict')
    if io_policy not in ('strict', 'warn'):
        raise ValueError('SAMPLING_IO_ADMISSION must be strict or warn')
    values['io_admission_policy'] = io_policy
    memory_policy = os.environ.get('SAMPLING_MEMORY_PRESSURE_POLICY', 'strict')
    if memory_policy not in ('strict', 'available'):
        raise ValueError('SAMPLING_MEMORY_PRESSURE_POLICY must be strict or available')
    values['memory_pressure_policy'] = memory_policy
    blocked = (values['available_gib'] < reserve
               or (load_limit > 0 and values['load1'] > load_limit * cpus)
               or (memory_policy == 'strict' and
                   values['memory_stall'] > float(os.environ.get('SAMPLING_MAX_MEMORY_PSI', '1')))
               or (io_policy == 'strict' and
                   values['io_stall'] > float(os.environ.get('SAMPLING_MAX_IO_PSI', '5'))))
    return blocked, values


def wait_for_resources(phase, interval=5., cancel=None):
    last = 0.
    while True:
        if cancel is not None and cancel():
            raise RuntimeError('Resource admission cancelled by the batch memory guard')
        blocked, values = resource_state()
        if not blocked:
            return
        if time.monotonic() - last >= 30:
            log_event('WAIT', phase, **values)
            last = time.monotonic()
        time.sleep(interval)


@contextmanager
def exclusive_gpu_slot(device):
    """GPUs 4--7 admit one project GPU process, including separate launchers."""
    if not str(device).startswith('cuda'):
        yield
        return
    local = int(str(device).split(':')[1]) if ':' in str(device) else 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    physical = str(local) if visible is None else visible.split(',')[local].strip()
    if physical not in ('4', '5', '6', '7'):
        yield
        return
    root = Path(os.environ.get('SAMPLING_RESOURCE_DIR',
        '/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/.resource_locks'))
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / f'exclusive_gpu_{physical}.lock').open('a')
    last = 0.
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - last >= 30:
                    log_event('WAIT', 'exclusive GPU', gpu=physical)
                    last = time.monotonic()
                time.sleep(5)
        while True:
            raw = subprocess.check_output(['nvidia-smi', '-i', physical,
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
            other = [int(line.strip()) for line in raw.splitlines()
                     if line.strip().isdigit() and int(line.strip()) != os.getpid()]
            if not other:
                break
            if time.monotonic() - last >= 30:
                log_event('WAIT', 'exclusive GPU occupied', gpu=physical, processes=other)
                last = time.monotonic()
            time.sleep(5)
        yield
    finally:
        handle.close()


@contextmanager
def resource_slot(kind, phase):
    """Separate experiment/loading/scoring limits, gated by available host resources."""
    variable = {'experiment': 'SAMPLING_MAX_TOTAL_JOBS',
                'scoring': 'SAMPLING_SCORING_CONCURRENCY'}.get(kind, 'SAMPLING_IO_CONCURRENCY')
    default_limit = {'experiment': '16', 'scoring': '4'}.get(kind, '8')
    limit = int(os.environ.get(variable, default_limit))
    if limit < 1:
        raise ValueError(f'{variable} must be positive')
    root = Path(os.environ.get('SAMPLING_RESOURCE_DIR',
        '/home/users/zsy/mydata/budding_yeast/benchmarks/budding_sampling/.resource_locks'))
    root.mkdir(parents=True, exist_ok=True)
    chosen = None
    last = 0.
    try:
        while chosen is None:
            # Do not occupy a scarce slot while waiting for host admission.
            wait_for_resources(phase)
            for index in range(limit):
                handle = (root / f'{kind}_{index}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    chosen = handle
                    break
                except BlockingIOError:
                    handle.close()
            if chosen is None:
                if time.monotonic() - last >= 30:
                    log_event('WAIT', phase, resource=kind, concurrent_limit=limit)
                    last = time.monotonic()
                time.sleep(.5)
        yield
    finally:
        if chosen is not None:
            chosen.close()
