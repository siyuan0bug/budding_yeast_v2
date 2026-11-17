"""Monitor explicitly registered batch children; never signal unrelated jobs."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .logging import log_event
from .resources import (memory_status, process_counters, _pressure, disk_counters,
                        disk_activity, severe_disk_congestion)


class PressureWatch:
    """Shared pressure timing for the batch launcher and resource probes."""
    def __init__(self, *, io_policy='strict', memory_policy='strict',
                 memory_limit=1., io_limit=5., grace=20., disk_grace=60.):
        if io_policy not in ('strict','disk_correlated'):
            raise ValueError('Unknown IO pressure policy')
        if memory_policy not in ('strict','available'):
            raise ValueError('Unknown memory pressure policy')
        if not all(math.isfinite(v) and v>0 for v in (memory_limit,io_limit,grace,disk_grace)):
            raise ValueError('Invalid pressure limits')
        self.io_policy=io_policy
        self.memory_policy=memory_policy
        self.memory_limit,self.io_limit=memory_limit,io_limit
        self.grace,self.disk_grace=grace,disk_grace
        self.previous_disk=self.previous_time=None
        self.memory_since=self.io_since=None

    def update(self, now, health, disks, *, active=True):
        rates=(disk_activity(self.previous_disk,disks,now-self.previous_time)
               if self.previous_disk is not None else {})
        congestion=severe_disk_congestion(rates)
        self.previous_disk,self.previous_time=disks,now
        memory_warning=active and health['memory_stall']>self.memory_limit
        available,reserve=health.get('available_gib'),health.get('reserve_gib')
        known_headroom=(available is not None and reserve is not None
                        and math.isfinite(available) and math.isfinite(reserve))
        memory_high=memory_warning and (self.memory_policy=='strict' or not known_headroom or available<reserve)
        correlated=self.io_policy=='disk_correlated' and congestion is not None
        io_high=active and health['io_stall']>self.io_limit and (not correlated or congestion)
        if memory_high:
            if self.memory_since is None:self.memory_since=now
        else:self.memory_since=None
        if io_high:
            if self.io_since is None:self.io_since=now
        else:self.io_since=None
        memory_stop=self.memory_since is not None and now-self.memory_since>self.grace
        io_grace=self.disk_grace if correlated else self.grace
        io_stop=self.io_since is not None and now-self.io_since>io_grace
        return dict(io_policy=self.io_policy,disk_activity=rates,severe_disk_congestion=congestion,
                    memory_policy=self.memory_policy,memory_pressure_warning=memory_warning,
                    memory_pressure_stop=memory_stop,io_pressure_stop=io_stop,
                    sustained_pressure=memory_stop or io_stop)


class BatchMemoryGuard:
    """Watch available memory/pressure; owned PSS is telemetry, never a quota."""
    def __init__(self, output, *, minimum_available_gib=None, interval=5.):
        self.output=Path(output)
        self.minimum=float(os.environ.get('SAMPLING_MIN_AVAILABLE_GIB','100')) if minimum_available_gib is None else minimum_available_gib
        self.pressure_grace=float(os.environ.get('SAMPLING_PRESSURE_GRACE_S','20'))
        self.io_limit=float(os.environ.get('SAMPLING_MAX_IO_PSI','5'))
        self.memory_pressure_limit=float(os.environ.get('SAMPLING_MAX_MEMORY_PSI','1'))
        self.io_policy=os.environ.get('SAMPLING_IO_GUARD_POLICY','strict')
        self.memory_policy=os.environ.get('SAMPLING_MEMORY_PRESSURE_POLICY','strict')
        self.disk_pressure_grace=float(os.environ.get('SAMPLING_DISK_PRESSURE_GRACE_S','60'))
        self.pressure_watch=PressureWatch(io_policy=self.io_policy,memory_policy=self.memory_policy,
            memory_limit=self.memory_pressure_limit,
            io_limit=self.io_limit,grace=self.pressure_grace,disk_grace=self.disk_pressure_grace)
        if (not all(math.isfinite(v) for v in (self.minimum,interval,self.pressure_grace,
                                              self.io_limit,self.memory_pressure_limit)) or
                self.minimum<0 or interval<=0 or self.pressure_grace<=0
                or self.io_limit<=0 or self.memory_pressure_limit<=0):
            raise ValueError('Invalid batch memory-guard limits')
        self.interval=interval
        self.children={};self.lock=threading.Lock();self.done=threading.Event()
        self.reason=None;self.peak=0.;self.started=None

    @property
    def aborted(self):
        return self.reason is not None

    @staticmethod
    def _terminate(child):
        if child.poll() is not None:
            return
        try:
            if Path(f'/proc/{child.pid}').stat().st_uid!=os.getuid() or os.getpgid(child.pid)!=child.pid:
                raise RuntimeError('Refusing to signal an unowned or non-isolated process group')
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        except (ProcessLookupError,FileNotFoundError):
            pass

    def stop(self, reason):
        self.reason=reason
        log_event('STOP','batch memory guard',reason=self.reason)
        with self.lock:children=list(self.children.values())
        for child in children:self._terminate(child)

    @contextmanager
    def track(self, child):
        with self.lock:
            self.children[child.pid]=child
            abort=self.aborted
        try:
            if abort:self._terminate(child)
            yield
        except BaseException:
            self._terminate(child)
            raise
        finally:
            with self.lock:self.children.pop(child.pid,None)

    def _watch(self):
        try:
            with self.output.open('a') as stream:
                while not self.done.is_set():
                    with self.lock:children=list(self.children.values())
                    jobs={str(c.pid):process_counters(c.pid) for c in children if c.poll() is None}
                    pss=sum(v['pss_gib'] for v in jobs.values())
                    available=memory_status()[1]/2**30
                    self.peak=max(self.peak,pss)
                    io_stall=_pressure('io','full')
                    memory_stall=_pressure('memory','some')
                    disks=disk_counters()
                    pressure=self.pressure_watch.update(time.monotonic(),
                        dict(io_stall=io_stall,memory_stall=memory_stall,
                             available_gib=available,reserve_gib=self.minimum),disks,active=bool(jobs))
                    sustained_pressure=pressure['sustained_pressure']
                    critical=bool(jobs) and (available<self.minimum or sustained_pressure)
                    row=dict(seconds=time.monotonic()-self.started,pss_gib=pss,available_gib=available,
                        jobs=jobs,io_stall=io_stall,memory_stall=memory_stall,
                        diskstats=disks,**pressure,action='stop_owned_batch' if critical else 'observe')
                    stream.write(json.dumps(row)+'\n');stream.flush()
                    if critical:
                        self.stop(f'owned PSS={pss:.2f}GiB (telemetry only), available={available:.2f}GiB '
                            f'(minimum {self.minimum}), IO full PSI={io_stall:.2f}%, memory PSI={memory_stall:.2f}%, '
                            f'sustained_pressure={sustained_pressure}')
                        return
                    self.done.wait(self.interval)
        except Exception as exc:
            self.stop(f'Batch memory monitoring failed: {exc}')

    def __enter__(self):
        self.output.parent.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic()
        self.thread=threading.Thread(target=self._watch,name='sampling-memory-guard',daemon=True)
        self.thread.start()
        return self

    def __exit__(self,kind,value,traceback):
        if kind is not None and not self.aborted:self.stop(f'Batch interrupted: {value}')
        self.done.set();self.thread.join()
        self.output.with_suffix('.summary.json').write_text(json.dumps(dict(
            peak_owned_pss_gib=self.peak,seconds=time.monotonic()-self.started,
            aborted=self.aborted,reason=self.reason,max_pss_gib=None,
            minimum_available_gib=self.minimum,pressure_grace_seconds=self.pressure_grace,
            io_full_psi_limit=self.io_limit,memory_psi_limit=self.memory_pressure_limit,
            io_policy=self.io_policy,memory_policy=self.memory_policy,
            disk_pressure_grace_seconds=self.disk_pressure_grace),indent=2)+'\n')
