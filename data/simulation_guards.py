"""Fail-closed solver validation and bounded worker/parent waits."""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, wait
from functools import wraps
import signal
import threading
import time

import numpy as np
from budding_yeast_v2.utils.runtime_logging import log_event


DEFAULT_SIM_TIMEOUT = 5.0


def validate_segment(sol, start, stop, num_vars):
    t = np.asarray(sol.t)
    y = np.asarray(sol.y)
    if (not sol.success or sol.status not in (0, 1) or t.ndim != 1 or len(t) == 0
            or y.shape != (num_vars, len(t)) or not np.isfinite(t).all()
            or not np.isfinite(y).all() or (np.diff(t) <= 0).any()
            or not np.isclose(t[0], start, rtol=0, atol=1e-10)
            or t[-1] < start or t[-1] > stop + 1e-10
            or (sol.status == 0 and not np.isclose(t[-1], stop, rtol=0, atol=1e-10))):
        raise ValueError('Failed, nonfinite, malformed, or incomplete ODE segment')


@contextmanager
def simulation_deadline(seconds):
    if not hasattr(signal, 'setitimer') or threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Bounded ODE simulation requires a POSIX worker main thread')
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('Simulation timeout must be finite and positive')
    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def alarm_handler(signum, frame):
        raise TimeoutError('ODE simulation wall-clock deadline exceeded')

    signal.signal(signal.SIGALRM, alarm_handler)
    deadline = min(seconds, old_timer[0]) if old_timer[0] > 0 else seconds
    signal.setitimer(signal.ITIMER_REAL, deadline)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:
            remaining = max(1e-6, old_timer[0] - (time.monotonic() - started))
            signal.setitimer(signal.ITIMER_REAL, remaining, old_timer[1])


def bounded_simulation(function):
    @wraps(function)
    def wrapped(name, *args, **kwargs):
        try:
            with simulation_deadline(DEFAULT_SIM_TIMEOUT):
                return function(name, *args, **kwargs)
        except (TimeoutError, RuntimeError):
            return name, None, 'Inviable: Numerical Failure', 'Pattern_C: Exponential-Arrest (==0)'
    return wrapped


def bounded_completed(executor, futures, stall_timeout=35.0):
    """A real parent watchdog, unlike result(timeout) on an already-done Future.

    Workers have individual simulation deadlines. If native code prevents signal
    delivery and the pool stops making progress, terminate its worker processes
    and abort generation; do NOT export a silently partial dataset.
    """
    if not np.isfinite(stall_timeout) or stall_timeout <= 0:
        raise ValueError('Parent stall timeout must be finite and positive')
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=stall_timeout, return_when=FIRST_COMPLETED)
        if not done:
            # Python 3.10 has no public terminate_workers(). Keep the compatibility
            # workaround local and join/kill only the executor's own children.
            processes = list((getattr(executor, '_processes', None) or {}).values())
            log_event('FAIL','process pool watchdog',stall_timeout_s=stall_timeout,
                      pending=len(pending),worker_pids=[process.pid for process in processes])
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=1.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
            executor._yeast_watchdog_aborted = True
            raise TimeoutError('ODE pool made no progress before parent deadline; generation aborted')
        yield from done


def bounded_ordered_map(executor, function, tasks, stall_timeout=35.0):
    """Bound submitted work as well as waits; preserve candidate input order.

    Submitting tens of thousands of tasks before consuming any result can block
    Python 3.10's process-pool wakeup/result pipes before the watchdog is entered.
    The window also bounds completed out-of-order trajectory buffers.
    """
    tasks = iter(tasks)
    window = max(1, 2 * int(getattr(executor, '_max_workers', 1)))
    indices = {}
    ready = {}
    next_index = submitted = 0
    exhausted = False

    def replenish():
        nonlocal submitted, exhausted
        while not exhausted and len(indices) + len(ready) < window:
            try:
                task = next(tasks)
            except StopIteration:
                exhausted = True
                break
            indices[executor.submit(function, task)] = submitted
            submitted += 1

    try:
        replenish()
        while indices:
            # Recreate the bounded wait after each completion so newly submitted
            # futures are watched too. No wait ever owns a growing global set.
            future = next(bounded_completed(executor, tuple(indices), stall_timeout))
            ready[indices.pop(future)] = future.result()
            while next_index in ready:
                yield ready.pop(next_index)
                next_index += 1
            replenish()
    finally:
        # A quota may stop consumption early. Cancel queued work and watchdog
        # running tasks BEFORE ProcessPoolExecutor.__exit__ waits on them.
        for future in indices:
            future.cancel()
        if not getattr(executor, '_yeast_watchdog_aborted', False):
            for _ in bounded_completed(executor, [f for f in indices if not f.done()], stall_timeout):
                pass
