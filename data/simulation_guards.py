"""Fail-closed solver validation and bounded worker/parent waits."""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, wait
from functools import wraps
import signal
import threading
import time

import numpy as np


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
    """Watch completion without changing ranking/selection order of map results."""
    indices = {executor.submit(function, task): i for i, task in enumerate(tasks)}
    ready = {}
    next_index = 0
    try:
        for future in bounded_completed(executor, indices, stall_timeout):
            ready[indices[future]] = future.result()
            while next_index in ready:
                yield ready.pop(next_index)
                next_index += 1
    finally:
        # A quota may stop consumption early. Cancel queued work and watchdog
        # running tasks BEFORE ProcessPoolExecutor.__exit__ waits on them.
        for future in indices:
            future.cancel()
        if not getattr(executor, '_yeast_watchdog_aborted', False):
            for _ in bounded_completed(executor, [f for f in indices if not f.done()], stall_timeout):
                pass
