"""Local experiment timestamps, phase durations and non-destructive stack dumps.

No file handler: launchers already redirect stdout/stderr to each train.log.
"""
from contextlib import contextmanager
from datetime import datetime
import faulthandler
import re
import signal
import sys
import threading
import time
from zoneinfo import ZoneInfo

TZ = ZoneInfo('Asia/Shanghai')
_STAMP = re.compile(r'^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d')
_OUTPUT_LOCK = threading.RLock()


def timestamp():
    return datetime.now(TZ).isoformat(sep=' ',timespec='milliseconds')


class TimestampedStream:
    """Prefix complete and fragmented text lines without swallowing stderr."""
    def __init__(self, stream):
        self.stream = stream
        self._line_start = True

    def write(self, text):
        if not isinstance(text,str):
            raise TypeError('Text stream requires str')
        with _OUTPUT_LOCK:
            for part in text.splitlines(keepends=True):
                if self._line_start and part.strip() and not _STAMP.match(part):
                    self.stream.write(f'[{timestamp()}] ')
                self.stream.write(part)
                self._line_start = part.endswith(('\n','\r'))
            self.stream.flush()
        return len(text)

    def flush(self):
        with _OUTPUT_LOCK:
            self.stream.flush()

    def __getattr__(self,name):
        return getattr(self.stream,name)


def install_timestamped_streams():
    for name in ('stdout','stderr'):
        stream=getattr(sys,name)
        if not isinstance(stream,TimestampedStream):
            setattr(sys,name,TimestampedStream(stream))


def log_event(event, phase, **fields):
    details=' '.join(f'{key}={value}' for key,value in fields.items())
    print(f'[{timestamp()}] [{event}] {phase}'+(f' | {details}' if details else ''),flush=True)


@contextmanager
def log_stage(phase, **fields):
    started=time.monotonic()
    log_event('START',phase,**fields)
    try:
        yield
    except BaseException as exc:
        # argparse --help exits successfully; do not report it as a failed run.
        event='END' if isinstance(exc,SystemExit) and not exc.code else 'FAIL'
        log_event(event,phase,**fields,elapsed_s=f'{time.monotonic()-started:.3f}',
                  error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        log_event('END',phase,**fields,elapsed_s=f'{time.monotonic()-started:.3f}')


def install_stack_diagnostics():
    """SIGUSR1 dumps all Python threads; never send it to unregistered old runs."""
    try:
        faulthandler.enable(file=sys.stderr,all_threads=True)
        if hasattr(signal,'SIGUSR1'):
            faulthandler.register(signal.SIGUSR1,file=sys.stderr,all_threads=True,chain=False)
    except (AttributeError,OSError,RuntimeError,ValueError):
        # Notebook/captured streams may not expose a real OS file descriptor.
        return False
    return True


class ProgressLog:
    def __init__(self,phase,total,interval=30.,**fields):
        self.phase,self.total,self.interval,self.fields=phase,int(total),interval,fields
        self.started=self.last=time.monotonic()

    def update(self,completed):
        now=time.monotonic()
        if completed==self.total or now-self.last>=self.interval:
            elapsed=now-self.started
            log_event('PROGRESS',self.phase,**self.fields,completed=int(completed),total=self.total,
                      elapsed_s=f'{elapsed:.1f}',rate_per_s=f'{completed/max(elapsed,1e-9):.2f}')
            self.last=now
