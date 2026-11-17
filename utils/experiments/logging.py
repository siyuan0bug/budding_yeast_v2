"""Local experiment timestamps, phase durations and non-destructive stack dumps.

No file handler: launchers already redirect stdout/stderr to each train.log.
"""
from contextlib import contextmanager,redirect_stdout,redirect_stderr
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
    """Flush output safely; optional legacy all-line timestamps."""
    def __init__(self, stream, *, all_lines=False):
        self.stream = stream
        self.all_lines = all_lines
        self._line_start = True

    def write(self, text):
        if not isinstance(text,str):
            raise TypeError('Text stream requires str')
        with _OUTPUT_LOCK:
            for part in text.splitlines(keepends=True):
                if self.all_lines and self._line_start and part.strip() and not _STAMP.match(part):
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


def install_timestamped_streams(*, all_lines=False):
    """Stage events carry explicit timestamps; ordinary output stays compact."""
    for name in ('stdout','stderr'):
        stream=getattr(sys,name)
        if not isinstance(stream,TimestampedStream):
            setattr(sys,name,TimestampedStream(stream,all_lines=all_lines))
        else:
            stream.all_lines=all_lines


def log_event(event, phase, **fields):
    details=' '.join(f'{key}={value}' for key,value in fields.items())
    print(f'[{timestamp()}] [{event}] {phase}'+(f' | {details}' if details else ''),flush=True)


def format_epoch_line(row, *, round_index, member, epochs, elapsed_s=None):
    """Console presentation only; CSV/W&B retain original names and precision."""
    fields=[f"train_loss={row['train_loss']:.6f}"]
    labels=(('val_loss_synth','val_synth'),('test_loss_synth','test_synth'),
            ('test_loss_real','test_real'),('test_real_denorm_rel_l2','test_real_denorm'))
    for key,label in labels:
        if key in row:fields.append(f"{label}={row[key]:.6f}")
    monitor=row.get('validation_metric','val_macro_rel_l2')
    if monitor=='val_macro_rel_l2':fields.append(f"val_macro_rel_l2={row['val_error']:.6f}")
    elif monitor=='val_sample_channel_rel_l2':fields.append(f"val_rel_l2_phys={row['val_error']:.6f}")
    elif 'val_loss_synth' not in row:fields.append(f"val_synth={row['val_error']:.6f}")
    fields.append(f"lr={row['lr']:.2e}")
    if elapsed_s is not None:fields.append(f"{elapsed_s:.1f}s")
    fields.append(f"round={round_index:02d} member={member} steps={int(row['optimizer_steps'])}")
    return f"[Epoch {int(row['epoch']):03d}/{int(epochs):03d}] "+' | '.join(fields)


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


@contextmanager
def tee_experiment_log(path):
    """Single-process CLI workflows retain console output and a local train.log.

    Batch launchers already redirect file descriptors and do not use this helper.
    It does not add timestamps or affect another process's logging handlers.
    """
    import logging
    from pathlib import Path
    import traceback
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',buffering=1) as file:
        class Tee:
            def __init__(self,console):self.console=console
            def write(self,value):
                with _OUTPUT_LOCK:self.console.write(value);file.write(value);file.flush()
                return len(value)
            def flush(self):
                with _OUTPUT_LOCK:self.console.flush();file.flush()
            def __getattr__(self,name):return getattr(self.console,name)
        stdout,stderr=sys.stdout,sys.stderr;out,err=Tee(stdout),Tee(stderr)
        changed=[]
        for logger in (logging.getLogger(),logging.getLogger('pytorch_lightning')):
            for handler in logger.handlers:
                if isinstance(handler,logging.StreamHandler):
                    original=handler.stream
                    if original is stderr or original is getattr(stderr,'stream',None):
                        handler.setStream(err);changed.append((handler,original))
                    elif original is stdout or original is getattr(stdout,'stream',None):
                        handler.setStream(out);changed.append((handler,original))
        try:
            with redirect_stdout(out),redirect_stderr(err):yield
        except BaseException:
            traceback.print_exc(file=file)
            raise
        finally:
            for handler,original in reversed(changed):handler.setStream(original)
