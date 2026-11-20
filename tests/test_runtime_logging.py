"""Train log timestamps preserve parsers, stderr and phase failure details."""
import io
import re
import sys

import pytest

from budding_yeast_v2.utils import runtime_logging as logs


PREFIX=r'\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\+08:00\] '


def test_fragmented_multiline_output_is_timestamped_once():
    output=io.StringIO();stream=logs.TimestampedStream(output,all_lines=True)
    assert stream.write('你好')==2
    stream.write(' world\nsecond\n')
    stream.write('[2026-10-29 12:00:00.000+08:00] existing\n')
    rows=output.getvalue().splitlines()
    assert re.fullmatch(PREFIX+'你好 world',rows[0])
    assert re.fullmatch(PREFIX+'second',rows[1])
    assert rows[2]=='[2026-10-29 12:00:00.000+08:00] existing'


def test_stdout_stderr_installation_is_idempotent_and_preserves_traceback(monkeypatch):
    out,err=io.StringIO(),io.StringIO()
    monkeypatch.setattr(sys,'stdout',out);monkeypatch.setattr(sys,'stderr',err)
    logs.install_timestamped_streams();first=sys.stdout
    logs.install_timestamped_streams();assert sys.stdout is first
    print('[Epoch 001/050] train_loss=0.1 | val_synth=0.2')
    print('RuntimeError: failure',file=sys.stderr)
    assert out.getvalue()=='[Epoch 001/050] train_loss=0.1 | val_synth=0.2\n'
    assert err.getvalue()=='RuntimeError: failure\n'
    assert re.search(r'\[Epoch\s+(\d+)/\d+\]',out.getvalue()).group(1)=='001'


def test_stage_end_failure_duration_and_successful_system_exit(capsys,monkeypatch):
    clock=iter([1.,3.,4.,9.,10.,11.])
    monkeypatch.setattr(logs.time,'monotonic',lambda:next(clock))
    with logs.log_stage('training',round=1):pass
    with pytest.raises(ValueError):
        with logs.log_stage('selection',budget=100):raise ValueError('bad candidate')
    with pytest.raises(SystemExit):
        with logs.log_stage('help'):raise SystemExit(0)
    text=capsys.readouterr().out
    assert '[END] training | round=1 elapsed_s=2.000' in text
    assert '[FAIL] selection | budget=100 elapsed_s=5.000 error=ValueError: bad candidate' in text
    assert '[END] help' in text


def test_progress_emits_periodically_and_at_completion(capsys,monkeypatch):
    clock=iter([0.,5.,31.,32.])
    monkeypatch.setattr(logs.time,'monotonic',lambda:next(clock))
    progress=logs.ProgressLog('ODE',100)
    progress.update(1);progress.update(50);progress.update(100)
    text=capsys.readouterr().out
    assert text.count('[PROGRESS]')==2
    assert 'completed=50 total=100' in text and 'completed=100 total=100' in text


def test_stack_diagnostics_declines_captured_stream_without_file_descriptor(monkeypatch):
    monkeypatch.setattr(sys,'stderr',io.StringIO())
    assert logs.install_stack_diagnostics() is False


def test_existing_loss_curve_parser_reads_timestamped_epochs(tmp_path,monkeypatch):
    from budding_yeast_v2.utils import curve_logger
    output=io.StringIO();stream=logs.TimestampedStream(output,all_lines=True)
    for i in (1,2):
        stream.write(f'[Epoch {i:03d}/050] train_loss=0.5 | val_synth=0.4 | '
            'test_synth=0.3 | test_real_mse=0.2 | test_real_denorm_rel_l2=0.1 | lr=1e-3 | 1.0s\n')
    path=tmp_path/'train.log';path.write_text(output.getvalue())
    seen=[]
    def record_plot(self,**kwargs):
        seen.append((self.epochs,self.train_loss,self.lr))
        return None
    monkeypatch.setattr(curve_logger.TrainingCurveLogger,'plot',record_plot)
    curve_logger.extract_and_plot_from_log(str(path))
    assert seen==[([1,2],[.5,.5],[.001,.001])]


def test_epoch_summary_is_compact_but_stage_has_timestamp(capsys):
    row=dict(epoch=4,train_loss=1.168340802192688,val_error=.832070561519249,
             validation_metric='val_macro_rel_l2',lr=.0004,optimizer_steps=64,
             val_macro_rel_l2=.832070561519249,val_loss_synth=1.0032075643539429)
    line=logs.format_epoch_line(row,round_index=0,member=0,epochs=100,elapsed_s=2.)
    assert line.startswith('[Epoch 004/100] train_loss=')
    assert 'train_loss=1.168341' in line and 'val_macro_rel_l2=0.832071' in line
    assert 'val_synth=1.003208' in line and 'lr=4.00e-04' in line
    assert 'val_error=' not in line and 'validation_metric=' not in line
    assert 'round=00 member=0 steps=64' in line
    assert row['val_error']==.832070561519249
    logs.log_event('START','training',round=0)
    assert re.match(PREFIX,capsys.readouterr().out)
