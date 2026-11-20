from datetime import datetime
import json
from pathlib import Path

from budding_yeast_v2.scripts.analysis.summarize_al_runtime import TZ,summarize


def fixture_run(tmp_path):
    run=tmp_path/'fno/random/seed_42';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps({'timestamp':'2026-10-23 09:24:08','args':{'model':'fno','al_strategy':'al4pde_random','seed':42,'num_al_rounds':4}}))
    (run/'summary').mkdir();(run/'summary/rounds_summary.json').write_text(json.dumps([{'round':i} for i in range(4)]))
    (run/'train.log').write_text('AL 轮 4/4\n[Epoch 200/200] train_loss=0.1')
    debug=run/'wandb/offline-run-test/logs';debug.mkdir(parents=True)
    (debug/'debug.log').write_text('2026-10-23 14:14:25,815 INFO MainThread:1 [wandb_run.py:_atexit_cleanup():2520] got exitcode: 0\n')
    logs=tmp_path/'batch_logs/a';logs.mkdir(parents=True)
    (logs/'gpu0.tsv').write_text('seed\\tmodel\\tstrategy\\texit_code\\texperiment_dir\n42\\tfno\\tal4pde_random\\t0\\t'+str(run)+'\n')
    return run


def test_completed_runtime_requires_rounds_wandb_and_batch_exit(tmp_path,monkeypatch):
    fixture_run(tmp_path)
    start=datetime(2026,10,23,9,23,49,tzinfo=TZ)
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:str(int(start.timestamp())))
    row=summarize(tmp_path)[0]
    assert row['completed'] and row['elapsed_hms']=='04:50:37'
    assert row['start_time'].endswith('+08:00')
    assert row['last_logged_epoch']==200 and row['batch_exit_zero']


def test_summary_without_successful_shutdown_is_not_a_complete_duration(tmp_path,monkeypatch):
    run=fixture_run(tmp_path)
    (run/'wandb/offline-run-test/logs/debug.log').write_text('2026-10-23 11:00:00,000 INFO config callback\n')
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'0')
    row=summarize(tmp_path)[0]
    assert not row['completed'] and row['elapsed_seconds'] is None
    assert row['start_time_source'].startswith('config timestamp')
