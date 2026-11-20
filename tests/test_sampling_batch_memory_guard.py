"""A raised concurrency cap must retain ownership-safe memory protection."""
import json
import signal
import subprocess
import sys
import time

import pytest
from budding_yeast_v2.utils.experiments import memory_guard,resources
from budding_yeast_v2.utils.experiments.memory_guard import BatchMemoryGuard,PressureWatch


def _device_snapshot(seconds, busy=.4, queue=4., await_ms=40.):
    return {'sdd':dict(read_ios=int(seconds*100),write_ios=0,read_sectors=int(seconds*2000),
        write_sectors=0,read_ms=int(seconds*100*await_ms),write_ms=0,
        busy_ms=int(seconds*1000*busy),weighted_ms=int(seconds*1000*queue))}


def test_correlated_pressure_does_not_stop_for_moderate_disks():
    watch=PressureWatch(io_policy='disk_correlated')
    for now in (0.,2.,21.,70.):
        row=watch.update(now,dict(io_stall=30.,memory_stall=0.),_device_snapshot(now))
        assert not row['sustained_pressure']


def test_correlated_pressure_stops_for_persistent_disk_congestion():
    watch=PressureWatch(io_policy='disk_correlated')
    for now in (0.,2.,59.):
        row=watch.update(now,dict(io_stall=30.,memory_stall=0.),
            _device_snapshot(now,busy=.99,queue=20.,await_ms=500.))
        assert not row['sustained_pressure']
    row=watch.update(61.,dict(io_stall=30.,memory_stall=0.),
        _device_snapshot(61.,busy=.99,queue=20.,await_ms=500.))
    assert row['io_pressure_stop'] and row['sustained_pressure']


def test_missing_disk_telemetry_falls_back_to_strict():
    watch=PressureWatch(io_policy='disk_correlated')
    watch.update(0.,dict(io_stall=30.,memory_stall=0.),{})
    assert watch.update(21.,dict(io_stall=30.,memory_stall=0.),{})['io_pressure_stop']


def test_memory_pressure_remains_guarded_when_disks_are_idle():
    watch=PressureWatch(io_policy='disk_correlated')
    watch.update(0.,dict(io_stall=0.,memory_stall=3.),_device_snapshot(0.))
    assert watch.update(21.,dict(io_stall=0.,memory_stall=3.),_device_snapshot(21.))['memory_pressure_stop']


def test_pressure_watch_does_not_combine_disjoint_memory_and_io_spikes():
    watch=PressureWatch()
    assert not watch.update(0.,dict(io_stall=0.,memory_stall=3.),{})['sustained_pressure']
    assert not watch.update(15.,dict(io_stall=30.,memory_stall=0.),{})['sustained_pressure']
    assert not watch.update(30.,dict(io_stall=0.,memory_stall=3.),{})['sustained_pressure']


def test_completed_workers_do_not_trigger_a_pressure_abort():
    watch=PressureWatch()
    for now in (0.,100.):
        assert not watch.update(now,dict(io_stall=30.,memory_stall=30.),{},active=False)['sustained_pressure']


def test_available_memory_policy_records_psi_without_stopping_with_headroom():
    watch=PressureWatch(memory_policy='available')
    for now in (0.,60.):
        row=watch.update(now,dict(io_stall=0.,memory_stall=3.,available_gib=530.,reserve_gib=100.),{})
        assert row['memory_pressure_warning'] and not row['sustained_pressure']


def test_available_memory_policy_keeps_conservative_fallback_when_unknown():
    watch=PressureWatch(memory_policy='available')
    watch.update(0.,dict(io_stall=0.,memory_stall=3.),{})
    assert watch.update(21.,dict(io_stall=0.,memory_stall=3.),{})['memory_pressure_stop']


@pytest.mark.parametrize('memory_policy',['strict','available'])
def test_guard_stops_only_tracked_process_tree_on_low_available_memory(tmp_path,monkeypatch,memory_policy):
    owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY',memory_policy)
    monkeypatch.setattr(memory_guard,'memory_status',lambda:(755*2**30,1*2**30))
    monkeypatch.setattr(memory_guard,'process_counters',lambda pid:dict(pss_gib=1.,rss_gib=1.))
    try:
        with BatchMemoryGuard(tmp_path/'resources.jsonl',interval=.01) as guard:
            with guard.track(owned):
                assert owned.wait(timeout=3)==-signal.SIGTERM
            assert guard.aborted
            assert unrelated.poll() is None
        assert json.loads((tmp_path/'resources.summary.json').read_text())['aborted']
    finally:
        for child in (owned,unrelated):
            if child.poll() is None:child.terminate()
            child.wait(timeout=3)


@pytest.mark.parametrize('available_gib',[110.,400.])
def test_large_owned_memory_does_not_stop_when_host_has_headroom(tmp_path,monkeypatch,available_gib):
    owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    # A stale old shell setting must not reinstate the retired 96 GiB cap.
    monkeypatch.setenv('SAMPLING_BATCH_MAX_PSS_GIB','96')
    monkeypatch.setenv('SAMPLING_MEMORY_PRESSURE_POLICY','available')
    monkeypatch.setenv('SAMPLING_PRESSURE_GRACE_S','.03')
    monkeypatch.delenv('SAMPLING_MIN_AVAILABLE_GIB',raising=False)
    monkeypatch.setattr(memory_guard,'memory_status',lambda:(755*2**30,available_gib*2**30))
    monkeypatch.setattr(memory_guard,'process_counters',lambda pid:dict(pss_gib=200.,rss_gib=200.))
    monkeypatch.setattr(memory_guard,'_pressure',lambda name,category:3. if name=='memory' else 0.)
    try:
        with BatchMemoryGuard(tmp_path/'large.jsonl',interval=.01) as guard:
            with guard.track(owned):
                time.sleep(.08)
                assert owned.poll() is None and not guard.aborted
        summary=json.loads((tmp_path/'large.summary.json').read_text())
        assert summary['peak_owned_pss_gib']==200. and summary['max_pss_gib'] is None
    finally:
        if owned.poll() is None:owned.terminate()
        owned.wait(timeout=3)


def test_resource_wait_can_be_cancelled_without_host_changes(monkeypatch):
    monkeypatch.setattr(resources,'resource_state',lambda:(True,{}))
    with pytest.raises(RuntimeError,match='cancelled'):
        resources.wait_for_resources('fixture',cancel=lambda:True)


def test_guard_does_not_abort_an_empty_queue_on_low_host_memory(tmp_path,monkeypatch):
    monkeypatch.setattr(memory_guard,'memory_status',lambda:(755*2**30,1*2**30))
    with BatchMemoryGuard(tmp_path/'resource.jsonl',interval=.01) as guard:
        time.sleep(.03)
    assert not guard.aborted


def test_exception_cleans_up_only_owned_isolated_children(tmp_path):
    owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    try:
        with pytest.raises(RuntimeError,match='fixture'):
            with BatchMemoryGuard(tmp_path/'resources.jsonl') as guard:
                with guard.track(owned):
                    raise RuntimeError('fixture')
        assert owned.wait(timeout=3)==-signal.SIGTERM
    finally:
        if owned.poll() is None:owned.terminate()
        owned.wait(timeout=3)


def test_sustained_io_pressure_stops_owned_job_without_oom(tmp_path,monkeypatch):
    owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    monkeypatch.setenv('SAMPLING_PRESSURE_GRACE_S','.03')
    monkeypatch.setattr(memory_guard,'memory_status',lambda:(755*2**30,500*2**30))
    monkeypatch.setattr(memory_guard,'process_counters',lambda pid:dict(pss_gib=1.,rss_gib=1.))
    monkeypatch.setattr(memory_guard,'_pressure',lambda name,category:10. if name=='io' else 0.)
    try:
        with BatchMemoryGuard(tmp_path/'io.jsonl',interval=.01) as guard:
            with guard.track(owned):
                assert owned.wait(timeout=3)==-signal.SIGTERM
            assert guard.aborted and 'sustained_pressure=True' in guard.reason
    finally:
        if owned.poll() is None:owned.terminate()
        owned.wait(timeout=3)
