"""The RAM watchdog may terminate only its explicitly owned experiment group."""
import json
import signal
import subprocess
import sys
from budding_yeast_v2.scripts.analysis import monitor_sampling_batch as monitor


def test_monitor_stops_only_owned_batch_on_critical_memory(tmp_path,monkeypatch):
    target=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],start_new_session=True)
    try:
        output=tmp_path/'memory.jsonl'
        monkeypatch.setattr(monitor,'memory_status',lambda:(755*2**30,1*2**30))
        monkeypatch.setattr(sys,'argv',['monitor','--pid',str(target.pid),'--output',str(output)])
        monitor.main()
        assert target.wait(timeout=3)==-signal.SIGTERM
        assert unrelated.poll() is None
        assert json.loads(output.read_text())['watchdog_action']=='stop_owned_batch'
    finally:
        for process in (target,unrelated):
            if process.poll() is None:process.terminate()
            process.wait(timeout=3)
