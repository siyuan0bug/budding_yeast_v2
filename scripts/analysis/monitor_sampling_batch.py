#!/usr/bin/env python3
"""Low-rate process-tree memory monitoring for one owned sampling batch."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.scripts.analysis.profile_sampling_resources import counters
from budding_yeast_v2.utils.experiments.resources import memory_status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=10.)
    parser.add_argument('--max-pss-gib',type=float,help='Optional diagnostic cap; default uses available memory only')
    parser.add_argument('--minimum-available-gib',type=float,default=100.)
    args=parser.parse_args()
    process=Path(f'/proc/{args.pid}')
    if process.stat().st_uid!=os.getuid():raise PermissionError('Only own batch may be monitored')
    start_identity=(process/'stat').read_text().split()[21]
    group=os.getpgid(args.pid)
    began=time.monotonic();peak=0.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('a') as stream:
        while process.exists():
            try:
                stat=(process/'stat').read_text().split()
                if stat[21]!=start_identity or stat[2]=='Z':break
                row=counters(args.pid)
                row.update(seconds=time.monotonic()-began,pid=args.pid,
                    available_gib=memory_status()[1]/2**30,load1=os.getloadavg()[0])
                children=set()
                for source in (process/'task').glob('*/children'):
                    try:children.update(map(int,source.read_text().split()))
                    except OSError:pass
                row['jobs']={str(pid):counters(pid) for pid in children}
            except (OSError,ValueError):break
            peak=max(peak,row['pss_gib'])
            critical=((args.max_pss_gib is not None and row['pss_gib']>args.max_pss_gib)
                or row['available_gib']<args.minimum_available_gib)
            row['watchdog_action']='stop_owned_batch' if critical else 'observe'
            stream.write(json.dumps(row)+'\n');stream.flush()
            if critical:
                # The tmux pane is its own process group; never signal this
                # monitor, unrelated sessions, or a reused PID/group.
                if group!=args.pid or os.getpgid(args.pid)!=group:
                    raise RuntimeError('Unsafe process group; refusing group signal')
                os.killpg(group,signal.SIGTERM)
                break
            time.sleep(args.interval)
    args.output.with_suffix('.summary.json').write_text(json.dumps(dict(peak_pss_gib=peak,
        seconds=time.monotonic()-began,pid=args.pid),indent=2))


if __name__=='__main__':main()
