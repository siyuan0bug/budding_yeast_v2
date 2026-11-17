"""Atomic named state sections; independent services do not overwrite each other."""
import fcntl
import json
from pathlib import Path
from budding_yeast_v2.data.dataset_archive import atomic_json


def read_section(path,key,default=None):
    path=Path(path)
    return json.loads(path.read_text()).get(key,default) if path.exists() else default


def update_section(path,key,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        state=json.loads(path.read_text()) if path.exists() else {}
        state[key]=value;atomic_json(path,state)
