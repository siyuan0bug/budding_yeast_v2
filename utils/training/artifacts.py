"""A training run's public outputs and private recovery state."""
from dataclasses import dataclass
from pathlib import Path
import csv
import os
import tempfile


def write_history(path, history):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(key for row in history for key in row))
    fd,temporary=tempfile.mkstemp(prefix='.history-',dir=path.parent,text=True)
    try:
        with os.fdopen(fd,'w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys)
            writer.writeheader();writer.writerows(history);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def read_history(path):
    rows=[]
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            rows.append({key:(int(value) if key in ('epoch','optimizer_steps') else
                value if key in ('validation_metric','loss_type','physical_rel_l2_version') else float(value))
                for key,value in row.items() if value!=''})
    return rows


@dataclass(frozen=True)
class TrainingArtifacts:
    output: Path
    state: Path

    @classmethod
    def create(cls,output,state=None):
        output=Path(output);state=Path(state) if state is not None else output/'_state'
        output.mkdir(parents=True,exist_ok=True);state.mkdir(parents=True,exist_ok=True)
        return cls(output,state)

    @property
    def model(self):return self.output/'model.ckpt'
    @property
    def resume(self):return self.state/'resume.ckpt'
    @property
    def last_weights(self):return self.state/'last_weights.ckpt'
    @property
    def history(self):return self.output/'history.csv'
    @property
    def metadata(self):return self.state/'training.json'
