"""Readable result paths shared by every sampling workflow; hashes stay in metadata."""
from dataclasses import asdict,is_dataclass
from pathlib import Path
import re

DATA=Path('/home/users/zsy/mydata/budding_yeast')
RESULTS=DATA/'experiments/budding_sampling'
SHARED=RESULTS/'shared/pool100k'
ALIASES={'lcmd_ll':'lcmd','physics_pre_power':'physics_pre','hggs_projected':'hggs',
         'peaks_regression':'peaks','ogas_async_projected':'ogas',
         'ours_dynamics_transport':'ours_transport','ours_intervention_closure':'ours_closure'}


def profile(config):
    c=asdict(config) if is_dataclass(config) else config
    def n(x):return f'{float(x):g}'
    epochs=c['epochs'];epochs='-'.join(map(str,epochs))
    value=f"w{c['width']}_n{c['n_blocks']}_m{c['modes']}_b{c['batch_size']}_lr{n(c['lr'])}_{c['loss']}_eps{n(c.get('relative_l2_eps',.01))}_e{epochs}"
    if not re.fullmatch(r'[A-Za-z0-9_.+-]+',value):raise ValueError('Unsafe experiment profile')
    return value


def method_name(method,kind='run',legacy=False):
    name=ALIASES.get(method,method)
    if method in ('ours_dynamics_transport','ours_intervention_closure','ours_mechanism_witness'):
        name+= '_generate' if kind=='generate' else '_pool'
    if legacy and method=='hggs_projected':name+='_legacy'
    return name


def experiment_directory(output,config,pool,method,kind='run',legacy=False):
    c=asdict(config) if is_dataclass(config) else config
    return Path(output)/c['model']/pool/method_name(method,kind,legacy)/profile(c)/f"seed_{c['seed']}"
