"""Isolated ensemble training processes, sharing only immutable dataset references."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import json,multiprocessing as mp,os
from pathlib import Path
import numpy as np


def train_member(config,round_index,protocol_path,pool,snapshot,output,member,identity,device):
    from .protocol import BenchmarkConfig,child_seed
    from .rounds import fit_round
    from ..contracts import PoolInputs,QueriedTraining
    from budding_yeast_v2.data.dataset_archive import SelectionArchive,TrajectoryStore
    from budding_yeast_v2.utils.experiments.logging import tee_experiment_log
    from budding_yeast_v2.utils.experiments.resources import resource_slot
    cfg=BenchmarkConfig(**config).validate();root=Path(protocol_path)
    definition=json.loads((root/'protocol.json').read_text());validation=[]
    with resource_slot('data','ensemble immutable input loading'):
        with SelectionArchive(snapshot) as archive:
            rows=archive.selection_rows
            raw=archive.store.read(rows,('params','y_init','data'))
            training=QueriedTraining(PoolInputs(rows,raw['params'],raw['y_init']),raw['data'])
        for name,source in definition['pools'].items():
            if name!=pool and cfg.validation_scope!='both':continue
            with np.load(root/f'{name}_split.npz') as split:ids=split['validation']
            store=TrajectoryStore(source['path']);raw=store.read(ids,('params','y_init','data'))
            validation.append((PoolInputs(ids,raw['params'],raw['y_init']),raw['data']))
    output=Path(output)
    with tee_experiment_log(output/'train.log'):
        model,_,_=fit_round(cfg,round_index,training,validation,output,
            child_seed(cfg.seed,'model',round_index,member),device=device,member=member,
            identity=identity,dataset_path=snapshot)
        model.cpu()
    return str(output/'model.ckpt')


@contextmanager
def parallel_members(config,round_index,protocol_path,pool,snapshot,directory,identity,device):
    if str(device).startswith('cuda'):
        local=int(str(device).split(':')[1]) if ':' in str(device) else 0
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        physical=str(local) if visible is None else visible.split(',')[local].strip()
        if physical in ('4','5','6','7'):
            raise ValueError('Parallel ensemble processes are forbidden on exclusive GPUs 4-7')
    executor=ProcessPoolExecutor(max_workers=config.ensemble_members-1,mp_context=mp.get_context('spawn'))
    futures=[];paths=[]
    try:
        for member in range(1,config.ensemble_members):
            output=Path(directory)/'_state'/f'round_{round_index:02d}'/f'ensemble_member_{member}'
            futures.append(executor.submit(train_member,asdict(config),round_index,str(protocol_path),pool,
                str(snapshot),str(output),member,identity,device))
        yield paths
        paths.extend(f.result() for f in futures)
    except BaseException:
        # Only this executor's children, never other users' GPU processes.
        for process in list((executor._processes or {}).values()):
            if process.is_alive():process.terminate()
            process.join(timeout=5)
        raise
    finally:executor.shutdown(wait=True,cancel_futures=True)
