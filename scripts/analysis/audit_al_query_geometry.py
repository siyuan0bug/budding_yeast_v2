"""Audit exact legal query availability in an existing pool; no model or ODE."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from budding_yeast_v2.utils.active_learning.strategies.response import background_graph
from budding_yeast_v2.utils.active_learning.strategies.query_design import interaction_blocks, invariance_edges


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path,indices=None):
    with np.load(path,allow_pickle=False) as archive:
        if indices is None:indices=np.arange(len(archive['params']))
        arrays={k:archive[k][indices] for k in ('params','y_init','candidate_ids')}
        for key in ('overrides_json','mutation_pairs_json'):
            arrays[key]=[json.loads(str(row)) for row in archive[key][indices]]
    return arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    with np.load(args.initial,allow_pickle=False) as archive:
        train=archive['split_train'];excluded=archive['candidate_ids'];excluded=excluded[excluded>=0]
    with np.load(args.pool,allow_pickle=False) as archive:
        keep=np.flatnonzero(~np.isin(archive['candidate_ids'],excluded))
    print(f'Reading {len(keep)} remaining candidates and {len(train)} synthetic training anchors',flush=True)
    pool=read(args.pool,keep);known=read(args.initial,train);n=len(pool['params'])
    params=np.concatenate((pool['params'],known['params'])).astype(np.float32)
    initials=np.concatenate((pool['y_init'],known['y_init']))
    overrides=pool['overrides_json']+known['overrides_json']
    metadata=[dict(pairs=row) for row in pool['mutation_pairs_json']+known['mutation_pairs_json']]
    ids=np.concatenate((pool['candidate_ids'],known['candidate_ids']))
    if len(np.unique(ids))!=len(ids):raise ValueError('Pool/initial candidate IDs overlap after pruning')
    print('Building exact background contrasts',flush=True)
    edges,distances=background_graph(params,initials,overrides,metadata,n)
    print(f'{len(edges)} contrasts; enumerating invariance and four-point queries',flush=True)
    invariants=invariance_edges(params,initials,overrides,edges)
    blocks=interaction_blocks(params,initials,overrides,metadata,edges)
    def describe(rows):
        return dict(actions=int(len(rows)),unique_remaining_candidates=int(len(np.unique(rows[rows<n]))),
                    unique_training_anchors=int(len(np.unique(rows[rows>=n]))))
    report=dict(pool=str(args.pool),initial=str(args.initial),pool_sha256=sha256(args.pool),
        initial_sha256=sha256(args.initial),remaining_candidates=n,training_anchors=len(train),
        response=describe(edges),proved_tem1_invariance=describe(invariants),exact_mc_ko_interaction=describe(blocks),
        seconds=time.monotonic()-started,scope='Input-only geometry, no predictions or new simulation')
    for name,rows in (('invariance_edges',invariants),('interaction_blocks',blocks)):
        np.savez_compressed(args.output/f'{name}.npz',candidate_ids=ids[rows],known_training=rows>=n)
    (args.output/'pool_geometry.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
