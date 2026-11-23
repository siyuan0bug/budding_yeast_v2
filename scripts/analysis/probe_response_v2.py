"""Training-side response/query probe with an existing initial-round FNO; no holdout evaluation."""
import argparse,json,multiprocessing as mp,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr

from budding_yeast_v2.data.dataset_archive import TrajectoryStore,atomic_json,file_sha256,_array_digest
from budding_yeast_v2.utils.sampling.benchmark.protocol import child_seed,effective_signatures
from budding_yeast_v2.utils.sampling.benchmark.runner import source_inventory
from budding_yeast_v2.utils.sampling.generation.runner import _acquire,_view,worker_lifecycle
from budding_yeast_v2.utils.sampling.generation.domain import FrozenBuddingDomain
from budding_yeast_v2.utils.sampling.generation.storage import GeneratedTrainingStore
from budding_yeast_v2.utils.sampling.generation.policies import GenerationPolicy
from budding_yeast_v2.utils.sampling.ours.response_v2 import METHODS
from budding_yeast_v2.utils.evaluation.checkpoints import load_sampling_checkpoint,Predictor
from budding_yeast_v2.utils.experiments.resources import resource_slot


def correlation(rows):
    if len(rows)<3 or np.ptp([r['score'] for r in rows])==0:return None
    return float(spearmanr([r['score'] for r in rows],[r['normalized_relative_l2'] for r in rows]).statistic)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--initial-run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=128);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--device',default='cpu');args=ap.parse_args()
    if not 16<=args.queries<=512 or not 1<=args.workers<=4:raise ValueError('Probe requires bounded query/worker limits')
    root=args.protocol;definition=json.loads((root/'protocol.json').read_text());seed=definition['config']['seed']
    assert args.initial_run.name==f'seed_{seed}'
    config=json.loads((args.initial_run/'config.json').read_text())
    if config['training']!=definition['config']:raise ValueError('Initial trained FNO protocol mismatch')
    sources=source_inventory();args.output.mkdir(parents=True,exist_ok=True)
    forbidden=set()
    with resource_slot('data','training-side probe input exclusion'):
        for name in definition['pools']:
            with np.load(root/f'{name}_inputs.npz',allow_pickle=False) as v,np.load(root/f'{name}_split.npz') as split:
                rows=np.r_[split['validation'],split['test']]
                forbidden.update(effective_signatures(v['params'][rows],v['y_init'][rows]))
    with np.load(root/'mutation_split.npz') as v:initial_rows=v['initial']
    checkpoint=args.initial_run/'rounds/round_00/model.ckpt'
    model,norm,_=load_sampling_checkpoint(checkpoint,args.device)
    predictor=Predictor([model],norm,500,args.device,128)
    snapshot=json.loads((args.initial_run/'rounds/round_00/dataset.json').read_text())
    training_manifest=json.loads(Path(snapshot['source']).read_text())
    if snapshot['rows']!=list(range(len(initial_rows))):raise ValueError('Initial checkpoint snapshot order changed')
    reports=[]
    for method in METHODS:
        directory=args.output/method
        if directory.exists():raise FileExistsError('Choose a fresh probe directory; no ambiguous partial reuse')
        (directory/'_state').mkdir(parents=True)
        with resource_slot('data','training-side probe initial labels'):
            store=GeneratedTrainingStore(definition['pools']['mutation']['path'],initial_rows,directory/'data','mutation',seed)
        for key in ('params','y_init','data'):
            if _array_digest(store.initial[key])!=training_manifest['fields'][key]['sha256']:
                raise ValueError('Initial checkpoint does not match shared queried cohort')
        initial=_view(store,metadata=True)
        domain=FrozenBuddingDomain('mutation',child_seed(seed,'generated_proposals'),directory/'_state/proposals',
            forbidden=forbidden,seen=set(effective_signatures(initial.params,initial.initial)))
        policy=GenerationPolicy(method,initial,store.initial['data'],child_seed(seed,'generation_policy'))
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                initializer=worker_lifecycle,initargs=(os.getpid(),)) as executor:
            _acquire(len(initial_rows)+args.queries,domain,store,policy,predictor,executor,directory,
                bank_size=128,query_batch_size=16,max_attempts_mult=30)
        journal=policy.controller.journal;actions=Counter(r['action'] for r in journal)
        known=[r for r in journal if r['score'] is not None]
        events=[json.loads(x) for x in (directory/'_state/acquisition.jsonl').read_text().splitlines()]
        after_bootstrap=events[1:]
        supported=sum(e['selected_supported'] for e in after_bootstrap)/sum(e['query_count'] for e in after_bootstrap)
        mechanics=(len(journal)==args.queries and supported>=.5 and actions.get('capacity_fallback',0)/args.queries<=.1
                   and all(r['score'] is None for r in journal if r['action'] in ('explore','support')))
        record=dict(method=method,valid_queries=len(journal),actions=dict(actions),
            post_bootstrap_supported_query_fraction=supported,
            estimated_vs_actual_task_spearman=correlation(known),mechanical_pass=mechanics,
            mean_normalized_prequery_relative=float(np.mean([r['normalized_relative_l2'] for r in journal])),
            mean_physical_prequery_relative=float(np.mean([r['physical_relative_l2'] for r in journal])))
        atomic_json(directory/'feedback.json',journal);atomic_json(directory/'report.json',record)
        reports.append(record);print(json.dumps(record),flush=True)
    if source_inventory()!=sources:raise RuntimeError('Probe sources changed')
    atomic_json(args.output/'report.json',dict(diagnostic_only=True,checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        protocol=str(root),seed=seed,holdout_labels_used=False,real_labels_used=False,biological_rules_changed=False,
        initial_labels_reused=len(initial_rows),new_queries=sum(r['valid_queries'] for r in reports),reports=reports,
        sources=sources,mechanical_pass=all(r['mechanical_pass'] for r in reports),
        # Predeclared training-side gate, not a guarantee of final model improvement.
        score_signal_pass=all(r['estimated_vs_actual_task_spearman'] is not None
                              and r['estimated_vs_actual_task_spearman']>.2 for r in reports)))

if __name__=='__main__':main()
