"""HGGS GF boundary ranking then pilot-residual global retention, training side only."""
import json
import numpy as np

from budding_yeast_v2.data.dataset_archive import atomic_json,fingerprint
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.evaluation.checkpoints import Predictor
from ..benchmark.runner import input_view
from ..benchmark.protocol import child_seed
from ..benchmark.rounds import fit_round
from ..contracts import QueriedTraining
from ..query import LabelOracle
from ..baseline.hggs import gradient_degree
from ..baseline.geometry import gaussian_projection


def prepare_hggs_initial(cohorts,root,cfg,directory,settings,*,device='cpu'):
    state=directory/'_state';state.mkdir(exist_ok=True);path=state/'hggs_gf.json'
    count=cfg.budgets[0]
    if count<4:raise ValueError('HGGS GF needs at least four initial training rows')
    with np.load(root/f'{cohorts.pool}_inputs.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    with np.load(root/f'{cohorts.pool}_split.npz') as z:allowed=z['candidates']
    oracle=LabelOracle(cohorts.source,allowed,state/'hggs_label_access.json',track='B')
    if path.exists():selected=json.loads(path.read_text())['selected_ids']
    else:
        rng=np.random.default_rng(child_seed(cfg.seed,'hggs_gf'))
        n=min(len(allowed),count*settings.coarse_multiplier)
        coarse=np.r_[cohorts.initial_rows,rng.choice(np.setdiff1d(allowed,cohorts.initial_rows),n-count,replace=False)]
        truth=oracle.scoring_reader(coarse)['data'];view=input_view(inputs,coarse)
        norm=NormalizationStats.fit(inputs['params'][cohorts.initial_rows],oracle.reveal(cohorts.initial_rows)['data'])
        physical=np.column_stack((view.params,view.initial));mean=physical.mean(0);scale=physical.std(0);scale[scale<1e-8]=1.
        raw_dimension=truth.shape[1]*truth.shape[2]
        projection=gaussian_projection(raw_dimension,min(settings.output_sketch,raw_dimension),cfg.seed)
        output=np.empty((len(coarse),projection.shape[1]))
        for start in range(0,len(coarse),cfg.prediction_batch_size):
            rows=truth[start:start+cfg.prediction_batch_size]
            output[start:start+len(rows)]=(norm.targets(rows).flatten(1)@projection).numpy()
        degrees=gradient_degree((physical-mean)/scale,output,settings.neighbors)
        boundary_count=min(count-2,max(2,int(count*settings.boundary_fraction)))
        order=np.argsort(-degrees,kind='stable');boundary=coarse[order[:boundary_count]]
        remaining=coarse[order[boundary_count:]]
        pilot=QueriedTraining(input_view(inputs,boundary),oracle.reveal(boundary)['data'])
        snapshot=state/'hggs_pilot/dataset.json'
        cohorts.source.snapshot(snapshot,boundary,provenance=dict(role='HGGS GF boundary pilot'))
        model,norm,_=fit_round(cfg,0,pilot,cohorts.validation,state/'hggs_pilot/primary',
            child_seed(cfg.seed,'hggs_gf_model'),device=device,dataset_path=snapshot,
            identity=dict(role='hggs_boundary_pilot',rows=fingerprint(boundary.tolist())))
        predictor=Predictor([model],norm,truth.shape[-1],device,cfg.prediction_batch_size)
        residual=np.empty(len(remaining))
        for start,prediction in predictor.batches(input_view(inputs,remaining)):
            target=norm.targets(oracle.scoring_reader(remaining[start:start+len(prediction)])['data'])
            residual[start:start+len(prediction)]=np.linalg.norm((prediction-target).flatten(1).numpy(),axis=1)
        retained=remaining[np.argsort(-residual,kind='stable')[:count-boundary_count]]
        selected=np.r_[boundary,retained].tolist();model.cpu()
        atomic_json(path,dict(selected_ids=selected,boundary_ids=boundary.tolist(),global_residual_ids=retained.tolist(),
            shared_random_initial_overlap=len(set(selected)&set(map(int,cohorts.initial_rows))),
            coarse_source_ids=coarse.tolist(),coarse_labels=len(coarse),neighbors=settings.neighbors,
            task_adaptations=['trajectory sketch replaces oscillation frequency','shared-source random coarse bank replaces paper LHS'],
            source_trajectories_copied=False,initial_cohort_matches_shared_random=False))
    if len(selected)!=count or len(set(selected))!=count or not set(selected)<=set(map(int,allowed)):
        raise ValueError('Saved HGGS GF selection violates training-side budget')
    cohorts.initial_rows=np.asarray(selected,dtype=int);cohorts.initial_inputs=input_view(inputs,cohorts.initial_rows)
    return json.loads(path.read_text())
