"""HGGS GF: boundary training followed by repeated current-model global retention."""
from dataclasses import replace
import json
import numpy as np

from budding_yeast_v2.data.dataset_archive import atomic_json,fingerprint,file_sha256
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.utils.evaluation.checkpoints import load_sampling_checkpoint
from ..benchmark.runner import input_view
from ..benchmark.protocol import child_seed
from ..benchmark.rounds import fit_round
from ..query import LabelOracle
from ..baseline.hggs import gradient_degree
from ..baseline.geometry import gaussian_projection
from .hggs_stages import GFProvider,HGGSStages


GF_VERSION='hggs_gf_multistage_current_model_v3'


def prepare_hggs_initial(cohorts,root,cfg,directory,settings,*,device='cpu'):
    state=directory/'_state';state.mkdir(exist_ok=True);path=state/'hggs_gf.json'
    count=cfg.budgets[0]
    if count<4:raise ValueError('HGGS GF needs at least four initial training rows')
    with np.load(root/f'{cohorts.pool}_inputs.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
    with np.load(root/f'{cohorts.pool}_split.npz') as z:allowed=z['candidates']
    oracle=LabelOracle(cohorts.source,allowed,state/'hggs_label_access.json',track='B')
    if path.exists():
        result=json.loads(path.read_text())
        if result.get('gf_version')!=GF_VERSION:raise ValueError('HGGS GF policy changed; use a new experiment directory')
        selected=result['selected_ids']
    else:
        rng=np.random.default_rng(child_seed(cfg.seed,'hggs_gf'))
        n=min(len(allowed),count*settings.coarse_multiplier)
        coarse=np.r_[cohorts.initial_rows,rng.choice(np.setdiff1d(allowed,cohorts.initial_rows),n-count,replace=False)]
        truth=oracle.scoring_reader(coarse)['data']
        reference=NormalizationStats.fit(inputs['params'][cohorts.initial_rows],oracle.reveal(cohorts.initial_rows)['data'])
        view=input_view(inputs,coarse);physical=np.column_stack((view.params,view.initial))
        scale=physical.std(0);scale[scale<1e-8]=1.
        dimension=truth.shape[1]*truth.shape[2]
        projection=gaussian_projection(dimension,min(settings.output_sketch,dimension),cfg.seed)
        output=np.empty((len(coarse),projection.shape[1]))
        for start in range(0,len(coarse),cfg.prediction_batch_size):
            block=truth[start:start+cfg.prediction_batch_size]
            output[start:start+len(block)]=(reference.targets(block).flatten(1)@projection).numpy()
        degrees=gradient_degree((physical-physical.mean(0))/scale,output,settings.neighbors)
        boundary_count=min(count-2,max(2,int(count*settings.boundary_fraction)))
        boundary=coarse[np.argsort(-degrees,kind='stable')[:boundary_count]]
        steps=min(settings.global_steps,count-boundary_count)
        additions=np.full(steps,(count-boundary_count)//steps);additions[:(count-boundary_count)%steps]+=1
        budgets=np.r_[boundary_count,boundary_count+np.cumsum(additions)].astype(int).tolist()
        epochs=[int(cfg.epochs[0])]*len(budgets)
        provider=GFProvider(cohorts.source,inputs,coarse,truth,boundary,cfg.prediction_batch_size,device)
        phases=state/'hggs_pilot/phases';phases.mkdir(parents=True,exist_ok=True)
        stages=HGGSStages(provider,budgets,epochs,phases,reference,cohorts.initial_rows,
            cohorts.validation,'val_loss_synth',keep_phase_weights=False,
            resume_path=state/'hggs_pilot/primary/_state/resume.ckpt')
        snapshot=state/'hggs_pilot/dataset.json'
        provider.snapshot(snapshot,boundary_count,0)
        train_cfg=replace(cfg,budgets=(boundary_count,),epochs=(sum(epochs),),
            warmup=(min(int(cfg.warmup[0]),sum(epochs)-1),))
        model,_,_=fit_round(train_cfg,0,provider.training(boundary_count),cohorts.validation,
            state/'hggs_pilot/primary',child_seed(cfg.seed,'hggs_gf_model'),device=device,
            dataset_path=snapshot,identity=dict(role='hggs_continuous_gf',gf_version=GF_VERSION,
                normalization_reference=fingerprint(cohorts.initial_rows.tolist())),
            data_adapter=stages.adapt_data,after_epoch_callbacks=(stages,),reload_dataloaders_every_n_epochs=1)
        model.cpu()
        progress=json.loads((phases/'stages.json').read_text())
        if not progress['complete'] or len(progress['records'])!=len(budgets):raise RuntimeError('Incomplete HGGS GF stages')
        provider.restore(progress['provider']);selected=provider.rows
        last=state/'hggs_pilot/primary/_state/last_weights.ckpt'
        _,_,payload=load_sampling_checkpoint(last,device='cpu')
        boundary_set=set(map(int,boundary))
        result=dict(gf_version=GF_VERSION,selected_ids=selected,boundary_ids=boundary.tolist(),
            global_residual_ids=[i for i in selected if i not in boundary_set],
            normalization_reference_ids=cohorts.initial_rows.tolist(),
            pilot_checkpoint=dict(policy='live_current_model_between_all_GF_stages',final_epoch=int(payload['epoch'])+1,
                final_sha256=file_sha256(last)),
            stages=progress['records'],shared_random_initial_overlap=len(set(selected)&set(map(int,cohorts.initial_rows))),
            coarse_source_ids=coarse.tolist(),coarse_labels=len(coarse),neighbors=settings.neighbors,
            global_steps=steps,task_adaptations=['trajectory sketch replaces oscillation frequency',
                'shared-source random coarse bank replaces paper LHS','shared-initial Z-score replaces fixed domain bounds',
                'FNO epoch budget replaces original full-batch MLP schedule'],
            source_trajectories_copied=False,initial_cohort_matches_shared_random=False)
        atomic_json(path,result)
    if len(selected)!=count or len(set(selected))!=count or not set(selected)<=set(map(int,allowed)):
        raise ValueError('Saved HGGS GF selection violates training-side budget')
    cohorts.initial_rows=np.asarray(selected,dtype=int);cohorts.initial_inputs=input_view(inputs,cohorts.initial_rows)
    return result
