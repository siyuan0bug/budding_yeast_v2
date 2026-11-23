"""Complete-pool data adapter; training execution is a shared service.

Only this adapter knows about selected row manifests, shared validation domains
and the current workflow's cold-start schedule. A future workflow can provide its
own prepared DataModule and reuse fit_model_round without changing any selector.
"""
from pathlib import Path
import numpy as np

from budding_yeast_v2.data.dataset_archive import open_dataset,fingerprint
from budding_yeast_v2.data.normalization import NormalizationStats
from budding_yeast_v2.data.preprocessing_contract import TRAIN_FIT_SCOPE
from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.utils.training.runner import fit_model_round
from budding_yeast_v2.utils.experiments.resources import resource_slot


def fit_round(config,round_index,training,validation,output,model_seed,*,device='cpu',
              member=0,weights=None,identity=None,wandb_run=None,dataset_path=None,state_dir=None,plot_context=None,monitoring=None,
              extra_callbacks=(),data_adapter=None,after_epoch_callbacks=(),reload_dataloaders_every_n_epochs=0,
              normalization_reference=None,normalization_reference_rows=None):
    if dataset_path is None:raise ValueError('Complete-pool rounds require an ordered training snapshot')
    dataset_path=Path(dataset_path).resolve()
    with resource_slot('data', 'training preprocessing'):
        with open_dataset(dataset_path) as archive:
            if (not np.array_equal(getattr(archive,'selection_rows',None),training.inputs.ids)
                    or not np.array_equal(archive['params'],training.inputs.params)
                    or not np.array_equal(archive['y_init'],training.inputs.initial)):
                raise ValueError('Training snapshot is not the authorized query set')
            stored=archive.get('sample_weights')
            if fingerprint(None if stored is None else np.asarray(stored).tolist())!=fingerprint(None if weights is None else np.asarray(weights).tolist()):
                raise ValueError('Snapshot weights disagree with the selected estimator')
        dm_kwargs=dict(dataset_path=str(dataset_path),batch_size_synth=config.batch_size,
            val_batch_size=config.prediction_batch_size,num_workers=0,seed=config.seed,
            use_adj_matrix=False,norm_type='log1p',fit_scope=TRAIN_FIT_SCOPE)
        dm=YeastDataModule(**dm_kwargs)
        dm.setup('fit')
        if normalization_reference is not None:
            # Explicit diagnostic: keep THIS dataset/split contract, replace
            # only statistics using a declared subset of its training inputs.
            refs=np.asarray(normalization_reference_rows,dtype=np.int64)
            if (refs.ndim!=1 or not len(refs) or len(set(refs))!=len(refs)
                    or not set(refs)<=set(training.inputs.ids) or normalization_reference.norm_type!=dm.norm_type):
                raise ValueError('Normalization reference must be a unique subset of current training rows')
            state=dm.preprocessing_state()
            state.update(normalization_reference.state())
            dm=YeastDataModule(**dm_kwargs,preprocessing_state=state)
            dm.setup('fit')
            identity=dict(identity or {},normalization_reference=dict(rows=refs.tolist(),
                stats_sha256=fingerprint({k:None if v is None else v.tolist()
                    for k,v in normalization_reference.state().items()}),fit_scope='declared_training_subset'))
        elif normalization_reference_rows is not None:
            raise ValueError('Reference rows without normalization statistics')
        if len(dm.train_dataset_synth)!=len(training.inputs.ids) or len(dm.test_dataset_real):
            raise ValueError('Data adapter changed the authorized training cohort')
        dm.drop_last_train=False
        dm.set_validation_data(validation,metric=config.validation_metric)
        normalization=NormalizationStats.from_datamodule(dm)
    # The adapter may query a resident cohort through LabelOracle, which takes
    # its own data slot. Do not retain preprocessing's slot across that query:
    # several dynamic jobs could otherwise each hold one slot while awaiting a
    # second. Fixed-pool callers do not install this hook.
    if data_adapter is not None:data_adapter(dm,normalization)
    epochs=int(config.epochs[round_index])
    def model_factory():
        model=YeastLitModule(model_name=config.model,num_vars=dm.num_vars,param_dim=dm.num_params,
            width=config.width,modes=config.modes,n_blocks=config.n_blocks,loss_type=config.loss,
            lr=config.lr,weight_decay=config.weight_decay,max_epochs=epochs,
            warmup_epochs=int(config.warmup[round_index]),min_lr=config.min_lr,norm_type=dm.norm_type,
            seq_len=dm.T_STEPS,system_id=dm.system_id)
        if config.loss=='rel_l2':model.criterion.eps=config.relative_l2_eps
        return model
    identity=dict(identity or {},round=int(round_index),member=int(member),
        experiment_seed=int(config.seed),derived_model_seed=int(model_seed),model=config.model,device=str(device))
    monitor={'normalized_loss':'val_loss_synth',
             'sample_channel_rel_l2':'val_sample_channel_rel_l2',
             'macro_trajectory_rel_l2':'val_macro_rel_l2'}[config.validation_metric]
    model,history=fit_model_round(model_factory,dm,output,model_seed,epochs=epochs,
        identity=identity,device=device,monitor=monitor,round_index=round_index,member=member,wandb_run=wandb_run,
        state_dir=state_dir,plot_context=plot_context,monitoring=monitoring,extra_callbacks=extra_callbacks,
        after_epoch_callbacks=after_epoch_callbacks,reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs)
    return model,normalization,history
