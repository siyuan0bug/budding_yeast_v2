"""Shared strict checkpoint/data loading for downstream model comparisons."""
from budding_yeast_v2.data.normalization import NormalizationStats
Normalizer = NormalizationStats  # Historical serialized/imported name only.
import inspect
import numpy as np
import torch
from budding_yeast_v2.models.yeast_lit_module import YeastLitModule
from budding_yeast_v2.data.preprocessing_contract import resolve_checkpoint_norm, resolve_checkpoint_fit_scope
from torch.utils.data import DataLoader

from budding_yeast_v2.data.yeast_datamodule import YeastDataModule
from budding_yeast_v2.data.sensitivity import preprocessing_fingerprint


def predict_real_checkpoints(checkpoints,dataset_path,device='cpu',batch_size=32):
    """Evaluate explicit LABEL=PATH inputs with identical preprocessing/cohorts.

    Incompatible normalization is rejected rather than mixing normalized inputs
    from one model with another model's statistics. No architecture fallback.
    """
    entries={}
    for item in checkpoints:
        label,separator,path=item.partition('=')
        if not separator or not label or not path or label in entries:
            raise ValueError('Each checkpoint must be a unique LABEL=PATH')
        entries[label]=path
    if not entries:
        raise ValueError('At least one checkpoint is required')
    predictions={}; target=None; reference_dm=None; fingerprint=None
    for label,path in entries.items():
        model,hp=load_model_from_checkpoint(path,device=device)
        dm=YeastDataModule(dataset_path=dataset_path,num_workers=0,use_adj_matrix=False,
            norm_type=hp['norm_type'],seed=hp['data_seed'],fit_scope=hp['normalization_fit_scope'],
            preprocessing_state=model._preprocessing_state)
        dm.setup('fit')
        current=preprocessing_fingerprint(dm)
        if fingerprint is not None and current != fingerprint:
            raise ValueError('Checkpoint normalization differs; evaluate models separately with eval.py')
        if reference_dm is not None and list(dm.test_indices[1]) != list(reference_dm.test_indices[1]):
            raise ValueError('Checkpoint evaluation cohorts differ')
        fingerprint=current;reference_dm=dm
        model.to(device).eval();model.bind_datamodule_stats(dm)
        outputs=[];targets=[]
        with torch.no_grad():
            offset=0
            for x,p,y in DataLoader(dm.test_dataset_real,batch_size=batch_size,shuffle=False):
                x,p,y=x.to(device),p.to(device),y.to(device)
                outputs.append(dm.denormalize_y(model(x,p),device).cpu())
                ids=dm.test_indices[1][offset:offset+len(x)]
                targets.append(torch.as_tensor(dm._raw_data[ids]).cpu())
                offset+=len(x)
        if not outputs:
            raise ValueError('Dataset has no real-mutant evaluation samples')
        predictions[label]=torch.cat(outputs)
        current_target=torch.cat(targets)
        if target is not None:
            torch.testing.assert_close(target,current_target)
        target=current_target
    names=[str(reference_dm._mutant_names[i]) for i in reference_dm.test_indices[1]]
    return predictions,target,names


def load_model_from_checkpoint(ckpt_path, device='cpu', *, graph_load_policy='current',
                               graph_view=None, adj_matrix_path=None, norm_type=None):
    if graph_load_policy == 'checkpoint' and (graph_view is not None or adj_matrix_path is not None):
        raise ValueError('Historical graph reproduction cannot also request a graph override')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if ckpt.get('format') == 'yeast_sampling_fno_v1':
        # Read-only support for the short-lived pre-consolidation format.
        nt=ckpt.get('normalization_type','log1p')
        if norm_type is not None and norm_type!=nt:
            raise ValueError('Requested normalization contradicts saved statistics')
        hp=dict(ckpt['model_config'],model_name='fno',norm_type=nt,
                normalization_fit_scope=ckpt.get('normalization_fit_scope','train_synthetic'),
                data_seed=ckpt.get('identity',{}).get('experiment_seed',42))
        model=YeastLitModule(**ckpt['model_config'],model_name='fno',norm_type=nt)
        model.model.load_state_dict(ckpt['state_dict'],strict=True)
        for key in ('mean_y','std_y','p_mean','p_std'):
            getattr(model,key).copy_(ckpt['normalization'][key].reshape_as(getattr(model,key)))
        model.norm_type=nt;model._stats_bound=True
        model._legacy_fit_scope=hp['normalization_fit_scope']
        model.set_parameter_stats(model.p_mean,model.p_std,nt)
        model.to(device).eval()
        return model,hp
    hp = dict(ckpt.get('hyper_parameters', {}))
    hp['norm_type'] = resolve_checkpoint_norm(ckpt, norm_type)
    hp['normalization_fit_scope'] = resolve_checkpoint_fit_scope(ckpt)
    dhp = ckpt.get('datamodule_hyper_parameters', {})
    saved_split = (ckpt.get('preprocessing_contract') or {}).get('split') or {}
    hp['data_seed'] = saved_split.get('seed', dhp.get('seed', hp.get('data_seed', 42)))
    accepted = inspect.signature(YeastLitModule.__init__).parameters
    init_kwargs = {key: value for key, value in hp.items() if key in accepted and key != 'self'}
    init_kwargs.update(
        adj_matrix=hp.get('adj_matrix') if adj_matrix_path is None else None,
        causal_matrix=hp.get('causal_matrix') if adj_matrix_path is None else None,
        adj_matrix_path=adj_matrix_path,
        graph_view=graph_view or hp.get('graph_view', 'hybrid'), graph_load_policy=graph_load_policy,
    )
    model = YeastLitModule(**init_kwargs)
    model.on_load_checkpoint(ckpt)
    incompatible = model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
    missing = [key for key in incompatible.missing_keys if key.startswith('model.')]
    unexpected = [key for key in incompatible.unexpected_keys if key.startswith('model.')]
    if missing or unexpected:
        raise RuntimeError(f'Incomplete model checkpoint: missing={missing}, unexpected={unexpected}')
    model.to(device).eval()
    return model, hp


def load_sampling_checkpoint(path, device='cpu'):
    model,_=load_model_from_checkpoint(path,device=device)
    return model,NormalizationStats.from_datamodule(model),torch.load(path,map_location='cpu',weights_only=False)


class Predictor:
    """Only models and fitted statistics; no store, oracle or holdout reference."""
    def __init__(self, members, normalization, steps, device, batch_size=128, sketch_seed=42):
        self.members = members
        self.normalization = normalization
        self.steps = steps
        self.device = device
        self.batch_size = batch_size
        self.sketch_seed = sketch_seed
        self.sketches = {}

    def batches(self, inputs, *, physical=False, latent=False, ensemble=False):
        for start in range(0, len(inputs.ids), self.batch_size):
            stop = min(start + self.batch_size, len(inputs.ids))
            x, p = self.normalization.encode(inputs.params[start:stop], inputs.initial[start:stop], self.steps)
            predictions = []
            for model in self.members if ensemble else self.members[:1]:
                model.to(self.device).eval()
                with torch.no_grad():
                    if latent:
                        backbone = getattr(model, 'model', model)
                        if 'return_features' not in inspect.signature(backbone.forward).parameters:
                            raise ValueError(f'{type(backbone).__name__} does not expose last-layer features; choose a Pred selector')
                        value = backbone(x.to(self.device), p.to(self.device), return_features=True)[1]
                    else:
                        forward = getattr(model,'forward_ic_time',model)
                        value = forward(x.to(self.device), p.to(self.device))
                    # Check before inverse-transform clamps: +Inf can otherwise
                    # become a finite concentration at the expm1 safety ceiling.
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Nonfinite model prediction before decoding; batch starts at {start}')
                    if physical:
                        value = self.normalization.decode(value)
                    value=value.float()
                    if not torch.isfinite(value).all():
                        raise FloatingPointError(f'Nonfinite decoded/cast prediction; batch starts at {start}')
                    predictions.append(value.cpu())
            yield start, torch.stack(predictions) if ensemble else predictions[0]

    def uncertainty(self, inputs):
        if len(self.members) < 2:
            raise ValueError('Uncertainty selection requires independently trained members')
        scores = np.empty(len(inputs.ids), dtype=np.float32)
        for start, predictions in self.batches(inputs, ensemble=True):
            # AL4PDE MSE disagreement around the ensemble mean.
            value = (predictions - predictions.mean(0, keepdim=True)).square().mean((0, 2, 3))
            scores[start:start + len(value)] = value.numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite uncertainty scores')
        return scores

    def features(self, inputs, dimension=512, latent=False):
        from budding_yeast_v2.utils.sampling.baseline.geometry import gaussian_projection
        output = np.empty((len(inputs.ids), dimension), dtype=np.float32)
        for start, values in self.batches(inputs, latent=latent):
            raw = values.transpose(1, 2).flatten(1)
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite model features')
            key = (latent, dimension)
            if key not in self.sketches:
                self.sketches[key] = gaussian_projection(raw.shape[1], dimension, self.sketch_seed)
            projected = raw @ self.sketches[key]
            output[start:start + len(projected)] = projected.numpy()
        return output

    def physical(self, inputs):
        return np.concatenate([values.numpy() for _, values in self.batches(inputs, physical=True)])
