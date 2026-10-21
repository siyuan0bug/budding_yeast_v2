"""Explicit reference-sensitivity sidecar for standard SC-FNO supervision."""
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def preprocessing_fingerprint(dm):
    values = {'norm_type':dm.norm_type}
    for name in ('mean_y','std_y','p_mean','p_std','s_c'):
        value = getattr(dm,name,None)
        values[name] = None if value is None else torch.as_tensor(value).cpu().tolist()
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()


class SensitivityDataset(Dataset):
    def __init__(self, base, rows, targets, parameter_indices):
        self.base, self.rows = base, np.asarray(rows)
        self.targets = targets
        self.parameter_indices = torch.as_tensor(parameter_indices,dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def __getitem__(self,index):
        x,p,y,pattern = self.base[index]
        return {'x':x,'p':p,'y':y,'pattern':pattern,
                'sensitivity_targets':torch.as_tensor(self.targets[self.rows[index]],dtype=torch.float32),
                'sensitivity_indices':self.parameter_indices}


def attach_sensitivities(dm, path):
    """Targets must use the exact dataset, normalization and fixed-y0 partials.

    NPZ keys: dy_dp (N,V,T,K), parameter_indices (K), data_sha256,
    params_sha256, preprocessing_sha256, initial_conditions_held_fixed=True.
    No test-time sensitivity data is consumed by the model or validation loop.
    """
    with np.load(path,allow_pickle=False) as archive:
        for name in ('data_sha256','params_sha256'):
            if str(archive[name].item()) != dm._dataset_contract[name]:
                raise ValueError(f'Sensitivity sidecar {name} does not match the source dataset')
        if str(archive['preprocessing_sha256'].item()) != preprocessing_fingerprint(dm):
            raise ValueError('Sensitivity targets do not match the current normalization statistics')
        if archive['initial_conditions_held_fixed'].item() is not True:
            raise ValueError('Sensitivity targets must hold initial conditions fixed')
        indices = archive['parameter_indices']
        if indices.ndim != 1 or len(indices)==0 or not np.issubdtype(indices.dtype,np.integer):
            raise ValueError('Expected integer parameter_indices vector')
        if len(np.unique(indices)) != len(indices) or np.any(indices<0) or np.any(indices>=dm.num_params):
            raise ValueError('Invalid sensitivity parameter indices')
        targets = archive['dy_dp']
        expected = (*dm._raw_data.shape,len(indices))
        if targets.shape != expected:
            raise ValueError(f'Sensitivity shape {targets.shape} does not match {expected}')
        if not np.isfinite(targets[dm._train_idx_synth]).all():
            raise ValueError('Training sensitivities must be finite; resolve event singularities explicitly')
    dm.train_dataset_synth = SensitivityDataset(
        dm.train_dataset_synth,dm._train_idx_synth,targets,indices)
