"""dataset services extracted from the pre-package implementation."""
import torch
from torch.utils.data import TensorDataset
from budding_yeast_v2.data.systems.budding.pattern_groups import abc_pattern_ids
import numpy as np
from budding_yeast_v2.data.label_metadata import trajectory_label_metadata


def pattern_target_ids(labels, system_id='budding'):
    """Native Fission absent patterns stay -1; Budding keeps historical ABC mapping."""
    if system_id == 'budding':
        return abc_pattern_ids(labels)
    if system_id != 'fission':
        raise ValueError(f'Unknown AL biological system {system_id!r}')
    mapping = {'Pattern_A': 0, 'Pattern_B': 1, 'Pattern_C': 2, 'None': -1}
    try:
        return np.asarray([mapping[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f'Unknown original Fission pattern: {exc.args[0]}') from exc


def _append_dm_label_metadata(dm, bd):
    """Keep supplementary row arrays aligned; training targets remain ABC."""
    if 'Y_real' not in bd:
        return
    fields = trajectory_label_metadata(np.asarray(bd['Y_real']), bd.get('biological_labels'),
                                       getattr(dm, 'system_id', 'budding'))
    arrays = dict(pattern_labels=np.asarray(bd['labels']), biological_labels=fields['labels'],
                  biological_labels_available=fields['biological_labels_available'])
    for key in ('levels','requested_levels','module_counts'):
        if key in bd:arrays[key]=np.asarray(bd[key],dtype=np.int16)
    for key in ('pattern_group_codes', 'pattern_group_labels', 'pattern_group_ids', 'observed_division_counts'):
        if key in fields:
            arrays[key] = fields[key]
    for key, values in arrays.items():
        existing = getattr(dm, key, None)
        if existing is not None:
            setattr(dm, key, np.concatenate((existing, values)))



def append_training_batch(self, dm, bd):
    old_tensors = dm.train_dataset_synth.tensors
    old_X, old_P, old_Y = old_tensors[0], old_tensors[1], old_tensors[2]
    if len(old_tensors) >= 4:
        new_pattern_ids = torch.as_tensor(pattern_target_ids(bd['labels'], getattr(dm, 'system_id', 'budding')),
            dtype=old_tensors[3].dtype, device=old_tensors[3].device)
        pattern_ids = torch.cat([old_tensors[3], new_pattern_ids], dim=0)
        dm.train_dataset_synth = TensorDataset(
            torch.cat([old_X, bd['X'].to(old_X.device)], dim=0),
            torch.cat([old_P, bd['P'].to(old_X.device)], dim=0),
            torch.cat([old_Y, bd['Y'].to(old_Y.device)], dim=0),
            pattern_ids
        )
    else:
        dm.train_dataset_synth = TensorDataset(
            torch.cat([old_X, bd['X'].to(old_X.device)], dim=0),
            torch.cat([old_P, bd['P'].to(old_X.device)], dim=0),
            torch.cat([old_Y, bd['Y'].to(old_Y.device)], dim=0),
        )
    if 'names' in bd and hasattr(dm, '_mutant_names'):
        self._append_dm_label_metadata(dm, bd)
        new_names = list(bd['names'])
        n_existing = len(dm._mutant_names)
        dm._mutant_names = list(dm._mutant_names) + new_names
        new_indices = np.arange(n_existing, n_existing + len(new_names))
        if hasattr(dm, '_train_idx_synth'):
            dm._train_idx_synth = np.concatenate([np.asarray(dm._train_idx_synth), new_indices])
        else:
            dm._train_idx_synth = new_indices
    self._al4pde_al_iter += 1
    self._train_size_history.append(len(dm.train_dataset_synth))
    n_added = int(bd['X'].shape[0])
    print(f"[al4pde v2.3] 追加 {n_added} 个样本 → 训练集 {len(dm.train_dataset_synth)} "
          f"(al_iter={self._al4pde_al_iter})")

    return n_added
