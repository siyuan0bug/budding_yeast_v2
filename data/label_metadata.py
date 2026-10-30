"""Supplementary row-label metadata, independent of model training targets."""
import numpy as np

UNKNOWN_BIOLOGY_LABEL = 'Unknown: Not recorded'


def biological_label_metadata(labels, n_samples):
    values = (np.full(n_samples, UNKNOWN_BIOLOGY_LABEL) if labels is None
              else np.asarray(labels, dtype=str))
    if values.shape != (n_samples,):
        raise ValueError('Biological labels must have one entry per trajectory')
    return dict(labels=values, biological_labels_available=values != UNKNOWN_BIOLOGY_LABEL)


def trajectory_label_metadata(data, labels, system_id):
    fields = biological_label_metadata(labels, len(data))
    if system_id == 'budding':
        from .systems.budding.pattern_groups import pattern_group_metadata
        fields.update(pattern_group_metadata(data))
    return fields


def dataset_label_summary(dm):
    """JSON-ready native labels for experiment records and W&B configuration."""
    from collections import Counter

    def counts(values, indices):
        return dict(Counter(str(values[idx]) for idx in indices))

    splits = {}
    for name, attr in (('train_synthetic', '_train_idx_synth'),
                       ('validation_synthetic', '_val_idx_synth'),
                       ('test_synthetic', '_test_idx_synth'), ('real', '_test_idx_real')):
        indices = getattr(dm, attr)
        split = dict(count=len(indices), biological_labels=counts(dm.biological_labels, indices),
                     pattern_labels=counts(dm.pattern_labels, indices),
                     biological_labels_available=int(dm.biological_labels_available[indices].sum()))
        if dm.pattern_group_codes is not None:
            split['pattern_groups'] = counts(dm.pattern_group_codes, indices)
        for key in ('levels','requested_levels','module_counts'):
            values=getattr(dm,key,None)
            if values is not None:split[key]=counts(values,indices)
        splits[name] = split
    return dict(system_id=dm.system_id, num_vars=dm.num_vars, num_params=dm.num_params,
                variable_names=list(dm.variable_names), horizon_minutes=float(dm.t_max),
                pattern_group_version=dm.pattern_group_version,
                missing_pattern_target=-1 if dm.system_id == 'fission' else None,
                splits=splits)
