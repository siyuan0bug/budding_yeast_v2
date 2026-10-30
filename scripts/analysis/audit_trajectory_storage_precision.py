"""Measure float16 storage rounding without changing any dataset or training run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from budding_yeast_v2.data.dataset_archive import open_dataset, _atomic_json


def audit(path):
    with open_dataset(path) as archive:
        values = archive['data']
        real = archive['is_real']
    groups = {key: dict(squared_error=0., squared_reference=0., sum_absolute_error=0., max_absolute_error=0.,
                        overflow_values=0, nonzero_rounded_to_zero=0, values=0)
              for key in ('all', 'real')}
    for start in range(0, len(values), 128):
        original = values[start:start+128]
        with np.errstate(over='ignore'):
            reconstructed = original.astype(np.float16).astype(np.float32)
        for name, mask in (('all', np.ones(len(original), dtype=bool)), ('real', real[start:start+128])):
            a, b = original[mask], reconstructed[mask]
            if not a.size:
                continue
            group = groups[name]
            error = b.astype(np.float64)-a
            group['values'] += a.size
            group['squared_error'] += float(np.square(error).sum())
            group['sum_absolute_error'] += float(np.abs(error).sum())
            group['squared_reference'] += float(np.square(a.astype(np.float64)).sum())
            group['max_absolute_error'] = max(group['max_absolute_error'], float(np.abs(error).max()))
            group['overflow_values'] += int((~np.isfinite(b)).sum())
            group['nonzero_rounded_to_zero'] += int(((a != 0) & (b == 0)).sum())
    for group in groups.values():
        group['relative_l2'] = (group['squared_error']/max(group['squared_reference'], 1e-300))**.5
        group['mean_absolute_error'] = group['sum_absolute_error']/max(group['values'], 1)
    return dict(dataset=str(path), shape=list(values.shape), original_dtype=str(values.dtype),
                trial_dtype='float16', maximum_physical_value=float(values.max()),
                minimum_physical_value=float(values.min()), groups=groups,
                note='Rounding audit only; no retraining or downstream label equivalence claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.dataset)
    # JSON cannot encode infinity as a standard number.
    def finite(value):
        if isinstance(value, dict):return {k: finite(v) for k, v in value.items()}
        if isinstance(value, float) and not np.isfinite(value):return str(value)
        return value
    report = finite(report)
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
