#!/usr/bin/env python3
"""Check finite-window labels with canonical MA integrations at longer horizons.

Diagnostic only: retains production ODE, initial-state rules, resets, tolerances
and five-second worker deadline. MASS>10 at a long horizon remains a biological
warning, even though the production exporter only mass-filters tf<=210.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent))
from budding_yeast_v2.data.systems.budding import simulation
from budding_yeast_v2.data.systems.budding.real_mutants import real_tasks
from budding_yeast_v2.scripts.analysis.dataset_diagnosis.audit_six_patterns import features


def probe(task):
    name, overrides, tf, output = task
    events = []
    original_solver = simulation.solve_ivp
    def traced_solver(*args, **kwargs):
        sol = original_solver(*args, **kwargs)
        if sol.t_events is not None:
            events.extend(float(t) for t in sol.t_events[0])
        return sol
    simulation.solve_ivp = traced_solver
    try:
        _, y, biology, old_pattern = simulation.simulate_mutant(name, overrides, tf=tf,
            ns=round(tf / 210 * 499) + 1)
    finally:
        simulation.solve_ivp = original_solver
    row = dict(mutant_name=name, tf_min=tf, complete=y is not None,
               original_label=biology, original_pattern=old_pattern,
               actual_division_events=len(events), division_times_min=';'.join(f'{t:.6f}' for t in events))
    if y is not None:
        f, _ = features(y[None])
        row.update({k:v[0].item() for k,v in f.items()})
        np.savez_compressed(Path(output) / f'{name.split("_")[0]}_{tf:g}min.npz',
                            data=y, time_grid=np.linspace(0,tf,y.shape[1]), division_times=events)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mutant-ids', nargs='*', help='Default all 126 canonical conditions')
    p.add_argument('--horizons', nargs='+', type=float, default=[210,420,630])
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(name, ovr, tf, str(args.output_dir)) for name,ovr in real_tasks()
        if not args.mutant_ids or name.split('_')[0] in args.mutant_ids for tf in args.horizons]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for row in pool.map(probe, tasks):
            rows.append(row)
            if len(rows) % 30 == 0: print(f'{len(rows)}/{len(tasks)} horizon integrations', flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / 'horizon_probes.csv', index=False)
    print(pd.crosstab(pd.DataFrame(rows).tf_min, pd.DataFrame(rows).pattern_code).to_string())
    print('Incomplete:',sum(not r['complete'] for r in rows))


if __name__ == '__main__': main()
