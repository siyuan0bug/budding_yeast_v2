#!/usr/bin/env python3
"""Repair confirmed numerical artifacts without changing inputs, IDs or splits."""
import argparse
import json
from pathlib import Path

import numpy as np

from budding_yeast_v2.data.dataset_archive import TrajectoryStore, atomic_json, file_sha256
from budding_yeast_v2.data.dataset_repairs import write_row_patch
from budding_yeast_v2.data.label_metadata import trajectory_label_metadata
from budding_yeast_v2.data.systems.budding.simulation import simulate_mutant, prepare_initial_conditions
from budding_yeast_v2.data.systems.budding.solver_policy import SOLVER_POLICY, numerical_flags
from budding_yeast_v2.utils.sampling.benchmark.protocol import derive_repaired_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--protocol-root', type=Path, required=True)
    parser.add_argument('--real-reference', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.protocol/'protocol.json').read_text())
    audit = json.loads((args.audit/'strict_solver_rechecks.json').read_text())
    repaired, evidence, exposure = {}, [], {}
    for name, definition in sorted(protocol['pools'].items()):
        store = TrajectoryStore(definition['path'], cache_chunks=1)
        if store.digest != definition['sha256']:
            raise ValueError('Original source changed')
        cases = sorted([r for r in audit['records'] if r['pool'] == name], key=lambda r:r['pool_row'])
        ids = np.asarray([r['pool_row'] for r in cases], dtype=np.int64)
        raw = store.read(ids, ('data','overrides_json','params','y_init','labels','pattern_labels','pattern_group_codes'))
        new_data, bio, abc, diagnostics = [], [], [], []
        for i, case in enumerate(cases):
            overrides = json.loads(str(raw['overrides_json'][i]))
            info = {}
            _, y, label, pattern = simulate_mutant('numerical_repair', overrides, diagnostics=info)
            if y is None:
                raise RuntimeError(f'Same-input repair failed; do not replace its identity: {name}/{ids[i]}')
            p, initial, _, _ = prepare_initial_conditions(overrides)
            if not np.array_equal(initial, raw['y_init'][i]):
                raise ValueError('Recomputed initial state changed')
            cast = y.astype(raw['data'].dtype)
            flags = numerical_flags(cast, p, initial)
            if flags['tem1_total_violation'] or flags['significant_negative'] or not np.isfinite(cast).all():
                raise ValueError('Confirmed artifact not resolved after integration')
            new_data.append(cast); bio.append(label); abc.append(pattern)
            info['numerical_repair'] = dict(source=str(store.path), row=int(ids[i]))
            diagnostics.append(json.dumps(info, sort_keys=True))
            entry = dict(pool=name, pool_row=int(ids[i]), old_minimum=float(raw['data'][i].min()),
                new_minimum=float(cast.min()), old_tem1_peak=float(raw['data'][i,28].max()),
                new_tem1_peak=float(cast[28].max()), tem1_total=p['Tem1T'],
                roles=case['roles'], training_exposure=case['training_exposure'], diagnostics=info)
            evidence.append(entry)
            for use in case['training_exposure']:
                key = f"{name}/{use['method']}"
                exposure[key] = min(exposure.get(key, use['round']), use['round'])
            print(f"Repaired {name} row {ids[i]}: TEM1 {entry['old_tem1_peak']:.6g} -> {entry['new_tem1_peak']:.6g}", flush=True)
        data = np.stack(new_data)
        changes = dict(data=data, pattern_labels=np.asarray(abc),
                       simulation_diagnostics_json=np.asarray(diagnostics))
        changes.update(trajectory_label_metadata(data, np.asarray(bio), 'budding'))
        # Pattern version/rules are shared scalar metadata, not corrected rows.
        changes = {k:v for k,v in changes.items() if 'pieces' in store.fields[k]}
        label_fields = trajectory_label_metadata(data, np.asarray(bio), 'budding')
        for i, row in enumerate(evidence[-len(ids):]):
            row['old_pattern_group'] = str(raw['pattern_group_codes'][i])
            row['new_pattern_group'] = str(label_fields['pattern_group_codes'][i])
            row['biological_label_changed'] = str(raw['labels'][i]) != bio[i]
            row['abc_label_changed'] = str(raw['pattern_labels'][i]) != abc[i]
        destination = args.output_root/(store.path.parent.name+'_numerical_qc')/'dataset.json'
        def progress(block, rows, total):
            if block % 50 == 0 or rows == total:
                print(f'{name}: rebuilt field hashes {rows}/{total}', flush=True)
        repaired[name] = write_row_patch(store.path, destination, ids, changes,
            provenance=dict(solver_policy=SOLVER_POLICY, audit_sha256=file_sha256(args.audit/'strict_solver_rechecks.json')),
            progress=progress)
        corrected = TrajectoryStore(repaired[name], cache_chunks=1)
        if corrected.count != store.count:
            raise ValueError('Repair changed pool size')
        # All row inputs and mutation provenance must remain bit-exact, not just approximately equal.
        unchanged = [k for k,v in store.fields.items() if 'pieces' in v and k not in changes]
        check = corrected.read(ids, tuple(unchanged)+('data',))
        before = store.read(ids, tuple(unchanged))
        for key in unchanged:
            if not np.array_equal(check[key], before[key]):
                raise ValueError(f'Repair modified row metadata: {key}')
        if not np.array_equal(check['data'], data):
            raise ValueError('Published repair differs from recomputed truth')
        if file_sha256(store.path) != definition['sha256']:
            raise ValueError('Original pool was modified')
    root = derive_repaired_protocol(args.protocol, repaired, args.protocol_root,
                                   real_reference=args.real_reference)
    report = dict(solver_policy=SOLVER_POLICY, corrected_rows=len(evidence), pools={k:str(v) for k,v in repaired.items()},
        protocol=str(root), real_reference=str(args.real_reference),
        original_pools_unchanged=True, counts_inputs_order_splits_unchanged=True,
        earliest_affected_training_rounds=exposure, records=evidence, formal_training_started=False)
    atomic_json(args.audit/'numerical_repair_completed.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
