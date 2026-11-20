"""Verify module/compound evidence using executed mutation and initializer code.

No ODE integration, real trajectories, prediction errors or acquisition tuning.
Module witnesses compare the intervention's complete changed-coordinate
projection, not equality of an entire multi-gene real background. Compound
reconstructions additionally compare the full effective input to the real row.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding import catalog
from budding_yeast_v2.data.systems.budding.distributions.ranges import EXACT_ZERO_TARGETS
from budding_yeast_v2.data.systems.budding.parameters import get_default_params
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.budding.simulation import prepare_initial_conditions
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES
from budding_yeast_v2.scripts.analysis.audit_budding_mutation_real_coverage import EVIDENCE
from budding_yeast_v2.scripts.analysis.plan_budding_lhs_domain import EndpointRNG


class ProbeRNG(EndpointRNG):
    rand = EndpointRNG.random


# Numeric witnesses chosen from the real overrides, not learned from errors.
# APC_Core-ts at alpha=0 is a boundary control; it is not a zero atom in MA.
MODULE_WITNESSES = {
    'CLN2': ('KO', 0., '3'), 'CLN3': ('KO', 0., '8'),
    'CLB2': ('KO', 0., '47'), 'CLB5': ('KO', 0., '63'),
    'SIC1': ('KO', 0., '27'), 'CDC6': ('KO', 0., '38'),
    'BCK2': ('KO', 0., '10'), 'SWI5': ('KO', 0., '43'),
    'CDC20': ('KO', 0., '74'), 'CDH1': ('KO', 0., '34'),
    'CDC14': ('GAL-OE', .8, '100'), 'NET1': ('ts', .05, '97'),
    'TEM1': ('KO', 0., '86'), 'CDC15': ('KO', 0., '92'),
    'PDS1': ('KO', 0., '80'), 'ESP1': ('ts', .002, '81'),
    'MAD2': ('CSL', 1., '112'), 'BUB2': ('CKO', 1., '113'),
    'APC_Core': ('ts', 0., '26'),
}
COMPOUND_WITNESSES = (
    ('CLB2', '50', [('CLB2', 'GAL-MC', 8.)]),
    ('CLB2', '62', [('CLB2', 'GAL-OE', .12), ('CLB2', 'db', .034/.4)]),
    ('CLB5', '72', [('CLB5', 'GAL-OE', .012), ('CLB5', 'db', 0.)]),
    ('SIC1', '29', [('SIC1', 'GAL-OE', .12), ('SIC1', 'db', 0.)]),
    ('PDS1', '83', [('PDS1', 'GAL-OE', .1), ('PDS1', 'db', 0.)]),
    ('PDS1', '84', [('PDS1', 'GAL-OE', .1), ('PDS1', 'db', 0.), ('ESP1', 'ts', .002)]),
)


def effective_state(overrides):
    params, initial, mad2_loss, bub2_loss = prepare_initial_conditions(overrides)
    state = {'p:'+key: float(params[key]) for key in get_default_params()}
    state.update({'y0:'+name: float(value) for name, value in zip(VARIABLE_NAMES, initial)})
    state.update({'control:MAD2_loss': float(mad2_loss), 'control:BUB2_loss': float(bub2_loss),
                  'control:BUB2_reset': 0. if bub2_loss else float(overrides.get('init_BUB2', .2)),
                  'control:timed_induction_104': float(params.get('is_mutant_104', False))})
    return state


def different(a, b):
    return sorted(key for key in a if not np.isclose(a[key], b[key], atol=1e-12, rtol=0))


def apply(backend, instructions):
    defaults, overrides = backend.defaults(), {}
    for gene, mode, alpha in instructions:
        backend.apply_gene_mutation(gene, mode, alpha, defaults, defaults.copy(), overrides,
                                    rng=ProbeRNG(False, .75))
    return overrides


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(output):
    output.mkdir(parents=True, exist_ok=True)
    baseline = effective_state({})
    real_states = {mid: effective_state(ovr) for mid, (_, ovr) in REAL_MUTANTS.items()}
    real_changes = {mid: set(different(state, baseline)) for mid, state in real_states.items()}
    all_real_changes = set().union(*real_changes.values())
    genes = set(catalog.GENE_NAMES) - {'MDT'}
    if genes != set(MODULE_WITNESSES):
        raise ValueError('Active catalog changed; explicitly review module witnesses')
    checked_compounds = {(gene, tuple(sorted(m for g, m, _ in instructions if g == gene)))
                         for gene, _, instructions in COMPOUND_WITNESSES
                         if len([g for g, _, _ in instructions if g == gene]) == 2}
    if checked_compounds != set(catalog.COMPOSITE_REAL_EVIDENCE):
        raise ValueError('Compound allowlist changed; explicitly review numeric witnesses')
    module_rows, mode_rows, coordinate_rows, compound_rows = [], [], [], []
    for profile in ('mutant_aware', 'mass'):
        backend = BuddingSystem(profile, profile)
        for gene in sorted(genes):
            mode, alpha, mid = MODULE_WITNESSES[gene]
            overrides = apply(backend, [(gene, mode, alpha)])
            synthetic = effective_state(overrides)
            keys = different(synthetic, baseline)
            mismatches = [k for k in keys if not np.isclose(synthetic[k], real_states[mid][k], atol=1e-12, rtol=0)]
            if not keys or mismatches:
                raise AssertionError((profile, gene, mid, mismatches))
            module_rows.append(dict(profile=profile, module=gene, witness_mode=mode, real_id=mid,
                real_name=REAL_MUTANTS[mid][0], changed_coordinates=';'.join(keys),
                synthetic_values_json=json.dumps({k: synthetic[k] for k in keys}),
                real_values_json=json.dumps({k: real_states[mid][k] for k in keys}),
                changed_projection_equal=True, entire_real_background_compared=False,
                caveat='MA boundary reconstruction, not a positive-probability zero atom'
                    if gene=='APC_Core' and profile=='mutant_aware' and (gene,mode) not in EXACT_ZERO_TARGETS else ''))
            for mode in catalog.GROUP_BY_NAME[gene]['modes']:
                draws, written = [], set()
                for upper in (False, True):
                    for branch in (.25, .75):
                        rng = ProbeRNG(upper, branch)
                        alpha = backend.sample_scaling_factor(mode, gene, rng=rng)
                        overrides = {}
                        backend.apply_gene_mutation(gene, mode, alpha, backend.defaults(), {}, overrides, rng=rng)
                        written.update(overrides)
                        draws.append(effective_state(overrides))
                changed = set().union(*(set(different(s, baseline)) for s in draws))
                unseen = sorted(changed - all_real_changes)
                mode_rows.append(dict(profile=profile, module=gene, mode=mode,
                    written_override_keys=';'.join(sorted(written)), actual_changed_coordinates=';'.join(sorted(changed)),
                    coordinates_never_changed_in_any_real=';'.join(unseen),
                    genotype_or_mechanism_real_ids=';'.join(EVIDENCE.get(gene, {}).get(mode, [])),
                    note='Shared coordinate overlap alone is not genotype/allele evidence'))
                for key in sorted(changed):
                    ids = [mid for mid in REAL_MUTANTS if key in real_changes[mid]]
                    coordinate_rows.append(dict(profile=profile, module=gene, mode=mode, coordinate=key,
                        wt=baseline[key], synthetic_lower=min(s[key] for s in draws),
                        synthetic_upper=max(s[key] for s in draws),
                        real_changed_ids=';'.join(ids), real_min=min(s[key] for s in real_states.values()),
                        real_max=max(s[key] for s in real_states.values())))
        for gene, mid, instructions in COMPOUND_WITNESSES:
            overrides = apply(backend, instructions)
            overrides['mdt'] = 150.
            state = effective_state(overrides)
            mismatches = different(state, real_states[mid])
            if mismatches:
                raise AssertionError((profile, mid, mismatches))
            compound_rows.append(dict(profile=profile, module=gene, real_id=mid, real_name=REAL_MUTANTS[mid][0],
                instructions_json=json.dumps(instructions), synthetic_overrides_json=json.dumps(overrides),
                real_overrides_json=json.dumps(REAL_MUTANTS[mid][1]), full_effective_input_equal=True,
                max_absolute_difference=max(abs(state[k]-real_states[mid][k]) for k in state),
                exact_zero_atom_in_db_rule=(profile=='mass' or (gene,'db') in EXACT_ZERO_TARGETS)
                    if gene in ('CLB5', 'SIC1', 'PDS1') else 'not_applicable',
                note='MA db=0 is a boundary reconstruction, not a positive-probability zero atom'
                     if profile=='mutant_aware' and gene in ('CLB5', 'SIC1', 'PDS1') and (gene,'db') not in EXACT_ZERO_TARGETS else ''))
    for filename, rows in [('module_parameter_witnesses.csv', module_rows), ('mode_parameter_inventory.csv', mode_rows),
                           ('coordinate_real_coverage.csv', coordinate_rows), ('compound_reconstructions.csv', compound_rows)]:
        write_csv(output/filename, rows)
    summary = dict(active_modules=len(genes), modules_without_any_parameter_witness=[],
        profiles=['mutant_aware', 'mass'], module_projection_checks_passed=len(module_rows),
        mode_footprints_audited=len(mode_rows), compound_full_input_checks_passed=len(compound_rows),
        novel_changed_coordinates=sorted({k for row in mode_rows for k in row['coordinates_never_changed_in_any_real'].split(';') if k}),
        full_input_comparisons_include='Every base parameter, 39 actual initial states, checkpoint/reset and timed-induction controls',
        tolerance=dict(absolute=1e-12, relative=0.),
        scope='Canonical 126 real definitions in this repository; no paper-wide biological validity claim',
        caveats=['Module witness is equality on the entire changed-coordinate projection, not the full multi-gene background',
                 'Mode support envelopes are not proof of joint-value coverage or genotype identity',
                 'Boundary reconstruction is not proof of positive sampling probability',
                 'APC_Core is a functional module; CDC20[APC-A] and APC_Core[APC-A] share one parameter mechanism'],
        production_files_modified=False, ode_calls=0)
    (output/'parameter_evidence_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)


if __name__ == '__main__':
    main()
