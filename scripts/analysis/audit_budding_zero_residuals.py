"""List real exact-zero channels replaced by mutant-aware residual distributions.

This compares actual production support components with semantically reviewed
same-mode real evidence. Shared parameter names alone are not allele evidence.
No simulation or production policy change is performed.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding.parameters import get_default_params
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.scripts.analysis.audit_budding_mutation_real_coverage import EVIDENCE
from budding_yeast_v2.scripts.analysis.plan_budding_lhs_domain import support_components


def audit():
    rows = []
    defaults = get_default_params()
    for component in support_components(BuddingSystem()):
        gene, mode, key = component['gene'], component['mode'], component['parameter']
        if mode not in ('db', 'ts', 'APC-A') or component['support_kind'] != 'continuous':
            continue
        evidence = EVIDENCE.get(gene, {}).get(mode, [])
        zero_ids = [mid for mid in evidence if REAL_MUTANTS[mid][1].get(key, defaults.get(key)) == 0.]
        if not zero_ids or component['upper'] <= 0.:
            continue
        rows.append(dict(gene=gene, mode=mode, branch=component['branch'], parameter=key,
            wt=defaults[key], representative_real_zero_ids=';'.join(zero_ids),
            current_residual_lower=component['lower'], current_residual_upper=component['upper'],
            explicit_zero_probability=0.,
            mechanism='degradation_channel_loss' if mode=='db' else
                      'synthesis_loss' if key.startswith('ks') else 'activity_channel_loss',
            caveat='Exact zero is a model intervention, not a zero protein concentration or proof of an in-vivo measurement'))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = audit()
    with (args.output/'exact_zero_vs_residual.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = Counter((r['gene'], r['mode'], r['branch']) for r in rows)
    summary = dict(status='current_rules_audit_read_only',
        gene_mode_branches=len(groups), parameter_component_rows=len(rows),
        unique_parameters=sorted({r['parameter'] for r in rows}),
        groups=[dict(gene=g, mode=m, branch=b, affected_parameters=count)
                for (g, m, b), count in sorted(groups.items())],
        independent_mechanisms=len({('APC-A',m,b) if m=='APC-A' else (g,m,b) for g,m,b in groups}),
        real_ids_are_representative_not_exhaustive=True,
        exclusions=['Already-exact KO/CA/CKO/PM and intrinsic GAL-CLN2 closures',
                    'CLB2-db: kdb2P=0 preserved, real kdb2pp=0.034 is nonzero',
                    'TEM1/ESP1-ts, NET1-ts/TAB6-1 have explicit nonzero real residuals',
                    'CDC14 activity-reduction branch has nonzero real activity',
                    'Modes without same-mode real evidence are not automatically equated to KO'],
        ode_calls=0, production_files_modified=False)
    (args.output/'zero_residual_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(summary=summary, channels=rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
