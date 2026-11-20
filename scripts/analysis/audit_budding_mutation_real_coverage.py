"""Audit synthetic local allele states against the canonical 126 real records.

Evidence is a curated genotype/mechanism crosswalk, not a claim of equality of
synthetic numerical rules, initial states or full multi-gene backgrounds. Real
trajectories and prediction errors are never read. This is an analysis tool;
the crosswalk must not be used to fit acquisition scores or pool probabilities.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding import catalog
from budding_yeast_v2.data.systems.budding.quotas import allele_options
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS


# Representative records; inclusion in a multi-gene real background is sufficient
# evidence for a constituent mode. This is not an exhaustive list of all real IDs.
EVIDENCE = {
    'CLN2': {'KO': ['3'], 'GAL-OE': ['4'], 'MC': ['15']},
    'CLN3': {'KO': ['8'], 'GAL-OE': ['9']},
    'CLB2': {'KO': ['47'], 'GAL-OE': ['25', '49', '124'],
             'GAL-MC': ['50'], 'db': ['55'], 'pLOF': ['48', '51', '52', '111']},
    'CLB5': {'KO': ['63'], 'GAL-OE': ['65'], 'MC': ['22'], 'db': ['68']},
    'SIC1': {'KO': ['27'], 'GAL-OE': ['28'], 'MC': ['57', '118'], 'db': ['29']},
    'CDC6': {'KO': ['38'], 'GAL-OE': ['121'], 'MC': ['59', '120']},
    'BCK2': {'KO': ['10'], 'MC': ['11', '24']},
    'SWI5': {'KO': ['43']},
    'CDC20': {'KO': ['71', '74'], 'ts': ['73'], 'GAL-OE': ['37', '77'],
              'MC': ['122'], 'APC-A': ['115']},
    'CDH1': {'KO': ['34'], 'CA': ['35']},
    'CDC14': {'ts': ['99', '105', '107'], 'GAL-OE': ['100'], 'MC': ['91', '96']},
    'NET1': {'TAB6-1': ['108'], 'GAL-OE': ['98'], 'ts': ['90', '95', '97']},
    'TEM1': {'KO': ['86'], 'GAL-OE': ['87'], 'MC': ['94'], 'ts': ['88']},
    'CDC15': {'KO': ['92'], 'GAL-OE': ['89'], 'MC': ['88', '93'], 'ts': ['95', '96']},
    'PDS1': {'KO': ['80'], 'GAL-OE': ['83', '84'], 'db': ['82']},
    'ESP1': {'ts': ['81'], 'GAL-OE': ['85']},
    'MAD2': {'CSL': ['112']},
    'BUB2': {'CKO': ['113']},
    'APC_Core': {'APC-A': ['115'], 'ts': ['26']},
}
COMPOUND_EVIDENCE = {
    ('CLB2', ('GAL-OE', 'db')): ['62'],
    ('CLB5', ('GAL-OE', 'db')): ['72'],
    ('SIC1', ('GAL-OE', 'db')): ['29'],
    ('PDS1', ('GAL-OE', 'db')): ['83', '84'],
}
NOTES = {
    ('CDC20', 'APC-A'): 'APC-A pathway alias, not a separate CDC20 gene allele; canonical ka20pp=0 is now preserved',
    ('APC_Core', 'APC-A'): 'APC-A pathway alias shares evidence and ka20pp=0 with CDC20[APC-A]',
    ('APC_Core', 'ts'): 'apc-ts functional mapping; canonical synthesis and CDH1 initial closures are now preserved',
    ('CLB2', 'pLOF'): 'CLB1 clb2 deletion retains partial module synthesis; mechanism-level mapping, not a literal pLOF genotype',
    ('CDC6', 'KO'): 'Model mapping for cdc6 deletion 2-49; not a claim of whole-gene deletion',
    ('SIC1', 'PM'): 'Synthetic kd2c1=kd3c1=0; real #29 db only sets kd3c1=0',
    ('CLB2', 'MC'): 'Real #50 is GAL-MC, not ordinary MC',
    ('CLN3', 'MC'): 'GAL-CLN3 exists; ordinary MC has no corresponding real genotype',
    ('NET1', 'ts'): 'net1-ts/TAB6-1 share binding mechanism; genotype names are kept distinct',
    ('CDC14', 'ts'): 'Real synthesis-loss and activity-loss backgrounds exist; not an assertion all synthetic branch strengths are calibrated',
    ('TEM1', 'MC'): 'Real #94 shares total-protein=5 with GAL-TEM1 #87 but is named multi-copy',
    ('MAD2', 'CSL'): 'Checkpoint-loss mapping for mad2 deletion',
    ('BUB2', 'CKO'): 'Checkpoint-loss mapping for bub2 deletion',
}


def paper_scope(gene, modes, ids):
    """Keep Table S5 membership distinct from paper predictions and concepts."""
    if len(modes)>1:
        return 'Table_S5_same_gene_compound' if ids else 'no_Table_S5_compound_evidence'
    mode=modes[0]
    if ids:
        return 'Table_S5_component_mapping_range_is_design' if mode=='pLOF' else 'Table_S5_type_or_mechanism_mapping'
    if (gene,mode) in {('CLN3','MC'),('CLB2','MC')}:
        return 'Chen2004_Table4_two_copy_prediction_not_in_TableS5'
    if (gene,mode)==('SIC1','PM'):
        return 'Chen2000_related_concept_current_two_parameter_rule_unverified'
    if mode=='DAmP':
        return 'external_DAmP_technique_current_gene_distribution_is_extension'
    return 'current_gene_parameter_recipe_not_found_in_reviewed_primary_tables'


def rows():
    genes = [g for g in catalog.GENE_NAMES if g != 'MDT']
    allowed = {(g, m) for g in genes for m in catalog.GROUP_BY_NAME[g]['modes']}
    evidence_keys = {(g, m) for g, modes in EVIDENCE.items() for m in modes}
    if not evidence_keys <= allowed:
        raise ValueError(f'Crosswalk no longer matches catalog: {evidence_keys - allowed}')
    all_compounds = {(g, a) for g in genes for a in allele_options(g) if len(a) > 1}
    if not set(COMPOUND_EVIDENCE) <= all_compounds:
        raise ValueError('Compound crosswalk no longer matches catalog')
    result = []
    for gene in genes:
        for modes in allele_options(gene):
            mode = '+'.join(modes)
            ids = (EVIDENCE.get(gene, {}).get(mode, []) if len(modes) == 1
                   else COMPOUND_EVIDENCE.get((gene, modes), []))
            names = [REAL_MUTANTS[mid][0] for mid in ids]  # Fail if evidence ID disappeared.
            keys = sorted(set().union(*(catalog.touch_set(gene, m) for m in modes)))
            note = NOTES.get((gene, mode), '')
            if gene in catalog.DISABLED_FUNCTIONAL_MODULES:
                note = 'Synthetic functional intervention; no corresponding allele in current canonical 126'
            result.append(dict(gene=gene, mode=mode,
                kind='single_mode' if len(modes) == 1 else 'same_gene_compound',
                represented_in_real=bool(ids), representative_real_ids=';'.join(ids),
                representative_real_names=' | '.join(names),
                parameter_touch_set=';'.join(keys), note=note,
                paper_evidence_scope=paper_scope(gene,modes,ids)))
    return result


def write_csv(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def real_compounds():
    """Scan every real genotype for same-gene GAL+db and multi-copy GAL.

    Keep GAL-CLN2 replacement of deleted native CLN1/2 synthesis separate; it
    is intrinsic to that GAL rule, not permission to combine arbitrary KO+GAL.
    """
    result = []
    for mid, (name, overrides) in REAL_MUTANTS.items():
        for gene in catalog.GENE_NAMES:
            if re.search(rf'GAL-{re.escape(gene)}-?db', name):
                modes = ('GAL-OE', 'db')
                if mid not in catalog.COMPOSITE_REAL_EVIDENCE.get((gene, modes), ()):
                    raise ValueError(f'Unreviewed real compound: {mid}: {name}')
                result.append(dict(real_id=mid, name=name, gene=gene, state='GAL-OE+db',
                    handling='Explicit same-gene compound allowlist', overrides_json=json.dumps(overrides)))
            if f'multi-copy GAL-{gene}' in name:
                if gene != 'CLB2':
                    raise ValueError(f'Unreviewed multi-copy GAL: {mid}: {name}')
                result.append(dict(real_id=mid, name=name, gene=gene, state='GAL-MC',
                    handling='Dedicated mode: GAL base 0.12 times integer copies; #50 is 8 copies',
                    overrides_json=json.dumps(overrides)))
        if 'GAL-CLN2' in name and overrides.get('ksn2pp') == 0.:
            result.append(dict(real_id=mid, name=name, gene='CLN2', state='GAL_with_native_channel_off',
                handling='Intrinsic GAL-CLN2 rule; do not generate a separate KO+GAL compound',
                overrides_json=json.dumps(overrides)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = rows()
    compounds = real_compounds()
    missing = [r for r in records if not r['represented_in_real']]
    summary = dict(canonical_real_records=len(REAL_MUTANTS),
        scope='Gene/mode constituent or corresponding functional mechanism, not exact input equality',
        evidence='Curated crosswalk to representative canonical IDs; not all matching real IDs',
        counts={kind: dict(total=sum(r['kind'] == kind for r in records),
                          absent=sum(r['kind'] == kind for r in missing))
                for kind in ('single_mode', 'same_gene_compound')},
        missing=[dict(gene=r['gene'], mode=r['mode'], kind=r['kind']) for r in missing],
        disabled_modules=sorted(catalog.DISABLED_FUNCTIONAL_MODULES),
        membership_does_not_imply_paper_wide_absence=True,
        real_compound_records=compounds,
        caveats=['Absence means absent from current canonical 126, not absent from literature',
                 'Presence does not validate every strength, initial state, or multi-gene background',
                 'APC-A appears under two catalog aliases but is one real mechanism',
                 'No catalog, probability, mutation, ODE, or acquisition rule is changed'])
    write_csv(args.output/'mutation_real_coverage.csv', records)
    write_csv(args.output/'mutations_absent_from_real.csv', missing)
    write_csv(args.output/'real_compound_records.csv', compounds)
    (args.output/'mutation_real_coverage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
