"""Active synthetic controls; inactive definitions remain available for review.

Real-mutant definitions and the canonical ODE are independent of this catalog.
"""
import numpy as np
from .distributions.ranges import RANGE_FINGERPRINT

CATALOG_VERSION = 'real_supported_composites_19_modules_2026_11_12_' + RANGE_FINGERPRINT[:12]
DISABLED_FUNCTIONAL_MODULES = frozenset({'PPX', 'Mcm1', 'SBF_MBF'})
# Compatibility for specialized samplers that used to add these graph nodes.
RESTORED_FUNCTIONAL_MODULES = frozenset()

BINDING_GROUPS = {'G1': {'name': 'CLN2',
        'params': ['ksn2p', 'ksn2pp'],
        'P_bio_G': 0.05,
        'modes': {'KO': 0.667, 'GAL-OE': 0.211, 'MC': 0.03, 'ts': 0.092}},
 'G2': {'name': 'CLN3',
        'params': ['C0'],
        'P_bio_G': 0.03,
        'modes': {'KO': 0.853, 'GAL-OE': 0.113, 'MC': 0.034}},
 'G3': {'name': 'CLB2',
        'params': ['ksb2p', 'ksb2pp', 'kdb2p', 'kdb2pp', 'kdb2P', 'kdib2'],
        'P_bio_G': 0.06,
        'modes': {'KO': 0.09,
                  'GAL-OE': 0.31,
                  'MC': 0.075,
                  'GAL-MC': 0.075,
                  'ts': 0.035,
                  'db': 0.31,
                  'pLOF': 0.15}},
 'G4': {'name': 'CLB5',
        'params': ['ksb5p', 'ksb5pp', 'kdb5pp'],
        'P_bio_G': 0.04,
        'modes': {'KO': 0.37, 'GAL-OE': 0.28, 'MC': 0.07, 'db': 0.28}},
 'G5': {'name': 'SIC1',
        'params': ['ksc1p', 'ksc1pp', 'kd3c1', 'kd2c1'],
        'P_bio_G': 0.05,
        'modes': {'KO': 0.44, 'GAL-OE': 0.35, 'MC': 0.07, 'db': 0.04, 'PM': 0.1}},
 'G6': {'name': 'CDC6',
        'params': ['ksf6p', 'ksf6pp', 'ksf6ppp'],
        'P_bio_G': 0.03,
        'modes': {'KO': 0.625, 'GAL-OE': 0.125, 'MC': 0.125, 'DAmP': 0.125}},
 'G7': {'name': 'BCK2', 'params': ['B0'], 'P_bio_G': 0.035, 'modes': {'KO': 0.8, 'MC': 0.2}},
 'G8': {'name': 'SWI5',
        'params': ['ksswip', 'ksswipp'],
        'P_bio_G': 0.03,
        'modes': {'KO': 0.8, 'GAL-OE': 0.2}},
 'G9': {'name': 'CDC20',
        'params': ['ks20p', 'ks20pp', 'ka20pp'],
        'P_bio_G': 0.06,
        'modes': {'KO': 0.105,
                  'GAL-OE': 0.16,
                  'MC': 0.055,
                  'ts': 0.055,
                  'APC-A': 0.57,
                  'DAmP': 0.055}},
 'G10': {'name': 'CDH1',
         'params': ['kscdh', 'kicdhpp', 'init_CDH1T', 'init_CDH1'],
         'P_bio_G': 0.06,
         'modes': {'KO': 0.962, 'CA': 0.038}},
 'G11': {'name': 'CDC14',
         'params': ['ks14', 'kppc1', 'kppf6', 'kaswi', 'kacdhpp'],
         'P_bio_G': 0.05,
         'modes': {'ts': 0.571, 'GAL-OE': 0.143, 'MC': 0.143, 'DAmP': 0.143}},
 'G12': {'name': 'NET1',
         'params': ['ksnet', 'kasrent', 'kasrentp'],
         'P_bio_G': 0.035,
         'modes': {'TAB6-1': 0.375, 'GAL-OE': 0.125, 'ts': 0.375, 'DAmP': 0.125}},
 'G13': {'name': 'TEM1',
         'params': ['Tem1T', 'ka15pp', 'init_Tem1'],
         'P_bio_G': 0.035,
         'modes': {'KO': 0.25, 'GAL-OE': 0.25, 'MC': 0.25, 'ts': 0.25}},
 'G14': {'name': 'CDC15',
         'params': ['Cdc15T', 'kkpnetpp', 'init_Cdc15'],
         'P_bio_G': 0.035,
         'modes': {'KO': 0.25, 'GAL-OE': 0.25, 'MC': 0.25, 'ts': 0.25}},
 'G15': {'name': 'PPX',
         'params': ['ksppx', 'kdppxpp', 'kppnetpp'],
         'P_bio_G': 0.035,
         'modes': {'KO': 0.25, 'GAL-OE': 0.25, 'ts': 0.25, 'db': 0.25}},
 'G16': {'name': 'PDS1',
         'params': ['kspdsp', 'ks1pdspp', 'ks2pdspp', 'kd2pdspp', 'kd3pdspp'],
         'P_bio_G': 0.04,
         'modes': {'KO': 0.43, 'GAL-OE': 0.27, 'db': 0.3}},
 'G17': {'name': 'ESP1',
         'params': ['ESP1T', 'kasesp', 'kdiesp'],
         'P_bio_G': 0.035,
         'modes': {'ts': 0.75, 'GAL-OE': 0.25}},
 'G18': {'name': 'MAD2', 'params': ['Vi20_active'], 'P_bio_G': 0.032, 'modes': {'CSL': 1.0}},
 'G19': {'name': 'BUB2',
         'params': ['BUB2_active', 'init_BUB2'],
         'P_bio_G': 0.032,
         'modes': {'CKO': 1.0}},
 'G20': {'name': 'SBF_MBF',
         'params': ['kasbf', 'ksn2pp', 'ksb5pp'],
         'P_bio_G': 0.035,
         'modes': {'KO': 1.0}},
 'G21': {'name': 'Mcm1',
         'params': ['kamcm'],
         'P_bio_G': 0.035,
         'modes': {'ts': 0.5, 'KO': 0.5}},
 'G22': {'name': 'APC_Core',
         'params': ['ka20pp', 'ks20p', 'ks20pp', 'kscdh', 'init_CDH1T', 'init_CDH1'],
         'P_bio_G': 0.035,
         'modes': {'APC-A': 0.9, 'ts': 0.1}},
 'G24': {'name': 'MDT', 'params': ['mdt'], 'P_bio_G': 0.032, 'modes': {'MDT': 1.0}}}

# Temporarily disabled by user decision: none of the canonical 126 genotypes
# targets these modules. Keep the definitions above for a future explicit opt-in;
# every sampler and mutation gate below sees only the active mapping.
DISABLED_BINDING_GROUPS = {k: g for k, g in BINDING_GROUPS.items()
                           if g['name'] in DISABLED_FUNCTIONAL_MODULES}
BINDING_GROUPS = {k: g for k, g in BINDING_GROUPS.items()
                  if g['name'] not in DISABLED_FUNCTIONAL_MODULES}

_total_g_weight = sum(g['P_bio_G'] for g in BINDING_GROUPS.values())

for g in BINDING_GROUPS.values():
    g['P_bio_G'] /= _total_g_weight
GROUP_BY_NAME = {g['name']: g for g in BINDING_GROUPS.values()}

GENE_NAMES = sorted(GROUP_BY_NAME.keys())

MODE_FREQ = 'class'

GENE_ROLE = {'CLN2': 'dose',
 'CLN3': 'dose',
 'BCK2': 'dose',
 'CLB2': 'dose',
 'CLB5': 'dose',
 'SIC1': 'dose',
 'PDS1': 'dose',
 'CDC6': 'dose',
 'SWI5': 'dose',
 'CDC20': 'dose',
 'CDH1': 'dose',
 'CDC14': 'enzyme',
 'CDC15': 'enzyme',
 'TEM1': 'enzyme',
 'PPX': 'enzyme',
 'SBF_MBF': 'tf',
 'Mcm1': 'tf',
 'ESP1': 'enzyme',
 'NET1': 'enzyme',
 'APC_Core': 'enzyme',
 'MAD2': 'event',
 'BUB2': 'event',
 'MDT': 'event'}
GENE_ROLE = {g: role for g, role in GENE_ROLE.items() if g in GROUP_BY_NAME}

MODE_CLASS = {'KO': 'loss',
 'GAL-OE': 'expression',
 'GAL-MC': 'expression',
 'MC': 'expression',
 'ts': 'activity',
 'DAmP': 'activity',
 'pLOF': 'activity',
 'APC-A': 'activity',
 'db': 'stability',
 'PM': 'stability',
 'CA': 'structure',
 'CSL': 'event',
 'CKO': 'event',
 'TAB6-1': 'event',
 'MDT': 'event'}

MODE_CLASS_QUOTA = {
    'loss': 0.28, 'expression': 0.28, 'activity': 0.18,
    'stability': 0.20, 'structure': 0.03, 'event': 0.03,
}

ROLE_AFFINITY = {
    'dose':   {'expression': 2.0, 'loss': 2.0, 'stability': 2.0,
               'activity': 1.0, 'structure': 1.0, 'event': 1.0},
    'enzyme': {'activity': 2.0, 'loss': 2.0,
               'expression': 1.0, 'stability': 1.0, 'structure': 1.0,
               'event': 1.0},
    'tf':     {'loss': 2.0, 'structure': 2.0,
               'expression': 1.0, 'activity': 1.0, 'stability': 1.0,
               'event': 1.0},
}


TOUCH = {('CLN2', 'KO'): {'ksn2pp'},
 ('CLN2', 'GAL-OE'): {'ksn2pp', 'ksn2p'},
 ('CLN2', 'MC'): {'ksn2pp'},
 ('CLN2', 'ts'): {'ksn2pp', 'ksn2p'},
 ('CLN3', 'KO'): {'C0'},
 ('CLN3', 'GAL-OE'): {'C0'},
 ('CLN3', 'MC'): {'C0'},
 ('CLB2', 'KO'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'GAL-OE'): {'ksb2p'},
 ('CLB2', 'GAL-MC'): {'ksb2p'},
 ('CLB2', 'MC'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'ts'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'db'): {'kdb2pp', 'kdb2P'},
 ('CLB2', 'pLOF'): {'ksb2p', 'ksb2pp'},
 ('CLB5', 'KO'): {'ksb5p', 'ksb5pp'},
 ('CLB5', 'GAL-OE'): {'ksb5p'},
 ('CLB5', 'MC'): {'ksb5p', 'ksb5pp'},
 ('CLB5', 'db'): {'kdb5pp'},
 ('SIC1', 'KO'): {'ksc1pp', 'ksc1p'},
 ('SIC1', 'GAL-OE'): {'ksc1p'},
 ('SIC1', 'MC'): {'ksc1pp', 'ksc1p'},
 ('SIC1', 'db'): {'kd3c1'},
 ('SIC1', 'PM'): {'kd2c1', 'kd3c1'},
 ('CDC6', 'KO'): {'ksf6p', 'ksf6pp', 'ksf6ppp'},
 ('CDC6', 'GAL-OE'): {'ksf6p'},
 ('CDC6', 'MC'): {'ksf6p', 'ksf6pp', 'ksf6ppp'},
 ('CDC6', 'DAmP'): {'ksf6p', 'ksf6pp', 'ksf6ppp'},
 ('BCK2', 'KO'): {'B0'},
 ('BCK2', 'MC'): {'B0'},
 ('SWI5', 'KO'): {'ksswip', 'ksswipp'},
 ('SWI5', 'GAL-OE'): {'ksswip'},
 ('CDC20', 'KO'): {'ks20p', 'ks20pp'},
 ('CDC20', 'GAL-OE'): {'ks20p'},
 ('CDC20', 'MC'): {'ks20p', 'ks20pp'},
 ('CDC20', 'ts'): {'ks20p', 'ks20pp'},
 ('CDC20', 'APC-A'): {'ka20pp'},
 ('CDC20', 'DAmP'): {'ks20p', 'ks20pp'},
 ('CDH1', 'KO'): {'init_CDH1T', 'init_CDH1', 'kscdh'},
 ('CDH1', 'CA'): {'kscdh', 'kicdhpp'},
 ('CDC14', 'ts'): {'kppf6', 'kacdhpp', 'ks14', 'kppc1', 'kaswi'},
 ('CDC14', 'GAL-OE'): {'ks14'},
 ('CDC14', 'MC'): {'ks14'},
 ('CDC14', 'DAmP'): {'ks14'},
 ('NET1', 'TAB6-1'): {'kasrent', 'kasrentp'},
 ('NET1', 'GAL-OE'): {'ksnet'},
 ('NET1', 'ts'): {'kasrent', 'kasrentp'},
 ('NET1', 'DAmP'): {'ksnet'},
 ('TEM1', 'KO'): {'ka15pp', 'init_Tem1', 'Tem1T'},
 ('TEM1', 'GAL-OE'): {'Tem1T'},
 ('TEM1', 'MC'): {'Tem1T'},
 ('TEM1', 'ts'): {'ka15pp', 'init_Tem1', 'Tem1T'},
 ('CDC15', 'KO'): {'init_Cdc15', 'Cdc15T', 'kkpnetpp'},
 ('CDC15', 'GAL-OE'): {'Cdc15T'},
 ('CDC15', 'MC'): {'Cdc15T'},
 ('CDC15', 'ts'): {'kkpnetpp'},
 ('PPX', 'KO'): {'ksppx'},
 ('PPX', 'GAL-OE'): {'ksppx'},
 ('PPX', 'ts'): {'kppnetpp'},
 ('PPX', 'db'): {'kdppxpp'},
 ('SBF_MBF', 'KO'): {'kasbf', 'ksn2pp', 'ksb5pp'},
 ('Mcm1', 'KO'): {'kamcm'},
 ('Mcm1', 'ts'): {'kamcm'},
 ('PDS1', 'KO'): {'ks2pdspp', 'ks1pdspp'},
 ('PDS1', 'GAL-OE'): {'kspdsp'},
 ('PDS1', 'db'): {'kd2pdspp', 'kd3pdspp'},
 ('ESP1', 'ts'): {'kdiesp', 'kasesp'},
 ('ESP1', 'GAL-OE'): {'ESP1T'},
 ('MAD2', 'CSL'): {'Vi20_active', 'Vi20'},
 ('BUB2', 'CKO'): {'BUB2_active', 'kbub2', 'init_BUB2'},
 ('APC_Core', 'APC-A'): {'ka20pp'},
 ('APC_Core', 'ts'): {'init_CDH1T', 'init_CDH1', 'kscdh', 'ks20pp', 'ks20p'},
 ('MDT', 'MDT'): {'mdt'}}

TOUCH = {pair: keys for pair, keys in TOUCH.items() if pair[0] in GROUP_BY_NAME}

def touch_set(gene, mode):
    return TOUCH.get((gene, mode), set())

# Same-gene compounds explicitly present in canonical real genotypes. #50 is
# represented separately by the CLB2 GAL-MC mode (GAL base times integer copies).
# Cross-gene combinations remain exploratory; disjoint parameter footprints
# alone do not authorize a new same-gene compound.
COMPOSITE_REAL_EVIDENCE = {
    ('CLB2', ('GAL-OE', 'db')): ('62',),
    ('CLB5', ('GAL-OE', 'db')): ('72',),
    ('SIC1', ('GAL-OE', 'db')): ('29',),
    ('PDS1', ('GAL-OE', 'db')): ('83', '84'),
}
COMPOSITE_ELIGIBLE = sorted({g for g, _ in COMPOSITE_REAL_EVIDENCE})


def composite_partners(gene, mode):
    return sorted({other for (g, modes) in COMPOSITE_REAL_EVIDENCE
                   if g == gene and mode in modes for other in modes if other != mode})


def expand_composites(pairs, rng):
    """Legacy 25% expansion branch, limited to the explicit real-supported list."""
    expanded = []
    for gene, mode in pairs:
        expanded.append((gene, mode))
        if gene in COMPOSITE_ELIGIBLE and rng.random() < .25:
            partners = composite_partners(gene, mode)
            if partners:
                expanded.append((gene, str(rng.choice(partners))))
    return expanded


def mutation_pairs_compatible(first, second):
    gene, mode = first
    other_gene, other_mode = second
    if gene == other_gene:
        if (gene, tuple(sorted((mode, other_mode)))) not in COMPOSITE_REAL_EVIDENCE:
            return False
    return not (touch_set(gene, mode) & touch_set(other_gene, other_mode))


def first_mutation_conflict(pairs):
    for index, pair in enumerate(pairs):
        for previous, other in enumerate(pairs[:index]):
            if not mutation_pairs_compatible(pair, other):
                return index, previous
    return None


def require_mutation(gene, mode):
    """Reject unsupported pairs before drawing randomness or editing inputs.

    This gate is shared by both profiles and candidate-pool auditing. Biological
    state names and the separate real-mutant table are not a sampling catalog.
    """
    if gene not in GROUP_BY_NAME or mode not in GROUP_BY_NAME[gene]['modes']:
        raise ValueError(f'Unsupported Budding synthetic mutation: {gene}[{mode}]')
