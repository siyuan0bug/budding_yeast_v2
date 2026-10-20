"""Legal gene/mode catalog and conflict footprints from MA."""
import numpy as np

BINDING_GROUPS = {
    'G1': {
        'name': 'CLN2', 'params': ['ksn2p', 'ksn2pp'],
        'P_bio_G': 0.05,
        'modes': {'KO': 0.667, 'GAL-OE': 0.211, 'MC': 0.030, 'ts': 0.092}
    },
    'G2': {
        'name': 'CLN3', 'params': ['C0'],
        'P_bio_G': 0.03,
        'modes': {'KO': 0.853, 'GAL-OE': 0.113, 'MC': 0.034}
    },
    'G3': {
        'name': 'CLB2', 'params': ['ksb2p', 'ksb2pp', 'kdb2p', 'kdb2pp', 'kdb2P', 'kdib2'],
        'P_bio_G': 0.06,
        'modes': {'KO': 0.090, 'GAL-OE': 0.310, 'MC': 0.075, 'ts': 0.035, 'db': 0.310, 'pLOF': 0.150, 'CA': 0.030}
    },
    'G4': {
        'name': 'CLB5', 'params': ['ksb5p', 'ksb5pp', 'kdb5pp'],
        'P_bio_G': 0.04,
        'modes': {'KO': 0.370, 'GAL-OE': 0.280, 'MC': 0.070, 'db': 0.280}
    },
    'G5': {
        'name': 'SIC1', 'params': ['ksc1p', 'ksc1pp', 'kd3c1', 'kd2c1'],
        'P_bio_G': 0.05,
        'modes': {'KO': 0.440, 'GAL-OE': 0.350, 'MC': 0.070, 'db': 0.040, 'PM': 0.100}
    },
    'G6': {
        'name': 'CDC6', 'params': ['ksf6p', 'ksf6pp', 'ksf6ppp'],
        'P_bio_G': 0.03,
        'modes': {'KO': 0.625, 'GAL-OE': 0.125, 'MC': 0.125, 'DAmP': 0.125}
    },
    'G7': {
        'name': 'BCK2', 'params': ['B0'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.800, 'MC': 0.200}
    },
    'G8': {
        'name': 'SWI5', 'params': ['ksswip', 'ksswipp'],
        'P_bio_G': 0.03,
        'modes': {'KO': 0.800, 'GAL-OE': 0.200}
    },
    'G9': {
        'name': 'CDC20', 'params': ['ks20p', 'ks20pp', 'ka20pp'],
        'P_bio_G': 0.06,
        'modes': {'KO': 0.105, 'GAL-OE': 0.160, 'MC': 0.055, 'ts': 0.055, 'APC-A': 0.570, 'DAmP': 0.055}
    },
    'G10': {
        'name': 'CDH1', 'params': ['kscdh', 'kicdhpp', 'init_CDH1T', 'init_CDH1'],
        'P_bio_G': 0.06,
        'modes': {'KO': 0.962, 'CA': 0.038}
    },
    'G11': {
        'name': 'CDC14', 'params': ['ks14', 'kppc1', 'kppf6', 'kaswi', 'kacdhpp'],
        'P_bio_G': 0.05,
        'modes': {'ts': 0.571, 'GAL-OE': 0.143, 'MC': 0.143, 'DAmP': 0.143}
    },
    'G12': {
        'name': 'NET1', 'params': ['ksnet', 'kasrent', 'kasrentp'],
        'P_bio_G': 0.035,
        'modes': {'TAB6-1': 0.375, 'GAL-OE': 0.125, 'ts': 0.375, 'DAmP': 0.125}
    },
    'G13': {
        'name': 'TEM1', 'params': ['Tem1T', 'ka15pp', 'init_Tem1'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.200, 'GAL-OE': 0.200, 'MC': 0.200, 'ts': 0.200, 'DAmP': 0.200}
    },
    'G14': {
        'name': 'CDC15', 'params': ['Cdc15T', 'kkpnetpp', 'init_Cdc15'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.200, 'GAL-OE': 0.200, 'MC': 0.200, 'ts': 0.200, 'DAmP': 0.200}
    },
    'G15': {
        'name': 'PPX', 'params': ['ksppx', 'kdppxpp', 'kppnetpp'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.250, 'GAL-OE': 0.250, 'ts': 0.250, 'db': 0.250}
    },
    'G16': {
        'name': 'PDS1', 'params': ['kspdsp', 'ks1pdspp', 'ks2pdspp', 'kd2pdspp', 'kd3pdspp'],
        'P_bio_G': 0.04,
        'modes': {'KO': 0.430, 'GAL-OE': 0.270, 'db': 0.300}
    },
    'G17': {
        'name': 'ESP1', 'params': ['ESP1T', 'kasesp', 'kdiesp'],
        'P_bio_G': 0.035,
        'modes': {'ts': 0.750, 'GAL-OE': 0.250}
    },
    'G18': {
        'name': 'MAD2', 'params': ['Vi20_active'],
        'P_bio_G': 0.032,
        'modes': {'CSL': 1.0}
    },
    'G19': {
        'name': 'BUB2', 'params': ['BUB2_active', 'init_BUB2'],
        'P_bio_G': 0.032,
        'modes': {'CKO': 1.0}
    },
    'G20': {
        'name': 'SBF_MBF', 'params': ['kasbf', 'kisbfpp', 'ksn2pp', 'ksb5pp'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.50, 'CA': 0.50}
    },
    'G21': {
        'name': 'Mcm1', 'params': ['kamcm'],
        'P_bio_G': 0.035,
        'modes': {'ts': 0.50, 'KO': 0.50}
    },
    'G22': {
        'name': 'APC_Core', 'params': ['ka20pp', 'ks20p', 'ks20pp', 'kscdh', 'init_CDH1T', 'init_CDH1'],
        'P_bio_G': 0.035,
        'modes': {'APC-A': 0.90, 'ts': 0.10}
    },
    'G23': {
        'name': 'ORI_BUD_SPN', 'params': ['ksori', 'ksbud', 'ksspn'],
        'P_bio_G': 0.032,
        'modes': {'perturb': 0.40, 'cdc24Δ': 0.30, 'cdc45Δ': 0.15, 'spindle_defect': 0.15}
    },
    'G24': {
        'name': 'MDT', 'params': ['mdt'],
        'P_bio_G': 0.032,
        'modes': {'MDT': 1.0}
    },
    'G25': {
        'name': 'CDC5_Proxy', 'params': ['ksppx'],
        'P_bio_G': 0.02,
        'modes': {'KO': 0.50, 'db': 0.50}
    },
    'G26': {
        'name': 'CDC28', 'params': ['ec1k2', 'ec1n2', 'ec1b5', 'ec1b2', 'ec1n3', 'ef6k2', 'ef6n2', 'ef6b5', 'ef6b2', 'ef6n3', 'eicdhn2', 'eicdhn3', 'eicdhb5', 'eicdhb2', 'esbfn2', 'esbfn3', 'esbfb5', 'eorib5', 'eorib2', 'ebudn2', 'ebudn3', 'ebudb5'],
        'P_bio_G': 0.02,
        'modes': {'ts': 0.50, 'pLOF': 0.50}
    },
    'G27': {
        'name': 'SCC1', 'params': ['kasesp'],
        'P_bio_G': 0.02,
        'modes': {'scc1-RR': 1.0}
    }
}

_total_g_weight = sum(g['P_bio_G'] for g in BINDING_GROUPS.values())

for g in BINDING_GROUPS.values():
    g['P_bio_G'] /= _total_g_weight
GROUP_BY_NAME = {g['name']: g for g in BINDING_GROUPS.values()}

GENE_NAMES = sorted(GROUP_BY_NAME.keys())

MODE_FREQ = 'class'

GENE_ROLE = {
    'CLN2': 'dose', 'CLN3': 'dose', 'BCK2': 'dose', 'CLB2': 'dose',
    'CLB5': 'dose', 'SIC1': 'dose', 'PDS1': 'dose', 'CDC6': 'dose',
    'SWI5': 'dose', 'CDC20': 'dose', 'CDH1': 'dose',
    'CDC28': 'enzyme', 'CDC14': 'enzyme', 'CDC15': 'enzyme', 'TEM1': 'enzyme',
    'ESP1': 'enzyme', 'NET1': 'enzyme', 'PPX': 'enzyme', 'CDC5_Proxy': 'enzyme',
    'APC_Core': 'enzyme',
    'SBF_MBF': 'tf', 'Mcm1': 'tf',
    'MAD2': 'event', 'BUB2': 'event', 'SCC1': 'event',
    'ORI_BUD_SPN': 'event', 'MDT': 'event',
}

MODE_CLASS = {
    'KO': 'loss', 'GAL-OE': 'expression', 'MC': 'expression',
    'ts': 'activity', 'DAmP': 'activity', 'pLOF': 'activity',
    'APC-A': 'activity',   # v5.1 reclass (was 'event'): APC-A zeroes ka20pp =
                           # catalytic inactivation of APC core activation —
                           # a loss-of-activity allele (same mechanism family
                           # as ts), not a switch/event mode. 10 real mutants
                           # (#115-124) ride this token.
    'db': 'stability', 'PM': 'stability',
    'CA': 'structure', 'CSL': 'event', 'CKO': 'event',
    'scc1-RR': 'event', 'TAB6-1': 'event', 'cdc24Δ': 'event',
    'cdc45Δ': 'event', 'spindle_defect': 'event', 'MDT': 'event',
    'perturb': 'event',
}

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

_CD28_EPS = ['ec1k2','ec1n2','ec1b5','ec1b2','ec1n3','ef6k2','ef6n2','ef6b5',
             'ef6b2','ef6n3','eicdhn2','eicdhn3','eicdhb5','eicdhb2','esbfn2',
             'esbfn3','esbfb5','eorib5','eorib2','ebudn2','ebudn3','ebudb5']

TOUCH = {
 ('CLN2','KO'): {'ksn2pp'}, ('CLN2','GAL-OE'): {'ksn2p','ksn2pp'},
 ('CLN2','MC'): {'ksn2pp'}, ('CLN2','ts'): {'ksn2p','ksn2pp'},
 ('CLN3','KO'): {'C0'}, ('CLN3','GAL-OE'): {'C0'}, ('CLN3','MC'): {'C0'},
 ('CLB2','KO'): {'ksb2p','ksb2pp'}, ('CLB2','GAL-OE'): {'ksb2p'},
 ('CLB2','MC'): {'ksb2p','ksb2pp'}, ('CLB2','ts'): {'ksb2p','ksb2pp'},
 ('CLB2','db'): {'kdb2P','kdb2pp'}, ('CLB2','pLOF'): {'ksb2p','ksb2pp'},
 ('CLB2','CA'): {'kdib2'},
 ('CLB5','KO'): {'ksb5p','ksb5pp'}, ('CLB5','GAL-OE'): {'ksb5p'},
 ('CLB5','MC'): {'ksb5p','ksb5pp'}, ('CLB5','db'): {'kdb5pp'},
 ('SIC1','KO'): {'ksc1p','ksc1pp'}, ('SIC1','GAL-OE'): {'ksc1p'},
 ('SIC1','MC'): {'ksc1p','ksc1pp'}, ('SIC1','db'): {'kd3c1'},
 ('SIC1','PM'): {'kd2c1','kd3c1'},
 ('CDC6','KO'): {'ksf6p','ksf6pp','ksf6ppp'}, ('CDC6','GAL-OE'): {'ksf6p'},
 ('CDC6','MC'): {'ksf6p','ksf6pp','ksf6ppp'},
 ('CDC6','DAmP'): {'ksf6p','ksf6pp','ksf6ppp'},
 ('BCK2','KO'): {'B0'}, ('BCK2','MC'): {'B0'},
 ('SWI5','KO'): {'ksswip','ksswipp'}, ('SWI5','GAL-OE'): {'ksswip'},
 ('CDC20','KO'): {'ks20p','ks20pp'}, ('CDC20','GAL-OE'): {'ks20p'},
 ('CDC20','MC'): {'ks20p','ks20pp'}, ('CDC20','ts'): {'ks20p','ks20pp'},
 ('CDC20','APC-A'): {'ka20pp'}, ('CDC20','DAmP'): {'ks20p','ks20pp'},
 ('CDH1','KO'): {'kscdh','init_CDH1T','init_CDH1'},
 ('CDH1','CA'): {'kicdhpp','kscdh'},
 ('CDC14','ts'): {'ks14','kppc1','kppf6','kaswi','kacdhpp'},
 ('CDC14','GAL-OE'): {'ks14'}, ('CDC14','MC'): {'ks14'},
 ('CDC14','DAmP'): {'ks14'},
 ('NET1','TAB6-1'): {'kasrent','kasrentp'}, ('NET1','GAL-OE'): {'ksnet'},
 ('NET1','ts'): {'kasrent','kasrentp'}, ('NET1','DAmP'): {'ksnet'},
 ('TEM1','KO'): {'Tem1T','ka15pp','init_Tem1'}, ('TEM1','GAL-OE'): {'Tem1T'},
 ('TEM1','MC'): {'Tem1T'}, ('TEM1','ts'): {'Tem1T','ka15pp','init_Tem1'},
 ('TEM1','DAmP'): {'Tem1T'},
 ('CDC15','KO'): {'Cdc15T','kkpnetpp','init_Cdc15'},
 ('CDC15','GAL-OE'): {'Cdc15T'}, ('CDC15','MC'): {'Cdc15T'},
 ('CDC15','ts'): {'kkpnetpp'}, ('CDC15','DAmP'): {'Cdc15T'},
 ('PPX','KO'): {'ksppx'}, ('PPX','GAL-OE'): {'ksppx'},
 ('PPX','ts'): {'kppnetpp'}, ('PPX','db'): {'kdppxpp'},
 ('PDS1','KO'): {'ks1pdspp','ks2pdspp'}, ('PDS1','GAL-OE'): {'kspdsp'},
 ('PDS1','db'): {'kd2pdspp','kd3pdspp'},
 ('ESP1','ts'): {'kasesp','kdiesp'}, ('ESP1','GAL-OE'): {'ESP1T'},
 ('MAD2','CSL'): {'Vi20','Vi20_active'},
 ('BUB2','CKO'): {'kbub2','BUB2_active','init_BUB2'},
 ('SBF_MBF','KO'): {'kasbf','ksn2pp','ksb5pp'},
 ('SBF_MBF','CA'): {'is_sbf_ca','kasbf','kisbfpp'},
 ('Mcm1','ts'): {'kamcm'}, ('Mcm1','KO'): {'kamcm'},
 ('APC_Core','APC-A'): {'ka20pp'},
 ('APC_Core','ts'): {'ks20p','ks20pp','kscdh','init_CDH1T','init_CDH1'},
 ('ORI_BUD_SPN','perturb'): {'ksori','ksbud','ksspn'},
 ('ORI_BUD_SPN','cdc24Δ'): {'ksbud'},
 ('ORI_BUD_SPN','cdc45Δ'): {'ksori'},
 ('ORI_BUD_SPN','spindle_defect'): {'ksspn'},
 ('MDT','MDT'): {'mdt'},
 ('CDC5_Proxy','KO'): {'ksppx'}, ('CDC5_Proxy','db'): {'kdppxpp'},
 ('CDC28','ts'): set(_CD28_EPS), ('CDC28','pLOF'): set(_CD28_EPS),
 ('SCC1','scc1-RR'): {'kasesp'},
}

def touch_set(gene, mode):
    return TOUCH.get((gene, mode), set())

COMPOSITE_ELIGIBLE = [n for n in GENE_NAMES
                      if any(m in GROUP_BY_NAME[n]['modes'] for m in ('GAL-OE', 'MC'))
                      and any(m in GROUP_BY_NAME[n]['modes'] for m in ('db', 'PM', 'CA'))]
