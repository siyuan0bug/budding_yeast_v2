"""Canonical MA real-mutant rules; the old MASS table is not imported."""
import copy
import numpy as np

REAL_MUTANTS = {
    '1': ('Wild-type (WT) on glucose (Glc)', {'mdt': 90.0}),
    '2.1': ('WT on galactose (Gal)', {'mdt': 150.0}),
    '2.2': ('WT on raffinose (Raff)', {'mdt': 160.0}),
    '3': ('cln1Δ cln2Δ', {'ksn2pp': 0.0}),
    '4': ('GAL-CLN2 cln1Δ cln2Δ', {'ksn2p': 0.12, 'ksn2pp': 0.0, 'mdt': 150.0}),
    '5': ('cln1Δ cln2Δ sic1Δ', {'ksn2pp': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0}),
    '6': ('cln1Δ cln2Δ cdh1Δ', {'ksn2pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '7': ('GAL-CLN2 cln1Δ cln2Δ cdh1Δ', {'ksn2p': 0.12, 'ksn2pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0}),
    '8': ('cln3Δ', {'C0': 0.0}),
    '9': ('GAL-CLN3', {'C0': 2.0, 'mdt': 150.0}),
    '10': ('bck2Δ', {'B0': 0.0}),
    '11': ('multi-copy BCK2', {'B0': 0.27}),
    '12': ('cln1Δ cln2Δ bck2Δ', {'ksn2pp': 0.0, 'B0': 0.0}),
    '13': ('cln3Δ bck2Δ', {'C0': 0.0, 'B0': 0.0}),
    '14': ('cln3Δ bck2Δ GAL-CLN2 cln1Δ cln2Δ', {'ksn2p': 0.12, 'ksn2pp': 0.0, 'C0': 0.0, 'B0': 0.0, 'mdt': 150.0}),
    '15': ('cln3Δ bck2Δ multi-copy CLN2', {'C0': 0.0, 'B0': 0.0, 'ksn2pp': 0.75}),
    '16': ('cln3Δ bck2Δ sic1Δ', {'C0': 0.0, 'B0': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0}),
    '17': ('cln1Δ cln2Δ cln3Δ', {'ksn2pp': 0.0, 'C0': 0.0}),
    '18': ('cln1Δ cln2Δ cln3Δ GAL-CLN2', {'ksn2p': 0.12, 'ksn2pp': 0.0, 'C0': 0.0, 'mdt': 150.0}),
    '19': ('cln1Δ cln2Δ cln3Δ GAL-CLN3', {'ksn2pp': 0.0, 'C0': 2.0, 'mdt': 150.0}),
    '20': ('cln1Δ cln2Δ cln3Δ sic1Δ', {'ksn2pp': 0.0, 'C0': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0}),
    '21': ('cln1Δ cln2Δ cln3Δ cdh1Δ', {'ksn2pp': 0.0, 'C0': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '22': ('cln1Δ cln2Δ cln3Δ multi-copy CLB5', {'ksn2pp': 0.0, 'C0': 0.0, 'ksb5p': 0.0032, 'ksb5pp': 0.020}),
    '23': ('cln1Δ cln2Δ cln3Δ GAL-CLB5', {'ksn2pp': 0.0, 'C0': 0.0, 'ksb5p': 0.012, 'mdt': 150.0}),
    '24': ('cln1Δ cln2Δ cln3Δ multi-copy BCK2', {'ksn2pp': 0.0, 'C0': 0.0, 'B0': 0.54}),
    '25': ('cln1Δ cln2Δ cln3Δ GAL-CLB2', {'ksn2pp': 0.0, 'C0': 0.0, 'ksb2p': 0.002, 'mdt': 150.0}),
    '26': ('cln1Δ cln2Δ cln3Δ apc-ts', {'ksn2pp': 0.0, 'C0': 0.0, 'ks20p': 0.0, 'ks20pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '27': ('sic1Δ', {'ksc1p': 0.0, 'ksc1pp': 0.0}),
    '28': ('GAL-SIC1', {'ksc1p': 0.12, 'mdt': 150.0}),
    '29': ('GAL-SIC1dbΔ', {'ksc1p': 0.12, 'kd3c1': 0.0, 'mdt': 150.0}),
    '30': ('GAL-SIC1 cln1Δ cln2Δ', {'ksc1p': 0.12, 'ksn2pp': 0.0, 'mdt': 150.0}),
    '31': ('GAL-SIC1 GAL-CLN2 cln1Δ cln2Δ', {'ksc1p': 0.12, 'ksn2p': 0.12, 'ksn2pp': 0.0, 'mdt': 150.0}),
    '32': ('GAL-SIC1 cln1Δ cln2Δ cdh1Δ', {'ksc1p': 0.12, 'ksn2pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0}),
    '33': ('GAL-SIC1 GAL-CLN2 cln1Δ cln2Δ cdh1Δ', {'ksc1p': 0.12, 'ksn2p': 0.12, 'ksn2pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0}),
    '34': ('cdh1Δ', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '35': ('Cdh1 constitutively active', {'kicdhpp': 0.0, 'kscdh': 0.03, 'mdt': 150.0}),
    '36': ('sic1Δ cdh1Δ', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '37': ('sic1Δ cdh1Δ GAL-CDC20', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ks20p': 10.0, 'mdt': 150.0}),
    '38': ('cdc6Δ 2-49', {'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0}),
    '39': ('sic1Δ cdc6Δ 2-49', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0}),
    '40': ('cdh1Δ cdc6Δ 2-49', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0}),
    '41': ('sic1Δ cdc6Δ 2-49 cdh1Δ', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0}),
    '42': ('sic1Δ cdc6Δ 2-49 cdh1Δ GAL-CDC20', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0, 'ks20p': 4.0, 'mdt': 150.0}),
    '43': ('swi5Δ', {'ksswip': 0.0, 'ksswipp': 0.0}),
    '44': ('swi5Δ GAL-CLB2', {'ksswip': 0.0, 'ksswipp': 0.0, 'ksb2p': 0.12, 'mdt': 150.0}),
    '45': ('swi5Δ cdh1Δ', {'ksswip': 0.0, 'ksswipp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '46': ('swi5Δ cdh1Δ GAL-SIC1', {'ksswip': 0.0, 'ksswipp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ksc1p': 0.12, 'mdt': 150.0}),
    '47': ('clb1Δ clb2Δ', {'ksb2p': 0.0, 'ksb2pp': 0.0}),
    '48': ('CLB1 clb2Δ', {'ksb2p': 0.00033, 'ksb2pp': 0.0132}),
    '49': ('GAL-CLB2', {'ksb2p': 0.12, 'mdt': 150.0}),
    '50': ('multi-copy GAL-CLB2', {'ksb2p': 0.96, 'mdt': 150.0}),
    '51': ('CLB1 clb2Δ cdh1Δ', {'ksb2p': 0.00033, 'ksb2pp': 0.0132, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0}),
    '52': ('CLB1 clb2Δ pds1Δ', {'ksb2p': 0.00033, 'ksb2pp': 0.0132, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0}),
    '53': ('GAL-CLB2 sic1Δ', {'ksb2p': 0.12, 'ksc1p': 0.0, 'ksc1pp': 0.0, 'mdt': 150.0}),
    '54': ('GAL-CLB2 cdh1Δ', {'ksb2p': 0.12, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0}),
    '55': ('CLB2-dbΔ', {'kdb2P': 0.0, 'kdb2pp': 0.034}),
    '56': ('CLB2-dbΔ on galactose', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'mdt': 150.0}),
    '57': ('CLB2-dbΔ multi-copy SIC1', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'ksc1p': 0.12, 'ksc1pp': 1.20}),
    '58': ('CLB2-dbΔ GAL-SIC1', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'ksc1p': 0.12, 'mdt': 150.0}),
    '59': ('CLB2-dbΔ multi-copy CDC6', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'ksf6p': 0.12, 'ksf6pp': 0.60, 'ksf6ppp': 0.02}),
    '60': ('CLB2-dbΔ clb5Δ', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'ksb5p': 0.0, 'ksb5pp': 0.0}),
    '61.1': ('CLB2-dbΔ clb5Δ on Gal', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'ksb5p': 0.0, 'ksb5pp': 0.0, 'mdt': 150.0}),
    '61.2': ('CLB2-dbΔ clb5Δ on Raff', {'kdb2P': 0.0, 'kdb2pp': 0.034, 'ksb5p': 0.0, 'ksb5pp': 0.0, 'mdt': 160.0}),
    '62': ('GAL-CLB2-dbΔ', {'ksb2p': 0.12, 'kdb2P': 0.0, 'kdb2pp': 0.034, 'mdt': 150.0}),
    '63': ('clb5Δ clb6Δ', {'ksb5p': 0.0, 'ksb5pp': 0.0}),
    '64': ('cln1Δ cln2Δ clb5Δ clb6Δ', {'ksn2pp': 0.0, 'ksb5p': 0.0, 'ksb5pp': 0.0}),
    '65': ('GAL-CLB5', {'ksb5p': 0.012, 'mdt': 150.0}),
    '66': ('GAL-CLB5 sic1Δ', {'ksb5p': 0.012, 'ksc1p': 0.0, 'ksc1pp': 0.0, 'mdt': 150.0}),
    '67': ('GAL-CLB5 cdh1Δ', {'ksb5p': 0.008, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0}),
    '68': ('CLB5-dbΔ', {'kdb5pp': 0.0}),
    '69': ('CLB5-dbΔ sic1Δ', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kdb5pp': 0.0}),
    '70': ('CLB5-dbΔ pds1Δ', {'kdb5pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0}),
    '71': ('CLB5-dbΔ pds1Δ cdc20Δ', {'kdb5pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0, 'ks20p': 0.0, 'ks20pp': 0.0}),
    '72': ('GAL-CLB5-dbΔ', {'ksb5p': 0.012, 'kdb5pp': 0.0, 'mdt': 150.0}),
    '73': ('cdc20-ts', {'ks20p': 0.0, 'ks20pp': 0.0}),
    '74': ('cdc20Δ clb5Δ', {'ks20p': 0.0, 'ks20pp': 0.0, 'ksb5p': 0.0, 'ksb5pp': 0.0}),
    '75': ('cdc20Δ pds1Δ', {'ks20p': 0.0, 'ks20pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0}),
    '76': ('cdc20Δ pds1Δ clb5Δ', {'ks20p': 0.0, 'ks20pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0, 'ksb5p': 0.0, 'ksb5pp': 0.0}),
    '77': ('GAL-CDC20', {'ks20p': 6.0, 'mdt': 150.0}),
    '78': ('cdc20-ts mad2Δ', {'ks20p': 0.0, 'ks20pp': 0.0, 'Vi20_active': 0.01}),
    '79': ('cdc20-ts bub2Δ', {'ks20p': 0.0, 'ks20pp': 0.0, 'kbub2': 0.0, 'BUB2_active': 0.0, 'init_BUB2': 0.0}),
    '80': ('pds1Δ', {'ks1pdspp': 0.0, 'ks2pdspp': 0.0}),
    '81': ('esp1-ts', {'kasesp': 0.1, 'kdiesp': 0.001}),
    '82': ('PDS1-dbΔ', {'kd2pdspp': 0.0, 'kd3pdspp': 0.0}),
    '83': ('GAL-PDS1-dbΔ', {'kspdsp': 0.10, 'kd2pdspp': 0.0, 'kd3pdspp': 0.0, 'mdt': 150.0}),
    '84': ('GAL-PDS1-dbΔ esp1-ts', {'kspdsp': 0.10, 'kd2pdspp': 0.0, 'kd3pdspp': 0.0, 'kasesp': 0.1, 'kdiesp': 0.001, 'mdt': 150.0}),
    '85': ('GAL-ESP1 cdc20-ts', {'ks20p': 0.0, 'ks20pp': 0.0, 'ESP1T': 3.0, 'mdt': 150.0}),
    '86': ('tem1Δ', {'Tem1T': 0.0, 'ka15pp': 0.0, 'init_Tem1': 0.0}),
    '87': ('GAL-TEM1', {'Tem1T': 5.0, 'mdt': 150.0}),
    '88': ('tem1-ts multi-copy CDC15', {'ka15pp': 0.002, 'Tem1T': 0.0, 'init_Tem1': 0.0, 'Cdc15T': 5.0}),
    '89': ('tem1-ts GAL-CDC15', {'ka15pp': 0.002, 'Tem1T': 0.0, 'init_Tem1': 0.0, 'Cdc15T': 15.0, 'mdt': 150.0}),
    '90': ('tem1-ts net1-ts', {'ka15pp': 0.002, 'Tem1T': 0.0, 'init_Tem1': 0.0, 'kasrent': 10.0, 'kasrentp': 0.05}),
    '91': ('tem1-ts multi-copy CDC14', {'ka15pp': 0.002, 'Tem1T': 0.0, 'init_Tem1': 0.0, 'ks14': 0.40}),
    '92': ('cdc15Δ', {'Cdc15T': 0.0, 'kkpnetpp': 0.0, 'init_Cdc15': 0.0}),
    '93': ('multi-copy CDC15', {'Cdc15T': 5.0}),
    '94': ('cdc15Δ multi-copy TEM1', {'Cdc15T': 0.0, 'kkpnetpp': 0.0, 'init_Cdc15': 0.0, 'Tem1T': 5.0}),
    '95': ('cdc15-ts net1-ts', {'kkpnetpp': 0.0, 'kasrent': 10.0, 'kasrentp': 0.05}),
    '96': ('cdc15-ts multi-copy CDC14', {'kkpnetpp': 0.0, 'ks14': 0.40}),
    '97': ('net1-ts', {'kasrent': 10.0, 'kasrentp': 0.05}),
    '98': ('GAL-NET1', {'ksnet': 0.42, 'mdt': 150.0}),
    '99': ('cdc14-ts', {'ks14': 0.0}),
    '100': ('GAL-CDC14', {'ks14': 0.80, 'mdt': 150.0}),
    '101': ('GAL-NET1 GAL-CDC14', {'ksnet': 0.252, 'ks14': 0.60, 'mdt': 150.0}),
    '102': ('net1Δ cdc20-ts', {'kasrent': 10.0, 'kasrentp': 0.05, 'ks20p': 0.0, 'ks20pp': 0.0}),
    '103': ('cdc14-ts GAL-SIC1', {'ks14': 0.0, 'ksc1p': 0.12, 'mdt': 150.0}),
    '104': ('cdc14-ts then GAL-SIC1', {'ks14': 0.0, 'mdt': 150.0, 'is_mutant_104': True}),
    '105': ('cdc14-ts sic1Δ', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kppc1': 0.64, 'kppf6': 0.64, 'kaswi': 0.32, 'kacdhpp': 0.128}),
    '106': ('cdc14-ts cdh1Δ', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'kppc1': 1.2, 'kppf6': 1.2, 'kaswi': 0.6, 'kacdhpp': 0.24}),
    '107': ('cdc14-ts GAL-CLN2', {'ksn2p': 0.12, 'ksn2pp': 0.0, 'kppc1': 0.4, 'kppf6': 0.4, 'kaswi': 0.2, 'kacdhpp': 0.08, 'mdt': 150.0}),
    '108': ('TAB6-1', {'kasrent': 10.0, 'kasrentp': 0.05}),
    '109': ('TAB6-1 cdc15-ts', {'kkpnetpp': 0.0, 'kasrent': 10.0, 'kasrentp': 0.05}),
    '110': ('TAB6-1 clb5Δ clb6Δ', {'ksb5p': 0.0, 'ksb5pp': 0.0, 'kasrent': 10.0, 'kasrentp': 0.05}),
    '111': ('TAB6-1 CLB1 clb2Δ', {'ksb2p': 0.00033, 'ksb2pp': 0.0132, 'kasrent': 10.0, 'kasrentp': 0.05}),
    '112': ('mad2Δ', {'Vi20_active': 0.01}),
    '113': ('bub2Δ', {'kbub2': 0.0, 'BUB2_active': 0.0, 'init_BUB2': 0.0}),
    '114': ('mad2Δ bub2Δ', {'Vi20_active': 0.01, 'kbub2': 0.0, 'BUB2_active': 0.0, 'init_BUB2': 0.0}),
    '115': ('APC-A', {'ka20pp': 0.0}),
    '116': ('APC-A cdh1Δ', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ka20pp': 0.0}),
    '117': ('APC-A cdh1Δ on galactose', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ka20pp': 0.0, 'mdt': 150.0}),
    '118': ('APC-A cdh1Δ multi-copy SIC1', {'ksc1p': 0.18, 'ksc1pp': 1.80, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ka20pp': 0.0}),
    '119': ('APC-A cdh1Δ GAL-SIC1', {'ksc1p': 0.24, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ka20pp': 0.0, 'mdt': 150.0}),
    '120': ('APC-A cdh1Δ multi-copy CDC6', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ka20pp': 0.0, 'ksf6p': 0.24, 'ksf6pp': 1.20, 'ksf6ppp': 0.04}),
    '121': ('APC-A cdh1Δ GAL-CDC6', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ksf6p': 0.36, 'ka20pp': 0.0, 'mdt': 150.0}),
    '122': ('APC-A cdh1Δ multi-copy CDC20', {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'ks20p': 0.15, 'ks20pp': 15.0, 'ka20pp': 0.0}),
    '123': ('APC-A sic1Δ', {'ksc1p': 0.0, 'ksc1pp': 0.0, 'ka20pp': 0.0}),
    '124': ('APC-A GAL-CLB2', {'ksb2p': 0.48, 'ka20pp': 0.0, 'mdt': 150.0})
}

mutant_rules = {f"{mid}_{mname}": movr for mid, (mname, movr) in REAL_MUTANTS.items()}

_CACHED_REAL_PARAMS = None

def get_cached_real_params(p_wt_default):
    global _CACHED_REAL_PARAMS
    if _CACHED_REAL_PARAMS is None:
        _CACHED_REAL_PARAMS = []
        for mid, (mname, movr) in REAL_MUTANTS.items():
            p_real = p_wt_default.copy()
            y0_real = {}
            for k, v in movr.items():
                if k.startswith('init_'):
                    y0_real[k] = v
                else:
                    p_real[k] = v
            _CACHED_REAL_PARAMS.append((p_real, y0_real))
    return _CACHED_REAL_PARAMS

def is_duplicate_real_mutant(ovr_candidate, p_wt_default):
    """
    136-dim Parameter Deduplication Check:
    Verify if ovr_candidate is 100% identical to any of the 126 Real Mutants in D_real.
    """
    p_cand = p_wt_default.copy()
    y0_cand_keys = {}
    for k, v in ovr_candidate.items():
        if k.startswith('init_'):
            y0_cand_keys[k] = v
        else:
            p_cand[k] = v

    cached_reals = get_cached_real_params(p_wt_default)
    for p_real, y0_real in cached_reals:
        if y0_cand_keys != y0_real:
            continue
        match = True
        for k, v in p_cand.items():
            if k not in p_real or abs(v - p_real[k]) > 1e-8 + 1e-5 * abs(p_real[k]):
                match = False
                break
        if match:
            return True

    return False

def real_tasks():
    return [(f"{mid}_{name}".replace(' ', '_'), copy.deepcopy(overrides))
            for mid, (name, overrides) in REAL_MUTANTS.items()]
