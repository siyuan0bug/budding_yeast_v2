"""Canonical MA parameter values and unchanged 141-column export order."""
def get_default_params():
    """Chen (2004) Table 2 Master Parameter Vector (133 Base Parameters)"""
    return {
        'B0':0.054, 'C0':0.4, 'Dn3':1.0, 'Jn3':6.0, 'ksn2p':0.0, 'ksn2pp':0.15, 'kdn2':0.12,
        'esbfn2':2.0, 'esbfn3':10.0, 'esbfb5':2.0, 'kasbf':0.38, 'kisbfp':0.6, 'kisbfpp':8.0, 'Jasbf':0.01, 'Jisbf':0.01,
        'ksb2p':0.001, 'ksb2pp':0.04, 'kdb2p':0.003, 'kdb2pp':0.4, 'kdb2P':0.15, 'kamcm':1.0, 'kimcm':0.15, 'Jamcm':0.1, 'Jimcm':0.1,
        'ksb5p':0.0008, 'ksb5pp':0.005, 'kdb5p':0.01, 'kdb5pp':0.16, 'ksc1p':0.012, 'ksc1pp':0.12, 'kd1c1':0.01, 'kd2c1':1.0, 'Jd2c1':0.05,
        'ec1k2':0.03, 'ec1n2':0.06, 'ec1b5':0.1, 'ec1b2':0.45, 'ec1n3':0.3, 'kasb2':50.0, 'kdib2':0.05, 'kasb5':50.0, 'kdib5':0.06,
        'kd3c1':1.0, 'kppc1':4.0, 'ksf6p':0.024, 'ksf6pp':0.12, 'ksf6ppp':0.004, 'kd1f6':0.01, 'kd2f6':1.0, 'Jd2f6':0.05,
        'ef6k2':0.03, 'ef6n2':0.06, 'ef6b5':0.1, 'ef6b2':0.55, 'ef6n3':0.3, 'kasf2':15.0, 'kdif2':0.5, 'kasf5':0.01, 'kdif5':0.01,
        'kd3f6':1.0, 'kppf6':4.0, 'ksswip':0.005, 'ksswipp':0.08, 'kdswi':0.08, 'kaswi':2.0, 'kiswi':0.05,
        'kaiep':0.1, 'kiiep':0.15, 'Jaiep':0.1, 'Jiiep':0.1, 'ks20p':0.006, 'ks20pp':0.6, 'kd20':0.3, 'ka20p':0.05, 'ka20pp':0.2, 'ki20p':0.01, 'ki20pp':8.0,
        'kscdh':0.01, 'kdcdh':0.01, 'kacdhp':0.01, 'kacdhpp':0.8, 'Jicdh':0.03, 'Jacdh':0.03, 'kicdhp':0.001, 'kicdhpp':0.08, 
        'eicdhn2':0.4, 'eicdhn3':0.25, 'eicdhb5':8.0, 'eicdhb2':1.2, 'ks14':0.2, 'kd14':0.1, 'ksnet':0.084, 'kdnet':0.03,
        'kasrent':200.0, 'kasrentp':1.0, 'kdirent':1.0, 'kdirentp':2.0, 'kppnetp':0.05, 'kppnetpp':3.0, 'kkpnetp':0.01, 'kkpnetpp':0.6,
        'kspdsp':0.0, 'ks1pdspp':0.03, 'ks2pdspp':0.055, 'kd1pdsp':0.01, 'kd2pdspp':0.2, 'kd3pdspp':0.04, 'kasesp':50.0, 'kdiesp':0.5, 'ESP1T':1.0,
        'ksppx':0.1, 'Jpds':0.04, 'kdppxp':0.17, 'kdppxpp':2.0, 'J20ppx':0.15, 'ksori':2.0, 'kdori':0.06, 'eorib5':0.9, 'eorib2':0.45,
        'ksbud':0.2, 'kdbud':0.06, 'ebudn2':0.25, 'ebudn3':0.05, 'ebudb5':1.0, 'ksspn':0.1, 'kdspn':0.06, 'Jspn':0.14,
        'Tem1T':1.0, 'jatem':0.1, 'Jitem':0.1, 'Cdc15T':1.0, 'ka15p':0.002, 'ka15pp':1.0, 'ka15P':0.001, 'ki15':0.5,
        'mdt':90.0, 'Vi20_active':8.0, 'BUB2_active':1.0
    }

_defaults = get_default_params()

_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})

PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])
