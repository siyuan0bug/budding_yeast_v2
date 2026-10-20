"""Fission 20 real-mutant identities; equal-parameter identities retained."""
import copy

REAL_MUTANTS = {
    'WT':                      ({}, 'no change'),
    'wee1ts':                  ({'Vwee': 0.08}, 'Vwee=0.08'),
    'wee1D':                   ({'Vwee': 0.0, 'Vwee_': 0.0}, 'Vwee=Vwee_=0'),
    'mik1D':                   ({'Vmik': 0.0, 'Vmik_': 0.0}, 'Vmik=Vmik_=0'),
    'mik1D wee1ts':            ({'Vmik': 0.0, 'Vmik_': 0.0, 'Vwee': 0.08},
                                'Vmik=Vmik_=0, Vwee=0.08 (cut)'),
    'pyp3D':                   ({'Vpyp': 0.0}, 'Vpyp=0'),
    'cdc25ts':                 ({'V25': 0.05, 'V25_': 0.00025},
                                'V25=0.05, V25_=0.00025 (G2 block)'),
    'cdc25D':                  ({'V25': 0.0, 'V25_': 0.0},
                                'V25=V25_=0 (G2 block)'),
    'wee1ts cdc25D':           ({'Vwee': 0.08, 'V25': 0.0, 'V25_': 0.0},
                                'Vwee=0.08, V25=V25_=0 (quantized 112/170)'),
    'wee1ts cdc25D pyp3D':     ({'Vwee': 0.08, 'V25': 0.0, 'V25_': 0.0, 'Vpyp': 0.0},
                                'G2 block (no Cdc25, no Pyp3)'),
    'mik1D wee1ts cdc25D':     ({'Vmik': 0.0, 'Vmik_': 0.0, 'Vwee': 0.08,
                                'V25': 0.0, 'V25_': 0.0}, 'cut lethal'),
    'wee1ts cdc25ts':          ({'Vwee': 0.08, 'V25': 0.05, 'V25_': 0.00025},
                                'Vwee=0.08, V25=0.05, V25_=0.00025'),
    'wee1D cdc25D':            ({'Vwee': 0.0, 'Vwee_': 0.0, 'V25': 0.0, 'V25_': 0.0},
                                'Vwee=Vwee_=0, V25=V25_=0'),
    'rum1D':                   ({'k5': 0.0}, 'k5=0'),
    'rum1D wee1ts':            ({'k5': 0.0, 'Vwee': 0.08}, 'k5=0, Vwee=0.08 (too small)'),
    'ste9D':                   ({'V2': 0.0, 'V2c': 0.0}, 'V2=V2c=0'),
    'cdc13dbD':                ({'V2': 0.0, 'V2c': 0.0},
                                'V2=V2c=0 (same as ste9D; non-degradable Cdc13)'),
    'ste9D wee1ts':            ({'V2': 0.0, 'V2c': 0.0, 'Vwee': 0.08},
                                'V2=V2c=0, Vwee=0.08 (too small)'),
    'ste9D rum1D':             ({'V2': 0.0, 'V2c': 0.0, 'k5': 0.0}, 'V2=V2c=0, k5=0'),
    'ste9D rum1D wee1ts':      ({'V2': 0.0, 'V2c': 0.0, 'k5': 0.0, 'Vwee': 0.08},
                                'V2=V2c=0, k5=0, Vwee=0.08 (too small)'),
}

def real_tasks():
    return [(name, copy.deepcopy(overrides)) for name, (overrides, _) in REAL_MUTANTS.items()]
