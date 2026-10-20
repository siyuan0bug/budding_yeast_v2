"""Fission 11 legal modules, parameter bindings and mode-class weights."""
SAMPLING_MODULES = {
    'Wee1':   {'params': ['Vwee', 'Vwee_'],           'modes': ['KO', 'ts', 'OP']},
    'Mik1':   {'params': ['Vmik', 'Vmik_'],           'modes': ['KO', 'ts', 'OP']},
    'Cdc25':  {'params': ['V25', 'V25_'],             'modes': ['KO', 'ts', 'OP']},
    'Pyp3':   {'params': ['Vpyp'],                    'modes': ['KO', 'ts', 'OP']},
    'Rum1':   {'params': ['k5'],                      'modes': ['KO', 'OP']},
    'Ste9':   {'params': ['V2', 'V2c'],               'modes': ['KO', 'ts', 'OP']},
    'SK':     {'params': ['SK'],                      'modes': ['damp']},
    'Cdc13':  {'params': ['k1'],                      'modes': ['ts', 'OP']},
    'Slp1':   {'params': ['kas', 'kad'],              'modes': ['ts', 'OP']},
    'PPsys':  {'params': ['V4', 'k3'],                'modes': ['ts', 'OP']},
    'Growth': {'params': ['mu'],                      'modes': ['scale']},
}

MODE_CLASS = {
    'KO': 'loss',          # 敲除
    'OP': 'expression',    # 过表达
    'ts': 'activity',      # 温度敏感 / 活性降低
    'damp': 'activity',    # SK 阻尼 (Rum1Δ 样表型)
    'scale': 'event',      # 生长速率改变
}

MODE_CLASS_QUOTA = {
    'loss': 0.28, 'expression': 0.28, 'activity': 0.18,
    'stability': 0.20, 'structure': 0.03, 'event': 0.03,
}
