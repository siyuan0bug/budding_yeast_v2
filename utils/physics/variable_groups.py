"""Historical Budding variable groups used by losses.

Group membership is preserved for existing experiments; this is not a universal
classification of ODE event variables. Unused projection helpers are archived in
utils/archive/biological_guard.py and are not applied by production training.
"""

# 39 Variable Definitions
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5,
 nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T,
 nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1,
 nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2) = range(39)

VARIABLE_NAMES = [
    'MASS', 'CLN2', 'CLB2', 'CLB5', 'SIC1', 'CDC6', 'C2', 'C5', 'F2', 'F5',
    'SIC1P', 'C2P', 'C5P', 'CDC6P', 'F2P', 'F5P', 'SWI5T', 'SWI5', 'IEP', 'CDC20T',
    'CDC20', 'CDH1T', 'CDH1', 'CDC14T', 'CDC14', 'NET1T', 'NET1', 'RENT', 'TEM1',
    'CDC15', 'PPX', 'PDS1', 'ESP1', 'ORI', 'BUD', 'SPN', 'Vi20', 'lte1', 'BUB2'
]

# 6 Macro Event / Jump Variables (subject to cell division / spindle / bud resets)
EVENT_VARS = [nMASS, nORI, nBUD, nSPN, nVi20, nlte1]

# Historical loss grouping: preserve its 33-channel membership.
CONTINUOUS_VARS = [v for v in range(39) if v not in EVENT_VARS]

# Historical zero-mask candidates; retained as metadata, not a projection rule.
POSSIBLE_ZERO_VARS = [nCDH1T, nCDH1, nTEM1, nCDC15, nBUB2]
