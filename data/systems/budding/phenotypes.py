"""Unchanged MA viability and trajectory-pattern classifiers."""
import numpy as np

def analyze_cell_viability_unified(eh, bud_div, mass_div, max_mass, max_clb2):
    if max_mass > 10.0:
        return "Inviable: G1 Arrest" if max_clb2 < 0.2 else "Inviable: M-phase Arrest"
    if 1 not in eh:
        return "Inviable: G1 Arrest" if max_clb2 < 0.2 else "Inviable: M-phase Arrest"
    div_indices = [i for i, e in enumerate(eh) if e == 1]
    last_div_idx = 0
    for i, h_idx in enumerate(div_indices):
        if bud_div[i] < 0.95: return "Inviable: Unbudded Division"
        if mass_div[i] < 0.8: return "Inviable: Premature Division"
        cycle_events = eh[last_div_idx:h_idx]
        cleaned = []
        for e in cycle_events:
            if not cleaned or cleaned[-1] != e: cleaned.append(e)
        if 3 not in cleaned or 4 not in cleaned: return "Inviable: Sequence Failure"
        i3, i4 = cleaned.index(3), cleaned.index(4)
        if i3 > i4: return "Inviable: Sequence Failure"
        if i > 0:
            if 2 not in cleaned: return "Inviable: Sequence Failure"
            if cleaned.index(2) > i3: return "Inviable: Sequence Failure"
        last_div_idx = h_idx + 1
    return "Viable"

def analyze_mass_pattern(mc):
    d = np.diff(mc)
    di = np.where(d < -0.1)[0]
    v = 0
    if len(di) > 0:
        v = 1; ld = di[0]
        for dd in di[1:]:
            if dd - ld > 10: v += 1
            ld = dd
    if v >= 2: return "Pattern_A: Multi-Division (>=2)"
    elif v == 1: return "Pattern_B: Single-Division (==1)"
    return "Pattern_C: Exponential-Arrest (==0)"
