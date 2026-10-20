"""Fission viability rules; distinct from numerical validity."""
from .constants import TOO_SMALL_M

def assign_labels(info, y_uniform, births_lab=None):
    div_times  = info['div_times']
    births     = births_lab if births_lab is not None else info['birth_masses']
    n_div      = len(div_times)
    status     = info['status']
    if status in ('NumericalFailure', 'MaxEvents'):
        return 'Inviable: Numerical Failure', None, status
    if n_div == 0:
        if info['max_MPF'] > 0.5:
            return 'Inviable: M-phase Arrest', None, status
        return 'Inviable: G2 Arrest', None, status
    if births and births[-1] < TOO_SMALL_M:   # 稳态 (最后一次) 出生质量, 原始尺度
        return 'Inviable: Too Small', None, status
    if n_div >= 2:
        pattern = 'Pattern_A'
    elif n_div == 1:
        pattern = 'Pattern_B'
    else:
        pattern = 'Pattern_C'
    return 'Viable', pattern, status
