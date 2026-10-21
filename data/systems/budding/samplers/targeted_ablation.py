"""Original targeted BAMS diagnostic skeletons and paired KO/hypomorph assignments."""
import numpy as np
from ..parameters import get_default_params
from ..distributions.mutant_aware import sample_scaling_factor
from ..mutations.mutant_aware import apply_gene_mutation

TARGETED_SKELETONS = [{'desc': 'M15_mc', 'lofs': ['CLN3', 'BCK2'], 'gofs': [('CLN2', 'MC')]}, {'desc': 'M15_gal', 'lofs': ['CLN3', 'BCK2'], 'gofs': [('CLN2', 'GAL-OE')]}, {'desc': 'M16', 'lofs': ['CLN3', 'BCK2', 'SIC1'], 'gofs': []}, {'desc': 'M13', 'lofs': ['CLN3', 'BCK2'], 'gofs': []}, {'desc': 'M22_mc', 'lofs': ['CLN2', 'CLN3'], 'gofs': [('CLB5', 'MC')]}, {'desc': 'M23_gal', 'lofs': ['CLN2', 'CLN3'], 'gofs': [('CLB5', 'GAL-OE')]}, {'desc': 'M24_mc', 'lofs': ['CLN2', 'CLN3'], 'gofs': [('BCK2', 'MC')]}, {'desc': 'M20', 'lofs': ['CLN2', 'CLN3', 'SIC1'], 'gofs': []}, {'desc': 'M33', 'lofs': ['CLN2', 'CDH1'], 'gofs': [('SIC1', 'GAL-OE'), ('CLN2', 'GAL-OE')]}, {'desc': 'M31', 'lofs': ['CLN2'], 'gofs': [('SIC1', 'GAL-OE'), ('CLN2', 'GAL-OE')]}, {'desc': 'M42', 'lofs': ['SIC1', 'CDC6', 'CDH1'], 'gofs': [('CDC20', 'GAL-OE')]}, {'desc': 'M58', 'lofs': [('CLB2', 'db')], 'gofs': [('SIC1', 'GAL-OE')]}, {'desc': 'M53', 'lofs': ['SIC1'], 'gofs': [('CLB2', 'GAL-OE')]}, {'desc': 'M124', 'lofs': [], 'gofs': [('APC_Core', 'APC-A'), ('CLB2', 'GAL-OE')]}, {'desc': 'M76', 'lofs': ['CDC20', 'PDS1', 'CLB5'], 'gofs': []}]

def sample_ts_alpha(g_name):
    """
    Calibrated ts decay factor matching mutant_aware.py:
    - Special genes (CDC20, PDS1, CLB5): <=1% boundary buffer U[0, 0.01]
    - CLB2-db: U[0.02125, 0.170]
    - Non-essential genes (CLN3, BCK2, CLN2, SIC1, CDH1, CDC6):
        50% generic ts: 0.20 * Beta(0.5, 2.0) (support (0, 0.20], mean 0.04)
        50% boundary buffer: U[0.0, 0.01] (support (0, 0.01], mean 0.005)
    """
    if g_name in ('CDC20', 'PDS1', 'CLB5'):
        return float(np.random.uniform(0.0, 0.01))
    if g_name == 'CLB2_db':
        return float(np.random.uniform(0.02125, 0.17))
    if np.random.rand() < 0.5:
        return float(0.2 * np.random.beta(0.5, 2.0))
    else:
        return float(np.random.uniform(0.0, 0.01))

def apply_targeted_mutation(g_entry, is_hypo, p_wt, ovr, decay_scalars):
    """
    Applies mutation according to biophysical categories:
    1. 纯敲除基因 (mode == 'KO'):
       - is_hypo = False (Arm A): 严格赋硬零点 0.0 (敲除基因不变，保持二值断崖)
       - is_hypo = True (Arm B): 赋连续弱化过渡态 (ts/Beta 衰减，消除硬断崖)
    2. 减弱基因 (mode in ('db', 'ts', 'DAmP', 'pLOF')):
       - 在 Arm A 与 Arm B 中，减弱倍数均动态改变 (从物理分布中抽取，不等于 real mutant 固定标称值)
       - Arm A 与 Arm B 共享同一个抽取的减弱倍数，确保严格的单变量控制！
    """
    if isinstance(g_entry, tuple):
        g_name, mode = g_entry
    else:
        g_name, mode = (g_entry, 'KO')
    if mode in ('db', 'ts', 'DAmP', 'pLOF'):
        alpha = decay_scalars.get((g_name, mode), sample_scaling_factor(mode, g_name))
        apply_gene_mutation(g_name, mode, alpha, p_wt, {}, ovr)
        return
    if not is_hypo:
        if g_name == 'CLN3':
            ovr['C0'] = 0.0
        elif g_name == 'BCK2':
            ovr['B0'] = 0.0
        elif g_name == 'CLN2':
            ovr['ksn2pp'] = 0.0
        elif g_name == 'SIC1':
            ovr['ksc1p'] = 0.0
            ovr['ksc1pp'] = 0.0
        elif g_name == 'CDH1':
            ovr['kscdh'] = 0.0
            ovr['init_CDH1T'] = 0.0
            ovr['init_CDH1'] = 0.0
        elif g_name == 'CDC6':
            ovr['ksf6p'] = 0.0
            ovr['ksf6pp'] = 0.0
            ovr['ksf6ppp'] = 0.0
        elif g_name == 'CLB5':
            ovr['ksb5p'] = 0.0
            ovr['ksb5pp'] = 0.0
        elif g_name == 'CDC20':
            ovr['ks20p'] = 0.0
            ovr['ks20pp'] = 0.0
        elif g_name == 'PDS1':
            ovr['kspdsp'] = 0.0
        elif g_name == 'CLB2':
            ovr['ksb2p'] = 0.0
            ovr['ksb2pp'] = 0.0
    else:
        alpha = sample_ts_alpha(g_name)
        if g_name == 'CLN3':
            ovr['C0'] = p_wt['C0'] * alpha
        elif g_name == 'BCK2':
            ovr['B0'] = p_wt['B0'] * alpha
        elif g_name == 'CLN2':
            ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'SIC1':
            ovr['ksc1p'] = p_wt['ksc1p'] * alpha
            ovr['ksc1pp'] = p_wt['ksc1pp'] * alpha
        elif g_name == 'CDH1':
            ovr['kscdh'] = p_wt['kscdh'] * alpha
            ovr['init_CDH1T'] = p_wt.get('init_CDH1T', 1.0) * alpha
            ovr['init_CDH1'] = p_wt.get('init_CDH1', 0.9304992) * alpha
        elif g_name == 'CDC6':
            ovr['ksf6p'] = p_wt['ksf6p'] * alpha
            ovr['ksf6pp'] = p_wt['ksf6pp'] * alpha
            ovr['ksf6ppp'] = p_wt['ksf6ppp'] * alpha
        elif g_name == 'CLB5':
            ovr['ksb5p'] = p_wt['ksb5p'] * alpha
            ovr['ksb5pp'] = p_wt['ksb5pp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha
            ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'PDS1':
            ovr['kspdsp'] = p_wt['kspdsp'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha
            ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha

def targeted_pair(count):
    p_wt = get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    sk = TARGETED_SKELETONS[count % len(TARGETED_SKELETONS)]
    has_gal = any((m == 'GAL-OE' for g, m in sk['gofs']))
    mdt = 150.0 if has_gal else float(np.random.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    gof_scalars = {}
    for g, m in sk['gofs']:
        gof_scalars[g, m] = sample_scaling_factor(m, g)
    decay_scalars = {}
    for lof in sk['lofs']:
        if isinstance(lof, tuple):
            g, m = lof
            if m in ('db', 'ts', 'DAmP', 'pLOF'):
                decay_scalars[g, m] = sample_scaling_factor(m, g)
    ovr_A = {'mdt': mdt}
    for lof in sk['lofs']:
        apply_targeted_mutation(lof, is_hypo=False, p_wt=p_wt, ovr=ovr_A, decay_scalars=decay_scalars)
    for (g, m), alpha in gof_scalars.items():
        apply_gene_mutation(g, m, alpha, p_wt, {}, ovr_A)
    ovr_B = {'mdt': mdt}
    for lof in sk['lofs']:
        apply_targeted_mutation(lof, is_hypo=True, p_wt=p_wt, ovr=ovr_B, decay_scalars=decay_scalars)
    for (g, m), alpha in gof_scalars.items():
        apply_gene_mutation(g, m, alpha, p_wt, {}, ovr_B)
    return (ovr_A, ovr_B, sk['desc'])
