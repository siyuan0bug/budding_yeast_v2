"""mass synthetic mutation assignment, independent of the canonical biological engine."""
import numpy as np
from ..catalog import require_mutation
from ..distributions.ranges import GAL_MC_BASE,GAL_MC_MIN,GAL_MC_MAX,NET1_BINDING_RESIDUAL,log_uniform

def apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr, *, rng=None):
    """
    Explicit, dedicated parameter modification engine per (Gene, Mode) pair
    matching Section 7 of Mutant-Aware Sampling Scheme v1.
    """
    require_mutation(g_name, m_chosen)
    rng = np.random if rng is None else rng
    if m_chosen == 'GAL-MC':
        if not float(alpha).is_integer() or not GAL_MC_MIN<=alpha<=GAL_MC_MAX:
            raise ValueError('CLB2 GAL-MC requires an integer copy count in 2..10')
        ovr['ksb2p'] = GAL_MC_BASE*float(alpha)
        p_mut['ksb2p'] = ovr['ksb2p']
    elif m_chosen == 'GAL-OE':
        if g_name == 'CLN2':
            p_mut['ksn2p'] = alpha
            p_mut['ksn2pp'] = 0.0
            ovr['ksn2p'] = p_mut['ksn2p']
            ovr['ksn2pp'] = 0.0
        elif g_name == 'CLN3':
            p_mut['C0'] = alpha
            ovr['C0'] = p_mut['C0']
        elif g_name == 'CLB2':
            p_mut['ksb2p'] = alpha
            ovr['ksb2p'] = p_mut['ksb2p']
        elif g_name == 'CLB5':
            p_mut['ksb5p'] = alpha
            ovr['ksb5p'] = p_mut['ksb5p']
        elif g_name == 'SIC1':
            p_mut['ksc1p'] = alpha
            ovr['ksc1p'] = p_mut['ksc1p']
        elif g_name == 'CDC6':
            p_mut['ksf6p'] = alpha
            ovr['ksf6p'] = p_mut['ksf6p']
        elif g_name == 'SWI5':
            p_mut['ksswip'] = alpha
            ovr['ksswip'] = p_mut['ksswip']
        elif g_name == 'CDC20':
            p_mut['ks20p'] = alpha
            ovr['ks20p'] = p_mut['ks20p']
        elif g_name == 'CDC14':
            p_mut['ks14'] = alpha
            ovr['ks14'] = p_mut['ks14']
        elif g_name == 'NET1':
            p_mut['ksnet'] = alpha
            ovr['ksnet'] = p_mut['ksnet']
        elif g_name == 'TEM1':
            p_mut['Tem1T'] = alpha
            ovr['Tem1T'] = p_mut['Tem1T']
        elif g_name == 'CDC15':
            p_mut['Cdc15T'] = alpha
            ovr['Cdc15T'] = p_mut['Cdc15T']
        elif g_name == 'PDS1':
            p_mut['kspdsp'] = alpha
            ovr['kspdsp'] = p_mut['kspdsp']
        elif g_name == 'ESP1':
            p_mut['ESP1T'] = alpha
            ovr['ESP1T'] = p_mut['ESP1T']
        elif g_name == 'PPX':
            p_mut['ksppx'] = alpha
            ovr['ksppx'] = alpha
    elif m_chosen == 'MC':
        if g_name == 'CLN2':
            ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'CLN3':
            ovr['C0'] = p_wt['C0'] * alpha
        elif g_name == 'BCK2':
            ovr['B0'] = p_wt['B0'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha
            ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
        elif g_name == 'CLB5':
            ovr['ksb5p'] = p_wt['ksb5p'] * alpha
            ovr['ksb5pp'] = p_wt['ksb5pp'] * alpha
        elif g_name == 'SIC1':
            ovr['ksc1p'] = p_wt['ksc1p'] * alpha
            ovr['ksc1pp'] = p_wt['ksc1pp'] * alpha
        elif g_name == 'CDC6':
            ovr['ksf6p'] = p_wt['ksf6p'] * alpha
            ovr['ksf6pp'] = p_wt['ksf6pp'] * alpha
            ovr['ksf6ppp'] = p_wt['ksf6ppp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha
            ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            ovr['ks14'] = p_wt['ks14'] * alpha
        elif g_name == 'TEM1':
            ovr['Tem1T'] = p_wt['Tem1T'] * alpha
        elif g_name == 'CDC15':
            ovr['Cdc15T'] = p_wt['Cdc15T'] * alpha
    elif m_chosen in ['DAmP', 'pLOF']:
        if g_name == 'CDC6':
            ovr['ksf6p'] = p_wt['ksf6p'] * alpha
            ovr['ksf6pp'] = p_wt['ksf6pp'] * alpha
            ovr['ksf6ppp'] = p_wt['ksf6ppp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha
            ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            ovr['ks14'] = p_wt['ks14'] * alpha
        elif g_name == 'NET1':
            ovr['ksnet'] = p_wt['ksnet'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha
            ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
    elif m_chosen == 'ts':
        if g_name == 'CLN2':
            ovr['ksn2p'] = p_wt['ksn2p'] * alpha
            ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha
            ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha
            ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            if rng.rand() < 0.5:
                ovr['ks14'] = 0.0
            else:
                ovr['ks14'] = p_wt['ks14']
                ovr['kppc1'] = p_wt['kppc1'] * alpha
                ovr['kppf6'] = p_wt['kppf6'] * alpha
                ovr['kaswi'] = p_wt['kaswi'] * alpha
                ovr['kacdhpp'] = p_wt['kacdhpp'] * alpha
        elif g_name == 'NET1':
            ovr['kasrent'] = p_wt['kasrent'] * alpha
            ovr['kasrentp'] = p_wt['kasrentp'] * alpha
        elif g_name == 'PPX':
            ovr['kppnetpp'] = 0.05
        elif g_name == 'Mcm1':
            ovr['kamcm'] = 0.13
        elif g_name == 'CDC15':
            ovr['kkpnetpp'] = 0.0
        elif g_name == 'TEM1':
            ovr['Tem1T'] = 0.0
            ovr['ka15pp'] = p_wt['ka15pp'] * alpha
            ovr['init_Tem1'] = 0.0
        elif g_name == 'ESP1':
            ovr['kasesp'] = p_wt['kasesp'] * alpha
            ovr['kdiesp'] = p_wt['kdiesp'] * alpha
        elif g_name == 'APC_Core':
            ovr['ks20p'] = 0.0
            ovr['ks20pp'] = 0.0
            ovr['kscdh'] = 0.0
            ovr['init_CDH1T'] = 0.0
            ovr['init_CDH1'] = 0.0
    elif m_chosen == 'db':
        if g_name == 'CLB2':
            ovr['kdb2P'] = 0.0
            ovr['kdb2pp'] = p_wt['kdb2pp'] * alpha
        elif g_name == 'CLB5':
            ovr['kdb5pp'] = 0.0
        elif g_name == 'PDS1':
            ovr['kd2pdspp'] = 0.0
            ovr['kd3pdspp'] = 0.0
        elif g_name == 'SIC1':
            ovr['kd3c1'] = 0.0
        elif g_name == 'PPX':
            ovr['kdppxpp'] = 0.0
    elif m_chosen == 'KO':
        if g_name == 'CLN2':
            ovr['ksn2pp'] = 0.0
        elif g_name == 'CLN3':
            ovr['C0'] = 0.0
        elif g_name == 'BCK2':
            ovr['B0'] = 0.0
        elif g_name == 'CLB2':
            ovr['ksb2p'] = 0.0
            ovr['ksb2pp'] = 0.0
        elif g_name == 'CLB5':
            ovr['ksb5p'] = 0.0
            ovr['ksb5pp'] = 0.0
        elif g_name == 'SIC1':
            ovr['ksc1p'] = 0.0
            ovr['ksc1pp'] = 0.0
        elif g_name in ['CDC6', 'pDEL']:
            ovr['ksf6p'] = 0.0
            ovr['ksf6pp'] = 0.0
            ovr['ksf6ppp'] = 0.0
        elif g_name == 'SWI5':
            ovr['ksswip'] = 0.0
            ovr['ksswipp'] = 0.0
        elif g_name == 'CDC20':
            ovr['ks20p'] = 0.0
            ovr['ks20pp'] = 0.0
        elif g_name == 'CDH1':
            ovr['kscdh'] = 0.0
            ovr['init_CDH1T'] = 0.0
            ovr['init_CDH1'] = 0.0
        elif g_name == 'TEM1':
            ovr['Tem1T'] = 0.0
            ovr['ka15pp'] = 0.0
            ovr['init_Tem1'] = 0.0
        elif g_name == 'CDC15':
            ovr['Cdc15T'] = 0.0
            ovr['kkpnetpp'] = 0.0
            ovr['init_Cdc15'] = 0.0
        elif g_name == 'PDS1':
            ovr['ks1pdspp'] = 0.0
            ovr['ks2pdspp'] = 0.0
        elif g_name == 'PPX':
            ovr['ksppx'] = 0.0
        elif g_name == 'Mcm1':
            ovr['kamcm'] = 0.0
        elif g_name == 'SBF_MBF':
            ovr.update(kasbf=0.0, ksn2pp=0.0, ksb5pp=0.0)
    elif m_chosen == 'CA':
        if g_name == 'CDH1':
            ovr['kicdhpp'] = 0.0
            ovr['kscdh'] = 0.03
    elif m_chosen == 'CSL':
        ovr['Vi20'] = 0.01
        ovr['Vi20_active'] = 0.01
    elif m_chosen == 'CKO':
        ovr['kbub2'] = 0.0
        ovr['BUB2_active'] = 0.0
        ovr['init_BUB2'] = 0.0
    elif m_chosen == 'APC-A' or (g_name in ['CDC20', 'APC_Core'] and m_chosen == 'APC-A'):
        ovr['ka20pp'] = 0.0
    elif m_chosen == 'TAB6-1':
        f = log_uniform(rng,NET1_BINDING_RESIDUAL)
        ovr['kasrent'] = p_wt['kasrent']*f
        ovr['kasrentp'] = p_wt['kasrentp']*f
    elif m_chosen == 'PM':
        ovr['kd2c1'] = 0.0
        ovr['kd3c1'] = 0.0
    elif m_chosen == 'pDEL':
        ovr['ksf6p'] = 0.0
        ovr['ksf6pp'] = 0.0
        ovr['ksf6ppp'] = 0.0
    elif m_chosen == 'MDT':
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0]))
