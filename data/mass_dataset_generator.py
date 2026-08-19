"""
MASS (Mutant-Aware Stratified Sampling v4.4) Dataset Generator
================================================================================
Strictly adheres to Mutant_Aware_Sampling_Scheme_v1_Chen_Primary_Backup.md 
and Chen (2004) Molecular Cell Cycle Model (Table 1, Table 2, Appendix B Table S5).

Export Alignment:
Matches single .npz file structure and field types of lhs_v1_origin.py:
- data           : 3D float array (N_samples, 39, 500) - 39 ODE state curves over 500 steps
- params         : 2D float array (N_samples, 136) - 136-dim parameter vectors + init_ states
- param_names    : 1D string array - 136 sorted parameter key names
- mutant_names   : 1D string array - Real mutant names from Appendix B & Synthetic mutant names ({genes}_{modes}_L{level}_{idx})
- labels         : 1D string array - Biological cell viability labels
- pattern_labels : 1D string array - Visual mass growth pattern labels
"""

import numpy as np
import math, os, sys, time, warnings, signal
import concurrent.futures
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

warnings.filterwarnings("ignore")

# ==============================================================================
# Part 1: Chen (2004) ODE Engine & Default State / Parameter Definitions
# ==============================================================================

# 39 State Variables
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, 
 nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, 
 nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, 
 nVi20, nlte1, nBUB2) = range(39)

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

def initial_conds():
    """Chen (2004) Table 2B Initial State Vector (39 States)"""
    y0 = np.zeros(39)
    y0[nMASS]  = 1.2060194; y0[nCLN2]  = 0.0652511; y0[nCLB2]  = 0.1469227; y0[nCLB5]  = 0.0518041
    y0[nSIC1]  = 0.0228776; y0[nCDC6]  = 0.1075804; y0[nC2]    = 0.2384047; y0[nC5]    = 0.0700814
    y0[nF2]    = 0.2360586; y0[nF5]    = 7.24514e-5;y0[nSIC1P] = 0.0064101; y0[nC2P]   = 0.0240340
    y0[nC5P]   = 0.0068783; y0[nCDC6P] = 0.0154863; y0[nF2P]   = 0.0273938; y0[nF5P]   = 7.90633e-6
    y0[nSWI5T] = 0.9764602; y0[nSWI5]  = 0.9561624; y0[nIEP]   = 0.1015391; y0[nCDC20T]= 1.9163399
    y0[nCDC20] = 0.4442965; y0[nCDH1T] = 1.0;       y0[nCDH1]  = 0.9304992; y0[nCDC14T]= 2.0
    y0[nCDC14] = 0.4683439; y0[nNET1T] = 2.7999999; y0[nNET1]  = 0.0186456; y0[nRENT]  = 1.0495476
    y0[nTEM1]  = 0.9038969; y0[nCDC15] = 0.6565329; y0[nPPX]   = 0.1231788; y0[nPDS1]  = 0.0256123
    y0[nESP1]  = 0.3013133; y0[nORI]   = 0.0009094; y0[nBUD]   = 0.0084734; y0[nSPN]   = 0.0305621
    y0[nVi20]  = 0.01;      y0[nlte1]  = 0.1000000; y0[nBUB2]  = 0.2000000
    return y0

def GK(Va, Vi, Ja, Ji):
    """Goldbeter-Koshland Ultra-sensitivity Function"""
    BB = Vi - Va + Ja * Vi + Ji * Va
    return 2 * Ji * Va / (BB + np.sqrt(np.maximum(0, BB**2 - 4 * (Vi - Va) * Ji * Va)))

def eqns(t, y, p):
    """
    Chen (2004) Table 1 Differential Equations & Auxiliary Functions.
    100% Strictly Aligned with Chen_2004_Extracted_Tables.md
    """
    dydt = np.zeros(39)
    (MASS, CLN2, CLB2, CLB5, SIC1, CDC6, C2, C5, F2, F5, SIC1P, C2P, C5P, CDC6P, F2P, F5P,
     SWI5T, SWI5, IEP, CDC20T, CDC20, CDH1T, CDH1, CDC14T, CDC14, NET1T, NET1, RENT,
     TEM1, CDC15, PPX, PDS1, ESP1, ORI, BUD, SPN, Vi20, lte1, BUB2) = y

    # Auxiliary Kinase & Mass-Action Conservation Relations
    kg = np.log(2) / p['mdt']
    BCK2 = p['B0'] * MASS
    CLN3 = p['C0'] * p['Dn3'] * MASS / (p['Jn3'] + p['Dn3'] * MASS)
    SIC1T = SIC1 + C2 + C5 + SIC1P + C2P + C5P
    CDC6T = CDC6 + F2 + F5 + CDC6P + F2P + F5P
    RENTP = CDC14T - RENT - CDC14
    NET1P = NET1T - NET1 - CDC14T + CDC14
    PE = p['ESP1T'] - ESP1

    # Activation / Degradation Auxiliary Rates
    if p.get('is_sbf_ca'):
        Vasbf = 0.38
    else:
        Vasbf = p['kasbf'] * (p['esbfn2']*CLN2 + p['esbfn3']*(CLN3+BCK2) + p['esbfb5']*CLB5)
    
    Vdb2 = p['kdb2p'] + p['kdb2pp']*CDH1 + p['kdb2P']*CDC20
    Vdb5 = p['kdb5p'] + p['kdb5pp']*CDC20
    Vkpc1 = p['kd1c1'] + p['kd2c1']*(p['ec1n3']*CLN3 + p['ec1k2']*BCK2 + p['ec1n2']*CLN2 + p['ec1b2']*CLB2 + p['ec1b5']*CLB5)/(p['Jd2c1'] + SIC1T)
    Vppc1 = p['kppc1']*CDC14
    Vkpf6 = p['kd1f6'] + p['kd2f6']*(p['ef6n3']*CLN3 + p['ef6k2']*BCK2 + p['ef6n2']*CLN2 + p['ef6b2']*CLB2 + p['ef6b5']*CLB5)/(p['Jd2f6'] + CDC6T)
    Vppf6 = p['kppf6']*CDC14
    Vaiep = p['kaiep']*CLB2
    Vacdh = p['kacdhp'] + p['kacdhpp']*CDC14
    Vicdh = p['kicdhp'] + p['kicdhpp']*(p['eicdhn3']*CLN3 + p['eicdhn2']*CLN2 + p['eicdhb5']*CLB5 + p['eicdhb2']*CLB2)
    Vppnet = p['kppnetp'] + p['kppnetpp']*PPX
    Vkpnet = (p['kkpnetp'] + p['kkpnetpp']*CDC15)*MASS
    Vdpds = p['kd1pdsp'] + p['kd2pdspp']*CDC20 + p['kd3pdspp']*CDH1
    Vdppx = p['kdppxp'] + p['kdppxpp']*(p['J20ppx']+CDC20)*p['Jpds']/(p['Jpds']+PDS1)

    # Goldbeter-Koshland Ultra-sensitive Switches
    SBF = GK(Vasbf, p['kisbfp']+p['kisbfpp']*CLB2, p['Jasbf'], p['Jisbf'])
    MBF = SBF  # In Chen (2004) ODE model, MBF and SBF share identical GK activation dynamics
    MCM1 = GK(p['kamcm']*CLB2, p['kimcm'], p['Jamcm'], p['Jimcm'])
    actual_ksc1p = 0.012 * 3 if (p.get('is_mutant_104') and t >= 180.0) else p['ksc1p']

    # Table 1 Differential Equations (0 .. 35)
    dydt[0] = kg * MASS                                                                                     # d[mass]/dt
    dydt[1] = (p['ksn2p'] + p['ksn2pp']*SBF)*MASS - p['kdn2']*CLN2                                          # d[Cln2]/dt
    dydt[2] = (p['ksb2p'] + p['ksb2pp']*MCM1)*MASS + p['kd3c1']*C2P + p['kdib2']*C2 + p['kd3f6']*F2P + p['kdif2']*F2 - (Vdb2 + p['kasb2']*SIC1 + p['kasf2']*CDC6)*CLB2 # d[Clb2]/dt
    dydt[3] = (p['ksb5p'] + p['ksb5pp']*MBF)*MASS + p['kd3c1']*C5P + p['kdib5']*C5 + p['kd3f6']*F5P + p['kdif5']*F5 - (Vdb5 + p['kasb5']*SIC1 + p['kasf5']*CDC6)*CLB5 # d[Clb5]/dt
    dydt[4] = (actual_ksc1p + p['ksc1pp']*SWI5) + (Vdb2 + p['kdib2'])*C2 + (Vdb5 + p['kdib5'])*C5 + Vppc1*SIC1P - (p['kasb2']*CLB2 + p['kasb5']*CLB5 + Vkpc1)*SIC1     # d[Sic1]/dt
    dydt[5] = (p['ksf6p'] + p['ksf6pp']*SWI5 + p['ksf6ppp']*SBF) + (Vdb2 + p['kdif2'])*F2 + (Vdb5 + p['kdif5'])*F5 + Vppf6*CDC6P - (p['kasf2']*CLB2 + p['kasf5']*CLB5 + Vkpf6)*CDC6 # d[Cdc6]/dt
    dydt[6] = p['kasb2']*CLB2*SIC1 + Vppc1*C2P - (p['kdib2'] + Vdb2 + Vkpc1)*C2                               # d[C2]/dt
    dydt[7] = p['kasb5']*CLB5*SIC1 + Vppc1*C5P - (p['kdib5'] + Vdb5 + Vkpc1)*C5                               # d[C5]/dt
    dydt[8] = p['kasf2']*CLB2*CDC6 + Vppf6*F2P - (p['kdif2'] + Vdb2 + Vkpf6)*F2                               # d[F2]/dt
    dydt[9] = p['kasf5']*CLB5*CDC6 + Vppf6*F5P - (p['kdif5'] + Vdb5 + Vkpf6)*F5                               # d[F5]/dt
    dydt[10] = Vkpc1*SIC1 - (Vppc1 + p['kd3c1'])*SIC1P + Vdb2*C2P + Vdb5*C5P                                 # d[Sic1P]/dt
    dydt[11] = Vkpc1*C2 - (Vppc1 + p['kd3c1'] + Vdb2)*C2P                                                     # d[C2P]/dt
    dydt[12] = Vkpc1*C5 - (Vppc1 + p['kd3c1'] + Vdb5)*C5P                                                     # d[C5P]/dt
    dydt[13] = Vkpf6*CDC6 - (Vppf6 + p['kd3f6'])*CDC6P + Vdb2*F2P + Vdb5*F5P                                 # d[Cdc6P]/dt
    dydt[14] = Vkpf6*F2 - (Vppf6 + p['kd3f6'] + Vdb2)*F2P                                                     # d[F2P]/dt
    dydt[15] = Vkpf6*F5 - (Vppf6 + p['kd3f6'] + Vdb5)*F5P                                                     # d[F5P]/dt
    dydt[16] = p['ksswip'] + p['ksswipp']*MCM1 - p['kdswi']*SWI5T                                             # d[Swi5_T]/dt
    dydt[17] = p['ksswip'] + p['ksswipp']*MCM1 + p['kaswi']*CDC14*(SWI5T - SWI5) - (p['kdswi'] + p['kiswi']*CLB2)*SWI5 # d[Swi5]/dt
    dydt[18] = Vaiep*(1 - IEP)/(p['Jaiep'] + 1 - IEP) - p['kiiep']*IEP/(p['Jiiep'] + IEP)                     # d[APC-P]/dt
    dydt[19] = (p['ks20p'] + p['ks20pp']*MCM1) - p['kd20']*CDC20T                                             # d[Cdc20_T]/dt
    dydt[20] = (p['ka20p'] + p['ka20pp']*IEP)*(CDC20T - CDC20) - (Vi20 + p['kd20'])*CDC20                     # d[Cdc20_A]/dt
    dydt[21] = p['kscdh'] - p['kdcdh']*CDH1T                                                                  # d[Cdh1_T]/dt
    dydt[22] = p['kscdh'] - p['kdcdh']*CDH1 + Vacdh*(CDH1T - CDH1)/(p['Jacdh'] + CDH1T - CDH1) - Vicdh*CDH1/(p['Jicdh'] + CDH1) # d[Cdh1]/dt
    dydt[23] = p['ks14'] - p['kd14']*CDC14T                                                                   # d[Cdc14_T]/dt
    dydt[24] = p['ks14'] - p['kd14']*CDC14 + (p['kdnet'] + p['kdirent'])*RENT + (p['kdnet'] + p['kdirentp'])*RENTP - (p['kasrent']*NET1 + p['kasrentp']*NET1P)*CDC14 # d[Cdc14]/dt
    dydt[25] = p['ksnet'] - p['kdnet']*NET1T                                                                  # d[Net1_T]/dt
    dydt[26] = p['ksnet'] - p['kdnet']*NET1 + (p['kd14'] + p['kdirent'])*RENT - p['kasrent']*NET1*CDC14 + Vppnet*NET1P - Vkpnet*NET1 # d[Net1]/dt
    dydt[27] = -(p['kd14'] + p['kdnet'] + p['kdirent'])*RENT + p['kasrent']*NET1*CDC14 + Vppnet*RENTP - Vkpnet*RENT # d[RENT]/dt
    dydt[28] = lte1*(p['Tem1T'] - TEM1)/(p['jatem'] + p['Tem1T'] - TEM1) - BUB2*TEM1/(p['Jitem'] + TEM1)       # d[Tem1]/dt
    dydt[29] = (p['ka15p']*(p['Tem1T'] - TEM1) + p['ka15pp']*TEM1 + p['ka15P']*CDC14)*(p['Cdc15T'] - CDC15) - p['ki15']*CDC15 # d[Cdc15]/dt
    dydt[30] = p['ksppx'] - Vdppx*PPX                                                                         # d[PPX]/dt
    dydt[31] = (p['kspdsp'] + p['ks1pdspp']*SBF + p['ks2pdspp']*MCM1) + p['kdiesp']*PE - (Vdpds + p['kasesp']*ESP1)*PDS1 # d[Pds1]/dt
    dydt[32] = -p['kasesp']*PDS1*ESP1 + (p['kdiesp'] + Vdpds)*PE                                              # d[Esp1]/dt
    dydt[33] = p['ksori']*(p['eorib5']*CLB5 + p['eorib2']*CLB2) - p['kdori']*ORI                              # d[ORI]/dt
    dydt[34] = p['ksbud']*(p['ebudn2']*CLN2 + p['ebudn3']*CLN3 + p['ebudb5']*CLB5) - p['kdbud']*BUD            # d[BUD]/dt
    dydt[35] = p['ksspn']*CLB2/(p['Jspn'] + CLB2) - p['kdspn']*SPN                                            # d[SPN]/dt
    dydt[36:39] = 0
    return dydt

# Discrete Events
def event1(t, y, p): return y[nCLB2] - 0.3
event1.terminal = True; event1.direction = -1
def event2(t, y, p): return y[nCLB2] + y[nCLB5] - 0.2
event2.terminal = True; event2.direction = -1
def event3(t, y, p): return y[nORI] - 1.0
event3.terminal = True; event3.direction = 1
def event4(t, y, p): return y[nSPN] - 1.0
event4.terminal = True; event4.direction = 1

# Cell Viability & Visual Pattern Classifier
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

def simulate_mutant(name, ovr, tf=210, ns=500):
    """Simulate a mutant cell trajectory over tf min with exact numerical protections"""
    p = get_default_params(); y0 = initial_conds()
    is_mad2_ko = False; is_bub2_ko = False

    for k, v in ovr.items():
        if k == 'init_CDH1T': y0[nCDH1T] = v
        elif k == 'init_CDH1': y0[nCDH1] = v
        elif k == 'init_BUB2': y0[nBUB2] = v
        elif k == 'init_ESP1': y0[nESP1] = v
        elif k == 'init_Tem1': y0[nTEM1] = v
        elif k == 'init_Cdc15': y0[nCDC15] = v
        elif k == 'init_PPX': y0[nPPX] = v
        elif k == 'init_CLN2': y0[nCLN2] = v
        elif k == 'init_CLB2': y0[nCLB2] = v
        elif k == 'init_CLB5': y0[nCLB5] = v
        elif k == 'init_SIC1': y0[nSIC1] = v
        elif k == 'init_CDC6': y0[nCDC6] = v
        elif k == 'init_CDC14': y0[nCDC14] = v
        elif k == 'init_NET1': y0[nNET1] = v
        elif k == 'init_PDS1': y0[nPDS1] = v
        elif k == 'init_CDC20': y0[nCDC20] = v
        else: p[k] = v
        if k == 'Vi20' and v == 0.01: is_mad2_ko = True
        if k == 'Vi20_active' and v == 0.01: is_mad2_ko = True
        if k == 'kbub2' and v == 0.0: is_bub2_ko = True
        if k == 'BUB2_active' and v == 0.0: is_bub2_ko = True

    if is_mad2_ko: p['Vi20_active'] = 0.01
    if is_bub2_ko: p['BUB2_active'] = 0.0; y0[nBUB2] = 0.0

    tr, yr = [0], [y0]; eh = []; bd = []; md = []; t0 = 0; lc = 0
    try:
        while t0 < tf and lc < 300:
            lc += 1
            sol = solve_ivp(eqns, [t0, tf], y0, args=(p,), method='BDF', 
                            events=[event1, event2, event3, event4], rtol=1e-4, atol=1e-6)
            tr.extend(sol.t[1:]); yr.extend(sol.y.T[1:])
            if sol.status == 1:
                t0, y0 = sol.t[-1], sol.y[:, -1].copy()
                ie = next((i+1 for i, te in enumerate(sol.t_events) if len(te) > 0 and np.isclose(te[-1], t0)), -1)
                if ie != -1: eh.append(ie)
                if ie == 1:
                    bd.append(y0[nBUD]); md.append(y0[nMASS]); y0[nBUD] = y0[nSPN] = 0
                    kg = np.log(2) / p['mdt']; y0[nMASS] = abs(1 + np.exp(-kg * ((1.026 / kg) - 32)) - 1) * y0[nMASS]
                    y0[nlte1] = 0.1; y0[nCLB2] -= 1e-6
                elif ie == 2: y0[nORI] = 0; y0[nCLB5] -= 1e-6
                elif ie == 3:
                    y0[nVi20] = p['Vi20_active'] if not is_mad2_ko else 0.01
                    y0[nBUB2] = p['BUB2_active'] if not is_bub2_ko else 0.0
                    y0[nORI] += 1e-6
                elif ie == 4:
                    y0[nVi20] = 0.01; y0[nlte1] = 1.0
                    y0[nBUB2] = ovr.get('init_BUB2', 0.2) if not is_bub2_ko else 0.0
                    y0[nSPN] += 1e-6
            else: break
    except Exception as e:
        return name, None, "Inviable: Numerical Failure", "Pattern_C: Exponential-Arrest (==0)"

    tr, yr = np.array(tr), np.array(yr).T
    if len(tr) < 2: return name, None, "Inviable: Numerical Failure", "Pattern_C: Exponential-Arrest (==0)"
    interp = interp1d(tr, yr, kind='linear', axis=1, fill_value="extrapolate")
    yu = interp(np.linspace(0, tf, ns))
    
    max_m = np.max(yu[nMASS])
    if tf <= 210 and max_m > 10.0:
        return name, None, "Discarded_Mass", None

    max_b2 = np.max(yu[nCLB2])
    bio_l = analyze_cell_viability_unified(eh, bd, md, max_m, max_b2)
    pat_l = analyze_mass_pattern(yu[nMASS])
    return name, yu, bio_l, pat_l

# ==============================================================================
# Part 2: Appendix B Table S5 All 126 Real Mutants Dictionary (Exact Names & Rules)
# ==============================================================================

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
    '88': ('tem1-ts multi-copy CDC15', {'ka15pp': 0.002, 'Tem1T': 0.0, 'Cdc15T': 5.0}),
    '89': ('tem1-ts GAL-CDC15', {'ka15pp': 0.002, 'Tem1T': 0.0, 'Cdc15T': 15.0, 'mdt': 150.0}),
    '90': ('tem1-ts net1-ts', {'ka15pp': 0.002, 'Tem1T': 0.0, 'kasrent': 10.0, 'kasrentp': 0.05}),
    '91': ('tem1-ts multi-copy CDC14', {'ka15pp': 0.002, 'Tem1T': 0.0, 'ks14': 0.40}),
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

# ==============================================================================
# Part 3: 27 Parameter Binding Groups (G1 ~ G27) & Hub Centrality Probabilities
# ==============================================================================

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
        'name': 'TEM1', 'params': ['Tem1T', 'ka15pp'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.200, 'GAL-OE': 0.200, 'MC': 0.200, 'ts': 0.200, 'DAmP': 0.200}
    },
    'G14': {
        'name': 'CDC15', 'params': ['Cdc15T', 'kkpnetpp'],
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

# Normalize P_bio_G weights
_total_g_weight = sum(g['P_bio_G'] for g in BINDING_GROUPS.values())
for g in BINDING_GROUPS.values():
    g['P_bio_G'] /= _total_g_weight

# ==============================================================================
# Part 4: Section 2.3.1 Continuous Biophysical Distribution Samplers
# ==============================================================================

def sample_scaling_factor(mode, g_name=None):
    if mode == 'GAL-OE':
        if g_name == 'CDC20':
            return 10.0 ** float(np.random.uniform(np.log10(0.50), np.log10(12.0)))
        elif g_name == 'CLB2':
            return 10.0 ** float(np.random.uniform(np.log10(0.002), np.log10(1.50)))
        elif g_name == 'SIC1':
            return 10.0 ** float(np.random.uniform(np.log10(0.03), np.log10(0.36)))
        elif g_name == 'CDC6':
            return 10.0 ** float(np.random.uniform(np.log10(0.10), np.log10(0.60)))
        elif g_name == 'CLB5':
            return 10.0 ** float(np.random.uniform(np.log10(0.004), np.log10(0.030)))
        elif g_name == 'PDS1':
            return 10.0 ** float(np.random.uniform(np.log10(0.02), np.log10(0.25)))
        elif g_name == 'CLN2':
            return 10.0 ** float(np.random.uniform(np.log10(0.03), np.log10(0.30)))
        elif g_name == 'CLN3':
            return 10.0 ** float(np.random.uniform(np.log10(1.0), np.log10(5.0)))
        elif g_name == 'TEM1':
            return 10.0 ** float(np.random.uniform(np.log10(2.0), np.log10(8.0)))
        elif g_name == 'CDC15':
            return 10.0 ** float(np.random.uniform(np.log10(3.0), np.log10(20.0)))
        elif g_name == 'ESP1':
            return 10.0 ** float(np.random.uniform(np.log10(1.5), np.log10(6.0)))
        elif g_name == 'CDC14':
            return 10.0 ** float(np.random.uniform(np.log10(0.30), np.log10(1.20)))
        elif g_name == 'NET1':
            return 10.0 ** float(np.random.uniform(np.log10(0.15), np.log10(0.60)))
        elif g_name in ['PPX', 'CDC5', 'CDC5_Proxy']:
            return 10.0 ** float(np.random.uniform(np.log10(0.20), np.log10(0.60)))
        elif g_name == 'SWI5':
            return 10.0 ** float(np.random.uniform(np.log10(0.02), np.log10(0.10)))
        else:
            log_alpha = np.random.uniform(np.log10(2.0), np.log10(30.0))
            return 10.0 ** float(log_alpha)
    elif mode == 'MC':
        log_alpha = np.random.uniform(np.log10(1.5), np.log10(30.0))
        return 10.0 ** float(log_alpha)
    elif mode == 'ts':
        if g_name == 'CDC14':
            return float(np.random.uniform(0.05, 0.35))
        u = np.random.beta(a=0.5, b=2.0)
        return float(0.20 * u)
    elif mode == 'DAmP':
        val = np.random.normal(loc=0.20, scale=0.05)
        return float(np.clip(val, 0.05, 0.40))
    elif mode == 'pLOF':
        return float(np.random.uniform(0.10, 0.50))
    elif mode == 'db':
        if g_name == 'CLB2':
            return float(np.random.uniform(0.04, 0.15))
        elif g_name == 'PDS1':
            return float(np.random.uniform(0.0, 0.05))
        elif g_name == 'CLB5':
            return 0.0
        elif g_name == 'SIC1':
            return 0.0
        else:
            return 0.0
    return 1.0

# ==============================================================================
# Part 5: Synthetic Mutant Sampler (MASS v4.4 Implementation)
# ==============================================================================

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

def apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr):
    """
    Explicit, dedicated parameter modification engine per (Gene, Mode) pair
    matching Section 7 of Mutant-Aware Sampling Scheme v1.
    """
    # 1. GAL-OE (Galactose Overexpression: GAL1 promoter drives basal synthesis)
    if m_chosen == 'GAL-OE':
        if g_name == 'CLN2':
            p_mut['ksn2p'] = alpha; p_mut['ksn2pp'] = 0.0
            ovr['ksn2p'] = p_mut['ksn2p']; ovr['ksn2pp'] = 0.0
        elif g_name == 'CLN3':
            p_mut['C0'] = alpha; ovr['C0'] = p_mut['C0']
        elif g_name == 'CLB2':
            p_mut['ksb2p'] = alpha; ovr['ksb2p'] = p_mut['ksb2p']
        elif g_name == 'CLB5':
            p_mut['ksb5p'] = alpha; ovr['ksb5p'] = p_mut['ksb5p']
        elif g_name == 'SIC1':
            p_mut['ksc1p'] = alpha; ovr['ksc1p'] = p_mut['ksc1p']
        elif g_name == 'CDC6':
            p_mut['ksf6p'] = alpha; ovr['ksf6p'] = p_mut['ksf6p']
        elif g_name == 'SWI5':
            p_mut['ksswip'] = alpha; ovr['ksswip'] = p_mut['ksswip']
        elif g_name == 'CDC20':
            p_mut['ks20p'] = alpha; ovr['ks20p'] = p_mut['ks20p']
        elif g_name == 'CDC14':
            p_mut['ks14'] = alpha; ovr['ks14'] = p_mut['ks14']
        elif g_name == 'NET1':
            p_mut['ksnet'] = alpha; ovr['ksnet'] = p_mut['ksnet']
        elif g_name == 'TEM1':
            p_mut['Tem1T'] = alpha; ovr['Tem1T'] = p_mut['Tem1T']
        elif g_name == 'CDC15':
            p_mut['Cdc15T'] = alpha; ovr['Cdc15T'] = p_mut['Cdc15T']
        elif g_name in ['PPX', 'CDC5', 'CDC5_Proxy']:
            p_mut['ksppx'] = alpha; ovr['ksppx'] = p_mut['ksppx']
        elif g_name == 'PDS1':
            p_mut['kspdsp'] = alpha; ovr['kspdsp'] = p_mut['kspdsp']
        elif g_name == 'ESP1':
            p_mut['ESP1T'] = alpha; ovr['ESP1T'] = p_mut['ESP1T']

    # 2. MC (Multi-copy: native promoter gene dosage amplification)
    elif m_chosen == 'MC':
        if g_name == 'CLN2':
            ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'CLN3':
            ovr['C0'] = p_wt['C0'] * alpha
        elif g_name == 'BCK2':
            ovr['B0'] = p_wt['B0'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha; ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
        elif g_name == 'CLB5':
            ovr['ksb5p'] = p_wt['ksb5p'] * alpha; ovr['ksb5pp'] = p_wt['ksb5pp'] * alpha
        elif g_name == 'SIC1':
            ovr['ksc1p'] = p_wt['ksc1p'] * alpha; ovr['ksc1pp'] = p_wt['ksc1pp'] * alpha
        elif g_name == 'CDC6':
            ovr['ksf6p'] = p_wt['ksf6p'] * alpha; ovr['ksf6pp'] = p_wt['ksf6pp'] * alpha; ovr['ksf6ppp'] = p_wt['ksf6ppp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha; ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            ovr['ks14'] = p_wt['ks14'] * alpha
        elif g_name == 'TEM1':
            ovr['Tem1T'] = p_wt['Tem1T'] * alpha
        elif g_name == 'CDC15':
            ovr['Cdc15T'] = p_wt['Cdc15T'] * alpha

    # 3. DAmP & pLOF (Promoter / mRNA Abundance Loss)
    elif m_chosen in ['DAmP', 'pLOF']:
        if g_name == 'CDC6':
            ovr['ksf6p'] = p_wt['ksf6p'] * alpha; ovr['ksf6pp'] = p_wt['ksf6pp'] * alpha; ovr['ksf6ppp'] = p_wt['ksf6ppp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha; ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            ovr['ks14'] = p_wt['ks14'] * alpha
        elif g_name == 'NET1':
            ovr['ksnet'] = p_wt['ksnet'] * alpha
        elif g_name == 'TEM1':
            ovr['Tem1T'] = p_wt['Tem1T'] * alpha
        elif g_name == 'CDC15':
            ovr['Cdc15T'] = p_wt['Cdc15T'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha; ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
        elif g_name == 'CDC28':
            cdk_eps = ['ec1k2', 'ec1n2', 'ec1b5', 'ec1b2', 'ec1n3', 'ef6k2', 'ef6n2', 'ef6b5', 'ef6b2', 'ef6n3', 'eicdhn2', 'eicdhn3', 'eicdhb5', 'eicdhb2', 'esbfn2', 'esbfn3', 'esbfb5', 'eorib5', 'eorib2', 'ebudn2', 'ebudn3', 'ebudb5']
            for ep in cdk_eps: ovr[ep] = p_wt[ep] * alpha

    # 4. ts (Temperature-Sensitive Thermal Inactivation)
    elif m_chosen == 'ts':
        if g_name == 'CLN2':
            ovr['ksn2p'] = p_wt['ksn2p'] * alpha; ovr['ksn2pp'] = p_wt['ksn2pp'] * alpha
        elif g_name == 'CLB2':
            ovr['ksb2p'] = p_wt['ksb2p'] * alpha; ovr['ksb2pp'] = p_wt['ksb2pp'] * alpha
        elif g_name == 'CDC20':
            ovr['ks20p'] = p_wt['ks20p'] * alpha; ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            # CDC14-ts dual physiological representations:
            # Mode A (50% prob): Restrictive temp (37°C) complete arrest: ks14 = 0.0, catalytic rates nominal WT
            # Mode B (50% prob): Semi-permissive temp (30-34°C) catalytic attenuation: ks14 = WT, 4 catalytic rates scaled by alpha in [0.05, 0.35]
            if np.random.rand() < 0.50:
                ovr['ks14'] = 0.0
            else:
                ovr['ks14'] = p_wt['ks14']
                ovr['kppc1'] = p_wt['kppc1'] * alpha
                ovr['kppf6'] = p_wt['kppf6'] * alpha
                ovr['kaswi'] = p_wt['kaswi'] * alpha
                ovr['kacdhpp'] = p_wt['kacdhpp'] * alpha
        elif g_name == 'NET1':
            # net1-ts / tab6-1: kasrent & kasrentp scaled by alpha (at alpha=0.05, gives 10.0 and 0.05)
            ovr['kasrent'] = p_wt['kasrent'] * alpha
            ovr['kasrentp'] = p_wt['kasrentp'] * alpha
        elif g_name in ['PPX', 'CDC5_Proxy']:
            # ppx-ts / cdc5-ts (#112): kppnetpp reduced from WT 3.0 to 0.05
            ovr['kppnetpp'] = 0.05
        elif g_name == 'CDC15':
            ovr['Cdc15T'] = 1.0; ovr['kkpnetpp'] = 0.0
        elif g_name == 'TEM1':
            ovr['Tem1T'] = 0.0; ovr['ka15pp'] = 0.002
        elif g_name == 'ESP1':
            ovr['kasesp'] = 0.1; ovr['kdiesp'] = 0.001
        elif g_name == 'APC_Core':
            ovr['ks20p'] = 0.0; ovr['ks20pp'] = 0.0; ovr['kscdh'] = 0.0; ovr['init_CDH1T'] = 0.0; ovr['init_CDH1'] = 0.0
        elif g_name == 'Mcm1':
            ovr['kamcm'] = 0.13
        elif g_name == 'CDC28':
            cdk_eps = ['ec1k2', 'ec1n2', 'ec1b5', 'ec1b2', 'ec1n3', 'ef6k2', 'ef6n2', 'ef6b5', 'ef6b2', 'ef6n3', 'eicdhn2', 'eicdhn3', 'eicdhb5', 'eicdhb2', 'esbfn2', 'esbfn3', 'esbfb5', 'eorib5', 'eorib2', 'ebudn2', 'ebudn3', 'ebudb5']
            for ep in cdk_eps: ovr[ep] = p_wt[ep] * alpha

    # 5. db (Destruction Box Degradation Resistance)
    elif m_chosen == 'db':
        if g_name == 'CLB2':
            # CLB2-dbΔ: kdb2P=0.0 (D-box lost, Cdc20 degradation eliminated), kdb2pp scaled by alpha (KEN-box residual, alpha~0.085 -> 0.034)
            ovr['kdb2P'] = 0.0; ovr['kdb2pp'] = p_wt['kdb2pp'] * alpha
        elif g_name == 'CLB5':
            ovr['kdb5pp'] = 0.0
        elif g_name == 'PDS1':
            ovr['kd2pdspp'] = 0.0; ovr['kd3pdspp'] = 0.0
        elif g_name == 'SIC1':
            ovr['kd3c1'] = 0.0
        elif g_name in ['PPX', 'CDC5_Proxy']:
            ovr['kdppxpp'] = 0.0

    # 6. KO (Knockout: complete loss of gene function)
    elif m_chosen == 'KO':
        if g_name == 'CLN2':
            ovr['ksn2pp'] = 0.0; ovr['init_CLN2'] = 0.0
        elif g_name == 'CLN3':
            ovr['C0'] = 0.0
        elif g_name == 'BCK2':
            ovr['B0'] = 0.0
        elif g_name == 'CLB2':
            ovr['ksb2p'] = 0.0; ovr['ksb2pp'] = 0.0; ovr['init_CLB2'] = 0.0
        elif g_name == 'CLB5':
            ovr['ksb5p'] = 0.0; ovr['ksb5pp'] = 0.0; ovr['init_CLB5'] = 0.0
        elif g_name == 'SIC1':
            ovr['ksc1p'] = 0.0; ovr['ksc1pp'] = 0.0; ovr['init_SIC1'] = 0.0
        elif g_name in ['CDC6', 'pDEL']:
            ovr['ksf6p'] = 0.0; ovr['ksf6pp'] = 0.0; ovr['ksf6ppp'] = 0.0; ovr['init_CDC6'] = 0.0
        elif g_name == 'SWI5':
            ovr['ksswip'] = 0.0; ovr['ksswipp'] = 0.0
        elif g_name == 'CDC20':
            ovr['ks20p'] = 0.0; ovr['ks20pp'] = 0.0
        elif g_name == 'CDH1':
            ovr['kscdh'] = 0.0; ovr['init_CDH1T'] = 0.0; ovr['init_CDH1'] = 0.0
        elif g_name == 'TEM1':
            ovr['Tem1T'] = 0.0; ovr['ka15pp'] = 0.0; ovr['init_Tem1'] = 0.0
        elif g_name == 'CDC15':
            ovr['Cdc15T'] = 0.0; ovr['kkpnetpp'] = 0.0; ovr['init_Cdc15'] = 0.0
        elif g_name in ['PPX', 'CDC5_Proxy']:
            ovr['ksppx'] = 0.0; ovr['init_PPX'] = 0.0
        elif g_name == 'PDS1':
            ovr['ks1pdspp'] = 0.0; ovr['ks2pdspp'] = 0.0; ovr['init_PDS1'] = 0.0
        elif g_name == 'SBF_MBF':
            ovr['kasbf'] = 0.0; ovr['ksn2pp'] = 0.0; ovr['ksb5pp'] = 0.0
        elif g_name == 'Mcm1':
            ovr['kamcm'] = 0.0
        elif g_name == 'ORI_BUD_SPN':
            ovr['ksori'] = 0.0; ovr['ksbud'] = 0.0; ovr['ksspn'] = 0.0
        elif g_name == 'CDC28':
            cdk_eps = ['ec1k2', 'ec1n2', 'ec1b5', 'ec1b2', 'ec1n3', 'ef6k2', 'ef6n2', 'ef6b5', 'ef6b2', 'ef6n3', 'eicdhn2', 'eicdhn3', 'eicdhb5', 'eicdhb2', 'esbfn2', 'esbfn3', 'esbfb5', 'eorib5', 'eorib2', 'ebudn2', 'ebudn3', 'ebudb5']
            for ep in cdk_eps: ovr[ep] = 0.0

    # 7. Structural Custom Modes (CA, APC-A, CSL, CKO, TAB6-1, scc1-RR, PM, MDT)
    elif m_chosen == 'CA':
        if g_name == 'CDH1':
            ovr['kicdhpp'] = 0.0; ovr['kscdh'] = 0.03
        elif g_name == 'CLB2':
            ovr['kdib2'] = 0.0
        elif g_name == 'SBF_MBF':
            ovr['is_sbf_ca'] = 1; ovr['kasbf'] = 0.38; ovr['kisbfpp'] = 0.0
    elif m_chosen == 'CSL':
        ovr['Vi20'] = 0.01; ovr['Vi20_active'] = 0.01
    elif m_chosen == 'CKO':
        ovr['kbub2'] = 0.0; ovr['BUB2_active'] = 0.0; ovr['init_BUB2'] = 0.0
    elif m_chosen == 'APC-A' or (g_name in ['CDC20', 'APC_Core'] and m_chosen == 'APC-A'):
        ovr['ka20pp'] = 0.0
    elif m_chosen == 'TAB6-1':
        ovr['kasrent'] = 10.0; ovr['kasrentp'] = 0.05
    elif m_chosen == 'scc1-RR':
        ovr['kasesp'] = 0.0
    elif m_chosen == 'PM':
        ovr['kd2c1'] = 0.0; ovr['kd3c1'] = 0.0
    elif m_chosen == 'pDEL':
        ovr['ksf6p'] = 0.0; ovr['ksf6pp'] = 0.0; ovr['ksf6ppp'] = 0.0
    elif m_chosen == 'perturb' and g_name == 'ORI_BUD_SPN':
        for k in ['ksori', 'ksbud', 'ksspn']:
            scale = float(np.clip(np.random.normal(1.0, 0.20), 0.1, 3.0))
            ovr[k] = p_wt[k] * scale
    elif m_chosen == 'cdc24Δ':
        ovr['ksbud'] = 0.0
    elif m_chosen == 'cdc45Δ':
        ovr['ksori'] = 0.0
    elif m_chosen == 'spindle_defect':
        ovr['ksspn'] = 0.0
    elif m_chosen == 'MDT':
        ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0]))

def sample_mass_mutant(level=1):
    """
    Generate a synthetic mutant sample by selecting binding groups and applying
    Section 7 master table lookups, Section 2.3 biophysical distributions,
    Section 2.4 carbon decoupling & Section 2.4.2 biological rescue motifs.
    Returns (ovr_dict, mutant_description_str).
    """
    p_wt = get_default_params()
    p_mut = p_wt.copy()
    ovr = {}

    has_gal_oe = False
    mut_descriptions = []

    # Step 1 & 2: Check for Biological Rescue Motif (Level 3 & 4 only, 50% probability)
    use_rescue_motif = (level in [3, 4] and np.random.rand() < 0.50)

    if use_rescue_motif:
        motif_type = np.random.choice(['M_EXIT', 'G1_S', 'ANAPHASE', 'PREMATURE'])
        selected_mut_pairs = []

        if motif_type == 'M_EXIT':
            driver_choices = [('CLB2', 'db'), ('CLB2', 'GAL-OE'), ('CDC14', 'ts'), ('TEM1', 'ts')]
            driver = driver_choices[np.random.choice(len(driver_choices))]
            suppressors_pool = [
                ('SIC1', np.random.choice(['MC', 'GAL-OE'])),
                ('CLB5', 'KO'),
                ('CDC6', 'MC'),
                ('CDC15', 'GAL-OE'),
                ('NET1', 'TAB6-1')
            ]
            k_supp = min(level - 1, len(suppressors_pool))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]
            if np.random.rand() < 0.30:
                ovr['mdt'] = float(np.random.choice([150.0, 160.0]))

        elif motif_type == 'G1_S':
            driver = ('CLN2', 'KO')
            suppressors_pool = [
                ('CLN3', 'KO'),
                ('BCK2', 'KO'),
                ('CLB5', 'GAL-OE'),
                ('CLB2', 'GAL-OE'),
                ('SIC1', 'KO')
            ]
            k_supp = min(level - 1, len(suppressors_pool))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]

        elif motif_type == 'ANAPHASE':
            driver = ('PDS1', 'db') if np.random.rand() < 0.6 else ('CDC20', 'ts')
            suppressors_pool = [
                ('ESP1', np.random.choice(['GAL-OE', 'MC'])),
                ('SCC1', 'scc1-RR'),
                ('MAD2', 'CSL'),
                ('BUB2', 'CKO')
            ]
            k_supp = min(level - 1, len(suppressors_pool))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]

        else: # PREMATURE
            driver = ('SIC1', 'KO')
            suppressors_pool = [
                ('CDH1', 'KO'),
                ('CDC20', np.random.choice(['GAL-OE', 'MC'])),
                ('CDC6', 'pDEL'),
                ('SWI5', 'KO')
            ]
            k_supp = min(level - 1, len(suppressors_pool))
            chosen_indices = np.random.choice(len(suppressors_pool), size=k_supp, replace=False)
            selected_mut_pairs = [driver] + [suppressors_pool[idx] for idx in chosen_indices]

        # Apply the structured rescue mutations
        for g_name, m_chosen in selected_mut_pairs:
            mut_descriptions.append(f"{g_name}[{m_chosen}]")
            if m_chosen == 'GAL-OE':
                has_gal_oe = True
            alpha = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
            apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    else:
        # Standard unconstrained exploratory sampling
        if level == 1: ki = 1
        elif level == 2: ki = 2
        elif level == 3: ki = np.random.randint(3, 6)
        else: ki = np.random.randint(4, 7)

        group_keys = list(BINDING_GROUPS.keys())
        group_weights = [BINDING_GROUPS[k]['P_bio_G'] for k in group_keys]
        selected_g_keys = np.random.choice(group_keys, size=min(ki, len(group_keys)), replace=False, p=group_weights)

        for g_key in selected_g_keys:
            g_info = BINDING_GROUPS[g_key]
            g_name = g_info['name']

            modes = list(g_info['modes'].keys())
            m_weights = [g_info['modes'][m] for m in modes]
            m_weights = np.array(m_weights) / sum(m_weights)
            m_chosen = np.random.choice(modes, p=m_weights)

            comp_mode = None
            if g_name in ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']:
                synth_modes = [m for m in ['GAL-OE', 'MC'] if m in modes]
                struc_modes = [m for m in ['db', 'PM', 'CA'] if m in modes]
                if m_chosen in synth_modes and struc_modes and np.random.rand() < 0.25:
                    comp_mode = np.random.choice(struc_modes)
                elif m_chosen in struc_modes and synth_modes and np.random.rand() < 0.25:
                    comp_mode = np.random.choice(synth_modes)

            if comp_mode:
                mut_descriptions.append(f"{g_name}[{m_chosen}+{comp_mode}]")
                if m_chosen == 'GAL-OE' or comp_mode == 'GAL-OE':
                    has_gal_oe = True

                alpha1 = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
                apply_gene_mutation(g_name, m_chosen, alpha1, p_wt, p_mut, ovr)
                alpha2 = sample_scaling_factor(comp_mode, g_name) if comp_mode in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
                apply_gene_mutation(g_name, comp_mode, alpha2, p_wt, p_mut, ovr)
            else:
                mut_descriptions.append(f"{g_name}[{m_chosen}]")
                if m_chosen == 'GAL-OE':
                    has_gal_oe = True

                alpha = sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
                apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    # Environmental Carbon Source Decoupling (Section 2.4.1)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    else:
        if 'mdt' not in ovr:
            carbon_choice = np.random.choice(['Glc', 'Gal', 'Raff'], p=[0.75, 0.18, 0.07])
            if carbon_choice == 'Gal':
                ovr['mdt'] = 150.0
            elif carbon_choice == 'Raff':
                ovr['mdt'] = 160.0
            else:
                ovr['mdt'] = 90.0

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str

# ==============================================================================
# Part 6: Complete Dataset Generation & .npz File Export (lhs_v1_origin.py Format)
# ==============================================================================

if os.name == 'nt':
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    OUTPUT_DIR = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    pass

DATASET_NAME = "mass_dataset_210min_500steps_dual_labels.npz"
DATASET_PATH = os.path.join(OUTPUT_DIR, DATASET_NAME)

def generate_mass_dataset(n_synthetic=20000, max_attempts=None, num_workers=32):
    """
    Generate complete dataset (126 Real Mutants + n_synthetic MASS Mutants)
    and export into single .npz file with exact field alignment to lhs_v1_origin.py.
    """
    print("=" * 80)
    print("MASS v4.4 Dataset Generator Protocol (lhs_v1_origin.py Export Alignment)")
    print("=" * 80)

    if max_attempts is None:
        max_attempts = max(50000, n_synthetic * 10)

    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    all_results = {}
    all_tasks = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}
    ordered_names = []

    # Phase 1: Prepare 126 Real Mutants (Exact Appendix B Table Names & Rules)
    print(f"\n[Phase 1/4] Preparing D_real (126 Historical Real Mutants from Appendix B)...")
    for mid, (mname, movr) in REAL_MUTANTS.items():
        sample_name = f"{mid}_{mname}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = movr
    print(f" -> Loaded {len(REAL_MUTANTS)} Real Mutants.")

    # Phase 2: Sample MASS Synthetic Mutants
    print(f"\n[Phase 2/4] Sampling {n_synthetic} Synthetic Mutants (MASS v4.4 Protocol)...")
    n_l1 = int(math.floor(0.40 * n_synthetic))
    n_l2 = int(math.floor(0.30 * n_synthetic))
    n_l3 = int(math.floor(0.20 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3

    print(f" -> Target Level Distribution: L1 (ki=1): {n_l1} | L2 (ki=2): {n_l2} | L3 (ki=3~5): {n_l3} | L4 (ki=4~6): {n_l4}")

    level_assignments = [1]*n_l1 + [2]*n_l2 + [3]*n_l3 + [4]*n_l4
    np.random.shuffle(level_assignments)

    attempts = 0
    synth_success = 0
    duplicates_blocked = 0

    while synth_success < n_synthetic and attempts < max_attempts:
        attempts += 1
        level = level_assignments[synth_success]
        ovr, desc_str = sample_mass_mutant(level=level)

        # 136-dim Parameter Deduplication Check against D_real
        if is_duplicate_real_mutant(ovr, defaults):
            duplicates_blocked += 1
            continue

        synth_success += 1
        sample_name = f"MASS_{desc_str}_L{level}_{synth_success}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = ovr

    print(f" -> Sampling Done: {synth_success} synthetic tasks generated in {attempts} attempts.")
    print(f"    - Filter [D_real Deduplication]: {duplicates_blocked} candidates blocked (duplicate with historical real mutants).")
    print(f" -> Total Simulation Queue: {len(ordered_names)} tasks ({len(REAL_MUTANTS)} Real + {synth_success} Synthetic).")

    # Phase 3: Parallel ODE Simulation Engine
    if num_workers is None:
        num_workers = 32

    print(f"\n[Phase 3/4] Simulating {len(ordered_names)} Mutant Trajectories on {num_workers} CPU Workers...")
    t_start = time.time()
    sim_counts = {
        'real_success': 0, 'synth_success': 0,
        'discarded_real': 0, 'discarded_synth': 0,
        'failed_real': 0, 'failed_synth': 0
    }

    log_step = max(50, len(ordered_names) // 40)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_name = {
            executor.submit(simulate_mutant, name, all_tasks[name], 210, 500): name
            for name in ordered_names
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_name)):
            name = future_to_name[future]
            is_synth = "_L" in name
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result(timeout=15.0)
                if y_uniform is not None:
                    all_results[m_name] = y_uniform
                    bio_labels_dict[m_name] = m_bio_label
                    pattern_labels_dict[m_name] = m_pattern_label
                    if is_synth: sim_counts['synth_success'] += 1
                    else: sim_counts['real_success'] += 1
                elif m_bio_label == "Discarded_Mass":
                    if is_synth: sim_counts['discarded_synth'] += 1
                    else: sim_counts['discarded_real'] += 1
                else:
                    if is_synth: sim_counts['failed_synth'] += 1
                    else: sim_counts['failed_real'] += 1
            except Exception:
                if is_synth: sim_counts['failed_synth'] += 1
                else: sim_counts['failed_real'] += 1

            if (idx + 1) % log_step == 0 or (idx + 1) == len(ordered_names):
                elapsed = max(0.001, time.time() - t_start)
                speed = (idx + 1) / elapsed
                valid_total = sim_counts['real_success'] + sim_counts['synth_success']
                disc_total = sim_counts['discarded_real'] + sim_counts['discarded_synth']
                fail_total = sim_counts['failed_real'] + sim_counts['failed_synth']
                pct = ((idx + 1) / len(ordered_names)) * 100
                print(f"  |-- [{idx + 1:5d}/{len(ordered_names)}] ({pct:5.1f}%) | Valid: {valid_total:5d} (Real:{sim_counts['real_success']:3d}, Synth:{sim_counts['synth_success']:5d}) | Discarded(Mass>10): {disc_total:4d} | Failed: {fail_total:3d} | Speed: {speed:5.1f} samples/s", flush=True)

    elapsed_time = time.time() - t_start
    print(f" -> Simulation Completed in {elapsed_time:.2f}s ({len(ordered_names) / max(0.001, elapsed_time):.1f} samples/s).", flush=True)

    # Phase 4: Format Packaging & .npz Export (Strictly Aligned with lhs_v1_origin.py)
    print(f"\n[Phase 4/4] Packaging & Saving Dataset Array Files...", flush=True)
    ordered_tensors = []
    final_names = []
    final_params = []
    final_bio_labels = []
    final_pattern_labels = []

    real_final_names = []
    synth_final_names = []
    real_bio_labels = []
    synth_bio_labels = []
    real_pattern_labels = []
    synth_pattern_labels = []
    synth_levels = []

    for name in ordered_names:
        if name in all_results:
            ordered_tensors.append(all_results[name])
            final_names.append(name)
            final_bio_labels.append(bio_labels_dict[name])
            final_pattern_labels.append(pattern_labels_dict[name])

            if "_L" in name:
                synth_final_names.append(name)
                synth_bio_labels.append(bio_labels_dict[name])
                synth_pattern_labels.append(pattern_labels_dict[name])
                # Extract level
                try:
                    lvl = int(name.split("_L")[1].split("_")[0])
                    synth_levels.append(lvl)
                except Exception:
                    synth_levels.append(-1)
            else:
                real_final_names.append(name)
                real_bio_labels.append(bio_labels_dict[name])
                real_pattern_labels.append(pattern_labels_dict[name])

            overrides = all_tasks[name]
            p_vec = [overrides.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)

    all_tensors = np.array(ordered_tensors)
    all_params = np.array(final_params)
    all_bio_labels = np.array(final_bio_labels)
    all_pattern_labels = np.array(final_pattern_labels)

    os.makedirs(os.path.dirname(os.path.abspath(DATASET_PATH)), exist_ok=True)
    np.savez_compressed(DATASET_PATH,
                        data=all_tensors,
                        params=all_params,
                        param_names=param_keys,
                        mutant_names=final_names,
                        labels=all_bio_labels,
                        pattern_labels=all_pattern_labels)

    total_valid = len(final_names)
    total_disc = sim_counts['discarded_real'] + sim_counts['discarded_synth']
    total_fail = sim_counts['failed_real'] + sim_counts['failed_synth']

    print("=" * 85, flush=True)
    print(f" MASS v4.4 DATASET GENERATION SUMMARY REPORT", flush=True)
    print("=" * 85, flush=True)
    print(f"Dataset Export Path               : {DATASET_PATH}", flush=True)
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}", flush=True)
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}", flush=True)
    print(f"Total Parameter Keys Count        : {len(param_keys)}", flush=True)
    print(f"Total Tasks Simulated             : {len(ordered_names)}", flush=True)
    print(f"Total Validated Samples Saved     : {total_valid} ({total_valid / len(ordered_names) * 100:.2f}% yield)", flush=True)
    print("-" * 85, flush=True)

    # 1. Filtering & Quality Audit
    print("\n[1. Filtering & Quality Control Audit]", flush=True)
    print(f"  * Filter 1 [D_real Deduplication Blocked] : {duplicates_blocked:5d} synthetic candidates")
    print(f"  * Filter 2 [Mass Divergence (Mass > 10)]  : {total_disc:5d} tasks (Real: {sim_counts['discarded_real']}, Synth: {sim_counts['discarded_synth']})")
    print(f"  * Filter 3 [ODE Solver Failures]          : {total_fail:5d} tasks (Real: {sim_counts['failed_real']}, Synth: {sim_counts['failed_synth']})")
    print(f"  * Valid Sample Yield Rate                 : {total_valid:5d} / {len(ordered_names)} ({total_valid / len(ordered_names) * 100:.1f}%)")

    # 2. Sample Composition
    print("\n[2. Sample Composition Breakdown]", flush=True)
    print(f"  * Real Mutants (Appendix B Table S5)      : {len(real_final_names):5d} / {len(REAL_MUTANTS)} ({len(real_final_names)/len(REAL_MUTANTS)*100:.1f}%)")
    print(f"  * Synthetic Mutants (MASS v4.4 Multi-Level): {len(synth_final_names):5d} / {synth_success} ({len(synth_final_names)/synth_success*100:.1f}%)")
    if synth_levels:
        u_lvl, c_lvl = np.unique(synth_levels, return_counts=True)
        for l_idx, count in zip(u_lvl, c_lvl):
            target_n = [n_l1, n_l2, n_l3, n_l4][l_idx-1] if 1 <= l_idx <= 4 else -1
            print(f"    - Level {l_idx} Synthetic Mutants         : {count:5d} / {target_n} ({count/target_n*100:.1f}%)")

    # 3. Mass Growth Pattern Distribution (Real vs Synth vs Total)
    print("\n[3. Mass Growth Pattern Labels Distribution (Real vs Synth)]", flush=True)
    pattern_categories = [
        "Pattern_A: Multi-Division (>=2)",
        "Pattern_B: Single-Division (==1)",
        "Pattern_C: Exponential-Arrest (==0)"
    ]
    header = f"  {'Pattern Label':<36} | {'Real Mutants':<15} | {'Synthetic Mutants':<18} | {'Overall Dataset':<18}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for pat in pattern_categories:
        c_r = real_pattern_labels.count(pat)
        pct_r = (c_r / max(1, len(real_pattern_labels))) * 100
        c_s = synth_pattern_labels.count(pat)
        pct_s = (c_s / max(1, len(synth_pattern_labels))) * 100
        c_t = list(all_pattern_labels).count(pat)
        pct_t = (c_t / max(1, len(all_pattern_labels))) * 100
        print(f"  {pat:<36} | {c_r:5d} ({pct_r:5.1f}%)   | {c_s:7d} ({pct_s:5.1f}%)    | {c_t:7d} ({pct_t:5.1f}%)")

    # 4. Biological Viability Labels Distribution (Real vs Synth vs Total)
    print("\n[4. Biological Viability Labels Distribution (Real vs Synth)]", flush=True)
    all_bio_unique = sorted(list(set(all_bio_labels)))
    v_header = f"  {'Viability Label':<32} | {'Real Mutants':<15} | {'Synthetic Mutants':<18} | {'Overall Dataset':<18}"
    print(v_header)
    print("  " + "-" * (len(v_header) - 2))
    for bio in all_bio_unique:
        c_r = real_bio_labels.count(bio)
        pct_r = (c_r / max(1, len(real_bio_labels))) * 100
        c_s = synth_bio_labels.count(bio)
        pct_s = (c_s / max(1, len(synth_bio_labels))) * 100
        c_t = list(all_bio_labels).count(bio)
        pct_t = (c_t / max(1, len(all_bio_labels))) * 100
        print(f"  {bio:<32} | {c_r:5d} ({pct_r:5.1f}%)   | {c_s:7d} ({pct_s:5.1f}%)    | {c_t:7d} ({pct_t:5.1f}%)")

    print("\n" + "=" * 85, flush=True)
    print("Sample Mutant Names Demonstration:", flush=True)
    for name in real_final_names[:3]:
        print(f"  - [Real]  {name}", flush=True)
    for name in synth_final_names[:5]:
        print(f"  - [Synth] {name}", flush=True)
    print("=" * 85 + "\n", flush=True)

    return DATASET_PATH

if __name__ == '__main__':
    # =========================================================================
    # 用户可调参数配置区 (随时在此处直接修改，或通过命令行参数指定)
    # =========================================================================
    NUM_SAMPLES = 100000   # 生成的采样样本数量
    NUM_WORKERS = 60      # 并行调用的 CPU 核心数 (当前服务器共有 64 核，设为 32 可预留一半核心给其他用户)
    # =========================================================================

    import argparse
    parser = argparse.ArgumentParser(description="MASS v4.4 Dataset Generator")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of synthetic samples to generate")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel CPU worker processes")
    args = parser.parse_args()

    print(f"Executing MASS v4.4 Dataset Generator (Samples: {args.samples}, Workers: {args.workers})...", flush=True)
    generate_mass_dataset(n_synthetic=args.samples, num_workers=args.workers)
    print("\n[SUCCESS] MASS Dataset Generator Execution Finished Cleanly!", flush=True)



