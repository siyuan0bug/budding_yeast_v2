"""
MASS (Mutant-Aware Stratified Sampling v5.1) Dataset Generator - UNIFIED MAIN
================================================================================
This is the single canonical engine + sampler (2026-09-06; v5.1 2026-09-08
real-mutant failure audit: R1-R5 composition rebalancing + ts/db anchor
envelopes). Fully self-contained: no imports from sibling project modules.
The structure-aware skeleton (uniform gene weights with degree/engine
ablation arms, structure-class quotas with random-arm mixing, L1 grid
rotation, mode-frequency arms, touch-set conflict resolution) is inlined in
Part 5.5 below.

Lineage (retired versions archived under codes/a6000v1/, never modify):
  v1  mass_dataset_generator.py  - rescue-motif engine (REMOVED: data leak,
                                   bans 3/4 of the zero-leakage spec)
  v1r mass_random_sampler.py     - motif banned, pure-random control
  v1w mass_wt_blind_sampler.py   - initial formula-derived wt-blind sampler
  v2  (intermediate)            - engine delegating to the sampler module
  v3  THIS FILE                 - v2 engine + sampler skeleton merged into
                                   one leak-free, dependency-free file

Zero-leakage selection rules (spec: Zero_Data_Leakage_and_Sampling_Specification.md v2.3):
  - gene weights  : uniform over 27 modules (v5.1 R2; degree+1 hub prior and
                    engine P_bio_G retained as ablation arms)
  - L1            : shuffled 84-cell (gene, mode) grid rotation (coverage)
  - L2/L3/L4      : 50% structure-class quotas from the 36-edge module graph
                    (C-A/C-B/C-C/C-D; cascade/star/antag_core/orthogonal)
                    + 50% i.i.d. uniform genes (v5.1 R3 softening)
  - mode          : --mode_freq {class, flat, role, s5} (class = mode-class
                    quota arm, v5.1 R1 DEFAULT; flat = uniform anti-leak;
                    role = Tier 2/3 affinity; s5 = LEAKS BY DESIGN control)
  - conflicts     : touch-set collision check + cross-redraw (Sect 3.1.6)
  - composite     : same-gene expression x structure dual mode, 25% on
                    mechanism-eligible genes (COMPOSITE_ELIGIBLE below)
  - carbon rule   : GAL -> mdt=150 forced; else {90,150,160} w.p. (0.75,0.18,0.07)
  - level quotas  : 40/30/20/10 mechanized (spec rule 4)
Strength values: sample_scaling_factor (Tier 2 literature bands, spec ch.6).
Allele-residual pairs are reparameterized as WT x f per spec sect 8.5
(PATCH channel, v2.1). v5.1: TEM1/ESP1/NET1 ts use anchor envelopes
LogU[anchor/4, anchor x 2] (declared E-Anchor exception, spec §5.5/§11.4);
PDS1-db residual band U[0, 0.01]; MC integer support {2..50}.
v5.2 (ts/db dedup-collapse audit fixes): (a) boundary-anchor buffer
f ~ U[0, 0.01] for the exact-zero ts/db/APC-A/TAB6-1 alleles whose fixed
constants collapsed L1 cells onto real-mutant points (CDC20-ts vs #73,
CDC14-ts Mode A vs #99, CLB5-db vs #68, TAB6-1 vs #108, APC-A vs #115;
70-84% dedup interception at Glc -> zero Glc-condition axis coverage);
(b) CLB2-db fold alpha ~ U[0.02125, 0.170] (2.1% - 17% WT) -> kdb2pp = 0.4*alpha
    in [0.0085, 0.068] = [anchor/4, anchor x 2] (real anchor kdb2pp=0.034 / alpha=0.085
    sits cleanly interior at 42.9th percentile, 9 strains #55-#62);
(c) TAB6-1 unified with the NET1-ts anchor envelope (same allele);
(d) MDT-cell carbon weights aligned to R5 (0.75,0.18,0.07).

Runs unchanged on the A6000 server (Linux: 60 workers, server output path)
and on a local Windows machine (auto: local core count, script-dir output);
--workers / --output_dir override the auto-detected defaults.

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
try:
    from .simulation_guards import bounded_simulation, validate_segment, bounded_completed
except ImportError:  # support python data/mutant_aware.py
    from simulation_guards import bounded_simulation, validate_segment, bounded_completed
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

_defaults = get_default_params()
_defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
PARAM_KEYS = sorted([k for k in _defaults.keys() if k != 'is_mutant_104'])

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

@bounded_simulation
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
    if p.get('Tem1T', 1.0) == 0.0: y0[nTEM1] = 0.0
    if p.get('Cdc15T', 1.0) == 0.0: y0[nCDC15] = 0.0

    tr, yr = [0], [y0]; eh = []; bd = []; md = []; t0 = 0; lc = 0
    try:
        while t0 < tf and lc < 300:
            lc += 1
            sol = solve_ivp(eqns, [t0, tf], y0, args=(p,), method='BDF', 
                            events=[event1, event2, event3, event4], rtol=1e-4, atol=1e-6)
            validate_segment(sol, t0, tf, len(y0))
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
    if (len(tr) < 2 or tr[-1] < tf or not np.isfinite(yr).all() or (np.diff(tr) <= 0).any()):
        return name, None, "Inviable: Numerical Failure", "Pattern_C: Exponential-Arrest (==0)"
    # Never synthesize an unintegrated tail by extrapolation.
    interp = interp1d(tr, yr, kind='linear', axis=1, bounds_error=True)
    yu = interp(np.linspace(0, tf, ns))
    if not np.isfinite(yu).all():
        return name, None, "Inviable: Numerical Failure", "Pattern_C: Exponential-Arrest (==0)"
    
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
        'name': 'TEM1', 'params': ['Tem1T', 'ka15pp', 'init_Tem1'],
        'P_bio_G': 0.035,
        'modes': {'KO': 0.200, 'GAL-OE': 0.200, 'MC': 0.200, 'ts': 0.200, 'DAmP': 0.200}
    },
    'G14': {
        'name': 'CDC15', 'params': ['Cdc15T', 'kkpnetpp', 'init_Cdc15'],
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

def sample_scaling_factor(mode, g_name=None, *, rng=None):
    """Use the caller's stream; omitted rng preserves legacy global sampling."""
    rng = np.random if rng is None else rng
    if mode == 'GAL-OE':
        if g_name == 'CDC20':
            return 10.0 ** float(rng.uniform(np.log10(0.50), np.log10(12.0)))
        elif g_name == 'CLB2':
            return 10.0 ** float(rng.uniform(np.log10(0.002), np.log10(1.50)))
        elif g_name == 'SIC1':
            return 10.0 ** float(rng.uniform(np.log10(0.03), np.log10(0.36)))
        elif g_name == 'CDC6':
            return 10.0 ** float(rng.uniform(np.log10(0.10), np.log10(0.60)))
        elif g_name == 'CLB5':
            return 10.0 ** float(rng.uniform(np.log10(0.004), np.log10(0.030)))
        elif g_name == 'PDS1':
            return 10.0 ** float(rng.uniform(np.log10(0.02), np.log10(0.25)))
        elif g_name == 'CLN2':
            # all 7 real GAL-CLN2 constructs: ksn2p=0.12; floor = anchor/2
            # (0.03 = anchor/4 is implausibly weak for a GAL1 construct)
            return 10.0 ** float(rng.uniform(np.log10(0.06), np.log10(0.30)))
        elif g_name == 'CLN3':
            return 10.0 ** float(rng.uniform(np.log10(1.0), np.log10(5.0)))
        elif g_name == 'TEM1':
            return 10.0 ** float(rng.uniform(np.log10(2.0), np.log10(8.0)))
        elif g_name == 'CDC15':
            return 10.0 ** float(rng.uniform(np.log10(3.0), np.log10(20.0)))
        elif g_name == 'ESP1':
            return 10.0 ** float(rng.uniform(np.log10(1.5), np.log10(6.0)))
        elif g_name == 'CDC14':
            return 10.0 ** float(rng.uniform(np.log10(0.30), np.log10(1.20)))
        elif g_name == 'NET1':
            return 10.0 ** float(rng.uniform(np.log10(0.15), np.log10(0.60)))
        elif g_name in ['PPX', 'CDC5', 'CDC5_Proxy']:
            return 10.0 ** float(rng.uniform(np.log10(0.20), np.log10(0.60)))
        elif g_name == 'SWI5':
            return 10.0 ** float(rng.uniform(np.log10(0.02), np.log10(0.10)))
        else:
            log_alpha = rng.uniform(np.log10(2.0), np.log10(30.0))
            return 10.0 ** float(log_alpha)
    elif mode == 'MC':
        # Chen 2004 Methods p.3848: "multiply ... by an integer, depending on
        # the number of additional copies" -> discrete integer copy count
        # n in {2..50}, P(n) ~ 1/n (discrete log-uniform).
        # Spec: §5.3 FOLD-M / §8.3 (v2.3; was {2..30} in v2.2, LogU[1.5,30]
        # before that). Upper endpoint 50 = 2u high-copy plasmid burden
        # (2u circles are maintained at 10-40+ copies/cell in S. cerevisiae,
        # E-Mech); real MC anchors top out at 25x (#122 CDC20 15.0/0.6).
        ns = np.arange(2, 31)
        w = 1.0 / ns
        return float(rng.choice(ns, p=w / w.sum()))
    elif mode == 'ts':
        if g_name == 'CDC14':
            return float(rng.uniform(0.05, 0.35))
        if g_name in ('TEM1', 'ESP1'):
            # v5.1 anchor envelope (spec §5.4 T3 / §11.4, v2.3 declared
            # E-Anchor exception, same epistemic status as the GAL bands):
            # Chen 2004 picked these constants for phenotype consistency
            # (ka15pp 1.0->0.002; kasesp 50->0.1, kdiesp 0.5->0.001, f=0.002),
            # i.e. calibration constants, not measurements. Under the generic
            # 0.2*Beta(0.5,2) arm the anchors sit at the 15th percentile
            # (median f=0.024 = 12x anchor), so the model's conditional mean
            # was ~20x the real residual -> catastrophic eval errors
            # (#89 tem1-ts MAE 22010 vs 0.1 for the fixed-constant engine).
            # Envelope LogU[anchor/4, anchor*2] (GAL band convention): anchor
            # is an interior point (P(f=anchor)=0), full-dim dedup still
            # blocks the real composites (#81/#84/#88-91 all carry partners).
            return 10.0 ** float(rng.uniform(np.log10(0.0005), np.log10(0.004)))
        if g_name == 'NET1':
            # v5.1 anchor envelope: net1-ts / tab6-1 anchors kasrent 200->10,
            # kasrentp 1.0->0.05 (f=0.05, 69th percentile of the Beta arm).
            # Envelope LogU[0.0125, 0.10] = [anchor/4, anchor*2] concentrates
            # the conditional mean (0.042) near the anchor and cuts the
            # conditional spread (kasrent in [2.5, 20] instead of (0, 40]).
            return 10.0 ** float(rng.uniform(np.log10(0.0125), np.log10(0.10)))
        if g_name in ('CDC20', 'CDC15', 'APC_Core'):
            # v5.2 boundary-anchor buffer (spec §5.4 T3, same rule as PDS1-db
            # D2): paper ts anchors for these alleles are exact zeros
            # (#73/#78/#79/#85 cdc20-ts; #95/#96 cdc15-ts; #26 apc-ts).
            # Exact-zero edits collapsed L1 CDC20[ts] onto real #73 (70% dedup
            # interception at Glc) -> the Glc-condition cdc20-ts axis had ZERO
            # training support (the a6000v1 baseline sampled random folds and
            # had no collapse -- this asymmetry, not the fold range itself,
            # is why base beat ma on the cdc20-ts family). f ~ U[0, 0.01] =
            # declared <=1% thermal-inactivation residual buffer; f=0 boundary
            # anchor unreachable (P=0), conditional mean 0.005 x WT.
            return float(rng.uniform(0.0, 0.01))
        u = rng.beta(a=0.5, b=2.0)
        return float(0.20 * u)
    elif mode == 'DAmP':
        val = rng.normal(loc=0.20, scale=0.05)
        return float(np.clip(val, 0.05, 0.40))
    elif mode == 'pLOF':
        return float(rng.uniform(0.10, 0.50))
    elif mode == 'db':
        if g_name == 'CLB2':
            # v5.2.1 / v2.3-calib (spec §5.4 D1, E-Anchor): real anchor kdb2pp=0.034
            # (all 9 CLB2-dbΔ strains #55-#62 share the paper constant; WT=0.40 ->
            # anchor fold alpha=0.085 = 8.5% WT). The target physical envelope is
            # kdb2pp in [anchor/4, anchor*2] = [0.0085, 0.068] (2.125% - 17.0% WT).
            # Since apply_gene_mutation sets kdb2pp = p_wt['kdb2pp'] * alpha (0.4 * alpha),
            # alpha must be sampled from [0.0085/0.4, 0.068/0.4] = U[0.02125, 0.170].
            # Real anchor alpha=0.085 sits at the 42.86% percentile (interior, P=0, dedup-safe).
            return float(rng.uniform(0.02125, 0.170))
        elif g_name == 'PDS1':
            # v5.1 (spec §5.4 D2, v2.3): residual band tightened U[0,0.05] ->
            # U[0, 0.01]. Pds1 has no KEN-box, so D-box deletion removes the
            # only APC recognition signal (mechanism-preferred value f=0 ==
            # paper anchor, boundary point); <=1% residual = declared
            # modelling buffer for unknown degradation-signal redundancy.
            # Conditional mean Vdpds 0.011 vs real anchor 0.010 (was 0.015).
            return float(rng.uniform(0.0, 0.01))
        elif g_name in ('CLB5', 'SIC1', 'PPX', 'CDC5_Proxy'):
            # v5.2: exact-zero db alleles -> <=1% residual buffer (same D2
            # rule as PDS1). The fixed f=0 collapsed L1 CLB5[db] onto real
            # #68 CLB5-dbΔ (84% dedup interception at Glc) and left the
            # low-residual kd*pp axis with only 3 carbon-condition support
            # points per allele. Boundary anchors stay unreachable (P=0).
            return float(rng.uniform(0.0, 0.01))
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

def apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr, *, rng=None):
    """
    Explicit, dedicated parameter modification engine per (Gene, Mode) pair
    matching Section 7 of Mutant-Aware Sampling Scheme v1.
    Stochastic allele branches use the supplied rng (legacy global if omitted).
    """
    rng = np.random if rng is None else rng
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
            # cdc20-1 ts at restrictive temp ~= complete loss (real #73:
            # ks20p=ks20pp=0, boundary anchor). v5.2: alpha ~ U[0,0.01]
            # buffer replaces the exact zeros -- exact-zero edits made every
            # Glc L1 draw an exact copy of #73 (dedup collapse, 70%
            # interception; the a6000v1 baseline's random-fold arm had no
            # collapse, which is why base outperformed ma on this family).
            ovr['ks20p'] = p_wt['ks20p'] * alpha; ovr['ks20pp'] = p_wt['ks20pp'] * alpha
        elif g_name == 'CDC14':
            # CDC14-ts dual physiological representations:
            # Mode A (50% prob): Restrictive temp (37°C) complete arrest:
            #   ks14 ~ WT x U[0,0.01] (v5.2 buffer; was exact 0.0, which
            #   collapsed onto real #99 cdc14-ts at Glc -> 50% x 75% dedup
            #   interception of the L1 CDC14[ts] cell), catalytic rates WT
            # Mode B (50% prob): Semi-permissive temp (30-34°C) catalytic attenuation: ks14 = WT, 4 catalytic rates scaled by alpha in [0.05, 0.35]
            if rng.random() < 0.50:
                ovr['ks14'] = p_wt['ks14'] * rng.uniform(0.0, 0.01)
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
            # PATCH (spec §8.5-2): paper constant 0.05 = WT 3.0 x f (anchor
            # f ~= 0.017 in-band); reparameterized as WT x f, ts fold
            # f ~ 0.20*Beta(0.5, 2.0), support (0, 0.20]
            ovr['kppnetpp'] = p_wt['kppnetpp'] * alpha
        elif g_name == 'CDC15':
            # cdc15-ts anchor kkpnetpp=0 (boundary, #95/#96). v5.2 buffer
            # alpha ~ U[0,0.01] x WT (exact 0.0 also thinned the #96 pair
            # ~17% whenever the MC copy count hit the anchor integer at Glc).
            ovr['kkpnetpp'] = p_wt['kkpnetpp'] * alpha
        elif g_name == 'TEM1':
            # tem1-2 ts: Tem1 pool absent (binary 0) + Tem1->Cdc15 activation
            # residual as a fold (paper anchor ka15pp=0.002 = WT 1.0 x f, f=0.002;
            # v5.1 envelope f ~ LogU[0.0005, 0.004], spec §5.4 T3 / §11.4)
            # Ban 2.1: Tem1T=0 requires init_Tem1=0 to prevent negative concentration (0 - 0.904 < 0)
            ovr['Tem1T'] = 0.0; ovr['ka15pp'] = p_wt['ka15pp'] * alpha; ovr['init_Tem1'] = 0.0
        elif g_name == 'ESP1':
            # PATCH (spec §8.5-2): paper constants {0.1, 0.001} = {50, 0.5} x
            # f (anchor f=0.002); v5.1 envelope f ~ LogU[0.0005, 0.004]
            ovr['kasesp'] = p_wt['kasesp'] * alpha; ovr['kdiesp'] = p_wt['kdiesp'] * alpha
        elif g_name == 'APC_Core':
            # apc-ts anchors (#26): ks20p/ks20pp/kscdh = 0 (boundary) and
            # CDH1 pools emptied. v5.2: the three rate constants share the
            # U[0,0.01] buffer alpha (exact zeros made the L3 analog of #26
            # an exact real copy -> dedup-blocked); init_CDH1T/init_CDH1
            # stay 0 (initial conditions, no residual concept).
            ovr['ks20p'] = p_wt['ks20p'] * alpha; ovr['ks20pp'] = p_wt['ks20pp'] * alpha
            ovr['kscdh'] = p_wt['kscdh'] * alpha
            ovr['init_CDH1T'] = 0.0; ovr['init_CDH1'] = 0.0
        elif g_name == 'Mcm1':
            # PATCH (spec §8.5-2): paper constant 0.13 = WT 1.0 x f (anchor
            # f=0.13 in-band); reparameterized as WT x f, f ~ 0.20*Beta(0.5, 2.0)
            ovr['kamcm'] = p_wt['kamcm'] * alpha
        elif g_name == 'CDC28':
            cdk_eps = ['ec1k2', 'ec1n2', 'ec1b5', 'ec1b2', 'ec1n3', 'ef6k2', 'ef6n2', 'ef6b5', 'ef6b2', 'ef6n3', 'eicdhn2', 'eicdhn3', 'eicdhb5', 'eicdhb2', 'esbfn2', 'esbfn3', 'esbfb5', 'eorib5', 'eorib2', 'ebudn2', 'ebudn3', 'ebudb5']
            for ep in cdk_eps: ovr[ep] = p_wt[ep] * alpha

    # 5. db (Destruction Box Degradation Resistance)
    elif m_chosen == 'db':
        if g_name == 'CLB2':
            # CLB2-dbΔ: kdb2P=0.0 (D-box lost, Cdc20 degradation eliminated),
            # kdb2pp = WT x alpha. alpha ~ U[0.02125, 0.170] -> kdb2pp in [0.0085, 0.068]
            # (v2.3 E-Anchor band [anchor/4, anchor*2]; real anchor 0.034 = WT 0.4 x 0.085 interior)
            ovr['kdb2P'] = 0.0; ovr['kdb2pp'] = p_wt['kdb2pp'] * alpha
        elif g_name == 'CLB5':
            # dbΔ paper zero = WT 0.16 x f; v5.2: f ~ U[0,0.01] boundary
            # buffer (was exact-zero allele; collapsed onto real #68)
            ovr['kdb5pp'] = p_wt['kdb5pp'] * alpha
        elif g_name == 'PDS1':
            # PATCH (spec §8.5-2 / §5.4 D2): dbΔ paper zeros = WT {0.2, 0.04} x f,
            # f ~ U[0, 0.01] (v5.1; was U[0, 0.05]) — f=0 boundary anchor ==
            # paper allele, <=1% residual = declared modelling buffer
            ovr['kd2pdspp'] = p_wt['kd2pdspp'] * alpha; ovr['kd3pdspp'] = p_wt['kd3pdspp'] * alpha
        elif g_name == 'SIC1':
            # dbΔ paper zero = WT 1.0 x f; v5.2: f ~ U[0,0.01] boundary
            # buffer (same D2 rule; #29 GAL-SIC1dbΔ anchor stays interior
            # via the continuous GAL partner)
            ovr['kd3c1'] = p_wt['kd3c1'] * alpha
        elif g_name in ['PPX', 'CDC5_Proxy']:
            # dbΔ paper zero = WT 2.0 x f; v5.2: f ~ U[0,0.01] boundary
            # buffer (no real anchor in D_real; consistency with D2)
            ovr['kdppxpp'] = p_wt['kdppxpp'] * alpha

    # 6. KO (Knockout: complete loss of gene function)
    elif m_chosen == 'KO':
        if g_name == 'CLN2':
            ovr['ksn2pp'] = 0.0
        elif g_name == 'CLN3':
            ovr['C0'] = 0.0
        elif g_name == 'BCK2':
            ovr['B0'] = 0.0
        elif g_name == 'CLB2':
            ovr['ksb2p'] = 0.0; ovr['ksb2pp'] = 0.0
        elif g_name == 'CLB5':
            ovr['ksb5p'] = 0.0; ovr['ksb5pp'] = 0.0
        elif g_name == 'SIC1':
            ovr['ksc1p'] = 0.0; ovr['ksc1pp'] = 0.0
        elif g_name in ['CDC6', 'pDEL']:
            ovr['ksf6p'] = 0.0; ovr['ksf6pp'] = 0.0; ovr['ksf6ppp'] = 0.0
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
            # KO = synthesis zero only (Chen p.3848 official semantics; ban 2.1:
            # PPX is one of the 11 dynamic proteins whose Table 2B initial value
            # must decay naturally, ~4 min half-life at kdppxp=0.17 -- like
            # CDC20/PDS1 KO, NOT like the official TEM1/CDC15 pool eliminations)
            ovr['ksppx'] = 0.0
        elif g_name == 'PDS1':
            ovr['ks1pdspp'] = 0.0; ovr['ks2pdspp'] = 0.0
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
        # v5.2: APC-A anchor ka20pp=0 (boundary, #115/#116/#117); buffer
        # U[0,0.01] x WT replaces the exact 0.0 (fixed zero collapsed L1
        # CDC20[APC-A] / APC_Core[APC-A] onto #115, 72-79% dedup
        # interception; the L2 analogs of #116/#117 were likewise blocked)
        ovr['ka20pp'] = p_wt['ka20pp'] * rng.uniform(0.0, 0.01)
    elif m_chosen == 'TAB6-1':
        # v5.2: tab6-1 IS the net1-ts allele (real #97 'net1-ts' and #108
        # 'TAB6-1' share the anchors kasrent 200->10, kasrentp 1.0->0.05,
        # i.e. f=0.05). Fixed constants made every Glc L1 draw an exact copy
        # of #108 (dedup collapse, 75% interception) and blocked the L2
        # analogs of #109/#110. Reuse the NET1 ts anchor envelope
        # LogU[anchor/4, anchor*2] (f interior, dedup-safe).
        f = 10.0 ** float(rng.uniform(np.log10(0.0125), np.log10(0.10)))
        ovr['kasrent'] = p_wt['kasrent'] * f; ovr['kasrentp'] = p_wt['kasrentp'] * f
    elif m_chosen == 'scc1-RR':
        ovr['kasesp'] = 0.0
    elif m_chosen == 'PM':
        ovr['kd2c1'] = 0.0; ovr['kd3c1'] = 0.0
    elif m_chosen == 'pDEL':
        ovr['ksf6p'] = 0.0; ovr['ksf6pp'] = 0.0; ovr['ksf6ppp'] = 0.0
    elif m_chosen == 'perturb' and g_name == 'ORI_BUD_SPN':
        for k in ['ksori', 'ksbud', 'ksspn']:
            scale = float(np.clip(rng.normal(1.0, 0.20), 0.1, 3.0))
            ovr[k] = p_wt[k] * scale
    elif m_chosen == 'cdc24Δ':
        ovr['ksbud'] = 0.0
    elif m_chosen == 'cdc45Δ':
        ovr['ksori'] = 0.0
    elif m_chosen == 'spindle_defect':
        ovr['ksspn'] = 0.0
    elif m_chosen == 'MDT':
        # Glc = standard lab medium (dominant); Gal/Raff = GAL-workflow media
        # v5.2: weights aligned with the R5 carbon rule (was 0.70/0.15/0.15,
        # inconsistent with the main-branch 0.75/0.18/0.07)
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))

# ==============================================================================
# Part 5.5: Structure-aware skeleton (inlined from mass_wt_blind_sampler v1.6,
# 2026-09-06 — the sampler module is retired, this file is fully self-contained)
# ==============================================================================

# --- 5.5a: Module regulation graph (36 edges of Scheme §3.1.1 — Tier 2 symbols) ---
MODULE_EDGES = [
    ('CLN3','SBF_MBF','+'), ('BCK2','SBF_MBF','+'), ('CLN2','SBF_MBF','+'),
    ('CLB5','SBF_MBF','+'), ('CLB2','SBF_MBF','-'),
    ('SBF_MBF','CLN2','+'), ('SBF_MBF','CLB5','+'), ('SBF_MBF','CDC6','+'),
    ('CLB2','Mcm1','+'), ('Mcm1','CLB2','+'), ('Mcm1','SWI5','+'),
    ('Mcm1','CDC20','+'), ('Mcm1','PDS1','+'),
    ('CLB2','CDC20','+'), ('CDC20','CLB2','-'), ('CDC20','CLB5','-'),
    ('CDC20','PDS1','-'),
    ('CDH1','CLB2','-'), ('CDH1','PDS1','-'),
    ('CDC14','CDH1','+'), ('CDC14','SIC1','+'), ('CDC14','CDC6','+'),
    ('CLB2','CDH1','-'), ('CLB5','CDH1','-'), ('CLN2','CDH1','-'),
    ('CLN3','CDH1','-'),
    ('SIC1','CLB2','-'), ('SIC1','CLB5','-'),
    ('CDC6','CLB2','-'), ('CDC6','CLB5','-'),
    ('SWI5','SIC1','+'), ('SWI5','CDC6','+'),
    ('NET1','CDC14','-'), ('PPX','NET1','+'), ('CDC15','NET1','+'),
    ('TEM1','CDC15','+'), ('BUB2','TEM1','-'), ('MAD2','CDC20','-'),
    ('PDS1','ESP1','-'),
    ('CDC28','SIC1','+'), ('CDC28','CDC6','+'), ('CDC28','CDH1','+'),
    ('CDC28','SBF_MBF','+'), ('CDC28','ORI_BUD_SPN','+'),
    ('CLB2','ORI_BUD_SPN','+'), ('CLB5','ORI_BUD_SPN','+'),
    # v6.0: APC_Core biological regulation edges (resolved graph disconnection)
    ('APC_Core','CLB2','-'), ('APC_Core','PDS1','-'),
    ('CDC20','APC_Core','+'), ('CDH1','APC_Core','+'),
    # v6.1: Direct ODE regulation edges (Chen 2004 Eq 17 & Eq 19)
    ('CLB2','SWI5','-'), ('CLB2','SIC1','-'), ('CLB5','SIC1','-'), ('CLN2','SIC1','-'),
]

GROUP_BY_NAME = {g['name']: g for g in BINDING_GROUPS.values()}
GENE_NAMES = sorted(GROUP_BY_NAME.keys())
# v5.1 (R2): default gene-weight arm = UNIFORM over the 27 modules.
# The v5.0 degree+1 arm over-sampled graph hubs (CLB2 8/113, CDC28 6/113 of
# total weight) and starved zero-degree modules (APC_Core 1/113 -> the
# APC-A channel was trained at ~1/4 of its uniform share; composition audit
# 2026-09-08: APC-A 1.3% vs 5.6% in the v1r baseline). Degree+1 and the
# engine's hand-set P_bio_G (0.03-0.06, no verifiable derivation) both stay
# reachable below as ablation arms (Scheme §8-4c).
GENE_WEIGHT_ARM = 'uniform'      # 'uniform' (v5.1 default) | 'degree' | 'engine'
if GENE_WEIGHT_ARM == 'engine':
    P_BIO = {n: GROUP_BY_NAME[n]['P_bio_G'] for n in GENE_NAMES}
elif GENE_WEIGHT_ARM == 'degree':
    # P_bio: mechanized degree centrality on the module graph (Tier 3
    # — pure equation-derived topology). w(G) = (undirected degree + 1),
    # normalized; +1 = Laplace smoothing for isolated modules.
    _deg = {n: 0 for n in GENE_NAMES}
    for _a, _b, _s in MODULE_EDGES:
        _deg[_a] += 1
        _deg[_b] += 1
    _raw = {n: _deg[n] + 1.0 for n in GENE_NAMES}
    _s = sum(_raw.values())
    P_BIO = {n: _raw[n] / _s for n in GENE_NAMES}
else:
    P_BIO = {n: 1.0 / len(GENE_NAMES) for n in GENE_NAMES}
_BIO_SUM = sum(P_BIO.values())

ADJ = {n: set() for n in GENE_NAMES}
for a, b, s in MODULE_EDGES:
    ADJ[a].add(b); ADJ[b].add(a)

def _bfs_dist(src):
    dist = {src: 0}; q = [src]
    while q:
        nxt = []
        for u in q:
            for v in ADJ[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1; nxt.append(v)
        q = nxt
    return dist

DIST = {n: _bfs_dist(n) for n in GENE_NAMES}
def graph_dist(a, b):
    return DIST[a].get(b, 99)

# --- 5.5b: L2 pair classes (Scheme §3.1.5) ---
# C-A: adjacent with a negative edge, or two '+' inputs of the same target
# (shared switch input-output ends), or mutual +/- pair (same switch both ends).
_NEG_PAIRS = {(a, b) for a, b, s in MODULE_EDGES if s == '-'}
_NEG_PAIRS |= {(b, a) for a, b, s in MODULE_EDGES if s == '-'}
_OUT = {n: set() for n in GENE_NAMES}
_OUT_NEG = {n: set() for n in GENE_NAMES}
_OUT_ALL = {n: set() for n in GENE_NAMES}
for a, b, s in MODULE_EDGES:
    if s == '+': _OUT[a].add(b)
    elif s == '-': _OUT_NEG[a].add(b)
    _OUT_ALL[a].add(b)

# Convergent pairs: pairs of genes that converge on shared downstream targets (Fan-in)
CONVERGENT_PAIRS = {}
for i, a in enumerate(GENE_NAMES):
    for b in GENE_NAMES[i+1:]:
        t = _OUT_ALL[a] & _OUT_ALL[b]
        if t:
            CONVERGENT_PAIRS[(a, b)] = sorted(t)

def _pair_class(a, b):
    if (a, b) in _NEG_PAIRS:
        return 'C-A'
    if _OUT[a] & _OUT[b]:
        return 'C-A'          # shared switch input ends
    d = graph_dist(a, b)
    if d <= 1:  return 'C-A'  # adjacent positive pair on same switch cascade
    if d <= 3:  return 'C-B'
    return 'C-D'

ALL_PAIRS = [(a, b) for i, a in enumerate(GENE_NAMES)
             for b in GENE_NAMES[i+1:]]
PAIR_LISTS = {'C-A': [], 'C-B': [], 'C-D': []}
for a, b in ALL_PAIRS:
    PAIR_LISTS[_pair_class(a, b)].append((a, b))

# Dynamic Hubs: all nodes with degree >= 3 + MDT
HUBS_DEG3 = [n for n in GENE_NAMES if len(ADJ[n]) >= 3]
STAR_LEAVES = {h: sorted(ADJ[h]) for h in HUBS_DEG3}
STAR_LEAVES['MDT'] = [n for n in GENE_NAMES if n != 'MDT']

# --- 5.5c: Mode-frequency arms (Scheme §9.6 / spec §6.8.2) ---
# MODE_FREQ is set by the CLI --mode_freq switch:
#   'class' — mode-class quotas, P(m|g) ∝ QUOTA[class(m)] (v5.1 DEFAULT;
#             class-level declared constants, anti-leak by construction —
#             the quota never sees per-gene test-set statistics)
#   'flat'  — uniform over modes given gene (v5.0 default, §3.1.4; kept as
#             ablation arm — its 1/k rule over-weights rare classes on
#             low-mode-count genes, e.g. CDH1[CA] at 50%)
#   'role'  — role-aware affinity weights (Tier 2/3, leak-free ablation B):
#             gene molecule-class x mode experiment-family affinity from the
#             Chen (2004) model structure; the qualitative map is Tier 2 model
#             knowledge, the x2 multiplier is a declared design constant. The
#             rule is SYSTEMATIC (class-level, never tuned per gene toward the
#             test set) — that is what keeps this arm leak-free while s5 is not.
#   's5'    — Table-S5-fitted mode weights (BINDING_GROUPS[g]['modes']).
#             DIAGNOSTIC ARM ONLY: re-introduces the real-genotype composition
#             leak ON PURPOSE (ban-3 negative control, quantifies leakage harm).
# All arms consume the exact same RNG call count; only the weights differ.
MODE_FREQ = 'class'

# Gene molecule classes from the Chen (2004) model structure (Tier 2; the
# classification is class-level only — never adjusted per gene toward the
# test set, which is exactly what distinguishes this arm from the s5 leak).
GENE_ROLE = {
    'CLN2': 'dose', 'CLN3': 'dose', 'BCK2': 'dose', 'CLB2': 'dose',
    'CLB5': 'dose', 'SIC1': 'dose', 'PDS1': 'dose', 'CDC6': 'dose',
    'SWI5': 'dose', 'CDC20': 'dose', 'CDH1': 'dose',
    'CDC28': 'enzyme', 'CDC14': 'enzyme', 'CDC15': 'enzyme', 'TEM1': 'enzyme',
    'ESP1': 'enzyme', 'NET1': 'enzyme', 'PPX': 'enzyme', 'CDC5_Proxy': 'enzyme',
    'APC_Core': 'enzyme',
    'SBF_MBF': 'tf', 'Mcm1': 'tf',
    'MAD2': 'event', 'BUB2': 'event', 'SCC1': 'event',
    'ORI_BUD_SPN': 'event', 'MDT': 'event',
}
MODE_CLASS = {
    'KO': 'loss', 'GAL-OE': 'expression', 'MC': 'expression',
    'ts': 'activity', 'DAmP': 'activity', 'pLOF': 'activity',
    'APC-A': 'activity',   # v5.1 reclass (was 'event'): APC-A zeroes ka20pp =
                           # catalytic inactivation of APC core activation —
                           # a loss-of-activity allele (same mechanism family
                           # as ts), not a switch/event mode. 10 real mutants
                           # (#115-124) ride this token.
    'db': 'stability', 'PM': 'stability',
    'CA': 'structure', 'CSL': 'event', 'CKO': 'event',
    'scc1-RR': 'event', 'TAB6-1': 'event', 'cdc24Δ': 'event',
    'cdc45Δ': 'event', 'spindle_defect': 'event', 'MDT': 'event',
    'perturb': 'event',
}
# v5.1 (R1): 'class' arm = mode-class quotas. P(m|g) ∝ QUOTA[class(m)],
# renormalized within each gene's legal mode table. Class-level DECLARED
# design constants (Tier 2 systematic rule, zero per-gene test-set contact —
# the same anti-leak standard as ROLE_AFFINITY): the classical-genetics
# vocabulary is loss/expression-dominant (KO and overexpression are the
# workhorses), ts is the classic yeast toolbox, db/PM residuals are rarer,
# CA is rarest. Fixes the flat arm's 1/k imbalance (CDH1[CA] 50% of CDH1
# draws, CA 9.9% overall vs 2.6% in the v1r baseline; composition audit
# 2026-09-08).
MODE_CLASS_QUOTA = {
    'loss': 0.28, 'expression': 0.28, 'activity': 0.18,
    'stability': 0.20, 'structure': 0.03, 'event': 0.03,
}
# role x mode-class affinity: affine classes x2 (declared soft-prior constant),
# all others x1. 'event' role genes keep flat (all their modes are event-class).
ROLE_AFFINITY = {
    'dose':   {'expression': 2.0, 'loss': 2.0, 'stability': 2.0,
               'activity': 1.0, 'structure': 1.0, 'event': 1.0},
    'enzyme': {'activity': 2.0, 'loss': 2.0,
               'expression': 1.0, 'stability': 1.0, 'structure': 1.0,
               'event': 1.0},
    'tf':     {'loss': 2.0, 'structure': 2.0,
               'expression': 1.0, 'activity': 1.0, 'stability': 1.0,
               'event': 1.0},
}

# --- 5.5d: Parameter-touch sets (same-parameter dual-mode ban, §3.1.6) ---
_CD28_EPS = ['ec1k2','ec1n2','ec1b5','ec1b2','ec1n3','ef6k2','ef6n2','ef6b5',
             'ef6b2','ef6n3','eicdhn2','eicdhn3','eicdhb5','eicdhb2','esbfn2',
             'esbfn3','esbfb5','eorib5','eorib2','ebudn2','ebudn3','ebudb5']

TOUCH = {
 ('CLN2','KO'): {'ksn2pp'}, ('CLN2','GAL-OE'): {'ksn2p','ksn2pp'},
 ('CLN2','MC'): {'ksn2pp'}, ('CLN2','ts'): {'ksn2p','ksn2pp'},
 ('CLN3','KO'): {'C0'}, ('CLN3','GAL-OE'): {'C0'}, ('CLN3','MC'): {'C0'},
 ('CLB2','KO'): {'ksb2p','ksb2pp'}, ('CLB2','GAL-OE'): {'ksb2p'},
 ('CLB2','MC'): {'ksb2p','ksb2pp'}, ('CLB2','ts'): {'ksb2p','ksb2pp'},
 ('CLB2','db'): {'kdb2P','kdb2pp'}, ('CLB2','pLOF'): {'ksb2p','ksb2pp'},
 ('CLB2','CA'): {'kdib2'},
 ('CLB5','KO'): {'ksb5p','ksb5pp'}, ('CLB5','GAL-OE'): {'ksb5p'},
 ('CLB5','MC'): {'ksb5p','ksb5pp'}, ('CLB5','db'): {'kdb5pp'},
 ('SIC1','KO'): {'ksc1p','ksc1pp'}, ('SIC1','GAL-OE'): {'ksc1p'},
 ('SIC1','MC'): {'ksc1p','ksc1pp'}, ('SIC1','db'): {'kd3c1'},
 ('SIC1','PM'): {'kd2c1','kd3c1'},
 ('CDC6','KO'): {'ksf6p','ksf6pp','ksf6ppp'}, ('CDC6','GAL-OE'): {'ksf6p'},
 ('CDC6','MC'): {'ksf6p','ksf6pp','ksf6ppp'},
 ('CDC6','DAmP'): {'ksf6p','ksf6pp','ksf6ppp'},
 ('BCK2','KO'): {'B0'}, ('BCK2','MC'): {'B0'},
 ('SWI5','KO'): {'ksswip','ksswipp'}, ('SWI5','GAL-OE'): {'ksswip'},
 ('CDC20','KO'): {'ks20p','ks20pp'}, ('CDC20','GAL-OE'): {'ks20p'},
 ('CDC20','MC'): {'ks20p','ks20pp'}, ('CDC20','ts'): {'ks20p','ks20pp'},
 ('CDC20','APC-A'): {'ka20pp'}, ('CDC20','DAmP'): {'ks20p','ks20pp'},
 ('CDH1','KO'): {'kscdh','init_CDH1T','init_CDH1'},
 ('CDH1','CA'): {'kicdhpp','kscdh'},
 ('CDC14','ts'): {'ks14','kppc1','kppf6','kaswi','kacdhpp'},
 ('CDC14','GAL-OE'): {'ks14'}, ('CDC14','MC'): {'ks14'},
 ('CDC14','DAmP'): {'ks14'},
 ('NET1','TAB6-1'): {'kasrent','kasrentp'}, ('NET1','GAL-OE'): {'ksnet'},
 ('NET1','ts'): {'kasrent','kasrentp'}, ('NET1','DAmP'): {'ksnet'},
 ('TEM1','KO'): {'Tem1T','ka15pp','init_Tem1'}, ('TEM1','GAL-OE'): {'Tem1T'},
 ('TEM1','MC'): {'Tem1T'}, ('TEM1','ts'): {'Tem1T','ka15pp','init_Tem1'},
 ('TEM1','DAmP'): {'Tem1T'},
 ('CDC15','KO'): {'Cdc15T','kkpnetpp','init_Cdc15'},
 ('CDC15','GAL-OE'): {'Cdc15T'}, ('CDC15','MC'): {'Cdc15T'},
 ('CDC15','ts'): {'kkpnetpp'}, ('CDC15','DAmP'): {'Cdc15T'},
 ('PPX','KO'): {'ksppx'}, ('PPX','GAL-OE'): {'ksppx'},
 ('PPX','ts'): {'kppnetpp'}, ('PPX','db'): {'kdppxpp'},
 ('PDS1','KO'): {'ks1pdspp','ks2pdspp'}, ('PDS1','GAL-OE'): {'kspdsp'},
 ('PDS1','db'): {'kd2pdspp','kd3pdspp'},
 ('ESP1','ts'): {'kasesp','kdiesp'}, ('ESP1','GAL-OE'): {'ESP1T'},
 ('MAD2','CSL'): {'Vi20','Vi20_active'},
 ('BUB2','CKO'): {'kbub2','BUB2_active','init_BUB2'},
 ('SBF_MBF','KO'): {'kasbf','ksn2pp','ksb5pp'},
 ('SBF_MBF','CA'): {'is_sbf_ca','kasbf','kisbfpp'},
 ('Mcm1','ts'): {'kamcm'}, ('Mcm1','KO'): {'kamcm'},
 ('APC_Core','APC-A'): {'ka20pp'},
 ('APC_Core','ts'): {'ks20p','ks20pp','kscdh','init_CDH1T','init_CDH1'},
 ('ORI_BUD_SPN','perturb'): {'ksori','ksbud','ksspn'},
 ('ORI_BUD_SPN','cdc24Δ'): {'ksbud'},
 ('ORI_BUD_SPN','cdc45Δ'): {'ksori'},
 ('ORI_BUD_SPN','spindle_defect'): {'ksspn'},
 ('MDT','MDT'): {'mdt'},
 ('CDC5_Proxy','KO'): {'ksppx'}, ('CDC5_Proxy','db'): {'kdppxpp'},
 ('CDC28','ts'): set(_CD28_EPS), ('CDC28','pLOF'): set(_CD28_EPS),
 ('SCC1','scc1-RR'): {'kasesp'},
}

def touch_set(gene, mode):
    return TOUCH.get((gene, mode), set())

# Composite (expression x structure) same-gene expansion eligibility: genes
# whose 84-grid mode table spans BOTH an expression-class mode (GAL-OE/MC --
# Law-A synthesis slots) and a structure-class mode (db/PM/CA -- Law-A
# degradation/phosphorylation slots). Pure ODE-slot criterion (Law A/B
# mapping), zero test-set contact -> {CLB2, CLB5, PDS1, PPX, SIC1}. The four
# real same-gene composites (GAL-SIC1dbΔ #29, GAL-CLB2-dbΔ #62,
# GAL-CLB5-dbΔ #72, GAL-PDS1-dbΔ #83) are all covered; PPX has no real anchor
# but is equally constructible (honest generalization, not mirroring the test
# list). CDH1 is excluded: its modes {KO, CA} lack an expression arm (#35
# Cdh1-CA is a single CA edit inside the engine).
COMPOSITE_ELIGIBLE = [n for n in GENE_NAMES
                      if any(m in GROUP_BY_NAME[n]['modes'] for m in ('GAL-OE', 'MC'))
                      and any(m in GROUP_BY_NAME[n]['modes'] for m in ('db', 'PM', 'CA'))]

# --- 5.5e: Exact-quota categorical picker ---
class QuotaPicker:
    """Exact quota categorical picker (equalized counts)."""
    def __init__(self, names, quotas):
        self.names = list(names)
        self.quotas = np.asarray(quotas, dtype=float)
        self.counts = np.zeros(len(self.names), dtype=int)
    def pick(self, weights=None):
        if weights is None:
            deficit = self.quotas * max(1, self.counts.sum()) - self.counts
            w = np.maximum(deficit, 1e-9)
        else:
            w = np.asarray(weights, dtype=float) + 1e-9
        i = int(np.random.choice(len(self.names), p=w / w.sum()))
        self.counts[i] += 1
        return self.names[i]

# --- 5.5f: Structure-aware skeleton sampling (Scheme §3.1) ---
STRUCT_L2 = {'C-A': 0.30, 'C-B': 0.30, 'C-C': 0.15, 'C-D': 0.25}
# v6.0: Convergent (Fan-in / synthetic lethality & rescue) arm added to resolve L3/L4 parallel blind spots
STRUCT_L34 = {'convergent': 0.30, 'cascade': 0.25, 'star': 0.25, 'antag_core': 0.15, 'orthogonal': 0.05}
_L2_PICKER = QuotaPicker(list(STRUCT_L2.keys()), list(STRUCT_L2.values()))
_L34_PICKER = QuotaPicker(list(STRUCT_L34.keys()), list(STRUCT_L34.values()))
# v5.1 (R3): structure-mix softening. The pure structure skeleton concentrates
# every L2+ draw on mechanistic pair geometry (adjacent/antagonistic edges,
# cascades, stars), which the 2026-09-08 composition audit flagged as
# over-concentrated relative to the unconstrained combination space the
# eval composites explore. 50% (declared constant) of L2/L3/L4 skeletons keep
# the structure-aware arm; the other 50% are i.i.d. uniform gene draws
# without replacement — the union spans both mechanistic structure and
# unconstrained combination space. L1 is unaffected (grid rotation).
STRUCT_MIX_P = 0.50

def _sample_random_genes(ki, boost=None):
    out = []
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def _weighted_gene(weights=None, exclude=()):
    names = [n for n in GENE_NAMES if n not in exclude]
    if weights is None:
        w = np.array([P_BIO[n] for n in names])
    else:
        w = np.array([weights.get(n, P_BIO[n]) for n in names])
    w = w / w.sum()
    return str(np.random.choice(names, p=w))

def _pair_weight(a, b, boost=None):
    w = P_BIO[a] * P_BIO[b] / (_BIO_SUM ** 2)
    if boost:
        w *= (boost.get(a, 1.0) + boost.get(b, 1.0)) / 2.0
    return w

def sample_L2_fallback(boost=None):
    a = _weighted_gene(boost)
    b = _weighted_gene(boost, exclude=(a,))
    return [a, b]

def _sample_L2(boost=None):
    cls = _L2_PICKER.pick()
    if cls == 'C-C':
        hub_candidates = HUBS_DEG3 + ['MDT']
        hub_weights = [float(len(ADJ[h])) for h in HUBS_DEG3] + [3.0]
        hw = np.array(hub_weights, dtype=float)
        center = str(np.random.choice(hub_candidates, p=hw / hw.sum()))
        leaves = [l for l in STAR_LEAVES[center]]
        if not leaves:
            return sample_L2_fallback(boost), 'C-C'
        w = np.array([P_BIO[l] * (boost.get(l, 1.0) if boost else 1.0) for l in leaves])
        leaf = str(np.random.choice(leaves, p=w / w.sum()))
        return [center, leaf], 'C-C'
    pairs = PAIR_LISTS[cls]
    w = np.array([_pair_weight(a, b, boost) for a, b in pairs])
    i = int(np.random.choice(len(pairs), p=w / w.sum()))
    return list(pairs[i]), cls

def _walk_cascade(ki, boost=None):
    start = _weighted_gene(boost)
    chain = [start]
    while len(chain) < ki:
        cur = chain[-1]
        # sorted: ADJ[cur] is a set whose iteration order depends on
        # per-process string hash randomization (PYTHONHASHSEED) — an
        # unsorted feed into np.random.choice breaks same-seed
        # reproducibility across processes
        neigh = sorted(n for n in ADJ[cur] if n not in chain)
        if neigh:
            nxt = str(np.random.choice(neigh))
        else:
            nxt = _weighted_gene(boost, exclude=tuple(chain))
        chain.append(nxt)
    return chain

def _sample_convergent(ki, boost=None):
    pair_keys = list(CONVERGENT_PAIRS.keys())
    if not pair_keys:
        return _sample_random_genes(ki, boost)
    w = np.array([P_BIO[a] * P_BIO[b] * (boost.get(a, 1.0) + boost.get(b, 1.0) if boost else 1.0) for a, b in pair_keys])
    idx = int(np.random.choice(len(pair_keys), p=w / w.sum()))
    a, b = pair_keys[idx]
    shared = CONVERGENT_PAIRS[(a, b)]
    target = str(np.random.choice(shared))
    
    # 2-hop biological pathway fan-in / downstream effector rescue (Tier 2/3 generic prior):
    # 50% chance: if target has downstream targets (_OUT_ALL[target]),
    # pick one as the downstream effector (e.g., SBF_MBF -> CLN2/CLB5)
    downstream = sorted(_OUT_ALL.get(target, set()) - {a, b, target})
    if downstream and np.random.rand() < 0.50:
        effector = str(np.random.choice(downstream))
        out = [a, b, effector]
    else:
        out = [a, b, target]
    
    if ki == 3:
        return out
    
    # ki > 3: draw downstream neighbors or regulators of target
    cand = sorted(set(ADJ[target]) - set(out))
    while len(out) < ki and cand:
        nxt = str(np.random.choice(cand))
        out.append(nxt)
        cand.remove(nxt)
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def _sample_star(ki, boost=None):
    hub_candidates = HUBS_DEG3 + ['MDT']
    hub_weights = [float(len(ADJ[h])) for h in HUBS_DEG3] + [3.0]
    hw = np.array(hub_weights, dtype=float)
    center = str(np.random.choice(hub_candidates, p=hw / hw.sum()))
    leaves = [l for l in STAR_LEAVES[center]]
    out = [center]
    pool = list(leaves) if leaves else list(GENE_NAMES)
    while len(out) < ki and pool:
        w = np.array([P_BIO[g] * (boost.get(g, 1.0) if boost else 1.0) for g in pool])
        i = int(np.random.choice(len(pool), p=w / w.sum()))
        out.append(pool.pop(i))
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def _sample_antag_core(ki, boost=None):
    core, _ = _sample_L2(boost)
    out = list(core)
    cand = set()
    for g in core:
        cand |= ADJ[g]
    cand -= set(out)
    while len(out) < ki:
        if cand:
            g = str(np.random.choice(sorted(cand)))
            cand.discard(g)
        else:
            g = _weighted_gene(boost, exclude=tuple(out))
        out.append(g)
    return out

def _sample_orthogonal(ki, boost=None):
    out = [_weighted_gene(boost)]
    tries = 0
    while len(out) < ki and tries < 60:
        tries += 1
        g = _weighted_gene(boost, exclude=tuple(out))
        if all(graph_dist(g, s) >= 3 for s in out):
            out.append(g)
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def sample_skeleton(level, boost=None):
    """Returns (genes list, structure_class). Level quotas are exact.
    v5.1 (R3): L2/L3/L4 draw the structure-aware arm with prob STRUCT_MIX_P
    (0.50), else the i.i.d. uniform 'random' arm."""
    if level == 1:
        return [_weighted_gene(boost)], 'L1'
    if np.random.rand() >= STRUCT_MIX_P:   # v5.1 random arm (R3)
        if level == 2:
            ki = 2
        elif level == 3:
            ki = int(np.random.randint(3, 6))
        else:
            ki = int(np.random.randint(4, 7))
        return _sample_random_genes(ki, boost), 'random'
    if level == 2:
        return _sample_L2(boost)
    ki = int(np.random.randint(3, 6)) if level == 3 else int(np.random.randint(4, 7))
    cls = _L34_PICKER.pick()
    if cls == 'convergent':   return _sample_convergent(ki, boost), cls
    if cls == 'cascade':      return _walk_cascade(ki, boost), cls
    if cls == 'star':         return _sample_star(ki, boost), cls
    if cls == 'antag_core':   return _sample_antag_core(ki, boost), cls
    return _sample_orthogonal(ki, boost), cls

# L1 (gene, mode) round-robin grid (Latin-square-like stratification, §3.1.4).
# Shuffled with a FIXED-seed RandomState: the shuffle runs at import time,
# BEFORE generate() seeds the global RNG — an unseeded shuffle here would make
# same-seed runs irreproducible (grid order differs across processes).
_L1_GRID = [(g, m) for g in GENE_NAMES for m in GROUP_BY_NAME[g]['modes']]
np.random.RandomState(20260831).shuffle(_L1_GRID)
_L1_IDX = {'i': 0}
def next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell

def _sample_mode(gene, *, rng=None):
    rng = np.random if rng is None else rng
    modes = list(GROUP_BY_NAME[gene]['modes'].keys())
    if MODE_FREQ == 's5':   # Ablation A: Table-S5-fitted mode weights (leaks by design)
        w = np.array([GROUP_BY_NAME[gene]['modes'][m] for m in modes], dtype=float)
        return str(rng.choice(modes, p=w / w.sum()))
    if MODE_FREQ == 'role':  # Ablation B: role-aware affinity weights (Tier 2/3)
        aff = ROLE_AFFINITY.get(GENE_ROLE.get(gene, 'event'), None)
        if aff is not None:
            w = np.array([aff.get(MODE_CLASS.get(m, 'event'), 1.0)
                          for m in modes], dtype=float)
            return str(rng.choice(modes, p=w / w.sum()))
        # event-role genes (or unmapped): flat, same RNG call as the flat arm
    if MODE_FREQ == 'class':  # v5.1 main arm: mode-class quotas (R1)
        w = np.array([MODE_CLASS_QUOTA.get(MODE_CLASS.get(m, 'event'), 1.0)
                      for m in modes], dtype=float)
        return str(rng.choice(modes, p=w / w.sum()))
    return str(rng.choice(modes))   # flattened uniform (anti-leak, §3.1.4)

def _resolve_conflicts(pairs, *, rng=None):
    """Same-parameter dual-mode ban + cross-gene overlap ban (§3.1.6).
    On collision between pair i and pair j: resample i's mode; if NO mode of
    gene_i can avoid the collision (e.g. CDH1 KO/CA both touch kscdh, which
    APC_Core-ts also touches), resample the PARTNER j instead; if neither
    side has a clean mode, drop pair i. After any drop the whole list is
    re-checked — a drop can expose a second, previously masked conflict
    (found by smoke test A2, 2026-09-06). Output is always conflict-free."""
    # Explicit streams also fix set traversal order across PYTHONHASHSEEDs.
    # Keep the omitted-rng path unchanged for existing sampler callers.
    pairs = list(pairs)
    rounds = 30
    while len(pairs) > 1:
        touched = {}
        conflict = None
        for i, (g, m) in enumerate(pairs):
            ts = touch_set(g, m)
            for pk in (sorted(ts) if rng is not None else ts):
                if pk in touched:
                    conflict = (i, touched[pk])
                    break
                touched[pk] = i
            if conflict is not None:
                break
        if conflict is None:
            return pairs
        i, j = conflict
        def _other_keys(idx):
            ks = [touch_set(g, m) for k, (g, m) in enumerate(pairs) if k != idx]
            return set().union(*ks) if ks else set()
        gi, gj = pairs[i][0], pairs[j][0]
        i_ok = any(not (touch_set(gi, mm) & _other_keys(i))
                   for mm in GROUP_BY_NAME[gi]['modes'])
        j_ok = any(not (touch_set(gj, mm) & _other_keys(j))
                   for mm in GROUP_BY_NAME[gj]['modes'])
        if rounds > 0 and i_ok:
            pairs[i] = (gi, _sample_mode(gi, rng=rng)); rounds -= 1
        elif rounds > 0 and j_ok:
            pairs[j] = (gj, _sample_mode(gj, rng=rng)); rounds -= 1
        else:
            # neither side can redraw clean (or redraw budget spent): drop
            # pair i and re-check the remaining list (loop continues)
            pairs.pop(i)
    return pairs

def _assign_convergent_modes(genes):
    """Universal Pathway Buffering / Genetic Suppressor mode assignment.
    For convergent multigene skeletons [a, b, eff, ...]:
    - First 2 parallel regulator inputs (a, b) receive Loss-of-Function (synthetic defect)
    - 3rd effector gene receives Dosage Suppression (OE) if activated downstream,
      or Bypass Suppression (LOF) if inhibited downstream.
    - Additional genes draw naturally from _sample_mode.
    Completely generic across the ODE topology, zero test-set hardcoding."""
    pairs = []
    for g in genes[:2]:
        modes = GROUP_BY_NAME[g]['modes']
        lofs = [m for m in ['KO', 'ts', 'DAmP', 'pLOF', 'db'] if m in modes]
        if lofs:
            m = 'KO' if 'KO' in lofs and np.random.rand() < 0.75 else str(np.random.choice(lofs))
        else:
            m = _sample_mode(g)
        pairs.append((g, m))

    if len(genes) >= 3:
        eff = genes[2]
        modes = GROUP_BY_NAME[eff]['modes']
        oes = [m for m in ['GAL-OE', 'MC'] if m in modes]
        lofs = [m for m in ['KO', 'ts', 'DAmP', 'pLOF', 'db'] if m in modes]
        if oes and (not lofs or np.random.rand() < 0.60):
            m = str(np.random.choice(oes))
        elif lofs:
            m = 'KO' if 'KO' in lofs and np.random.rand() < 0.75 else str(np.random.choice(lofs))
        else:
            m = _sample_mode(eff)
        pairs.append((eff, m))

    for g in genes[3:]:
        pairs.append((g, _sample_mode(g)))

    return pairs

def sample_mass_mutant(level=1):
    """
    v2 leak-free mutant selection (replaces the v1 rescue-motif path).

    v1 leakage (REMOVED, 2026-09-06): 8 rescue-motif templates were hand-built
    against the 126 real mutants (code comments mapped motifs to strain numbers
    #13-#126 = ban-4 targeted hardcoding) and the gene/mode draws used
    Table-S5-fitted P_bio_G / mode weights (ban-3 composition statistics).

    Selection = the structure-aware skeleton inlined in Part 5.5 above
    (all Tier 2/3 sources, zero test-set contact):
      - gene weights  : uniform over 27 modules (v5.1 R2; degree+1 and engine
                        P_bio_G retained as ablation arms)
      - L1            : shuffled 84-cell (gene, mode) grid rotation (coverage)
      - L2/L3/L4      : 50% structure-class quotas from the 36-edge module
                        graph (C-A/C-B/C-C/C-D; cascade/star/antag_core/
                        orthogonal) + 50% i.i.d. uniform genes (v5.1 R3)
      - mode          : --mode_freq arm (class quotas DEFAULT = v5.1 R1;
                        flat = uniform anti-leak; role = Tier 2/3 affinity;
                        s5 = leak-by-design control)
      - conflicts     : touch-set collision check + mode redraw (Sect 3.1.6)
      - carbon rule   : GAL -> mdt=150 forced; else {90,150,160}
                        w.p. (0.75,0.18,0.07) (v5.1 R5, v1r-aligned)
    Strength values: engine sample_scaling_factor (Tier 2 literature bands;
    v5.1: TEM1/ESP1/NET1 ts anchor envelopes LogU[anchor/4, anchor x 2],
    PDS1-db residual U[0, 0.01], MC integer support {2..50}).
    Level quotas: set by generate_mass_dataset (40/30/20/10 mechanized).
    """
    p_wt = get_default_params()
    p_mut = p_wt.copy()
    ovr = {}

    has_gal_oe = False
    mut_descriptions = []

    # Phase 1: structure-aware skeleton (genes) + mode arm
    if level == 1:
        g0, m0 = next_L1_cell()
        if MODE_FREQ != 'flat':
            # s5/role arms: keep the grid's gene marginal, re-draw the mode
            # from the arm's weights — only the mode-frequency channel changes
            m0 = _sample_mode(g0)
        pairs = [(g0, m0)]
    else:
        genes, _cls = sample_skeleton(level)
        if _cls == 'convergent' and np.random.rand() < 0.60:
            pairs = _assign_convergent_modes(genes)
        else:
            pairs = [(g, _sample_mode(g)) for g in genes]

    # same-gene composite: expression x structure dual mode, 25% (declared
    # free constant, minority-form prior; eligibility = mechanism criterion
    # at COMPOSITE_ELIGIBLE above; spec §7 composite rule)
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = GROUP_BY_NAME.get(g, {}).get('modes', [])
        if g in COMPOSITE_ELIGIBLE and np.random.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(np.random.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(np.random.choice(synth))))
    pairs = expanded

    # same-parameter dual-mode ban: redraw conflicting modes (Sect 3.1.6)
    pairs = _resolve_conflicts(pairs)
    if not pairs:            # defensive: cannot happen (a single
        return sample_mass_mutant(level)  # pair never self-conflicts)

    for g_name, m_chosen in pairs:
        mut_descriptions.append(f"{g_name}[{m_chosen}]")
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        alpha = sample_scaling_factor(m_chosen, g_name) \
            if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
        apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)

    # carbon source rule (v5.1 / R5): GAL -> Gal forced; else Glc-dominant
    # weights aligned with the v1r baseline (mass_random_sampler.py):
    # Glc 0.75 / Gal 0.18 / Raff 0.07 (was 0.70/0.15/0.15 — the flatter Raff
    # share over-trained a carbon source carried by only 2/126 real strains)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0],
                                            p=[0.75, 0.18, 0.07]))

    desc_str = "+".join(mut_descriptions)
    return ovr, desc_str

def rescue_collision_candidate(ovr, desc_str, p_wt, *, rng=None):
    """
    Rescues a candidate sample that hit a 100% collision with D_real.
    Branch 1 (70%): Glucose preservation with continuous mode switch (DAmP / ts / pLOF).
    Branch 2 (30%): Carbon shift to Galactose (150) or Raffinose (160).
    Forward the caller's rng to all nested scaling/mutation draws.
    """
    rng = np.random if rng is None else rng
    tokens = desc_str.split('+')
    parsed = []
    for tok in tokens:
        if '[' in tok and tok.endswith(']'):
            parts = tok.split('[')
            g = parts[0]
            m = parts[1][:-1]
            parsed.append((g, m))
        else:
            parsed.append((tok, 'KO'))

    if not parsed:
        return None, desc_str

    p_mut = p_wt.copy()
    new_ovr = {}

    # 70% chance: Mode continuous switch on Glucose
    if rng.random() < 0.70:
        ko_genes = [g for g, m in parsed if m == 'KO' and g in GROUP_BY_NAME]
        if ko_genes:
            target_g = str(rng.choice(ko_genes))
            avail = [m for m in ['DAmP', 'ts', 'pLOF'] if m in GROUP_BY_NAME[target_g]['modes']]
            if avail:
                target_m = str(rng.choice(avail))
                new_pairs = [(g, target_m if g == target_g else m) for g, m in parsed]
                new_desc = []
                for g_name, m_chosen in new_pairs:
                    new_desc.append(f"{g_name}[{m_chosen}]")
                    alpha = sample_scaling_factor(m_chosen, g_name, rng=rng) \
                        if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
                    apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, new_ovr, rng=rng)
                new_ovr['mdt'] = ovr.get('mdt', 90.0)
                return new_ovr, "+".join(new_desc)

    # 30% chance or fallback: Carbon Shift (Gal 150 or Raff 160)
    shift_mdt = float(rng.choice([150.0, 160.0]))
    shifted_ovr = dict(ovr)
    shifted_ovr['mdt'] = shift_mdt
    return shifted_ovr, desc_str + f"_MDT[{int(shift_mdt)}]"

# ==============================================================================
# Part 6: Complete Dataset Generation & .npz File Export (lhs_v1_origin.py Format)
# ==============================================================================

def generate_mass_dataset(n_synthetic=100000, level_ratios=[0.25, 0.35, 0.25, 0.15],
                          max_attempts=None, num_workers=None,
                          output_name=None, output_dir=None):
    """
    Generate complete dataset (126 Real Mutants + n_synthetic MASS Mutants)
    and export into single .npz file with exact field alignment to lhs_v1_origin.py.
    """
    if output_name is None:
        output_name = f"mutant_aware_{n_synthetic}_210m_500s.npz"

    if output_dir is None:
        if os.name == 'nt':
            output_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            output_dir = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

    dataset_path = os.path.join(output_dir, output_name)

    print("=" * 85)
    print(f"MASS v5.0 Dataset Generator Protocol (Unified Main, lhs_v1_origin.py Export Alignment)")
    print(f"Target Output: {dataset_path}")
    print("=" * 85)

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
    print(f"\n[Phase 2/4] Sampling {n_synthetic} Synthetic Mutants (MASS v4.5 Protocol)...")
    r1, r2, r3, r4 = level_ratios
    total_r = r1 + r2 + r3 + r4
    r1, r2, r3, r4 = r1 / total_r, r2 / total_r, r3 / total_r, r4 / total_r

    n_l1 = int(math.floor(r1 * n_synthetic))
    n_l2 = int(math.floor(r2 * n_synthetic))
    n_l3 = int(math.floor(r3 * n_synthetic))
    n_l4 = n_synthetic - n_l1 - n_l2 - n_l3

    print(f" -> Target Level Distribution: L1 (ki=1): {n_l1} ({r1*100:.1f}%) | L2 (ki=2): {n_l2} ({r2*100:.1f}%) | L3 (ki=3~5): {n_l3} ({r3*100:.1f}%) | L4 (ki=4~6): {n_l4} ({r4*100:.1f}%)")

    level_assignments = [1]*n_l1 + [2]*n_l2 + [3]*n_l3 + [4]*n_l4
    np.random.shuffle(level_assignments)

    attempts = 0
    synth_success = 0
    duplicates_blocked = 0
    duplicates_rescued = 0

    while synth_success < n_synthetic and attempts < max_attempts:
        attempts += 1
        level = level_assignments[synth_success]
        ovr, desc_str = sample_mass_mutant(level=level)

        # 136-dim Parameter Deduplication Check against D_real
        if is_duplicate_real_mutant(ovr, defaults):
            # v6.0: Collision Rescue. When hitting an exact real-mutant combination,
            # rescue via continuous mode-switch (70%) or carbon shift (30%)
            # to preserve core interaction coverage without data leakage.
            rescued_ovr, rescued_desc = rescue_collision_candidate(ovr, desc_str, defaults)
            if rescued_ovr is not None and not is_duplicate_real_mutant(rescued_ovr, defaults):
                ovr = rescued_ovr
                desc_str = rescued_desc
                duplicates_rescued += 1
            else:
                duplicates_blocked += 1
                continue

        synth_success += 1
        sample_name = f"MASS_{desc_str}_L{level}_{synth_success}".replace(" ", "_")
        ordered_names.append(sample_name)
        all_tasks[sample_name] = ovr

    print(f" -> Sampling Done: {synth_success} synthetic tasks generated in {attempts} attempts.")
    print(f"    - Filter [D_real Deduplication]: {duplicates_blocked} candidates blocked, {duplicates_rescued} candidates rescued.")
    print(f" -> Total Simulation Queue: {len(ordered_names)} tasks ({len(REAL_MUTANTS)} Real + {synth_success} Synthetic).")

    # Phase 3: Parallel ODE Simulation Engine
    if num_workers is None:
        total_cores = os.cpu_count() or 4
        num_workers = min(60, total_cores)

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
        for idx, future in enumerate(bounded_completed(executor, future_to_name)):
            name = future_to_name[future]
            is_synth = "_L" in name
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result()
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

    os.makedirs(os.path.dirname(os.path.abspath(dataset_path)), exist_ok=True)
    np.savez_compressed(dataset_path,
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
    print(f" MASS v4.5 DATASET GENERATION SUMMARY REPORT", flush=True)
    print("=" * 85, flush=True)
    print(f"Dataset Export Path               : {dataset_path}", flush=True)
    print(f"Data Tensor Shape (N, Vars, Steps): {all_tensors.shape}", flush=True)
    print(f"Params Matrix Shape (N, P_dim)    : {all_params.shape}", flush=True)
    print(f"Total Parameter Keys Count        : {len(param_keys)}", flush=True)
    print(f"Total Tasks Simulated             : {len(ordered_names)}", flush=True)
    print(f"Total Validated Samples Saved     : {total_valid} ({total_valid / len(ordered_names) * 100:.2f}% yield)", flush=True)
    print("-" * 85, flush=True)

    # 1. Filtering & Quality Audit
    print("\n[1. Filtering & Quality Control Audit]", flush=True)
    print(f"  * Filter 1 [D_real Deduplication Blocked] : {duplicates_blocked:5d} synthetic candidates")
    print(f"  * Filter 2 [Mass Divergence (Mass > 10)]  : {total_disc:5d} tasks")
    print(f"  * Filter 3 [ODE Solver Failures]          : {total_fail:5d} tasks")
    print(f"  * Valid Sample Yield Rate                 : {total_valid:5d} / {len(ordered_names)} ({total_valid / len(ordered_names) * 100:.1f}%)")

    # 2. Sample Composition
    print("\n[2. Sample Composition Breakdown]", flush=True)
    print(f"  * Real Mutants (Appendix B Table S5)      : {len(real_final_names):5d} / {len(REAL_MUTANTS)} ({len(real_final_names)/len(REAL_MUTANTS)*100:.1f}%)")
    print(f"  * Synthetic Mutants (MASS v4.5 Multi-Level): {len(synth_final_names):5d} / {synth_success} ({len(synth_final_names)/synth_success*100:.1f}%)")
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

    return dataset_path


if __name__ == '__main__':
    # =========================================================================
    # 用户可调参数配置区 (随时在此处直接修改，或通过命令行参数指定)
    # =========================================================================
    NUM_SAMPLES = 100000                           # 生成的采样样本数量
    NUM_WORKERS = min(60, os.cpu_count() or 4)    # 并行 CPU 核心数（自动适配：服务器 64 核取 60，本地取本机核数；--workers 可覆盖）
    LEVEL_RATIOS = [0.25, 0.35, 0.25, 0.15]       # v6.0: 优化重平衡配额（压缩L1单突变冗余，释放配额至并联汇聚L3/L4与推拉L2）
    
    # 导出的数据集文件名 (默认自动携带样本数，也可直接在此自定义修改)
    DATASET_NAME = f"mutant_aware_{NUM_SAMPLES}_210m_500s.npz"

    # 导出的数据集目录路径 (自动适配系统环境，也可直接在此自定义写死绝对路径)
    if os.name == 'nt':
        OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        OUTPUT_DIR = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"
    # =========================================================================

    import argparse
    parser = argparse.ArgumentParser(description="MASS v5.0 Dataset Generator (Unified Main)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of synthetic samples to generate")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel CPU worker processes")
    parser.add_argument("--level_ratios", type=float, nargs=4, default=LEVEL_RATIOS, help="Proportions for Level 1, 2, 3, 4 tiers")
    parser.add_argument("--mode_freq", choices=['class', 'flat', 'role', 's5'], default='class',
                        help="Mode-frequency arm: class=mode-class quotas (v5.1/v5.2 DEFAULT, R1; class-level declared constants, anti-leak); flat=uniform 1/k (ablation baseline, v5.0 default); role=role-aware affinity weights (Tier 2/3, leak-free ablation B); s5=Table-S5-fitted weights (LEAKS BY DESIGN, ban-3 negative-control ablation A)")
    parser.add_argument("--output_name", type=str, default=None, help="Export dataset .npz filename (default: auto with sample count)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Export directory path")
    args = parser.parse_args()

    # 动态确定输出文件名：如果命令行显式指定了 --output_name 则使用指定的；否则自动使用基于 args.samples 的文件名
    if args.output_name is not None:
        final_output_name = args.output_name
    else:
        if args.samples == NUM_SAMPLES:
            final_output_name = DATASET_NAME
        else:
            final_output_name = f"mutant_aware_{args.samples}_210m_500s.npz"

    # mode-frequency arm (spec §6.8.2): module-level switch consumed by
    # _sample_mode / next_L1_cell path inside sample_mass_mutant
    MODE_FREQ = args.mode_freq
    if args.mode_freq == 'flat' and not final_output_name.endswith('_flatmode.npz'):
        final_output_name = final_output_name.replace('.npz', '_flatmode.npz')
    elif args.mode_freq == 'role' and not final_output_name.endswith('_rolemode.npz'):
        final_output_name = final_output_name.replace('.npz', '_rolemode.npz')
    elif args.mode_freq == 's5' and not final_output_name.endswith('_s5mode.npz'):
        final_output_name = final_output_name.replace('.npz', '_s5mode.npz')

    print(f"Executing MASS v5.0 Dataset Generator (Samples: {args.samples}, Workers: {args.workers}, ModeFreq: {args.mode_freq}, Output: {final_output_name})...", flush=True)
    generate_mass_dataset(
        n_synthetic=args.samples,
        level_ratios=args.level_ratios,
        num_workers=args.workers,
        output_name=final_output_name,
        output_dir=args.output_dir
    )
    print("\n[SUCCESS] MASS Dataset Generator Execution Finished Cleanly!", flush=True)

