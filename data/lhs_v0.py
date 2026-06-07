import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import qmc
import concurrent.futures
import time
import os
import warnings

warnings.filterwarnings("ignore") 

# ==========================================
# 0. 全局配置参数
# ==========================================
SIM_TF = int(os.environ.get("SIM_TF", 210))       
SIM_STEPS = int(os.environ.get("SIM_STEPS", 500)) 
NUM_LHS_SAMPLES = 50000  # 全局盲抽样本数
SIM_TIMEOUT = 5.0 

DATASET_NAME = f"lhs_blind_{SIM_TF}min_{SIM_STEPS}steps_dual_labels.npz"

# ==========================================
# 1. 严格离散保护名单与变量映射
# ==========================================
(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, 
 nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, 
 nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, 
 nVi20, nlte1, nBUB2) = range(39)

DISCRETE_STRICT_PARAMS = [
    'init_CDH1T', 'init_CDH1', 'kscdh', 'kdcdh', 
    'ks14', 'kd14', 'ksnet', 'kdnet', 
    'mdt', 'Vi20_active', 'BUB2_active', 'init_BUB2', 'is_mutant_104', 'ESP1T'
]

def initial_conds():
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
    y0[nVi20]  = 0.0099999; y0[nlte1]  = 0.1000000; y0[nBUB2]  = 0.2000000
    return y0

def get_default_params():
    return {
        'B0':0.054, 'C0':0.4, 'Dn3':1.0, 'Jn3':6.0, 'ksn2p':0.0, 'ksn2pp':0.15, 'kdn2':0.12,
        'esbfn2':2.0, 'esbfn3':10.0, 'esbfb5':2.0, 'kasbf':0.38, 'kisbfp':0.6, 'kisbfpp':8.0, 'Jasbf':0.01, 'Jisbf':0.01,
        'ksb2p':0.001, 'ksb2pp':0.04, 'kdb2p':0.003, 'kdb2pp':0.4, 'kdb2P':0.15, 'kamcm':1.0, 'kimcm':0.15, 'Jamcm':0.1, 'Jimcm':0.1,
        'ksb5p':0.0008, 'ksb5pp':0.005, 'kdb5p':0.01, 'kdb5pp':0.16, 'ksc1p':0.012, 'ksc1pp':0.12, 'kd1c1':0.01, 'kd2c1':1.0, 'Jd2c1':0.05,
        'ec1k2':0.03, 'ec1n2':0.06, 'ec1b5':0.1, 'ec1b2':0.45, 'ec1n3':0.3, 'kasb2':50.0, 'kdib2':0.05, 'kasb5':50.0, 'kdib5':0.06,
        'kd3c1':1.0, 'kppc1':4.0, 'ksf6p':0.024, 'ksf6pp':0.12, 'ksf6ppp':0.004, 'kd1f6':0.01, 'kd2f6':1.0, 'Jd2f6':0.05,
        'ef6k2':0.03, 'ef6n2':0.06, 'ef6b5':0.1, 'ef6b2':0.55, 'ef6n3':0.0, 'kasf2':15.0, 'kdif2':0.5, 'kasf5':0.01, 'kdif5':0.01,
        'kd3f6':1.0, 'kppf6':4.0, 'ksswip':0.005, 'ksswipp':0.08, 'kdswi':0.08, 'kaswi':2.0, 'kiswi':0.05,
        'kaiep':0.1, 'kiiep':0.15, 'Jaiep':0.1, 'Jiiep':0.1, 'ks20p':0.006, 'ks20pp':0.6, 'kd20':0.3, 'ka20p':0.05, 'ka20pp':0.2, 'ki20p':0.01, 'ki20pp':8.0,
        'kscdh':0.01, 'kdcdh':0.01, 'kacdhp':0.01, 'kacdhpp':0.8, 'Jicdh':0.03, 'Jacdh':0.03, 'kicdhp':0.001, 'kicdhpp':0.08, 
        'eicdhn2':0.4, 'eicdhn3':0.25, 'eicdhb5':8.0, 'eicdhb2':1.2, 'ks14':0.2, 'kd14':0.1, 'ksnet':0.084, 'kdnet':0.03,
        'kasrent':200.0, 'kasrentp':1.0, 'kdirent':1.0, 'kdirentp':2.0, 'kppnetp':0.05, 'kppnetpp':3.0, 'kkpnetp':0.01, 'kkpnetpp':0.6,
        'kspdsp':0.0, 'ks1pdspp':0.03, 'ks2pdspp':0.055, 'kd1pdsp':0.01, 'kd2pdspp':0.2, 'kd3pdspp':0.04, 'kasesp':50.0, 'kdiesp':0.5, 'ESP1T':1.0,
        'ksppx':0.1, 'Jpds':0.04, 'kdppxp':0.17, 'kdppxpp':2.0, 'J20ppx':0.15, 'ksori':2.0, 'kdori':0.06, 'eorib5':0.9, 'eorib2':0.45,
        'ksbud':0.2, 'kdbud':0.06, 'ebudn2':0.25, 'ebudn3':0.05, 'ebudb5':1.0, 'ksspn':0.1, 'kdspn':0.06, 'Jspn':0.14,
        'Tem1T':1.0, 'jatem':0.1, 'Jitem':0.1, 'Cdc15T':1.0, 'ka15p':0.002, 'ka15pp':1.0, 'ka15P':0.001, 'ki15':0.5,
        'mdt': 90.0, 'Vi20_active': 8.0, 'BUB2_active': 1.0  
    }

def GK(Va, Vi, Ja, Ji):
    BB = Vi - Va + Ja * Vi + Ji * Va
    return 2 * Ji * Va / (BB + np.sqrt(BB**2 - 4 * (Vi - Va) * Ji * Va))

def eqns(t, y, p):
    dydt = np.zeros(39)
    MASS, CLN2, CLB2, CLB5, SIC1, CDC6, C2, C5, F2, F5, SIC1P, C2P, C5P, CDC6P, F2P, F5P, SWI5T, SWI5, IEP, CDC20T, CDC20, CDH1T, CDH1, CDC14T, CDC14, NET1T, NET1, RENT, TEM1, CDC15, PPX, PDS1, ESP1, ORI, BUD, SPN, Vi20, lte1, BUB2 = y
    
    kg = np.log(2) / p['mdt']
    BCK2 = p['B0'] * MASS
    CLN3 = p['C0'] * p['Dn3'] * MASS / (p['Jn3'] + p['Dn3'] * MASS)
    SIC1T = SIC1 + C2 + C5 + SIC1P + C2P + C5P
    CDC6T = CDC6 + F2 + F5 + CDC6P + F2P + F5P
    RENTP = CDC14T - RENT - CDC14
    NET1P = NET1T - NET1 - CDC14T + CDC14
    PE = p['ESP1T'] - ESP1
    
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
    
    SBF = GK(Vasbf, p['kisbfp']+p['kisbfpp']*CLB2, p['Jasbf'], p['Jisbf'])
    MCM1 = GK(p['kamcm']*CLB2, p['kimcm'], p['Jamcm'], p['Jimcm'])
    
    # 🌟 Boolean 参数安全获取
    actual_ksc1p = 0.012 * 3 if (p.get('is_mutant_104', False) and t >= 180.0) else p['ksc1p']
    
    dydt[0] = kg * MASS
    dydt[1] = (p['ksn2p'] + p['ksn2pp']*SBF)*MASS - p['kdn2']*CLN2
    dydt[2] = (p['ksb2p'] + p['ksb2pp']*MCM1)*MASS + (p['kd3c1']*C2P + p['kd3f6']*F2P) + (p['kdib2']*C2 + p['kdif2']*F2) - (Vdb2 + p['kasb2']*SIC1 + p['kasf2']*CDC6)*CLB2
    dydt[3] = (p['ksb5p'] + p['ksb5pp']*SBF)*MASS + (p['kd3c1']*C5P + p['kd3f6']*F5P) + (p['kdib5']*C5 + p['kdif5']*F5) - (Vdb5 + p['kasb5']*SIC1 + p['kasf5']*CDC6)*CLB5
    dydt[4] = (actual_ksc1p + p['ksc1pp']*SWI5) + (Vdb2*C2 + Vdb5*C5) + (p['kdib2']*C2 + p['kdib5']*C5) + Vppc1*SIC1P - (p['kasb2']*CLB2 + p['kasb5']*CLB5 + Vkpc1)*SIC1
    dydt[5] = (p['ksf6p'] + p['ksf6pp']*SWI5 + p['ksf6ppp']*SBF) + (Vdb2*F2 + Vdb5*F5) + (p['kdif2']*F2 + p['kdif5']*F5) + Vppf6*CDC6P - (p['kasf2']*CLB2 + p['kasf5']*CLB5 + Vkpf6)*CDC6
    dydt[6] = p['kasb2']*CLB2*SIC1 + Vppc1*C2P - (p['kdib2'] + Vdb2 + Vkpc1)*C2
    dydt[7] = p['kasb5']*CLB5*SIC1 + Vppc1*C5P - (p['kdib5'] + Vdb5 + Vkpc1)*C5
    dydt[8] = p['kasf2']*CLB2*CDC6 + Vppf6*F2P - (p['kdif2'] + Vdb2 + Vkpf6)*F2
    dydt[9] = p['kasf5']*CLB5*CDC6 + Vppf6*F5P - (p['kdif5'] + Vdb5 + Vkpf6)*F5
    dydt[10] = Vkpc1*SIC1 - (Vppc1 + p['kd3c1'])*SIC1P + Vdb2*C2P + Vdb5*C5P
    dydt[11] = Vkpc1*C2 - (Vppc1 + p['kd3c1'] + Vdb2)*C2P
    dydt[12] = Vkpc1*C5 - (Vppc1 + p['kd3c1'] + Vdb5)*C5P
    dydt[13] = Vkpf6*CDC6 - (Vppf6 + p['kd3f6'])*CDC6P + Vdb2*F2P + Vdb5*F5P
    dydt[14] = Vkpf6*F2 - (Vppf6 + p['kd3f6'] + Vdb2)*F2P
    dydt[15] = Vkpf6*F5 - (Vppf6 + p['kd3f6'] + Vdb5)*F5P
    dydt[16] = p['ksswip'] + p['ksswipp']*MCM1 - p['kdswi']*SWI5T
    dydt[17] = p['ksswip'] + p['ksswipp']*MCM1 + p['kaswi']*CDC14*(SWI5T - SWI5) - (p['kiswi']*CLB2 + p['kdswi'])*SWI5
    dydt[18] = Vaiep*(1 - IEP)/(p['Jaiep'] + 1 - IEP) - p['kiiep']*IEP/(p['Jiiep'] + IEP)
    dydt[19] = (p['ks20p'] + p['ks20pp']*MCM1) - p['kd20']*CDC20T
    dydt[20] = (p['ka20p'] + p['ka20pp']*IEP)*(CDC20T - CDC20) - (Vi20 + p['kd20'])*CDC20
    dydt[21] = p['kscdh'] - p['kdcdh']*CDH1T
    dydt[22] = p['kscdh'] - p['kdcdh']*CDH1 + Vacdh*(CDH1T - CDH1)/(p['Jacdh'] + CDH1T - CDH1) - Vicdh*CDH1/(p['Jicdh'] + CDH1)
    dydt[23] = p['ks14'] - p['kd14']*CDC14T
    dydt[24] = (p['kdirent']*RENT + p['kdirentp']*RENTP) - (p['kasrent']*NET1 + p['kasrentp']*NET1P)*CDC14 + p['ks14'] - p['kd14']*CDC14 + p['kdnet']*(RENT + RENTP)
    dydt[25] = p['ksnet'] - p['kdnet']*NET1T
    dydt[26] = p['kdirent']*RENT - p['kasrent']*NET1*CDC14 + Vppnet*NET1P - Vkpnet*NET1 + p['ksnet'] - p['kdnet']*NET1 + p['kd14']*RENT
    dydt[27] = -p['kdirent']*RENT + p['kasrent']*NET1*CDC14 + Vppnet*RENTP - Vkpnet*RENT - (p['kd14'] + p['kdnet'])*RENT
    dydt[28] = lte1*(p['Tem1T'] - TEM1)/(p['jatem'] + p['Tem1T'] - TEM1) - BUB2*TEM1/(p['Jitem'] + TEM1)
    dydt[29] = (p['ka15p']*(p['Tem1T'] - TEM1) + p['ka15pp']*TEM1 + p['ka15P']*CDC14)*(p['Cdc15T'] - CDC15) - p['ki15']*CDC15
    dydt[30] = p['ksppx'] - Vdppx*PPX
    dydt[31] = (p['kspdsp'] + p['ks1pdspp']*SBF + p['ks2pdspp']*MCM1) + p['kdiesp']*PE - (Vdpds + p['kasesp']*ESP1)*PDS1
    dydt[32] = -p['kasesp']*PDS1*ESP1 + (p['kdiesp'] + Vdpds)*PE
    dydt[33] = p['ksori']*(p['eorib5']*CLB5 + p['eorib2']*CLB2) - p['kdori']*ORI
    dydt[34] = p['ksbud']*(p['ebudn2']*CLN2 + p['ebudn3']*CLN3 + p['ebudb5']*CLB5) - p['kdbud']*BUD
    dydt[35] = p['ksspn']*CLB2/(p['Jspn'] + CLB2) - p['kdspn']*SPN
    dydt[36:39] = 0 
    return dydt

# ==========================================
# 2. 离散事件与双重打标系统 
# ==========================================
def event1(t, y, p): return y[nCLB2] - 0.3
event1.terminal = True; event1.direction = -1

def event2(t, y, p): return y[nCLB2] + y[nCLB5] - 0.2
event2.terminal = True; event2.direction = -1

def event3(t, y, p): return y[nORI] - 1.0
event3.terminal = True; event3.direction = 1

def event4(t, y, p): return y[nSPN] - 1.0
event4.terminal = True; event4.direction = 1

def analyze_cell_viability_unified(event_history, bud_at_div, mass_at_div, max_mass, max_clb2):
    if max_mass > 10.0:
        return "Inviable: G1 Arrest" if max_clb2 < 0.2 else "Inviable: M-phase Arrest"
            
    if 1 not in event_history:
        return "Inviable: G1 Arrest" if max_clb2 < 0.2 else "Inviable: M-phase Arrest"

    div_indices = [i for i, e in enumerate(event_history) if e == 1]
    last_div_idx = 0
    for i, h_idx in enumerate(div_indices):
        if bud_at_div[i] < 0.95: return "Inviable: Unbudded Division"
        if mass_at_div[i] < 0.8: return "Inviable: Premature Division"
            
        cycle_events = event_history[last_div_idx:h_idx]
        cleaned = []
        for e in cycle_events:
            if not cleaned or cleaned[-1] != e:
                cleaned.append(e)
                
        if 3 not in cleaned or 4 not in cleaned: return "Inviable: Sequence Failure"
        idx3, idx4 = cleaned.index(3), cleaned.index(4)
        if idx3 > idx4: return "Inviable: Sequence Failure"
        
        if i > 0: 
            if 2 not in cleaned: return "Inviable: Sequence Failure"
            if cleaned.index(2) > idx3: return "Inviable: Sequence Failure"
                
        last_div_idx = h_idx + 1 
    return "Viable"

def analyze_mass_pattern(mass_curve):
    diff = np.diff(mass_curve)
    drop_indices = np.where(diff < -0.1)[0]
    
    valid_drops = 0
    if len(drop_indices) > 0:
        valid_drops = 1
        last_drop = drop_indices[0]
        for d in drop_indices[1:]:
            if d - last_drop > 10: 
                valid_drops += 1
            last_drop = d
            
    if valid_drops >= 2: return "Pattern_A: Multi-Division (>=2)"
    elif valid_drops == 1: return "Pattern_B: Single-Division (==1)"
    else: return "Pattern_C: Exponential-Arrest (==0)"

def build_discrete_pools(mutants_dict):
    """自动扫描并构建单变量和成对变量的合法真值集合"""
    defaults = get_default_params()
    defaults.update({
        'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 
        'init_BUB2': 0.2, 'BUB2_active': 1.0, 'is_mutant_104': False, 'ESP1T': 1.0
    })

    single_pools = {p: set([defaults[p]]) for p in DISCRETE_STRICT_PARAMS 
                    if p not in ['init_CDH1T', 'init_CDH1', 'init_BUB2', 'BUB2_active']}
    
    cdh1_pairs = set([(defaults['init_CDH1T'], defaults['init_CDH1'])])
    bub2_pairs = set([(defaults['init_BUB2'], defaults['BUB2_active'])])

    for rules in mutants_dict.values():
        for p in single_pools.keys():
            if p in rules:
                single_pools[p].add(rules[p])
        
        if 'init_CDH1T' in rules or 'init_CDH1' in rules:
            val_T = rules.get('init_CDH1T', defaults['init_CDH1T'])
            val_A = rules.get('init_CDH1', defaults['init_CDH1'])
            cdh1_pairs.add((val_T, val_A))
            
        if 'init_BUB2' in rules or 'BUB2_active' in rules:
            val_init = rules.get('init_BUB2', defaults['init_BUB2'])
            val_act = rules.get('BUB2_active', defaults['BUB2_active'])
            bub2_pairs.add((val_init, val_act))

    single_pools = {p: list(v) for p, v in single_pools.items()}
    return single_pools, list(cdh1_pairs), list(bub2_pairs)

def simulate_mutant(mutant_name, param_overrides, tf=SIM_TF, num_steps=SIM_STEPS):
    p = get_default_params()
    y0 = initial_conds()
    
    for k, v in param_overrides.items():
        if k == 'init_CDH1T': y0[nCDH1T] = v
        elif k == 'init_CDH1': y0[nCDH1] = v
        elif k == 'init_BUB2': y0[nBUB2] = v
        else:
            p[k] = v
        
    t_raw, y_raw = [0], [y0]
    event_history = []
    bud_at_div, mass_at_div = [], []

    t0, loop_count = 0, 0
    start_cpu_time = time.time()
    
    def eqns_with_timeout(t, y, p_args):
        if time.time() - start_cpu_time > SIM_TIMEOUT:
            raise TimeoutError("Fortran ODE Stiff Deadlock Intercepted!")
        return eqns(t, y, p_args)

    while t0 < tf and loop_count < 300: 
        loop_count += 1
        sol = solve_ivp(eqns_with_timeout, [t0, tf], y0, args=(p,), method='BDF', 
                        events=[event1, event2, event3, event4], 
                        rtol=1e-4, atol=1e-6)
        
        t_raw.extend(sol.t[1:])
        y_raw.extend(sol.y.T[1:])
        
        if sol.status == 1:
            t0, y0 = sol.t[-1], sol.y[:, -1].copy()
            ie = next((i+1 for i, te in enumerate(sol.t_events) if len(te) > 0 and np.isclose(te[-1], t0)), -1)

            if ie != -1: event_history.append(ie)
            
            if ie == 1:
                bud_at_div.append(y0[nBUD]); mass_at_div.append(y0[nMASS])
                y0[nBUD] = y0[nSPN] = 0
                kg = np.log(2) / p['mdt']
                y0[nMASS] = abs(1 + np.exp(-kg * ((1.026 / kg) - 32)) - 1) * y0[nMASS]
                y0[nlte1] = 0.1; y0[nCLB2] -= 1e-6  
            elif ie == 2:
                y0[nORI] = 0; y0[nCLB5] -= 1e-6  
            elif ie == 3:
                y0[nVi20] = p['Vi20_active']; y0[nBUB2] = p['BUB2_active']; y0[nORI] += 1e-6
            elif ie == 4:
                y0[nVi20] = 0.01; y0[nlte1] = 1.0; 
                y0[nBUB2] = param_overrides.get('init_BUB2', 0.2) 
                y0[nSPN] += 1e-6   
        else:
            break

    t_raw, y_raw = np.array(t_raw), np.array(y_raw).T 
    interpolator = interp1d(t_raw, y_raw, kind='linear', axis=1, fill_value="extrapolate")
    y_uniform = interpolator(np.linspace(0, tf, num_steps))

    max_mass = np.max(y_uniform[nMASS])
    if SIM_TF <= 210 and max_mass > 10.0:
        return mutant_name, None, "Discarded_Mass", None

    bio_label = analyze_cell_viability_unified(event_history, bud_at_div, mass_at_div, np.max(y_uniform[nMASS]), np.max(y_uniform[nCLB2]))
    pattern_label = analyze_mass_pattern(y_uniform[nMASS])
    
    return mutant_name, y_uniform, bio_label, pattern_label

# ==========================================
# 3. 核心突变体规则字典 (请用你完整的 127 种突变字典替换此处！)
# ==========================================
mutant_rules = {
    '1_WT_Glc': {},
    '2.1_WT_Gal': {'mdt': 150.0},
    '2.2_WT_Raff': {'mdt': 160.0},
    '3_cln1_cln2_KO': {'ksn2pp': 0.0},
    '4_GAL_CLN2_cln1_cln2_KO': {'ksn2p': 0.12, 'ksn2pp': 0.0, 'mdt': 150.0},
    '5_cln1_cln2_sic1_KO': {'ksn2pp': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0},
    '6_cln1_cln2_cdh1_KO': {'ksn2pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '7_GAL_CLN2_cln1_cln2_cdh1_KO': {'ksn2p': 0.12, 'ksn2pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0},
    '8_cln3_KO': {'C0': 0.0},
    '9_GAL_CLN3': {'C0': 0.4 * 5, 'mdt': 150.0},
    '10_bck2_KO': {'B0': 0.0},
    '11_multi_BCK2': {'B0': 0.054 * 5},
    '12_cln1_cln2_bck2_KO': {'ksn2pp': 0.0, 'B0': 0.0},
    '13_cln3_bck2_KO': {'C0': 0.0, 'B0': 0.0},
    '14_cln3_bck2_GAL_CLN2_cln1_cln2': {'C0': 0.0, 'B0': 0.0, 'ksn2p': 0.12, 'ksn2pp': 0.0, 'mdt': 150.0},
    '15_cln3_bck2_multi_CLN2': {'C0': 0.0, 'B0': 0.0, 'ksn2pp': 0.15 * 5},
    '16_cln3_bck2_sic1_KO': {'C0': 0.0, 'B0': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0},
    '17_cln1_cln3_KO': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0}, 
    '18_cln1_cln3_GAL_CLN2': {'ksn2p': 0.12, 'ksn2pp': 0.0, 'C0': 0.0, 'mdt': 150.0},
    '19_cln1_cln2_cln3_GAL_CLN3': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.4 * 5, 'mdt': 150.0},
    '20_cln1_cln2_cln3_sic1': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0},
    '21_cln1_cln2_cln3_cdh1': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '22_cln1_cln2_cln3_multi_CLB5': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'ksb5p': 0.0008 * 4, 'ksb5pp': 0.005 * 4},
    '23_cln1_cln2_cln3_GAL_CLB5': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'ksb5p': 0.0008 * 15, 'mdt': 150.0},
    '24_cln1_cln2_cln3_multi_BCK2': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'B0': 0.054 * 10},
    '25_cln1_cln2_cln3_GAL_CLB2': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'ksb2p': 0.001 * 2, 'mdt': 150.0},
    '26_cln1_cln2_cln3_apc_ts': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'C0': 0.0, 'kscdh': 0.0, 'ks20p': 0.0, 'ks20pp': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '27_sic1_KO': {'ksc1p': 0.0, 'ksc1pp': 0.0},
    '28_GAL_SIC1': {'ksc1p': 0.012 * 10, 'mdt': 150.0},
    '29_GAL_SIC1_db': {'ksc1p': 0.012 * 10, 'kd3c1': 0.0, 'mdt': 150.0},
    '30_GAL_SIC1_cln1_cln2': {'ksc1p': 0.012 * 10, 'ksn2pp': 0.0, 'mdt': 150.0},
    '31_GAL_SIC1_GAL_CLN2_cln1_cln2': {'ksc1p': 0.012 * 10, 'ksn2p': 0.12, 'ksn2pp': 0.0, 'mdt': 150.0},
    '32_GAL_SIC1_cln1_cln2_cdh1': {'ksc1p': 0.12, 'ksc1pp': 0.12, 'ksn2pp': 0.0, 'kscdh': 0.0, 'mdt': 150.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '33_GAL_SIC1_GAL_CLN2_cdh1': {'ksc1p': 0.12, 'ksn2p': 0.12, 'ksn2pp': 0.0, 'kscdh': 0.0, 'mdt': 150.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '34_cdh1_KO': {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '35_Cdh1_constitutively_active': {'kicdhpp': 0.0, 'kscdh': 0.01 * 3, 'mdt': 150.0},
    '36_sic1_cdh1': {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '37_sic1_cdh1_GAL_CDC20': {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'ks20p': 10.0, 'mdt': 150.0},
    '38_cdc6_42_49': {'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0},
    '39_sic1_cdc6_42_49': {'ksc1p': 0.0, 'ksc1pp': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0},
    '40_cdh1_cdc6_42_49': {'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '41_sic1_cdc6_cdh1': {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kscdh': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '42_sic1_cdc6_cdh1_GAL_CDC20': {'ksc1p': 0.0, 'ksc1pp': 0.0, 'ksf6p': 0.0, 'ksf6pp': 0.0, 'ksf6ppp': 0.0, 'kscdh': 0.0, 'ks20p': 4.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0},
    '43_swi5_KO': {'ksswip': 0.0, 'ksswipp': 0.0},
    '44_swi5_GAL_CLB2': {'ksswip': 0.0, 'ksswipp': 0.0, 'ksb2p': 0.12, 'mdt': 150.0},
    '45_swi5_cdh1': {'ksswip': 0.0, 'ksswipp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '46_swi5_cdh1_GAL_SIC1': {'ksswip': 0.0, 'ksswipp': 0.0, 'kscdh': 0.0, 'ksc1p': 0.012 * 10, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0},
    '47_clb1_clb2_KO': {'ksb2p': 0.0, 'ksb2pp': 0.0},
    '48_CLB1_clb2': {'ksb2p': 0.001 * 0.33, 'ksb2pp': 0.04 * 0.33},
    '49_GAL_CLB2': {'ksb2p': 0.12, 'mdt': 150.0},
    '50_multi_GAL_CLB2': {'ksb2p': 0.96, 'mdt': 150.0},
    '51_CLB1_clb2_cdh1': {'ksb2p': 0.001 * 0.33, 'ksb2pp': 0.04 * 0.33, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '52_CLB1_clb2_pds1': {'ksb2p': 0.001 * 0.33, 'ksb2pp': 0.04 * 0.33, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0},
    '53_GAL_CLB2_sic1': {'ksb2p': 0.12, 'ksc1p': 0.0, 'ksc1pp': 0.0, 'mdt': 150.0},
    '54_GAL_CLB2_cdh1': {'ksb2p': 0.12, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0},
    '55_CLB2_db': {'kdb2P': 0.0, 'kdb2pp': 0.03},
    '56_CLB2_db_Gal': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'mdt': 150.0},
    '57_CLB2_db_multi_SIC1': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksc1p': 0.012 * 10, 'ksc1pp': 0.12 * 10},
    '58_CLB2_db_GAL_SIC1': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksc1p': 0.012 * 10, 'mdt': 150.0},
    '59_CLB2_db_multi_CDC6': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksf6p': 0.024 * 5, 'ksf6pp': 0.12 * 5, 'ksf6ppp': 0.004 * 5},
    '60_CLB2_db_clb5': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksb5p': 0.0, 'ksb5pp': 0.0},
    '61.1_CLB2_db_clb5_Gal': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksb5p': 0.0, 'ksb5pp': 0.0, 'mdt': 150.0},
    '61.2_CLB2_db_clb5_Raff': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksb5p': 0.0, 'ksb5pp': 0.0, 'mdt': 160.0},
    '62_GAL_CLB2_db': {'kdb2P': 0.0, 'kdb2pp': 0.03, 'ksb2pp': 0.12, 'mdt': 150.0},
    '63_clb5_clb6_KO': {'ksb5p': 0.0, 'ksb5pp': 0.0},
    '64_cln1_cln2_clb5_clb6': {'ksn2p': 0.0, 'ksn2pp': 0.0, 'ksb5p': 0.0, 'ksb5pp': 0.0},
    '65_GAL_CLB5': {'ksb5p': 0.0008 * 15, 'mdt': 150.0},
    '66_GAL_CLB5_sic1': {'ksb5p': 0.0008 * 15, 'ksc1p': 0.0, 'ksc1pp': 0.0, 'mdt': 150.0},
    '67_GAL_CLB5_cdh1': {'ksb5p': 0.0008 * 10, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0},
    '68_CLB5_db': {'kdb5pp': 0.0},
    '69_CLB5_db_sic1': {'kdb5pp': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0},
    '70_CLB5_db_pds1': {'kdb5pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0},
    '71_CLB5_db_pds1_cdc20': {'kdb5pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0, 'ks20p': 0.0, 'ks20pp': 0.0},
    '72_GAL_CLB5_db': {'ksb5p': 0.0008 * 15, 'kdb5pp': 0.0, 'mdt': 150.0},
    '73_cdc20_ts': {'ks20p': 0.0, 'ks20pp': 0.0},
    '74_cdc20_clb5': {'ks20p': 0.0, 'ks20pp': 0.0, 'ksb5p': 0.0, 'ksb5pp': 0.0}, 
    '75_cdc20_pds1': {'ks20p': 0.0, 'ks20pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0},
    '76_cdc20_pds1_clb5': {'ks20p': 0.0, 'ks20pp': 0.0, 'ksb5p': 0.0, 'ksb5pp': 0.0, 'ks1pdspp': 0.0, 'ks2pdspp': 0.0},
    '77_GAL_CDC20': {'ks20p': 6.0, 'mdt': 150.0},
    '78_cdc20_ts_mad2': {'ks20p': 0.0, 'ks20pp': 0.0, 'Vi20_active': 0.01},
    '79_cdc20_ts_bub2': {'ks20p': 0.0, 'ks20pp': 0.0, 'BUB2_active': 0.0, 'init_BUB2': 0.0},
    '80_pds1_KO': {'ks1pdspp': 0.0, 'ks2pdspp': 0.0},
    '81_esp1_ts': {'kasesp': 0.1, 'kdiesp': 0.002},
    '82_PDS1_db': {'kd2pdspp': 0.0, 'kd3pdspp': 0.0},
    '83_GAL_PDS1_db': {'kspdsp': 0.1, 'kd2pdspp': 0.0, 'kd3pdspp': 0.0, 'mdt': 150.0},
    '84_GAL_PDS1_db_esp1_ts': {'kspdsp': 0.1, 'kd2pdspp': 0.0, 'kd3pdspp': 0.0, 'kasesp': 0.1, 'kdiesp': 0.002, 'mdt': 150.0},
    '85_GAL_ESP1_cdc20_ts': {'ks20p': 0.0, 'ks20pp': 0.0, 'ESP1T': 3.0, 'mdt': 150.0}, 
    '86_tem1_KO': {'ka15pp': 0.002, 'Tem1T': 0.0},
    '87_GAL_TEM1': {'Tem1T': 5.0, 'mdt': 150.0},
    '88_tem1_ts_multi_CDC15': {'ka15pp': 0.002, 'Tem1T': 0.0, 'Cdc15T': 5.0},
    '89_tem1_ts_GAL_CDC15': {'ka15pp': 0.002, 'Tem1T': 0.0, 'Cdc15T': 15.0, 'mdt': 150.0},
    '90_tem1_net1_ts': {'ka15pp': 0.002, 'Tem1T': 0.0, 'kasrent': 10.0, 'kasrentp': 0.05},
    '91_tem1_ts_multi_CDC14': {'ka15pp': 0.002, 'Tem1T': 0.0, 'ks14': 0.4},
    '92_cdc15_KO': {'kkpnetpp': 0.0, 'Cdc15T': 0.0},
    '93_multi_CDC15': {'Cdc15T': 5.0},
    '94_cdc15_ts_multi_TEM1': {'kkpnetpp': 0.0, 'Cdc15T': 0.0, 'Tem1T': 5.0},
    '95_cdc15_net1_ts': {'kkpnetpp': 0.0, 'kasrentp': 0.05, 'kasrent': 10.0},
    '96_cdc15_ts_multi_CDC14': {'kkpnetpp': 0.0, 'ks14': 0.4},
    '97_net1_ts': {'kasrentp': 0.05, 'kasrent': 10.0},
    '98_GAL_NET1': {'ksnet': 0.084 * 5, 'mdt': 150.0},
    '99_cdc14_ts': {'ks14': 0.0},
    '100_GAL_CDC14': {'ks14': 0.2 * 4, 'mdt': 150.0},
    '101_GAL_NET1_GAL_CDC14': {'ksnet': 0.084 * 3, 'ks14': 0.2 * 3, 'mdt': 150.0},
    '102_net1_cdc20_ts': {'kasrentp': 0.05, 'kasrent': 10.0, 'ks20p': 0.0, 'ks20pp': 0.0},
    '103_cdc14_ts_GAL_SIC1': {'ks14': 0.0, 'ksc1p': 0.012 * 10, 'mdt': 150.0},
    '104_cdc14_ts_then_GAL_SIC1': {'ks14': 0.0, 'is_mutant_104': True, 'mdt': 150.0}, 
    '105_cdc14_ts_sic1': {'ksc1p': 0.0, 'ksc1pp': 0.0, 'kppc1': 0.64, 'kppf6': 0.64, 'kaswi': 0.32, 'kacdhpp': 0.128},
    '106_cdc14_ts_cdh1': {'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'kppc1': 1.2, 'kppf6': 1.2, 'kaswi': 0.6, 'kacdhpp': 0.13},
    '107_cdc14_ts_GAL_CLN2': {'ksn2p': 0.12, 'ksn2pp': 0.0, 'kppc1': 0.8, 'kppf6': 0.8, 'kaswi': 0.4, 'kacdhpp': 0.16, 'mdt': 150.0},
    '108_TAB6_1': {'kasrentp': 0.05, 'kasrent': 10.0},
    '109_TAB6_1_cdc15': {'kasrentp': 0.05, 'kasrent': 10.0, 'kkpnetpp': 0.0},
    '110_TAB6_1_clb5_clb6': {'kasrentp': 0.05, 'kasrent': 10.0, 'ksb5p': 0.0, 'ksb5pp': 0.0},
    '111_TAB6_1_CLB1_clb2': {'kasrentp': 0.05, 'kasrent': 10.0, 'ksb2p': 0.001 * 0.33, 'ksb2pp': 0.04 * 0.33},
    '112_mad2_KO': {'Vi20_active': 0.01},
    '113_bub2_KO': {'BUB2_active': 0.0, 'init_BUB2': 0.0},
    '114_mad2_bub2': {'Vi20_active': 0.01, 'BUB2_active': 0.0, 'init_BUB2': 0.0},
    '115_APC_A': {'ka20pp': 0.0},
    '116_APC_A_cdh1': {'ka20pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '117_APC_A_cdh1_Gal': {'ka20pp': 0.0, 'kscdh': 0.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0, 'mdt': 150.0},
    '118_APC_A_cdh1_multi_SIC1': {'ka20pp': 0.0, 'kscdh': 0.0, 'ksc1p': 0.012 * 15, 'ksc1pp': 0.12 * 15, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '119_APC_A_cdh1_GAL_SIC1': {'ka20pp': 0.0, 'kscdh': 0.0, 'ksc1p': 0.012 * 20, 'mdt': 150.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '120_APC_A_cdh1_multi_CDC6': {'ka20pp': 0.0, 'kscdh': 0.0, 'ksf6p': 0.024 * 10, 'ksf6pp': 0.12 * 10, 'ksf6ppp': 0.004 * 10, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '121_APC_A_cdh1_GAL_CDC6': {'ka20pp': 0.0, 'kscdh': 0.0, 'ksf6p': 0.024 * 15, 'mdt': 150.0, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '122_APC_A_cdh1_multi_CDC20': {'ka20pp': 0.0, 'kscdh': 0.0, 'ks20p': 0.006 * 25, 'ks20pp': 0.6 * 25, 'init_CDH1T': 0.0, 'init_CDH1': 0.0},
    '123_APC_A_sic1': {'ka20pp': 0.0, 'ksc1p': 0.0, 'ksc1pp': 0.0},
    '124_APC_A_GAL_CLB2': {'ka20pp': 0.0, 'ksb2p': 0.48, 'mdt': 150.0}
}

# ==========================================
# 4. 【核心重构】生物锁死 + 先验概率盲抽引擎
# ==========================================
def build_discrete_pools(mutants_dict):
    """自动扫描并构建单变量和成对变量的历史合法真值集合"""
    defaults = get_default_params()
    defaults.update({
        'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 
        'init_BUB2': 0.2, 'BUB2_active': 1.0, 'is_mutant_104': False, 'ESP1T': 1.0
    })

    single_pools = {p: set([defaults[p]]) for p in DISCRETE_STRICT_PARAMS 
                    if p not in ['init_CDH1T', 'init_CDH1', 'init_BUB2', 'BUB2_active']}
    
    cdh1_pairs = set([(defaults['init_CDH1T'], defaults['init_CDH1'])])
    bub2_pairs = set([(defaults['init_BUB2'], defaults['BUB2_active'])])

    for rules in mutants_dict.values():
        for p in single_pools.keys():
            if p in rules:
                single_pools[p].add(rules[p])
        
        if 'init_CDH1T' in rules or 'init_CDH1' in rules:
            val_T = rules.get('init_CDH1T', defaults['init_CDH1T'])
            val_A = rules.get('init_CDH1', defaults['init_CDH1'])
            cdh1_pairs.add((val_T, val_A))
            
        if 'init_BUB2' in rules or 'BUB2_active' in rules:
            val_init = rules.get('init_BUB2', defaults['init_BUB2'])
            val_act = rules.get('BUB2_active', defaults['BUB2_active'])
            bub2_pairs.add((val_init, val_act))

    single_pools = {p: list(v) for p, v in single_pools.items()}
    return single_pools, list(cdh1_pairs), list(bub2_pairs)


def generate_lhs_tasks(num_lhs_samples):
    """融合了 v1 离散物理保护与 v0 概率稀疏掩码的全新全局盲抽引擎"""
    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2, 'BUB2_active': 1.0, 'is_mutant_104': False, 'ESP1T': 1.0})
    
    # 提取离散真值图鉴
    single_pools, cdh1_pairs, bub2_pairs = build_discrete_pools(mutant_rules)
    
    target_params = sorted(list(defaults.keys()))
    dim = len(target_params)
    
    # 1. 扫描整个突变体字典，统计频次和连续变量的历史极值
    mutation_counts = {key: 0 for key in target_params}
    historical_c_values = {key: [] for key in target_params}
    
    for rules in mutant_rules.values():
        for key, val in rules.items():
            mutation_counts[key] += 1
            if key not in DISCRETE_STRICT_PARAMS:
                historical_c_values[key].append(val)
                
    max_count = max(mutation_counts.values()) if mutation_counts.values() else 1
    p_mutate = {}
    c_bounds = {}
    
    # 2. 🌟 采纳用户的天才建议：计算概率分布，未突变过的参数设为 0
    for key in target_params:
        count = mutation_counts[key]
        if count == 0:
            p_mutate[key] = 0.0  # 绝对的生物学背景锁定！
        else:
            p_mutate[key] = 0.2 + 0.6 * (count / max_count)
            
        # 预先推导连续变量的动态极值边界
        if key not in DISCRETE_STRICT_PARAMS and count > 0:
            vals = historical_c_values[key] + [defaults[key]]
            min_v, max_v = min(vals), max(vals)
            l_bound = 0.0 if min_v <= 0 else min_v * 0.1
            u_bound = 0.01 if max_v <= 0 else max_v * 1.5
            if u_bound <= l_bound:
                u_bound = l_bound + 0.01
            c_bounds[key] = (l_bound, u_bound)
            
    # 3. 准备 LHS 矩阵和概率掩码阵列
    sampler = qmc.LatinHypercube(d=dim, seed=2026)
    sample_unit = sampler.random(n=num_lhs_samples)
    mask_unit = np.random.rand(num_lhs_samples, dim)
    
    lhs_tasks_dict = {}
    
    for i in range(num_lhs_samples):
        overrides = {}
        sampled_cdh1, sampled_bub2 = False, False
        
        for j, key in enumerate(target_params):
            # 🌟 概率稀疏掩码：如果不幸没有摇中突变概率，直接跳过（保持默认 WT）
            if mask_unit[i, j] > p_mutate[key]:
                continue
                
            # 摇中突变概率，开始进入 v1.1 级别的硬核物理保护分发！
            if key in DISCRETE_STRICT_PARAMS:
                
                # 成对锁死盲抽逻辑 (Pair-Locked Blind Masking)
                if key in ['init_CDH1T', 'init_CDH1']:
                    if not sampled_cdh1:
                        pair = cdh1_pairs[np.random.choice(len(cdh1_pairs))]
                        if not np.isclose(pair[0], defaults['init_CDH1T']) or not np.isclose(pair[1], defaults['init_CDH1']):
                            overrides['init_CDH1T'] = pair[0]
                            overrides['init_CDH1'] = pair[1]
                        sampled_cdh1 = True
                        
                elif key in ['init_BUB2', 'BUB2_active']:
                    if not sampled_bub2:
                        pair = bub2_pairs[np.random.choice(len(bub2_pairs))]
                        if not np.isclose(pair[0], defaults['init_BUB2']) or not np.isclose(pair[1], defaults['BUB2_active']):
                            overrides['init_BUB2'] = pair[0]
                            overrides['BUB2_active'] = pair[1]
                        sampled_bub2 = True
                        
                # 严格单体图鉴盲抽 (含 Bool 保护)
                else:
                    val = np.random.choice(single_pools[key])
                    # 处理 numpy bool 写入 JSON/Dict 的小毛病
                    if key == 'is_mutant_104': val = bool(val) 
                    
                    if not np.isclose(val, defaults[key], rtol=1e-5, atol=1e-7):
                        overrides[key] = val
            else:
                # 连续参数空间的高维探索
                l_bound, u_bound = c_bounds[key]
                val = l_bound + sample_unit[i, j] * (u_bound - l_bound)
                overrides[key] = float(val)

        lhs_tasks_dict[f"LHS_Sample_{i:05d}"] = overrides
        
    print("\n" + "="*50)
    print("📊 [重构版探索者引擎] 概率稀疏与生物图鉴锁定完成")
    print(f"🔹 纳入采样参数: {dim} 个")
    print(f"🔹 未知常数背景已冻结 (Mutation Prob = 0.0)")
    print(f"🔹 离散与成对质量守恒已受历史图鉴保护")
    print("="*50 + "\n")
    
    return lhs_tasks_dict

# ==========================================
# 5. 主程序：大一统数据整合与保存
# ==========================================
if __name__ == '__main__':
    
    all_tasks = {} 
    all_tasks.update(mutant_rules)
    
    # 将 LHS 任务并入大一统任务字典
    lhs_tasks = generate_lhs_tasks(NUM_LHS_SAMPLES)
    all_tasks.update(lhs_tasks)
    
    ordered_names = list(all_tasks.keys())
    results = {}
    bio_labels_dict = {}
    pattern_labels_dict = {}
    
    print(f"🔥 启动集群: 共计 {len(ordered_names)} 个细胞，单点超时设定 {SIM_TIMEOUT}s...")
    start_time = time.time()
    
    timeout_count = 0
    fail_count = 0
    mass_drop_count = 0  
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_name = {
            executor.submit(simulate_mutant, name, all_tasks[name], SIM_TF, SIM_STEPS): name 
            for name in ordered_names
        }
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_name)):
            name = future_to_name[future]
            try:
                m_name, y_uniform, m_bio_label, m_pattern_label = future.result(timeout=SIM_TIMEOUT + 10.0)
                
                if y_uniform is not None:
                    results[m_name] = y_uniform
                    bio_labels_dict[m_name] = m_bio_label
                    pattern_labels_dict[m_name] = m_pattern_label
                elif m_bio_label == "Discarded_Mass":
                    mass_drop_count += 1
                    
            except concurrent.futures.TimeoutError:
                print(f"⚠️ [{name}] ODE 求解死锁，强行切断 (超 {SIM_TIMEOUT}s)")
                timeout_count += 1
            except Exception as e:
                print(f"❌ [{name}] 求解失败: {e}")
                fail_count += 1
                
            if (idx + 1) % 500 == 0:
                print(f"  └─ 进度: {idx + 1} / {len(ordered_names)}")

    print(f"\n💥 统计: 成功保留 {len(results)} 个，因MASS过大扔掉 {mass_drop_count} 个，超时 {timeout_count} 个，失败 {fail_count} 个。")

    # ==========================================
    # 🌟 [新增] 第三级防御：动态浓度异常剔除系统 (Data Pruning)
    # ==========================================
    RELATIVE_MULTIPLIER = 2.0 
    ABSOLUTE_SAFE_LIMIT = 2.0 
    
    # 1. 提取本次模拟中所有存活的 Real Mutants 的极值包围网
    real_max_vals = []
    for name in ordered_names:
        if name in results and "LHS" not in name:
            real_max_vals.append(np.max(results[name], axis=1))
            
    concentration_drop_count = 0
    if real_max_vals:
        real_envelope = np.max(real_max_vals, axis=0)
        dynamic_thresholds = np.maximum(real_envelope * RELATIVE_MULTIPLIER, ABSOLUTE_SAFE_LIMIT)
        
        # 2. 扫描 LHS 样本，抓捕浓度爆炸的怪物细胞
        names_to_delete = set()
        for name in ordered_names:
            if name in results and "LHS" in name:
                lhs_max_vals = np.max(results[name], axis=1)
                for var_idx in range(1, 39):
                    if lhs_max_vals[var_idx] > dynamic_thresholds[var_idx]:
                        names_to_delete.add(name)
                        break 
                        
        # 3. 内存销毁
        for name in names_to_delete:
            del results[name] 
            concentration_drop_count += 1
            
        print(f"\n🛡️ 动态浓度剔除系统执行完毕！")
        print(f"  └─ 成功拦截并销毁了 {concentration_drop_count} 个浓度爆炸的怪物细胞。")
    else:
        print("\n⚠️ 未检测到可用的 Real Mutant 基准，跳过动态浓度清洗。")

    # ==========================================
    # 6. 对齐参数矩阵与格式保存
    # ==========================================
    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104']) 
    
    ordered_tensors = []
    final_names = []
    final_params = []
    final_bio_labels = [] 
    final_pattern_labels = [] 
    
    for name in ordered_names:
        if name in results:
            ordered_tensors.append(results[name])
            final_names.append(name)
            final_bio_labels.append(bio_labels_dict[name]) 
            final_pattern_labels.append(pattern_labels_dict[name])
            
            overrides = all_tasks[name]
            p_vec = [overrides.get(k, defaults[k]) for k in param_keys]
            final_params.append(p_vec)
            
    all_tensors = np.array(ordered_tensors)
    all_params = np.array(final_params)
    all_bio_labels = np.array(final_bio_labels)
    all_pattern_labels = np.array(final_pattern_labels)
    
    np.savez_compressed(DATASET_NAME, 
                        data=all_tensors, 
                        params=all_params,
                        param_names=param_keys,
                        mutant_names=final_names,
                        labels=all_bio_labels,             
                        pattern_labels=all_pattern_labels) 
                        
    print(f"\n📊 保存成功！文件: {DATASET_NAME}")
    print(f"✅ 耗时: {time.time() - start_time:.2f} 秒")
    print(f"📊 张量维度 (Samples, Vars, Timesteps): {all_tensors.shape}")
    print(f"📊 参数维度 (Samples, Param_Count): {all_params.shape}")
    
    print("\n🔬 【Biological Labels】 生物学机理标签分布:")
    u_bio, c_bio = np.unique(all_bio_labels, return_counts=True)
    for lbl, count in zip(u_bio, c_bio):
        print(f"  - {lbl}: {count} 个")

    print("\n📈 【Pattern Labels】 视觉形态学标签分布:")
    u_pat, c_pat = np.unique(all_pattern_labels, return_counts=True)
    for lbl, count in zip(u_pat, c_pat):
        print(f"  - {lbl}: {count} 个")