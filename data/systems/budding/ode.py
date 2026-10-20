"""Unchanged MA RHS, Goldbeter-Koshland helper and four event guards."""
import numpy as np
from .states import nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2
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

def event1(t, y, p): return y[nCLB2] - 0.3

event1.terminal = True; event1.direction = -1

def event2(t, y, p): return y[nCLB2] + y[nCLB5] - 0.2

event2.terminal = True; event2.direction = -1

def event3(t, y, p): return y[nORI] - 1.0

event3.terminal = True; event3.direction = 1

def event4(t, y, p): return y[nSPN] - 1.0

event4.terminal = True; event4.direction = 1
