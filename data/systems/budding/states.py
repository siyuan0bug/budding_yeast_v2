"""State order and initial values extracted from mutant_aware.py; no legacy import."""
import numpy as np

(nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, 
 nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, 
 nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, 
 nVi20, nlte1, nBUB2) = range(39)
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

VARIABLE_NAMES = ('MASS CLN2 CLB2 CLB5 SIC1 CDC6 C2 C5 F2 F5 SIC1P C2P C5P CDC6P F2P F5P '
                  'SWI5T SWI5 IEP CDC20T CDC20 CDH1T CDH1 CDC14T CDC14 NET1T NET1 RENT '
                  'TEM1 CDC15 PPX PDS1 ESP1 ORI BUD SPN Vi20 lte1 BUB2').split()
INIT_Y0_MAP = {'init_' + name: i for i, name in enumerate(VARIABLE_NAMES)}
INIT_Y0_MAP.update({'init_Tem1': nTEM1, 'init_Cdc15': nCDC15})
