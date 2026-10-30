"""MA single-sample solver; initialization factored without changing assignments/order."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from ...simulation_guards import bounded_simulation, validate_segment
from .parameters import get_default_params
from .states import initial_conds
from .states import nMASS, nCLN2, nCLB2, nCLB5, nSIC1, nCDC6, nC2, nC5, nF2, nF5, nSIC1P, nC2P, nC5P, nCDC6P, nF2P, nF5P, nSWI5T, nSWI5, nIEP, nCDC20T, nCDC20, nCDH1T, nCDH1, nCDC14T, nCDC14, nNET1T, nNET1, nRENT, nTEM1, nCDC15, nPPX, nPDS1, nESP1, nORI, nBUD, nSPN, nVi20, nlte1, nBUB2
from .ode import eqns, event1, event2, event3, event4
from .phenotypes import analyze_cell_viability_unified, analyze_mass_pattern

def prepare_initial_conditions(ovr):
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

    return p, y0, is_mad2_ko, is_bub2_ko

@bounded_simulation
def simulate_mutant(name, ovr, tf=210, ns=500):
    p, y0, is_mad2_ko, is_bub2_ko = prepare_initial_conditions(ovr)
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
    max_b2 = np.max(yu[nCLB2])
    bio_l = analyze_cell_viability_unified(eh, bd, md, max_m, max_b2)
    pat_l = analyze_mass_pattern(yu[nMASS])
    return name, yu, bio_l, pat_l
