"""mutant_aware synthetic mutation assignment, independent of the canonical biological engine."""
import numpy as np

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
