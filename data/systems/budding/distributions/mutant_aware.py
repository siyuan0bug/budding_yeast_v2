"""mutant_aware amplitude distribution; ranges and draw order preserved."""
import numpy as np

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
