"""WT-Blind specialized QSS/inversion/ledger algorithms, using its pruned specialized topology and original strength policy."""
import numpy as np
import math
from ..parameters import get_default_params
from ..ode import GK
from ..catalog import BINDING_GROUPS

EPS = 1e-12
M_BIRTH = 1.2060194
M_M = 2 * M_BIRTH
SIM_WINDOW = 210.0
N_BINS = 5
TRAJ_MAX = 1000.0
N_LOG_BANDS = 8
BOUNDARY_LO, BOUNDARY_HI = (0.5, 2.0)
MASS_LIMIT = 10.0
MODULE_EDGES = [('CLB2', 'CDC20', '+'),
 ('CDC20', 'CLB2', '-'),
 ('CDC20', 'CLB5', '-'),
 ('CDC20', 'PDS1', '-'),
 ('CDH1', 'CLB2', '-'),
 ('CDH1', 'PDS1', '-'),
 ('CDC14', 'CDH1', '+'),
 ('CDC14', 'SIC1', '+'),
 ('CDC14', 'CDC6', '+'),
 ('CLB2', 'CDH1', '-'),
 ('CLB5', 'CDH1', '-'),
 ('CLN2', 'CDH1', '-'),
 ('CLN3', 'CDH1', '-'),
 ('SIC1', 'CLB2', '-'),
 ('SIC1', 'CLB5', '-'),
 ('CDC6', 'CLB2', '-'),
 ('CDC6', 'CLB5', '-'),
 ('SWI5', 'SIC1', '+'),
 ('SWI5', 'CDC6', '+'),
 ('NET1', 'CDC14', '-'),
 ('CDC15', 'NET1', '+'),
 ('TEM1', 'CDC15', '+'),
 ('BUB2', 'TEM1', '-'),
 ('MAD2', 'CDC20', '-'),
 ('PDS1', 'ESP1', '-')]
GROUP_BY_NAME = {g['name']: g for g in BINDING_GROUPS.values()}
GENE_NAMES = sorted(GROUP_BY_NAME.keys())
USE_ENGINE_P_BIO = False
if USE_ENGINE_P_BIO:
    P_BIO = {n: GROUP_BY_NAME[n]['P_bio_G'] for n in GENE_NAMES}
else:
    _deg = {n: 0 for n in GENE_NAMES}
    for _a, _b, _s in MODULE_EDGES:
        _deg[_a] += 1
        _deg[_b] += 1
    _raw = {n: _deg[n] + 1.0 for n in GENE_NAMES}
    _s = sum(_raw.values())
    P_BIO = {n: _raw[n] / _s for n in GENE_NAMES}
_BIO_SUM = sum(P_BIO.values())
MODE_FREQ = 'flat'
FOLD_DIST = 'flat'

def _tier2_fold(mode, gene):
    """Tier-2 fold draw mirroring engine sample_scaling_factor semantics
    (mass_dataset_generator.py:578). Real constructs cluster here because
    they are typical biological constructs, not because the test-set list
    was consulted."""
    if mode == 'MC':
        return float(10.0 ** np.random.uniform(np.log10(1.5), np.log10(30.0)))
    if mode == 'ts':
        if gene == 'CDC14':
            return float(np.random.uniform(0.05, 0.35))
        return float(0.2 * np.random.beta(0.5, 2.0))
    if mode == 'DAmP':
        return float(np.clip(np.random.normal(0.2, 0.05), 0.05, 0.4))
    if mode == 'pLOF':
        return float(np.random.uniform(0.1, 0.5))
    if mode == 'db':
        if gene == 'CLB2':
            return float(np.random.uniform(0.04, 0.15))
        if gene == 'PDS1':
            return float(np.random.uniform(0.0, 0.05))
        return 0.0
    return 1.0
ADJ = {n: set() for n in GENE_NAMES}
for a, b, s in MODULE_EDGES:
    ADJ[a].add(b)
    ADJ[b].add(a)

def _bfs_dist(src):
    dist = {src: 0}
    q = [src]
    while q:
        nxt = []
        for u in q:
            for v in ADJ[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        q = nxt
    return dist
DIST = {n: _bfs_dist(n) for n in GENE_NAMES}

def graph_dist(a, b):
    return DIST[a].get(b, 99)
_NEG_PAIRS = {(a, b) for a, b, s in MODULE_EDGES if s == '-'} | {(b, a) for a, b, s in MODULE_EDGES if s == '-'}
_OUT = {n: set() for n in GENE_NAMES}
for a, b, s in MODULE_EDGES:
    if s == '+':
        _OUT[a].add(b)

def _pair_class(a, b):
    if (a, b) in _NEG_PAIRS:
        return 'C-A'
    if _OUT[a] & _OUT[b]:
        return 'C-A'
    d = graph_dist(a, b)
    if d <= 1:
        return 'C-A'
    if d <= 3:
        return 'C-B'
    return 'C-D'
ALL_PAIRS = [(a, b) for i, a in enumerate(GENE_NAMES) for b in GENE_NAMES[i + 1:]]
PAIR_LISTS = {'C-A': [], 'C-B': [], 'C-D': []}
for a, b in ALL_PAIRS:
    PAIR_LISTS[_pair_class(a, b)].append((a, b))
STAR_CENTERS = ['CLB2', 'MDT']
STAR_LEAVES = {'CLB2': sorted(ADJ['CLB2']), 'MDT': [n for n in GENE_NAMES if n != 'MDT']}
TOUCH = {('CLN2', 'KO'): {'ksn2pp'},
 ('CLN2', 'GAL-OE'): {'ksn2pp', 'ksn2p'},
 ('CLN2', 'MC'): {'ksn2pp'},
 ('CLN2', 'ts'): {'ksn2pp', 'ksn2p'},
 ('CLN3', 'KO'): {'C0'},
 ('CLN3', 'GAL-OE'): {'C0'},
 ('CLN3', 'MC'): {'C0'},
 ('CLB2', 'KO'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'GAL-OE'): {'ksb2p'},
 ('CLB2', 'MC'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'ts'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'db'): {'kdb2pp', 'kdb2P'},
 ('CLB2', 'pLOF'): {'ksb2p', 'ksb2pp'},
 ('CLB2', 'CA'): {'kdib2'},
 ('CLB5', 'KO'): {'ksb5p', 'ksb5pp'},
 ('CLB5', 'GAL-OE'): {'ksb5p'},
 ('CLB5', 'MC'): {'ksb5p', 'ksb5pp'},
 ('CLB5', 'db'): {'kdb5pp'},
 ('SIC1', 'KO'): {'ksc1pp', 'ksc1p'},
 ('SIC1', 'GAL-OE'): {'ksc1p'},
 ('SIC1', 'MC'): {'ksc1pp', 'ksc1p'},
 ('SIC1', 'db'): {'kd3c1'},
 ('SIC1', 'PM'): {'kd2c1', 'kd3c1'},
 ('CDC6', 'KO'): {'ksf6p', 'ksf6pp', 'ksf6ppp'},
 ('CDC6', 'GAL-OE'): {'ksf6p'},
 ('CDC6', 'MC'): {'ksf6p', 'ksf6pp', 'ksf6ppp'},
 ('CDC6', 'DAmP'): {'ksf6p', 'ksf6pp', 'ksf6ppp'},
 ('BCK2', 'KO'): {'B0'},
 ('BCK2', 'MC'): {'B0'},
 ('SWI5', 'KO'): {'ksswip', 'ksswipp'},
 ('SWI5', 'GAL-OE'): {'ksswip'},
 ('CDC20', 'KO'): {'ks20p', 'ks20pp'},
 ('CDC20', 'GAL-OE'): {'ks20p'},
 ('CDC20', 'MC'): {'ks20p', 'ks20pp'},
 ('CDC20', 'ts'): {'ks20p', 'ks20pp'},
 ('CDC20', 'APC-A'): {'ka20pp'},
 ('CDC20', 'DAmP'): {'ks20p', 'ks20pp'},
 ('CDH1', 'KO'): {'init_CDH1T', 'init_CDH1', 'kscdh'},
 ('CDH1', 'CA'): {'kscdh', 'kicdhpp'},
 ('CDC14', 'ts'): {'kppf6', 'kacdhpp', 'ks14', 'kppc1', 'kaswi'},
 ('CDC14', 'GAL-OE'): {'ks14'},
 ('CDC14', 'MC'): {'ks14'},
 ('CDC14', 'DAmP'): {'ks14'},
 ('NET1', 'TAB6-1'): {'kasrent', 'kasrentp'},
 ('NET1', 'GAL-OE'): {'ksnet'},
 ('NET1', 'ts'): {'kasrent', 'kasrentp'},
 ('NET1', 'DAmP'): {'ksnet'},
 ('TEM1', 'KO'): {'ka15pp', 'init_Tem1', 'Tem1T'},
 ('TEM1', 'GAL-OE'): {'Tem1T'},
 ('TEM1', 'MC'): {'Tem1T'},
 ('TEM1', 'ts'): {'ka15pp', 'Tem1T'},
 ('CDC15', 'KO'): {'init_Cdc15', 'Cdc15T', 'kkpnetpp'},
 ('CDC15', 'GAL-OE'): {'Cdc15T'},
 ('CDC15', 'MC'): {'Cdc15T'},
 ('CDC15', 'ts'): {'kkpnetpp'},
 ('PDS1', 'KO'): {'ks2pdspp', 'ks1pdspp'},
 ('PDS1', 'GAL-OE'): {'kspdsp'},
 ('PDS1', 'db'): {'kd2pdspp', 'kd3pdspp'},
 ('ESP1', 'ts'): {'kdiesp', 'kasesp'},
 ('ESP1', 'GAL-OE'): {'ESP1T'},
 ('MAD2', 'CSL'): {'Vi20_active', 'Vi20'},
 ('BUB2', 'CKO'): {'BUB2_active', 'kbub2', 'init_BUB2'},
 ('APC_Core', 'APC-A'): {'ka20pp'},
 ('APC_Core', 'ts'): {'init_CDH1T', 'init_CDH1', 'kscdh', 'ks20pp', 'ks20p'},
 ('MDT', 'MDT'): {'mdt'}}

def touch_set(gene, mode):
    return TOUCH.get((gene, mode), set())
FOLD_MODES = {('BCK2', 'MC'),
 ('CDC14', 'DAmP'),
 ('CDC14', 'MC'),
 ('CDC14', 'ts'),
 ('CDC15', 'MC'),
 ('CDC20', 'DAmP'),
 ('CDC20', 'MC'),
 ('CDC20', 'ts'),
 ('CDC6', 'DAmP'),
 ('CDC6', 'MC'),
 ('CLB2', 'DAmP'),
 ('CLB2', 'MC'),
 ('CLB2', 'db'),
 ('CLB2', 'pLOF'),
 ('CLB2', 'ts'),
 ('CLB5', 'MC'),
 ('CLN2', 'MC'),
 ('CLN2', 'ts'),
 ('CLN3', 'MC'),
 ('NET1', 'DAmP'),
 ('NET1', 'ts'),
 ('SIC1', 'MC'),
 ('TEM1', 'MC')}
RESIDUAL_MODES = {'ts', 'DAmP', 'pLOF', 'db'}
COMPOSITE_ELIGIBLE = ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']
ALLELE_RESIDUAL = {('CDC15', 'ts'): ['kkpnetpp'],
 ('TEM1', 'ts'): ['Tem1T', 'ka15pp'],
 ('ESP1', 'ts'): ['kasesp', 'kdiesp'],
 ('CLB5', 'db'): ['kdb5pp'],
 ('PDS1', 'db'): ['kd2pdspp', 'kd3pdspp'],
 ('SIC1', 'db'): ['kd3c1'],
 ('SIC1', 'PM'): ['kd2c1', 'kd3c1']}

class LogBandSampler:
    """log10-uniform band sampler with exact per-band quota equalization."""

    def __init__(self, lo, hi, n_bands=N_LOG_BANDS, zero_prob=0.0, boundary=None, boundary_prob=0.0):
        self.edges = np.linspace(lo, hi, n_bands + 1)
        self.n = n_bands
        self.counts = np.zeros(n_bands, dtype=int)
        self.zero_prob = zero_prob
        self.boundary = boundary
        self.boundary_prob = boundary_prob
        self.n_zero = 0

    def sample(self, band_bias=None):
        """band_bias: optional callable band_index -> extra probability weight
        (used by backlog directed generation)."""
        if self.zero_prob > 0 and np.random.rand() < self.zero_prob:
            self.n_zero += 1
            return 0.0
        if self.boundary is not None and np.random.rand() < self.boundary_prob:
            return float(10 ** np.random.uniform(np.log10(self.boundary[0]), np.log10(self.boundary[1])))
        if band_bias is None:
            m = self.counts.min()
            cand = np.where(self.counts == m)[0]
            i = int(np.random.choice(cand))
        else:
            w = np.asarray(band_bias, dtype=float) + 1e-09
            i = int(np.random.choice(self.n, p=w / w.sum()))
        self.counts[i] += 1
        return float(10 ** np.random.uniform(self.edges[i], self.edges[i + 1]))

class QuotaPicker:
    """Exact quota categorical picker (equalized counts)."""

    def __init__(self, names, quotas):
        self.names = list(names)
        self.quotas = np.asarray(quotas, dtype=float)
        self.counts = np.zeros(len(self.names), dtype=int)

    def pick(self, weights=None):
        if weights is None:
            deficit = self.quotas * max(1, self.counts.sum()) - self.counts
            w = np.maximum(deficit, 1e-09)
        else:
            w = np.asarray(weights, dtype=float) + 1e-09
        i = int(np.random.choice(len(self.names), p=w / w.sum()))
        self.counts[i] += 1
        return self.names[i]

def qss_g1(p, m, max_iter=80, tol=0.0001, damp=0.5):
    """G1-phase (pre-Start, MCM1=0 basal branch) fixed point. Tier 3 algebra."""
    CLN3 = p['C0'] * p['Dn3'] * m / (p['Jn3'] + p['Dn3'] * m)
    BCK2 = p['B0'] * m
    CDH1T = p['kscdh'] / p['kdcdh']
    CDH1 = CDH1T * GK(p['kacdhp'], p['kicdhp'], p['Jacdh'], p['Jicdh'])
    CDC20T = p['ks20p'] / max(p['kd20'], EPS)
    CDC20a = CDC20T * p['ka20p'] / max(p['ka20p'] + 0.01 + p['kd20'], EPS)
    SIC1T = p['ksc1p'] / max(p['kd1c1'], EPS)
    CDC6T = p['ksf6p'] / max(p['kd1f6'], EPS)
    Vdb2 = p['kdb2p'] + p['kdb2pp'] * CDH1 + p['kdb2P'] * CDC20a
    kappa2 = Vdb2 + p['kasb2'] * SIC1T + p['kasf2'] * CDC6T
    Vdb5 = p['kdb5p'] + p['kdb5pp'] * CDC20a
    SBF, CLN2, CLB5, Clb2 = (0.1, 0.05, 0.05, 0.05)
    for _ in range(max_iter):
        if p.get('is_sbf_ca'):
            Vasbf = 0.38
        else:
            Vasbf = p['kasbf'] * (p['esbfn2'] * CLN2 + p['esbfn3'] * (CLN3 + BCK2) + p['esbfb5'] * CLB5)
        Visbf = p['kisbfp'] + p['kisbfpp'] * Clb2
        SBF_n = GK(Vasbf, Visbf, p['Jasbf'], p['Jisbf'])
        CLN2_n = (p['ksn2p'] + p['ksn2pp'] * SBF_n) * m / max(p['kdn2'], EPS)
        CLB5_n = (p['ksb5p'] + p['ksb5pp'] * SBF_n) * m / max(Vdb5, EPS)
        Clb2_n = p['ksb2p'] * m / max(kappa2, EPS)
        SBF = (1 - damp) * SBF + damp * SBF_n
        CLN2 = (1 - damp) * CLN2 + damp * CLN2_n
        CLB5 = (1 - damp) * CLB5 + damp * CLB5_n
        Clb2 = (1 - damp) * Clb2 + damp * Clb2_n
        err = max(abs(SBF_n - SBF), abs(CLN2_n - CLN2) / max(CLN2_n, EPS), abs(CLB5_n - CLB5) / max(CLB5_n, EPS), abs(Clb2_n - Clb2) / max(Clb2_n, EPS))
        if err < tol:
            break
    else:
        if not np.isfinite(CLN2 + CLB5 + Clb2 + SBF):
            return None
    if not np.isfinite(CLN2 + CLB5 + Clb2 + SBF):
        return None
    return dict(SBF=SBF, CLN2=CLN2, CLB5=CLB5, Clb2=Clb2, Vasbf=Vasbf, Visbf=Visbf, Vdb5=Vdb5, CLN3=CLN3, BCK2=BCK2, CDC20a=CDC20a, CDH1=CDH1)

def qss_M(p, m, phase='meta', max_iter=200, tol=0.0001, damp=0.5):
    """M-phase fixed point, two event-automaton branches (Tier 2 assignments):
      phase='meta' : SAC armed (Vi20=Vi20_active, BUB2=BUB2_active, lte1=0.1)
                     -> MEN off, CDC14 sequestered, Clb2 high.
      phase='ana'  : post-SPN exit state (Vi20=0.01, BUB2=init, lte1=1.0)
                     -> MEN on, CDC14 released, CDC20 active.
    Positive feedbacks (SBF<->CLN2, Clb2<->MCM1) solved by damped iteration;
    sequestration uses FREE SIC1/CDC6 fractions (the spec's SIC1T_eff), since
    most total SIC1 is bound in C2/C5 complexes when Clb2 is high.
    Returns None on divergence/NaN (Scheme: NaN -> NaN bucket)."""
    CLN3 = p['C0'] * p['Dn3'] * m / (p['Jn3'] + p['Dn3'] * m)
    BCK2 = p['B0'] * m
    if phase == 'meta':
        Vi20 = p['Vi20_active']
        BUB2 = p['BUB2_active']
        lte1 = 0.1
        v = dict(SBF=0.5, CLN2=0.1, CLB5=0.1, MCM1=0.9, Clb2=0.5, IEP=0.9, CDC20T=1.5, CDC20a=0.1, CDH1T=1.0, CDH1=0.2, SWI5=0.2, SIC1T=0.02, CDC6T=0.01, CDC14T=2.0, CDC14=0.2, TEM1a=0.1, CDC15a=0.2, PPX=0.1, PDS1=0.3, ESP1=0.2, ORI=1.0, SPN=0.5, PE=0.8, kappa2=1.0, Vkpc1=0.5, Vkpf6=0.5, Vdb2=0.5, Vdpds=0.1, Vac15=0.1, Vicdh=0.05)
    else:
        Vi20 = 0.01
        BUB2 = p.get('init_BUB2', 0.2)
        lte1 = 1.0
        v = dict(SBF=0.5, CLN2=0.1, CLB5=0.1, MCM1=0.5, Clb2=0.3, IEP=0.9, CDC20T=1.5, CDC20a=1.2, CDH1T=1.0, CDH1=0.8, SWI5=0.8, SIC1T=0.01, CDC6T=0.01, CDC14T=2.0, CDC14=1.5, TEM1a=0.9, CDC15a=0.6, PPX=0.1, PDS1=0.05, ESP1=0.8, ORI=1.0, SPN=1.0, PE=0.2, kappa2=1.0, Vkpc1=0.5, Vkpf6=0.5, Vdb2=0.5, Vdpds=0.1, Vac15=0.5, Vicdh=0.05)
    for _ in range(max_iter):
        old = v.copy()
        if p.get('is_sbf_ca'):
            Vasbf = 0.38
        else:
            Vasbf = p['kasbf'] * (p['esbfn2'] * v['CLN2'] + p['esbfn3'] * (CLN3 + BCK2) + p['esbfb5'] * v['CLB5'])
        Visbf = p['kisbfp'] + p['kisbfpp'] * v['Clb2']
        v['SBF'] = GK(Vasbf, Visbf, p['Jasbf'], p['Jisbf'])
        v['CLN2'] = (p['ksn2p'] + p['ksn2pp'] * v['SBF']) * m / max(p['kdn2'], EPS)
        Vdb5 = p['kdb5p'] + p['kdb5pp'] * v['CDC20a']
        v['CLB5'] = (p['ksb5p'] + p['ksb5pp'] * v['SBF']) * m / max(Vdb5, EPS)
        v['MCM1'] = GK(p['kamcm'] * v['Clb2'], p['kimcm'], p['Jamcm'], p['Jimcm'])
        v['CDC20T'] = (p['ks20p'] + p['ks20pp'] * v['MCM1']) / max(p['kd20'], EPS)
        act20 = p['ka20p'] + p['ka20pp'] * v['IEP']
        v['CDC20a'] = v['CDC20T'] * act20 / max(act20 + Vi20 + p['kd20'], EPS)
        v['Vdb2'] = p['kdb2p'] + p['kdb2pp'] * v['CDH1'] + p['kdb2P'] * v['CDC20a']
        v['CDC14T'] = p['ks14'] / max(p['kd14'], EPS)
        v['TEM1a'] = p['Tem1T'] * lte1 / max(lte1 + BUB2, EPS)
        v['Vac15'] = p['ka15p'] * max(p['Tem1T'] - v['TEM1a'], 0.0) + p['ka15pp'] * v['TEM1a'] + p['ka15P'] * v['CDC14']
        v['CDC15a'] = p['Cdc15T'] * v['Vac15'] / max(v['Vac15'] + p['ki15'], EPS)
        Vkpnet = (p['kkpnetp'] + p['kkpnetpp'] * v['CDC15a']) * m
        Vppnet = p['kppnetp'] + p['kppnetpp'] * v['PPX']
        v['CDC14'] = v['CDC14T'] * Vkpnet / max(Vkpnet + Vppnet, EPS)
        Vdppx = p['kdppxp'] + p['kdppxpp'] * (p['J20ppx'] + v['CDC20a']) * p['Jpds'] / (p['Jpds'] + v['PDS1'])
        v['PPX'] = p['ksppx'] / max(Vdppx, EPS)
        v['CDH1T'] = p['kscdh'] / max(p['kdcdh'], EPS)
        Vacdh = p['kacdhp'] + p['kacdhpp'] * v['CDC14']
        v['Vicdh'] = p['kicdhp'] + p['kicdhpp'] * (p['eicdhn3'] * CLN3 + p['eicdhn2'] * v['CLN2'] + p['eicdhb5'] * v['CLB5'] + p['eicdhb2'] * v['Clb2'])
        v['CDH1'] = v['CDH1T'] * GK(Vacdh, v['Vicdh'], p['Jacdh'], p['Jicdh'])
        SWI5T = (p['ksswip'] + p['ksswipp'] * v['MCM1']) / max(p['kdswi'], EPS)
        v['SWI5'] = SWI5T * p['kaswi'] * v['CDC14'] / max(p['kaswi'] * v['CDC14'] + p['kiswi'] * v['Clb2'], EPS)
        cyc1 = p['ec1n3'] * CLN3 + p['ec1k2'] * BCK2 + p['ec1n2'] * v['CLN2'] + p['ec1b2'] * v['Clb2'] + p['ec1b5'] * v['CLB5']
        v['Vkpc1'] = p['kd1c1'] + p['kd2c1'] * cyc1 / (p['Jd2c1'] + v['SIC1T'])
        v['SIC1T'] = (p['ksc1p'] + p['ksc1pp'] * v['SWI5']) / max(v['Vkpc1'], EPS)
        cyc6 = p['ef6n3'] * CLN3 + p['ef6k2'] * BCK2 + p['ef6n2'] * v['CLN2'] + p['ef6b2'] * v['Clb2'] + p['ef6b5'] * v['CLB5']
        v['Vkpf6'] = p['kd1f6'] + p['kd2f6'] * cyc6 / (p['Jd2f6'] + v['CDC6T'])
        v['CDC6T'] = (p['ksf6p'] + p['ksf6pp'] * v['SWI5'] + p['ksf6ppp'] * v['SBF']) / max(v['Vkpf6'], EPS)
        koff1 = p['kdib2'] + v['Vdb2'] + v['Vkpc1']
        SIC1_free = v['SIC1T'] * koff1 / max(koff1 + p['kasb2'] * v['Clb2'] + p['kasb5'] * v['CLB5'], EPS)
        koff6 = p['kdif2'] + v['Vdb2'] + v['Vkpf6']
        CDC6_free = v['CDC6T'] * koff6 / max(koff6 + p['kasf2'] * v['Clb2'] + p['kasf5'] * v['CLB5'], EPS)
        v['kappa2'] = v['Vdb2'] + p['kasb2'] * SIC1_free + p['kasf2'] * CDC6_free
        v['Clb2'] = (p['ksb2p'] + p['ksb2pp'] * v['MCM1']) * m / max(v['kappa2'], EPS)
        v['IEP'] = GK(p['kaiep'] * v['Clb2'], p['kiiep'], p['Jaiep'], p['Jiiep'])
        v['Vdpds'] = p['kd1pdsp'] + p['kd2pdspp'] * v['CDC20a'] + p['kd3pdspp'] * v['CDH1']
        v['PE'] = max(p['ESP1T'] - v['ESP1'], 0.0)
        v['PDS1'] = (p['kspdsp'] + p['ks1pdspp'] * v['SBF'] + p['ks2pdspp'] * v['MCM1'] + p['kdiesp'] * v['PE']) / max(v['Vdpds'] + p['kasesp'] * v['ESP1'], EPS)
        v['ESP1'] = p['ESP1T'] * (p['kdiesp'] + v['Vdpds']) / max(p['kasesp'] * v['PDS1'] + p['kdiesp'] + v['Vdpds'], EPS)
        v['ORI'] = p['ksori'] * (p['eorib5'] * v['CLB5'] + p['eorib2'] * v['Clb2']) / max(p['kdori'], EPS)
        v['SPN'] = p['ksspn'] * v['Clb2'] / (p['Jspn'] + v['Clb2']) / max(p['kdspn'], EPS)
        for k in v:
            v[k] = (1 - damp) * old[k] + damp * v[k]
        err = max((abs(v[k] - old[k]) / max(abs(old[k]), EPS) for k in v))
        if err < tol:
            return v
    if any((not np.isfinite(x) for x in v.values())):
        return None
    return v
D_NAMES = ['D1_RStart', 'D2_RORI', 'D3_RM', 'D4_RC20', 'D5_Rdeg', 'D6_RMEN', 'D7_RRENT', 'D8_Bcyc', 'D9_RSPN', 'D10_Rsep']

def _critical_volume(p, m_lo=M_BIRTH, m_hi=MASS_LIMIT):
    """D1: solve R_Start(m)=Vasbf/Visbf=1 (GK Va=Vi structural condition).
    Upper bracket = engine mass-divergence bound (max_m > 10 -> Discarded):
    critical volumes beyond it would be rolled back in Phase 4 anyway, so
    they go straight to NaN bucket here (cheaper, same outcome)."""

    def R(m):
        fp = qss_g1(p, m)
        if fp is None:
            return None
        return fp['Vasbf'] / max(fp['Visbf'], EPS)
    r_lo = R(m_lo)
    if r_lo is None:
        return np.nan
    if r_lo >= 1.0:
        return m_lo
    r_hi = R(m_hi)
    if r_hi is None:
        return np.nan
    if r_hi < 1.0:
        return np.inf
    lo, hi = (m_lo, m_hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = R(mid)
        if r is None:
            return np.nan
        if r >= 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def compute_D(p, mdt):
    """Self-consistent D' vector (10 dims). NaN entries -> NaN bucket path.
    All concentrations come from the candidate's OWN fixed points; the only
    constants are Tier-2 event thresholds (0.3, 1.0) and the window length.
    D1/D2/D3/D9/D10 from the metaphase branch (SAC armed), D5/D6/D7 from the
    anaphase branch (SAC off, MEN on) — the spec's 'SAC on/off two branches'."""
    D = np.full(10, np.nan)
    d1 = _critical_volume(p)
    if np.isnan(d1):
        return D
    fp_meta = qss_M(p, M_M, 'meta')
    fp_ana = qss_M(p, M_M, 'ana')
    if fp_meta is None or fp_ana is None:
        return D
    D[0] = d1
    D[1] = fp_meta['ORI']
    D[2] = p['kamcm'] * fp_meta['Clb2'] / max(p['kimcm'], EPS)
    act20 = p['ka20p'] + p['ka20pp'] * fp_meta['IEP']
    D[3] = act20 / max(0.01 + p['kd20'], EPS)
    kclr = p['kdb2P'] * fp_ana['CDC20a'] + p['kdb2pp'] * fp_ana['CDH1']
    if kclr <= EPS or fp_meta['Clb2'] <= 0.3:
        D[4] = np.inf
    else:
        D[4] = math.log(fp_meta['Clb2'] / 0.3) / kclr
    D[5] = (p['ka15pp'] * fp_ana['TEM1a'] + p['ka15P'] * fp_ana['CDC14']) / max(p['ki15'], EPS)
    D[6] = (p['kkpnetp'] + p['kkpnetpp'] * fp_ana['CDC15a']) * M_M / max(p['kppnetp'] + p['kppnetpp'] * fp_ana['PPX'], EPS)
    D[7] = SIM_WINDOW * math.log(2) / mdt
    D[8] = fp_meta['SPN']
    D[9] = fp_meta['PDS1'] * p['kasesp'] / max(p['kdiesp'] + fp_meta['Vdpds'], EPS)
    return D
GAL_ALPHA_CAP_MULT = 100.0
_NATIVE_SCALE_PARAMS = {'CLN2': ('ksn2p', 'ksn2pp'),
 'CLN3': ('C0',),
 'CLB5': ('ksb5p', 'ksb5pp'),
 'CLB2': ('ksb2p', 'ksb2pp'),
 'SIC1': ('ksc1p', 'ksc1pp'),
 'CDC6': ('ksf6p', 'ksf6pp', 'ksf6ppp'),
 'SWI5': ('ksswip', 'ksswipp'),
 'CDC20': ('ks20p', 'ks20pp'),
 'CDC14': ('ks14',),
 'NET1': ('ksnet',),
 'TEM1': ('Tem1T',),
 'CDC15': ('Cdc15T',),
 'PDS1': ('kspdsp', 'ks1pdspp', 'ks2pdspp'),
 'ESP1': ('ESP1T',)}

def gal_alpha_cap(gene, p_wt):
    """alpha_cap = 100 x gene's native maximum synthesis capacity (Tier 2)."""
    scale = sum((max(p_wt.get(k, 0.0), 0.0) for k in _NATIVE_SCALE_PARAMS.get(gene, ())))
    if scale <= 0.0:
        scale = 1.0
    return GAL_ALPHA_CAP_MULT * scale
GAL_ALPHA_BIO = {'CDC20': (0.5, 12.0),
 'CLB2': (0.002, 1.5),
 'SIC1': (0.03, 0.36),
 'CDC6': (0.1, 0.6),
 'CLB5': (0.004, 0.03),
 'PDS1': (0.02, 0.25),
 'CLN2': (0.03, 0.3),
 'CLN3': (1.0, 5.0),
 'TEM1': (2.0, 8.0),
 'CDC15': (3.0, 20.0),
 'ESP1': (1.5, 6.0),
 'CDC14': (0.3, 1.2),
 'NET1': (0.15, 0.6),
 'SWI5': (0.02, 0.1)}
GAL_BIO_FALLBACK = (2.0, 30.0)
GAL_STATS = {'eff_guided': 0, 'bio_draw': 0}

def _gal_theta_and_invert(gene, E, p, fpG1, fpM):
    """Per-gene structural threshold theta (Tier 2 / own fixed point) and the
    one-step algebraic inversion alpha(E). All concentrations are the
    candidate's OWN fixed-point values (Tier 3 algebra)."""
    if gene == 'CLN2':
        th = p['kisbfp'] / max(p['kasbf'] * p['esbfn2'], EPS)
        return (th, E * th * p['kdn2'] / M_BIRTH)
    if gene == 'CLN3':
        th = p['kisbfp'] / max(p['kasbf'] * p['esbfn3'], EPS)
        return (th, E * th * (p['Jn3'] + p['Dn3'] * M_BIRTH) / max(p['Dn3'] * M_BIRTH, EPS))
    if gene == 'CLB5':
        th = p['kisbfp'] / max(p['kasbf'] * p['esbfb5'], EPS)
        return (th, E * th * fpG1['Vdb5'] / M_BIRTH - p['ksb5pp'] * fpG1['SBF'])
    if gene == 'CLB2':
        th = p['kimcm'] / max(p['kamcm'], EPS)
        return (th, E * th * fpM['kappa2'] / M_M - p['ksb2pp'] * fpM['MCM1'])
    if gene == 'SIC1':
        th = fpM['Vdb2'] / max(p['kasb2'], EPS)
        return (th, E * th * fpM['Vkpc1'] - p['ksc1pp'] * fpM['SWI5'])
    if gene == 'CDC6':
        th = fpM['Vdb2'] / max(p['kasf2'], EPS)
        return (th, E * th * fpM['Vkpf6'] - p['ksf6pp'] * fpM['SWI5'] - p['ksf6ppp'] * fpM['SBF'])
    if gene == 'SWI5':
        th = max(p['ksc1p'], 0.0) / max(p['ksc1pp'], EPS)
        return (th, E * th * p['kdswi'] - p['ksswipp'] * fpM['MCM1'])
    if gene == 'CDC20':
        th = max((p['kdb2p'] + p['kdb2pp'] * fpM['CDH1']) / max(p['kdb2P'], 1e-06), 0.1)
        return (th, E * th * p['kd20'] - p['ks20pp'] * fpM['MCM1'])
    if gene == 'CDC14':
        th = max(p['kicdhp'] + p['kicdhpp'] * 0.05, EPS) / max(p['kacdhpp'], EPS)
        f14 = fpM['CDC14'] / max(fpM['CDC14T'], EPS)
        return (th, E * th * p['kd14'] / max(f14, 0.05))
    if gene == 'NET1':
        th = fpM['CDC14T']
        return (th, E * th * p['kdnet'])
    if gene == 'TEM1':
        th = p['ki15'] / max(p['ka15pp'], EPS)
        return (th, E * th * (0.1 + p.get('init_BUB2', 0.2)) / 0.1)
    if gene == 'CDC15':
        th = max((p['kppnetp'] + p['kppnetpp'] * fpM['PPX'] - p['kkpnetp'] * M_M) / max(p['kkpnetpp'] * M_M, EPS), 0.05)
        frac = fpM['Vac15'] / max(fpM['Vac15'] + p['ki15'], EPS)
        return (th, E * th / max(frac, 0.05))
    if gene == 'PDS1':
        th = p['kdiesp'] / max(p['kasesp'], EPS)
        return (th, E * th * (fpM['Vdpds'] + p['kasesp'] * fpM['ESP1']) - p['ks1pdspp'] * fpM['SBF'] - p['ks2pdspp'] * fpM['MCM1'] - p['kdiesp'] * fpM['PE'])
    if gene == 'ESP1':
        th = fpM['PDS1']
        return (th, E * th)
    return (1.0, E)

def apply_gal_inversions(gal_list, p_wt, p_mut, ovr, *, ma):
    """v3.8 hybrid: effect-guided alpha bounded by the per-gene promoter-
    strength bio-range (Tier 2). Iterative self-consistency is kept (2-3
    damped rounds), but the inverted alpha is accepted ONLY when it lands
    inside the bio-range; otherwise a log-uniform draw WITHIN the bio-range
    replaces it (physical prior). This removes the v3.7 pathology where a
    mis-scaled inversion collapsed alpha to ~0 (PDS1/CDC6: GAL == KO) or
    pegged it at the 100x cap (CLN2), leaving the bio-realistic region
    empty. gal_list: [(gene, E)]. Returns (converged, alphas)."""
    if not gal_list:
        return (True, {})
    alphas = {g: 0.0 for g, _ in gal_list}
    for g, _ in gal_list:
        ma.apply_gene_mutation(g, 'GAL-OE', 0.0, p_wt, p_mut, ovr)
    converged = False
    frozen = {}
    for _round in range(3):
        p_eff = p_wt.copy()
        p_eff.update(ovr)
        fpG1 = qss_g1(p_eff, M_BIRTH)
        fpM = qss_M(p_eff, M_M)
        if fpG1 is None or fpM is None:
            return (False, alphas)
        new_alphas = {}
        for g, E in gal_list:
            if g in frozen:
                new_alphas[g] = frozen[g]
                continue
            _, a = _gal_theta_and_invert(g, E, p_eff, fpG1, fpM)
            a = max(a, 0.0)
            lo, hi = GAL_ALPHA_BIO.get(g, GAL_BIO_FALLBACK)
            if lo <= a <= hi:
                GAL_STATS['eff_guided'] += 1
            else:
                a = float(10.0 ** np.random.uniform(np.log10(lo), np.log10(hi)))
                frozen[g] = a
                GAL_STATS['bio_draw'] += 1
            new_alphas[g] = a
        conv = all((abs(new_alphas[g] - alphas[g]) <= 0.05 * max(new_alphas[g], alphas[g], 1e-09) for g in new_alphas if new_alphas[g] > 1e-09))
        alphas = new_alphas
        for g in gal_list:
            ma.apply_gene_mutation(g[0], 'GAL-OE', alphas[g[0]], p_wt, p_mut, ovr)
        if conv and _round >= 1:
            converged = True
            break
    return (converged, alphas)
LEVEL_QUOTAS = [0.4, 0.3, 0.2, 0.1]
STRUCT_L2 = {'C-A': 0.3, 'C-B': 0.3, 'C-C': 0.15, 'C-D': 0.25}
STRUCT_L34 = {'cascade': 0.45, 'star': 0.25, 'antag_core': 0.2, 'orthogonal': 0.1}

def _weighted_gene(weights=None, exclude=()):
    names = [n for n in GENE_NAMES if n not in exclude]
    if weights is None:
        w = np.array([P_BIO[n] for n in names])
    else:
        w = np.array([weights.get(n, P_BIO[n]) for n in names])
    w = w / w.sum()
    return str(np.random.choice(names, p=w))

def _pair_weight(a, b, boost=None):
    w = P_BIO[a] * P_BIO[b] / _BIO_SUM ** 2
    if boost:
        w *= (boost.get(a, 1.0) + boost.get(b, 1.0)) / 2.0
    return w

def _sample_L2(boost=None):
    cls = QuotaPicker_picker_L2()
    if cls == 'C-C':
        center = str(np.random.choice(STAR_CENTERS, p=[2/3, 1/3]))
        leaves = [l for l in STAR_LEAVES[center]]
        if not leaves:
            return (sample_L2_fallback(boost), 'C-C')
        w = np.array([P_BIO[l] * (boost.get(l, 1.0) if boost else 1.0) for l in leaves])
        leaf = str(np.random.choice(leaves, p=w / w.sum()))
        return ([center, leaf], 'C-C')
    pairs = PAIR_LISTS[cls]
    w = np.array([_pair_weight(a, b, boost) for a, b in pairs])
    i = int(np.random.choice(len(pairs), p=w / w.sum()))
    return (list(pairs[i]), cls)
_L2_PICKER = QuotaPicker(list(STRUCT_L2.keys()), list(STRUCT_L2.values()))

def QuotaPicker_picker_L2():
    return _L2_PICKER.pick()
_L34_PICKER = QuotaPicker(list(STRUCT_L34.keys()), list(STRUCT_L34.values()))

def sample_L2_fallback(boost=None):
    a = _weighted_gene(boost)
    b = _weighted_gene(boost, exclude=(a,))
    return [a, b]

def _walk_cascade(ki, boost=None):
    start = _weighted_gene(boost)
    chain = [start]
    while len(chain) < ki:
        cur = chain[-1]
        neigh = sorted((n for n in ADJ[cur] if n not in chain))
        if neigh:
            nxt = str(np.random.choice(neigh))
        else:
            nxt = _weighted_gene(boost, exclude=tuple(chain))
        chain.append(nxt)
    return chain

def _sample_star(ki, boost=None):
    center = str(np.random.choice(STAR_CENTERS, p=[2/3, 1/3]))
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
        if all((graph_dist(g, s) >= 3 for s in out)):
            out.append(g)
    while len(out) < ki:
        out.append(_weighted_gene(boost, exclude=tuple(out)))
    return out

def sample_skeleton(level, boost=None):
    """Returns (genes list, structure_class). Level quotas are exact."""
    if level == 1:
        return ([_weighted_gene(boost)], 'L1')
    if level == 2:
        return _sample_L2(boost)
    ki = 3 if level == 3 else 3
    cls = _L34_PICKER.pick()
    if cls == 'cascade':
        return (_walk_cascade(ki, boost), cls)
    if cls == 'star':
        return (_sample_star(ki, boost), cls)
    if cls == 'antag_core':
        return (_sample_antag_core(ki, boost), cls)
    return (_sample_orthogonal(ki, boost), cls)
_L1_GRID = [(g, m) for g in GENE_NAMES for m in GROUP_BY_NAME[g]['modes']]
np.random.RandomState(20260831).shuffle(_L1_GRID)
_L1_IDX = {'i': 0}

def next_L1_cell():
    cell = _L1_GRID[_L1_IDX['i'] % len(_L1_GRID)]
    _L1_IDX['i'] += 1
    return cell
PRIMARY_COORDS = {'CLN3': [1],
 'BCK2': [1],
 'CLN2': [1],
 'CLB5': [1, 2],
 'CLB2': [3, 5],
 'SIC1': [3, 5],
 'CDC6': [3],
 'SWI5': [3],
 'CDC20': [4, 5],
 'MAD2': [4],
 'APC_Core': [4, 5],
 'CDH1': [5],
 'CDC14': [5, 6],
 'ESP1': [10],
 'PDS1': [10],
 'NET1': [6, 7],
 'TEM1': [6],
 'CDC15': [6, 7],
 'MDT': [8],
 'BUB2': [6]}

def _sample_mode(gene):
    modes = list(GROUP_BY_NAME[gene]['modes'].keys())
    if MODE_FREQ == 's5':
        w = np.array([GROUP_BY_NAME[gene]['modes'][m] for m in modes], dtype=float)
        return str(np.random.choice(modes, p=w / w.sum()))
    return str(np.random.choice(modes))

def _resolve_conflicts(pairs):
    """Same-parameter dual-mode ban + cross-gene overlap ban (§3.1.6).
    Resample conflicting modes; drop pairs that cannot resolve."""
    pairs = list(pairs)
    for _ in range(30):
        touched = {}
        conflict = None
        for i, (g, m) in enumerate(pairs):
            ts = touch_set(g, m)
            for pk in ts:
                if pk in touched:
                    conflict = i
                    break
                touched[pk] = i
            if conflict is not None:
                break
        if conflict is None:
            return pairs
        g, m = pairs[conflict]
        pairs[conflict] = (g, _sample_mode(g))
    if len(pairs) > 1:
        pairs.pop(conflict)
    return pairs

def build_candidate(level, samplers, boost=None, fold_bias_fn=None, *, ma):
    """Phases 1-2: skeleton -> modes -> folds / GAL inversion -> ovr dict.
    Returns dict(ovr, desc, genes_modes, struct_class, mdt, gal_conv) or None."""
    genes, struct_class = sample_skeleton(level, boost)
    if level == 1:
        g0, m0 = next_L1_cell()
        if MODE_FREQ == 's5':
            m0 = _sample_mode(g0)
        genes, pairs = ([g0], [(g0, m0)])
    else:
        pairs = [(g, _sample_mode(g)) for g in genes]
        expanded = []
        for g, m in pairs:
            expanded.append((g, m))
            if g in COMPOSITE_ELIGIBLE and np.random.rand() < 0.25:
                gmode = GROUP_BY_NAME[g]['modes']
                synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
                struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
                if m in synth and struc:
                    expanded.append((g, str(np.random.choice(struc))))
                elif m in struc and synth:
                    expanded.append((g, str(np.random.choice(synth))))
        pairs = expanded
    pairs = _resolve_conflicts(pairs)
    if not pairs:
        return None
    p_wt = get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    gal_list = []
    desc_parts = []
    has_gal = False
    edits = []
    for g, m in pairs:
        desc_parts.append(f'{g}[{m}]')
        if m == 'GAL-OE':
            has_gal = True
            E = samplers['E'].sample(fold_bias_fn('E') if fold_bias_fn else None)
            gal_list.append((g, E))
            edits.append((g, m, E))
        elif (g, m) in FOLD_MODES:
            if FOLD_DIST == 'tier2':
                f = _tier2_fold(m, g) if ma.distribution_profile == 'mass' else ma.sample_scaling_factor(m, g)
            else:
                f = samplers['f_res' if m in RESIDUAL_MODES else 'f_mc'].sample(fold_bias_fn(m) if fold_bias_fn else None)
            ma.apply_gene_mutation(g, m, f, p_wt, p_mut, ovr)
            edits.append((g, m, f))
        else:
            ma.apply_gene_mutation(g, m, 1.0, p_wt, p_mut, ovr)
            if (g, m) in ALLELE_RESIDUAL:
                if FOLD_DIST == 'tier2' and m in ('ts', 'db'):
                    f = _tier2_fold(m, g) if ma.distribution_profile == 'mass' else ma.sample_scaling_factor(m, g)
                else:
                    f = samplers['f_res'].sample(fold_bias_fn(m) if fold_bias_fn else None)
                for pk in ALLELE_RESIDUAL[g, m]:
                    ovr[pk] = p_wt[pk] * f
                edits.append((g, m, f))
            else:
                edits.append((g, m, None))
    gal_conv, _alphas = (True, {})
    if gal_list:
        gal_conv, _alphas = apply_gal_inversions(gal_list, p_wt, p_mut, ovr, ma=ma)
    if 'mdt' not in ovr:
        if has_gal:
            ovr['mdt'] = 150.0
        else:
            ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0]))
    return dict(ovr=ovr, desc='+'.join(desc_parts), genes_modes=pairs, struct_class=struct_class, level=level, mdt=ovr['mdt'], gal_conv=gal_conv, edits=edits)

class Ledger:

    def __init__(self, n_target, n_bins=N_BINS, nan_cap_frac=0.05):
        self.n_bins = n_bins
        self.edges = None
        self.counts = np.zeros((10, n_bins), dtype=int)
        self.cap = int(math.ceil(1.3 * n_target / n_bins))
        self.dir_counts = np.zeros((10, 3), dtype=int)
        self.nan_count = 0
        self.nan_cap = int(math.ceil(nan_cap_frac * n_target))
        self.boundary_count = np.zeros(10, dtype=int)
        self.nan_by_mode = {}

    def fit_pilot(self, D_mat):
        """D_mat: (n_pilot, 10) with NaN/inf entries allowed."""
        edges = []
        for d in range(10):
            col = D_mat[:, d]
            fin = col[np.isfinite(col)]
            if len(fin) < 20:
                lo, hi = (0.0, 1.0)
            else:
                lo, hi = (float(np.min(fin)), float(np.max(fin)))
            qs = np.quantile(fin, np.linspace(0, 1, self.n_bins + 1)) if len(fin) >= 20 else np.linspace(lo, hi, self.n_bins + 1)
            qs = np.maximum.accumulate(qs)
            for j in range(1, len(qs) - 1):
                if qs[j] <= qs[j - 1]:
                    base = qs[j - 1] if np.isfinite(qs[j - 1]) else 0.0
                    qs[j] = base + abs(base) * 1e-09 + 1e-12
            qs[0], qs[-1] = (-np.inf, np.inf)
            edges.append(qs)
        self.edges = edges

    def bin_of(self, dim, val):
        if np.isnan(val):
            return -1
        if np.isinf(val):
            return self.n_bins - 1
        return int(np.searchsorted(self.edges[dim], val, side='right') - 1)

    def cells(self, D):
        """Returns list of (dim, bin) for finite dims."""
        out = []
        for d in range(10):
            b = self.bin_of(d, D[d])
            if b >= 0:
                out.append((d, b))
        return out

    def in_boundary(self, D):
        flags = []
        for d in range(10):
            v = D[d]
            flags.append(bool(np.isfinite(v) and BOUNDARY_LO <= v <= BOUNDARY_HI))
        return flags
SIGN = {('CLN3', 'GAL-OE'): {0: -1},
 ('CLN3', 'MC'): {0: -1},
 ('CLN3', 'KO'): {0: 1},
 ('BCK2', 'MC'): {0: -1},
 ('BCK2', 'KO'): {0: 1},
 ('CLN2', 'GAL-OE'): {0: -1},
 ('CLN2', 'MC'): {0: -1},
 ('CLN2', 'ts'): {0: 1},
 ('CLN2', 'KO'): {0: 1},
 ('CLB5', 'GAL-OE'): {0: -1, 1: 1},
 ('CLB5', 'MC'): {0: -1, 1: 1},
 ('CLB5', 'KO'): {0: 1, 1: -1},
 ('CLB5', 'db'): {0: -1, 1: 1},
 ('CLB2', 'GAL-OE'): {2: 1, 4: 1},
 ('CLB2', 'MC'): {2: 1, 4: 1},
 ('CLB2', 'KO'): {2: -1, 4: 1},
 ('CLB2', 'ts'): {2: -1, 4: 1},
 ('CLB2', 'pLOF'): {2: -1, 4: 1},
 ('CLB2', 'DAmP'): {2: -1, 4: 1},
 ('CLB2', 'db'): {4: 1},
 ('CLB2', 'CA'): {2: 1, 4: 1},
 ('SIC1', 'GAL-OE'): {2: -1, 4: -1},
 ('SIC1', 'MC'): {2: -1, 4: -1},
 ('SIC1', 'KO'): {2: 1, 4: 1},
 ('SIC1', 'db'): {2: -1, 4: -1},
 ('SIC1', 'PM'): {2: -1, 4: -1},
 ('CDC6', 'GAL-OE'): {2: -1},
 ('CDC6', 'MC'): {2: -1},
 ('CDC6', 'KO'): {2: 1},
 ('CDC6', 'DAmP'): {2: 1},
 ('SWI5', 'GAL-OE'): {2: -1},
 ('SWI5', 'KO'): {2: 1},
 ('CDC20', 'GAL-OE'): {3: 1, 4: -1},
 ('CDC20', 'MC'): {3: 1, 4: -1},
 ('CDC20', 'KO'): {3: -1, 4: 1},
 ('CDC20', 'ts'): {3: -1, 4: 1},
 ('CDC20', 'DAmP'): {3: -1, 4: 1},
 ('CDC20', 'APC-A'): {3: -1, 4: 1},
 ('MAD2', 'CSL'): {3: 1},
 ('BUB2', 'CKO'): {5: 1},
 ('APC_Core', 'APC-A'): {3: -1, 4: 1},
 ('APC_Core', 'ts'): {3: -1, 4: 1},
 ('CDH1', 'KO'): {4: 1},
 ('CDH1', 'CA'): {4: -1},
 ('CDC14', 'GAL-OE'): {4: -1, 5: 1},
 ('CDC14', 'MC'): {4: -1, 5: 1},
 ('CDC14', 'ts'): {4: 1, 5: -1},
 ('CDC14', 'DAmP'): {4: 1, 5: -1},
 ('ESP1', 'ts'): {9: -1},
 ('ESP1', 'GAL-OE'): {9: 1},
 ('PDS1', 'GAL-OE'): {9: 1},
 ('PDS1', 'KO'): {9: -1},
 ('PDS1', 'db'): {9: 1},
 ('NET1', 'TAB6-1'): {5: 1, 6: 1},
 ('NET1', 'ts'): {5: 1, 6: 1},
 ('NET1', 'DAmP'): {5: 1, 6: 1},
 ('NET1', 'GAL-OE'): {5: -1, 6: -1},
 ('TEM1', 'GAL-OE'): {5: 1},
 ('TEM1', 'MC'): {5: 1},
 ('TEM1', 'KO'): {5: -1},
 ('TEM1', 'ts'): {5: -1},
 ('CDC15', 'GAL-OE'): {6: 1},
 ('CDC15', 'MC'): {6: 1},
 ('CDC15', 'KO'): {6: -1},
 ('CDC15', 'ts'): {6: -1},
 ('MDT', 'MDT'): {}}
NEUTRAL_LOG_TOL = 0.05

def direction_signs(edits):
    """edits: [(gene, mode, val)] where val = fold f (scaling) or target E
    (GAL-OE) or None (structural). Returns {dim: 0/1/2} with 0=neutral,
    1=up, 2=down, computed on primary coordinates only (§3.5-3)."""
    out = {}
    for g, m, val in edits:
        for d, s in SIGN.get((g, m), {}).items():
            if val is not None and val > 0 and (abs(math.log10(val)) < NEUTRAL_LOG_TOL):
                out[d] = 0
            else:
                out[d] = 1 if s > 0 else 2
    return out

def make_samplers():
    return {'f_res': LogBandSampler(-3.0, 0.0, zero_prob=0.2), 'f_mc': LogBandSampler(0.0, 2.0, zero_prob=0.0), 'E': LogBandSampler(-1.0, 1.0, zero_prob=0.0, boundary=(BOUNDARY_LO, BOUNDARY_HI), boundary_prob=0.3)}

def _rollback(accepted, name, ledger, seen_sigs=None):
    """Numerical failure / mass divergence -> sample not stored and its ledger
    cells + direction counts rolled back (Scheme §3.5-4); its genotype
    signature is freed so the same edit can be re-attempted with a new fold."""
    for cand in accepted:
        if cand.get('name') == name:
            for d, b in cand.get('cells', []):
                ledger.counts[d, b] = max(0, ledger.counts[d, b] - 1)
            for d, s in cand.get('dirs', {}).items():
                ledger.dir_counts[d, s] = max(0, ledger.dir_counts[d, s] - 1)
            if not cand.get('cells') and (not cand.get('dirs')):
                ledger.nan_count = max(0, ledger.nan_count - 1)
            if seen_sigs is not None and cand.get('sig') is not None:
                seen_sigs.discard(cand['sig'])
            return
