"""
mass_wt_blind_sampler.py — MASS v6.0 (WB-MASS) WT-Blind Mutant-Aware Sampler
================================================================================
Implements Mutant_Aware_Sampling_Scheme_v3_WT_Blind.md (Scheme v3.5).

Zero-leakage contract (Tier model, §1 of the spec):
- Tier 1 FORBIDDEN  : 126 real mutants' genotype/phenotype/frequency; ANY WT
  simulation output (no WT trajectory is ever run; no WT-calibrated constant
  appears in any sampling decision).
- Tier 2 ALLOWED    : equations & event thresholds (0.3 / 0.2 / 1.0), 27 binding
  groups, mode->parameter edit rules, simulation window 210 min, self-sampled
  values (mdt, folds f, target effects E).
- Tier 3 UNAVOIDABLE: the 133 baseline parameter numbers, used ONLY inside
  (a) algebraic evaluation of the candidate's OWN parameter vector
  (QSS fixed points, D coordinates, GAL inversion), (b) the ODE black box.
  No sampling decision reads a baseline value as a design anchor.

The ONLY contact with the 126 real mutants is is_duplicate_real_mutant()
(pure exclusion, §1.4 replacement test: swap the test batch -> algorithm
unchanged except the dedup list).

Pipeline:
  Phase 0  pilot: uniform skeleton + fold/effect sampling -> self-consistent
           fixed points -> D' vectors -> per-dim quantile bin edges.
  Phase 1  structure-aware skeleton (module graph, Level quotas 40/30/20/10,
           structure-class quotas, P_bio genes, flattened modes, carbon rule).
  Phase 2  parameter strengths: scaling modes -> log-band fold quota with
           zero inflation; GAL-OE -> self-consistent effect inversion.
  Phase 3  coverage ledger acceptance (5 bins/dim, direction 45/45/10 on
           primary coordinates, NaN bucket <=5%, backlog directed generation).
  Phase 4  black-box ODE simulation + dual labels + npz export
           (numerical failure / mass divergence -> ledger rollback).

Export format is field-compatible with mass_dataset_generator.py (v4.5):
  data (N,39,500), params (N,136), param_names, mutant_names, labels,
  pattern_labels. Real mutants are NOT included (they are the test set).
"""

import numpy as np
import math, os, sys, time, argparse, warnings
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mass_dataset_generator import (
    get_default_params, GK, apply_gene_mutation,
    is_duplicate_real_mutant, simulate_mutant, BINDING_GROUPS,
    REAL_MUTANTS)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
warnings.filterwarnings("ignore")

EPS = 1e-12
M_BIRTH = 1.2060194   # initial MASS from Chen(2004) Table 2B (Tier 2 spec)
M_M     = 2 * M_BIRTH # late-cycle mass = exact structural doubling of birth
SIM_WINDOW = 210.0    # Tier 2 (engine simulate_mutant tf=210)
N_BINS = 5            # ledger bins per D dimension
TRAJ_MAX = 1e3        # numerical validity gate (v3.7, Tier 2): physiological
                      # states are O(1) (event thresholds <=1, mass limit 10);
                      # any |y| > 1e3 means the integrator diverged on this
                      # combo — no biological meaning, rolled back at
                      # generation time and replaced by a resample.
N_LOG_BANDS = 8       # fold / effect log10 bands
BOUNDARY_LO, BOUNDARY_HI = 0.5, 2.0   # structural threshold band (Tier 2)
MASS_LIMIT = 10.0     # engine mass-divergence discard bound (max_m > 10)

# =============================================================================
# Part 1: Module regulation graph (36 edges of Scheme §3.1.1 — Tier 2 symbols)
# =============================================================================

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
]

GROUP_BY_NAME = {g['name']: g for g in BINDING_GROUPS.values()}
GENE_NAMES = sorted(GROUP_BY_NAME.keys())
# P_bio: mechanized degree centrality on the 36-edge module graph (Tier 2 —
# pure equation-derived topology). w(G) = (undirected degree + 1), normalized;
# the +1 is Laplace smoothing so isolated modules keep a uniform-prior weight.
# The engine's hand-set P_bio_G values (0.03–0.06, no verifiable derivation)
# are retained ONLY as an ablation arm (Scheme §8-4c) — per the project rule
# that every number in the sampler needs a Tier 1/2/3 attribution.
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

# Ablation A switch (Scheme §9.6): mode-frequency source.
#   'flat' — uniform over modes given gene (anti-leak default, §3.1.4)
#   's5'   — Table S5-fitted mode weights (BINDING_GROUPS[g]['modes']), the
#            same source mass_random_sampler uses. DIAGNOSTIC ARM ONLY: this
#            re-introduces the real-genotype-list leak ON PURPOSE, to isolate
#            the mode-frequency channel's contribution to real-mutant metrics
#            (prediction: rescues the Pattern_C KO-stack cluster 13/17/21/26/88/89).
# All other machinery (ledger, level quotas, folds, GAL inversion, dup bans)
# is untouched in both arms; the 'flat' arm consumes the exact same RNG calls.
MODE_FREQ = 'flat'

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

# C-A: adjacent with a negative edge, or two '+' inputs of the same target
# (shared switch input-output ends), or mutual +/- pair (same switch both ends).
_NEG_PAIRS = {(a, b) for a, b, s in MODULE_EDGES if s == '-'} | \
             {(b, a) for a, b, s in MODULE_EDGES if s == '-'}
_OUT = {n: set() for n in GENE_NAMES}
for a, b, s in MODULE_EDGES:
    if s == '+': _OUT[a].add(b)

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

STAR_CENTERS = ['CDC28', 'CLB2', 'MDT']
STAR_LEAVES = {'CDC28': sorted(ADJ['CDC28']), 'CLB2': sorted(ADJ['CLB2']),
               'MDT': [n for n in GENE_NAMES if n != 'MDT']}

# =============================================================================
# Part 2: Mode semantics (parameter-touch sets for conflict check + fold kinds)
# =============================================================================

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
 ('TEM1','MC'): {'Tem1T'}, ('TEM1','ts'): {'Tem1T','ka15pp'},
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

# modes where the sampled fold f multiplies a rate (Scheme §3.2.1)
FOLD_MODES = {
    ('CLN2','MC'), ('CLN3','MC'), ('BCK2','MC'), ('CLB2','MC'),
    ('CLB5','MC'), ('SIC1','MC'), ('CDC6','MC'), ('CDC20','MC'),
    ('CDC14','MC'), ('TEM1','MC'), ('CDC15','MC'),
    ('CLN2','ts'), ('CLB2','ts'), ('CDC20','ts'), ('CDC14','ts'),
    ('NET1','ts'), ('CDC28','ts'),
    ('CDC6','DAmP'), ('CDC20','DAmP'), ('CDC14','DAmP'), ('NET1','DAmP'),
    ('TEM1','DAmP'), ('CDC15','DAmP'), ('CLB2','DAmP'), ('CDC28','DAmP'),
    ('CLB2','pLOF'), ('CDC28','pLOF'),
    ('CLB2','db'),
}
RESIDUAL_MODES = {'ts', 'DAmP', 'pLOF', 'db'}   # f in [1e-3, 1], zero-inflated
COMPOSITE_ELIGIBLE = ['CLB2', 'CLB5', 'PDS1', 'SIC1', 'CDH1']

# Tier-2 allele-residual patches (Scheme v3.5 §3.2.3): mutations whose engine
# edit hard-codes a paper allele constant are re-parameterized as WT x f with
# f from the residual fold sampler (zero-inflated: f=0 == the paper's complete
# allele). Justification, Chen (2004) p.3848: ts at semi-permissive temperature
# = "rate constants ... reduced to a FRACTION of the wild-type values" — the
# paper's own continuous-residual semantics. Every paper allele is one point
# inside the range (e.g. ESP1-ts: f=0.002 reproduces kasesp=0.1 & kdiesp=0.001
# exactly; PPX-ts: 0.05/3.0 ~= 0.017; Mcm1-ts: 0.13; all db zeros: f=0).
# Modes NOT here stay binary by design (CA/CSL/CKO/APC-A/scc1-RR/cdc24Δ/
# cdc45Δ/spindle_defect/TAB6-1/KO/APC_Core-ts): dominant alleles or event
# knockouts with no strength dimension in the literature.
ALLELE_RESIDUAL = {
    ('PPX','ts'):        ['kppnetpp'],
    ('CDC5_Proxy','ts'): ['kppnetpp'],
    ('CDC15','ts'):      ['kkpnetpp'],
    ('Mcm1','ts'):       ['kamcm'],
    ('TEM1','ts'):       ['Tem1T', 'ka15pp'],
    ('ESP1','ts'):       ['kasesp', 'kdiesp'],
    ('CLB5','db'):       ['kdb5pp'],
    ('PDS1','db'):       ['kd2pdspp', 'kd3pdspp'],
    ('SIC1','db'):       ['kd3c1'],
    ('SIC1','PM'):       ['kd2c1', 'kd3c1'],
    ('PPX','db'):        ['kdppxpp'],
    ('CDC5_Proxy','db'): ['kdppxpp'],
}

# =============================================================================
# Part 3: Quota samplers (log bands with exact equalization, §3.2.1 / §3.2.2)
# =============================================================================

class LogBandSampler:
    """log10-uniform band sampler with exact per-band quota equalization."""
    def __init__(self, lo, hi, n_bands=N_LOG_BANDS, zero_prob=0.0,
                 boundary=None, boundary_prob=0.0):
        self.edges = np.linspace(lo, hi, n_bands + 1)
        self.n = n_bands
        self.counts = np.zeros(n_bands, dtype=int)
        self.zero_prob = zero_prob
        self.boundary = boundary            # (lo, hi) linear-space band
        self.boundary_prob = boundary_prob
        self.n_zero = 0

    def sample(self, band_bias=None):
        """band_bias: optional callable band_index -> extra probability weight
        (used by backlog directed generation)."""
        if self.zero_prob > 0 and np.random.rand() < self.zero_prob:
            self.n_zero += 1
            return 0.0
        if self.boundary is not None and np.random.rand() < self.boundary_prob:
            return float(10 ** np.random.uniform(np.log10(self.boundary[0]),
                                                 np.log10(self.boundary[1])))
        if band_bias is None:
            m = self.counts.min()
            cand = np.where(self.counts == m)[0]
            i = int(np.random.choice(cand))
        else:
            w = np.asarray(band_bias, dtype=float) + 1e-9
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
            w = np.maximum(deficit, 1e-9)
        else:
            w = np.asarray(weights, dtype=float) + 1e-9
        i = int(np.random.choice(len(self.names), p=w / w.sum()))
        self.counts[i] += 1
        return self.names[i]

# =============================================================================
# Part 4: Self-consistent QSS fixed points (Scheme §3.4 — no WT snapshot)
# =============================================================================

def qss_g1(p, m, max_iter=80, tol=1e-4, damp=0.5):
    """G1-phase (pre-Start, MCM1=0 basal branch) fixed point. Tier 3 algebra."""
    CLN3 = p['C0'] * p['Dn3'] * m / (p['Jn3'] + p['Dn3'] * m)
    BCK2 = p['B0'] * m
    # G1-frozen auxiliaries (cyclin-low phase)
    CDH1T = p['kscdh'] / p['kdcdh']
    CDH1 = CDH1T * GK(p['kacdhp'], p['kicdhp'], p['Jacdh'], p['Jicdh'])
    CDC20T = p['ks20p'] / max(p['kd20'], EPS)
    CDC20a = CDC20T * p['ka20p'] / max(p['ka20p'] + 0.01 + p['kd20'], EPS)
    SIC1T = p['ksc1p'] / max(p['kd1c1'], EPS)       # SWI5 low in G1
    CDC6T = p['ksf6p'] / max(p['kd1f6'], EPS)
    Vdb2 = p['kdb2p'] + p['kdb2pp'] * CDH1 + p['kdb2P'] * CDC20a
    kappa2 = Vdb2 + p['kasb2'] * SIC1T + p['kasf2'] * CDC6T
    Vdb5 = p['kdb5p'] + p['kdb5pp'] * CDC20a
    SBF, CLN2, CLB5, Clb2 = 0.1, 0.05, 0.05, 0.05
    for _ in range(max_iter):
        if p.get('is_sbf_ca'):
            Vasbf = 0.38
        else:
            Vasbf = p['kasbf'] * (p['esbfn2'] * CLN2 +
                                  p['esbfn3'] * (CLN3 + BCK2) +
                                  p['esbfb5'] * CLB5)
        Visbf = p['kisbfp'] + p['kisbfpp'] * Clb2
        SBF_n = GK(Vasbf, Visbf, p['Jasbf'], p['Jisbf'])
        CLN2_n = (p['ksn2p'] + p['ksn2pp'] * SBF_n) * m / max(p['kdn2'], EPS)
        CLB5_n = (p['ksb5p'] + p['ksb5pp'] * SBF_n) * m / max(Vdb5, EPS)
        Clb2_n = p['ksb2p'] * m / max(kappa2, EPS)
        SBF = (1-damp)*SBF + damp*SBF_n
        CLN2 = (1-damp)*CLN2 + damp*CLN2_n
        CLB5 = (1-damp)*CLB5 + damp*CLB5_n
        Clb2 = (1-damp)*Clb2 + damp*Clb2_n
        err = max(abs(SBF_n-SBF), abs(CLN2_n-CLN2)/max(CLN2_n,EPS),
                  abs(CLB5_n-CLB5)/max(CLB5_n,EPS), abs(Clb2_n-Clb2)/max(Clb2_n,EPS))
        if err < tol:
            break
    else:
        if not np.isfinite(CLN2 + CLB5 + Clb2 + SBF):
            return None
    if not np.isfinite(CLN2 + CLB5 + Clb2 + SBF):
        return None
    return dict(SBF=SBF, CLN2=CLN2, CLB5=CLB5, Clb2=Clb2,
                Vasbf=Vasbf, Visbf=Visbf, Vdb5=Vdb5, CLN3=CLN3, BCK2=BCK2,
                CDC20a=CDC20a, CDH1=CDH1)

def qss_M(p, m, phase='meta', max_iter=200, tol=1e-4, damp=0.5):
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
        # warm start on the HIGH-Clb2 branch of the Clb2<->MCM1 bistable
        # switch (the low branch is the G1 state, handled by qss_g1).
        v = dict(SBF=0.5, CLN2=0.1, CLB5=0.1, MCM1=0.9, Clb2=0.5, IEP=0.9,
                 CDC20T=1.5, CDC20a=0.1, CDH1T=1.0, CDH1=0.2, SWI5=0.2,
                 SIC1T=0.02, CDC6T=0.01, CDC14T=2.0, CDC14=0.2, TEM1a=0.1,
                 CDC15a=0.2, PPX=0.1, PDS1=0.3, ESP1=0.2, ORI=1.0, SPN=0.5,
                 PE=0.8, kappa2=1.0, Vkpc1=0.5, Vkpf6=0.5, Vdb2=0.5,
                 Vdpds=0.1, Vac15=0.1, Vicdh=0.05)
    else:
        Vi20 = 0.01
        BUB2 = p.get('init_BUB2', 0.2)
        lte1 = 1.0
        v = dict(SBF=0.5, CLN2=0.1, CLB5=0.1, MCM1=0.5, Clb2=0.3, IEP=0.9,
                 CDC20T=1.5, CDC20a=1.2, CDH1T=1.0, CDH1=0.8, SWI5=0.8,
                 SIC1T=0.01, CDC6T=0.01, CDC14T=2.0, CDC14=1.5, TEM1a=0.9,
                 CDC15a=0.6, PPX=0.1, PDS1=0.05, ESP1=0.8, ORI=1.0, SPN=1.0,
                 PE=0.2, kappa2=1.0, Vkpc1=0.5, Vkpf6=0.5, Vdb2=0.5,
                 Vdpds=0.1, Vac15=0.5, Vicdh=0.05)
    for _ in range(max_iter):
        old = v.copy()
        if p.get('is_sbf_ca'):
            Vasbf = 0.38
        else:
            Vasbf = p['kasbf'] * (p['esbfn2'] * v['CLN2'] +
                                  p['esbfn3'] * (CLN3 + BCK2) +
                                  p['esbfb5'] * v['CLB5'])
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
        # MEN chain (TEM1 -> CDC15 -> NET1 phosphorylation -> CDC14 release)
        v['CDC14T'] = p['ks14'] / max(p['kd14'], EPS)
        v['TEM1a'] = p['Tem1T'] * lte1 / max(lte1 + BUB2, EPS)
        v['Vac15'] = (p['ka15p'] * max(p['Tem1T'] - v['TEM1a'], 0.0) +
                      p['ka15pp'] * v['TEM1a'] + p['ka15P'] * v['CDC14'])
        v['CDC15a'] = p['Cdc15T'] * v['Vac15'] / max(v['Vac15'] + p['ki15'], EPS)
        Vkpnet = (p['kkpnetp'] + p['kkpnetpp'] * v['CDC15a']) * m
        Vppnet = p['kppnetp'] + p['kppnetpp'] * v['PPX']
        v['CDC14'] = v['CDC14T'] * Vkpnet / max(Vkpnet + Vppnet, EPS)
        Vdppx = p['kdppxp'] + p['kdppxpp'] * (p['J20ppx'] + v['CDC20a']) * \
                p['Jpds'] / (p['Jpds'] + v['PDS1'])
        v['PPX'] = p['ksppx'] / max(Vdppx, EPS)
        # CDH1 switch
        v['CDH1T'] = p['kscdh'] / max(p['kdcdh'], EPS)
        Vacdh = p['kacdhp'] + p['kacdhpp'] * v['CDC14']
        v['Vicdh'] = p['kicdhp'] + p['kicdhpp'] * (
            p['eicdhn3'] * CLN3 + p['eicdhn2'] * v['CLN2'] +
            p['eicdhb5'] * v['CLB5'] + p['eicdhb2'] * v['Clb2'])
        v['CDH1'] = v['CDH1T'] * GK(Vacdh, v['Vicdh'], p['Jacdh'], p['Jicdh'])
        # SIC1 / CDC6 (SWI5 active form; free fractions for sequestration)
        SWI5T = (p['ksswip'] + p['ksswipp'] * v['MCM1']) / max(p['kdswi'], EPS)
        v['SWI5'] = SWI5T * p['kaswi'] * v['CDC14'] / max(
            p['kaswi'] * v['CDC14'] + p['kiswi'] * v['Clb2'], EPS)
        cyc1 = (p['ec1n3'] * CLN3 + p['ec1k2'] * BCK2 + p['ec1n2'] * v['CLN2'] +
                p['ec1b2'] * v['Clb2'] + p['ec1b5'] * v['CLB5'])
        v['Vkpc1'] = p['kd1c1'] + p['kd2c1'] * cyc1 / (p['Jd2c1'] + v['SIC1T'])
        v['SIC1T'] = (p['ksc1p'] + p['ksc1pp'] * v['SWI5']) / max(v['Vkpc1'], EPS)
        cyc6 = (p['ef6n3'] * CLN3 + p['ef6k2'] * BCK2 + p['ef6n2'] * v['CLN2'] +
                p['ef6b2'] * v['Clb2'] + p['ef6b5'] * v['CLB5'])
        v['Vkpf6'] = p['kd1f6'] + p['kd2f6'] * cyc6 / (p['Jd2f6'] + v['CDC6T'])
        v['CDC6T'] = (p['ksf6p'] + p['ksf6pp'] * v['SWI5'] +
                      p['ksf6ppp'] * v['SBF']) / max(v['Vkpf6'], EPS)
        # free fractions (binding equilibrium with dissociation + clearance)
        koff1 = p['kdib2'] + v['Vdb2'] + v['Vkpc1']
        SIC1_free = v['SIC1T'] * koff1 / max(koff1 + p['kasb2'] * v['Clb2'] +
                                             p['kasb5'] * v['CLB5'], EPS)
        koff6 = p['kdif2'] + v['Vdb2'] + v['Vkpf6']
        CDC6_free = v['CDC6T'] * koff6 / max(koff6 + p['kasf2'] * v['Clb2'] +
                                             p['kasf5'] * v['CLB5'], EPS)
        v['kappa2'] = v['Vdb2'] + p['kasb2'] * SIC1_free + p['kasf2'] * CDC6_free
        v['Clb2'] = (p['ksb2p'] + p['ksb2pp'] * v['MCM1']) * m / max(v['kappa2'], EPS)
        v['IEP'] = GK(p['kaiep'] * v['Clb2'], p['kiiep'], p['Jaiep'], p['Jiiep'])
        # PDS1 / ESP1 sequestration pair
        v['Vdpds'] = (p['kd1pdsp'] + p['kd2pdspp'] * v['CDC20a'] +
                      p['kd3pdspp'] * v['CDH1'])
        v['PE'] = max(p['ESP1T'] - v['ESP1'], 0.0)
        v['PDS1'] = (p['kspdsp'] + p['ks1pdspp'] * v['SBF'] +
                     p['ks2pdspp'] * v['MCM1'] + p['kdiesp'] * v['PE']) / \
                    max(v['Vdpds'] + p['kasesp'] * v['ESP1'], EPS)
        v['ESP1'] = p['ESP1T'] * (p['kdiesp'] + v['Vdpds']) / \
                    max(p['kasesp'] * v['PDS1'] + p['kdiesp'] + v['Vdpds'], EPS)
        # events-driven checkpoints
        v['ORI'] = p['ksori'] * (p['eorib5'] * v['CLB5'] +
                                 p['eorib2'] * v['Clb2']) / max(p['kdori'], EPS)
        v['SPN'] = p['ksspn'] * v['Clb2'] / (p['Jspn'] + v['Clb2']) / max(p['kdspn'], EPS)
        # damping
        for k in v:
            v[k] = (1 - damp) * old[k] + damp * v[k]
        err = max(abs(v[k] - old[k]) / max(abs(old[k]), EPS) for k in v)
        if err < tol:
            return v
    if any(not np.isfinite(x) for x in v.values()):
        return None
    return v   # not converged within max_iter: usable but flagged by audit

# =============================================================================
# Part 5: 10-D driving-force coordinates (Scheme §3.3 / §3.4)
# =============================================================================

D_NAMES = ['D1_RStart', 'D2_RORI', 'D3_RM', 'D4_RC20', 'D5_Rdeg', 'D6_RMEN',
           'D7_RRENT', 'D8_Bcyc', 'D9_RSPN', 'D10_Rsep']

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
    if r_lo is None: return np.nan
    if r_lo >= 1.0:  return m_lo
    r_hi = R(m_hi)
    if r_hi is None: return np.nan
    if r_hi < 1.0:   return np.inf
    lo, hi = m_lo, m_hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        r = R(mid)
        if r is None: return np.nan
        if r >= 1.0: hi = mid
        else: lo = mid
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
        return D  # solver failure -> NaN bucket
    fp_meta = qss_M(p, M_M, 'meta')
    fp_ana = qss_M(p, M_M, 'ana')
    if fp_meta is None or fp_ana is None:
        return D
    D[0] = d1
    D[1] = fp_meta['ORI']                                # event threshold 1
    D[2] = p['kamcm'] * fp_meta['Clb2'] / max(p['kimcm'], EPS)
    act20 = p['ka20p'] + p['ka20pp'] * fp_meta['IEP']
    D[3] = act20 / max(0.01 + p['kd20'], EPS)            # SAC-off permissive branch
    kclr = p['kdb2P'] * fp_ana['CDC20a'] + p['kdb2pp'] * fp_ana['CDH1']
    if kclr <= EPS or fp_meta['Clb2'] <= 0.3:
        D[4] = np.inf                                    # never reaches 0.3
    else:
        D[4] = math.log(fp_meta['Clb2'] / 0.3) / kclr    # event threshold 0.3
    D[5] = (p['ka15pp'] * fp_ana['TEM1a'] + p['ka15P'] * fp_ana['CDC14']) / \
           max(p['ki15'], EPS)
    D[6] = (p['kkpnetp'] + p['kkpnetpp'] * fp_ana['CDC15a']) * M_M / \
           max(p['kppnetp'] + p['kppnetpp'] * fp_ana['PPX'], EPS)
    D[7] = SIM_WINDOW * math.log(2) / mdt
    D[8] = fp_meta['SPN']                                # event threshold 1
    # ESP1 sequestration/release drive ratio (structural GK-like Va/Vi = 1
    # condition, Tier 2; self-consistent version of the spec's D10 — the
    # spec's T_ref normalizer is undefined, this form needs no extra constant)
    D[9] = fp_meta['PDS1'] * p['kasesp'] / max(p['kdiesp'] + fp_meta['Vdpds'], EPS)
    return D

# =============================================================================
# Part 6: GAL-OE self-consistent effect inversion (Scheme §3.2.2)
# =============================================================================

# --- GAL alpha biophysical cap (v3.7 fix: parameter pollution) ---------------
# GAL promoter overexpression is bounded by promoter strength: Chen (2004)'s own
# GAL constructs sit at ~1-10x the gene's native synthesis capacity (GAL-CLN2
# 0.12 vs native 0.15; GAL-CLN3 2.0 vs 0.4; GAL-SIC1 10x k's,c1), the strongest
# GAL/plasmid systems reach ~100x. Tier 2 biophysical bound (Scheme §1
# whitelist item 4). Without it, multi-mutation backgrounds distort the fixed
# point and the algebraic inversion emits 1e8..1e28 alphas that (a) enter the
# npz even on the NaN-bucket path and (b) blow up dataset-level param std.
GAL_ALPHA_CAP_MULT = 100.0
_NATIVE_SCALE_PARAMS = {
    'CLN2': ('ksn2p', 'ksn2pp'),
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
    'PPX': ('ksppx',),
    'CDC5_Proxy': ('ksppx',),
    'PDS1': ('kspdsp', 'ks1pdspp', 'ks2pdspp'),
    'ESP1': ('ESP1T',),
}

def gal_alpha_cap(gene, p_wt):
    """alpha_cap = 100 x gene's native maximum synthesis capacity (Tier 2)."""
    scale = sum(max(p_wt.get(k, 0.0), 0.0)
                for k in _NATIVE_SCALE_PARAMS.get(gene, ()))
    if scale <= 0.0:
        scale = 1.0   # unreachable: every gene has a nonzero native term
    return GAL_ALPHA_CAP_MULT * scale

# v3.8: per-gene GAL promoter-strength bio-ranges (Tier 2 white-list item 4:
# "GAL promoter induction strength, plasmid copy number, and other literature
# magnitude ranges" — the same documented ranges the engine's own
# sample_scaling_factor encodes). Empirical motivation (see Scheme §9.6):
# the pure effect-inversion mis-scales alpha by orders of magnitude on
# several genes (PDS1/CDC6 -> alpha~0, i.e. GAL degenerates to KO; CLN2 ->
# pegged at the 100x cap), leaving the bio-realistic alpha region EMPTY in
# the dataset — real GAL constructs (e.g. GAL-PDS1 kspdsp=0.1) then sit
# >400 std away from the training support and become unlearnable.
GAL_ALPHA_BIO = {
    'CDC20': (0.50, 12.0), 'CLB2': (0.002, 1.50), 'SIC1': (0.03, 0.36),
    'CDC6': (0.10, 0.60), 'CLB5': (0.004, 0.030), 'PDS1': (0.02, 0.25),
    'CLN2': (0.03, 0.30), 'CLN3': (1.0, 5.0), 'TEM1': (2.0, 8.0),
    'CDC15': (3.0, 20.0), 'ESP1': (1.5, 6.0), 'CDC14': (0.30, 1.20),
    'NET1': (0.15, 0.60), 'PPX': (0.20, 0.60), 'CDC5': (0.20, 0.60),
    'CDC5_Proxy': (0.20, 0.60), 'SWI5': (0.02, 0.10),
}
GAL_BIO_FALLBACK = (2.0, 30.0)   # undocumented genes: engine's own fallback

# module-level counters, reset per run, reported in the audit
GAL_STATS = {'eff_guided': 0, 'bio_draw': 0}

def _gal_theta_and_invert(gene, E, p, fpG1, fpM):
    """Per-gene structural threshold theta (Tier 2 / own fixed point) and the
    one-step algebraic inversion alpha(E). All concentrations are the
    candidate's OWN fixed-point values (Tier 3 algebra)."""
    if gene == 'CLN2':
        th = p['kisbfp'] / max(p['kasbf'] * p['esbfn2'], EPS)
        return th, E * th * p['kdn2'] / M_BIRTH            # ksn2pp zeroed by GAL
    if gene == 'CLN3':
        th = p['kisbfp'] / max(p['kasbf'] * p['esbfn3'], EPS)
        return th, E * th * (p['Jn3'] + p['Dn3'] * M_BIRTH) / max(p['Dn3'] * M_BIRTH, EPS)
    if gene == 'CLB5':
        th = p['kisbfp'] / max(p['kasbf'] * p['esbfb5'], EPS)
        return th, E * th * fpG1['Vdb5'] / M_BIRTH - p['ksb5pp'] * fpG1['SBF']
    if gene == 'CLB2':
        th = p['kimcm'] / max(p['kamcm'], EPS)
        return th, E * th * fpM['kappa2'] / M_M - p['ksb2pp'] * fpM['MCM1']
    if gene == 'SIC1':
        th = fpM['Vdb2'] / max(p['kasb2'], EPS)
        return th, E * th * fpM['Vkpc1'] - p['ksc1pp'] * fpM['SWI5']
    if gene == 'CDC6':
        th = fpM['Vdb2'] / max(p['kasf2'], EPS)
        return th, E * th * fpM['Vkpf6'] - p['ksf6pp'] * fpM['SWI5'] - p['ksf6ppp'] * fpM['SBF']
    if gene == 'SWI5':
        th = max(p['ksc1p'], 0.0) / max(p['ksc1pp'], EPS)
        return th, E * th * p['kdswi'] - p['ksswipp'] * fpM['MCM1']
    if gene == 'CDC20':
        th = max((p['kdb2p'] + p['kdb2pp'] * fpM['CDH1']) / max(p['kdb2P'], 1e-6), 0.1)
        return th, E * th * p['kd20'] - p['ks20pp'] * fpM['MCM1']
    if gene == 'CDC14':
        th = max(p['kicdhp'] + p['kicdhpp'] * 0.05, EPS) / max(p['kacdhpp'], EPS)
        f14 = fpM['CDC14'] / max(fpM['CDC14T'], EPS)
        return th, E * th * p['kd14'] / max(f14, 0.05)
    if gene == 'NET1':
        th = fpM['CDC14T']                               # stoichiometric balance
        return th, E * th * p['kdnet']
    if gene == 'TEM1':
        th = p['ki15'] / max(p['ka15pp'], EPS)
        return th, E * th * (0.1 + p.get('init_BUB2', 0.2)) / 0.1
    if gene == 'CDC15':
        th = max((p['kppnetp'] + p['kppnetpp'] * fpM['PPX'] - p['kkpnetp'] * M_M) /
                 max(p['kkpnetpp'] * M_M, EPS), 0.05)
        frac = fpM['Vac15'] / max(fpM['Vac15'] + p['ki15'], EPS)
        return th, E * th / max(frac, 0.05)
    if gene in ('PPX', 'CDC5_Proxy'):
        th = p['kppnetp'] / max(p['kppnetpp'], EPS)
        Vdppx = p['kdppxp'] + p['kdppxpp'] * (p['J20ppx'] + fpM['CDC20a']) * \
                p['Jpds'] / (p['Jpds'] + fpM['PDS1'])
        return th, E * th * max(Vdppx, EPS)
    if gene == 'PDS1':
        th = p['kdiesp'] / max(p['kasesp'], EPS)
        return th, E * th * (fpM['Vdpds'] + p['kasesp'] * fpM['ESP1']) - \
               p['ks1pdspp'] * fpM['SBF'] - p['ks2pdspp'] * fpM['MCM1'] - \
               p['kdiesp'] * fpM['PE']
    if gene == 'ESP1':
        th = fpM['PDS1']
        return th, E * th
    return 1.0, E  # unreachable fallback

def apply_gal_inversions(gal_list, p_wt, p_mut, ovr):
    """v3.8 hybrid: effect-guided alpha bounded by the per-gene promoter-
    strength bio-range (Tier 2). Iterative self-consistency is kept (2-3
    damped rounds), but the inverted alpha is accepted ONLY when it lands
    inside the bio-range; otherwise a log-uniform draw WITHIN the bio-range
    replaces it (physical prior). This removes the v3.7 pathology where a
    mis-scaled inversion collapsed alpha to ~0 (PDS1/CDC6: GAL == KO) or
    pegged it at the 100x cap (CLN2), leaving the bio-realistic region
    empty. gal_list: [(gene, E)]. Returns (converged, alphas)."""
    if not gal_list:
        return True, {}
    alphas = {g: 0.0 for g, _ in gal_list}
    # round 0: apply with alpha=0, then invert from that state
    for g, _ in gal_list:
        apply_gene_mutation(g, 'GAL-OE', 0.0, p_wt, p_mut, ovr)
    converged = False
    frozen = {}   # v3.8.1: gene -> bio-range draw, drawn ONCE and frozen.
    # A physical prior needs no self-consistency: re-drawing every round made
    # the 5% convergence test fail almost surely for every out-of-range gene
    # (80.7% of inversions), flagging those candidates gal_conv=False ->
    # pilot NaN 6.2% -> 24.4% and ~20% of main-phase attempts wasted after
    # the NaN cap filled. Frozen draws repeat exactly (delta=0), so only the
    # effect-guided (in-range) genes must satisfy the 5% stability test.
    for _round in range(3):
        p_eff = p_wt.copy(); p_eff.update(ovr)
        fpG1 = qss_g1(p_eff, M_BIRTH)
        fpM = qss_M(p_eff, M_M)
        if fpG1 is None or fpM is None:
            return False, alphas
        new_alphas = {}
        for g, E in gal_list:
            if g in frozen:
                new_alphas[g] = frozen[g]
                continue
            _, a = _gal_theta_and_invert(g, E, p_eff, fpG1, fpM)
            a = max(a, 0.0)
            lo, hi = GAL_ALPHA_BIO.get(g, GAL_BIO_FALLBACK)
            if lo <= a <= hi:
                GAL_STATS['eff_guided'] += 1   # effect-guided, in bio-range
            else:
                # inversion out of the physically meaningful range -> replace
                # with a log-uniform draw inside the bio-range (Tier 2 prior),
                # drawn once and frozen for the rest of this candidate
                a = float(10.0 ** np.random.uniform(np.log10(lo), np.log10(hi)))
                frozen[g] = a
                GAL_STATS['bio_draw'] += 1
            new_alphas[g] = a
        # convergence: all relative changes < 5%. Check every gene with a
        # NONZERO NEW alpha — skipping genes by old-alpha silently ignores
        # 0 -> large jumps and could declare premature convergence.
        conv = all(abs(new_alphas[g] - alphas[g]) <=
                   0.05 * max(new_alphas[g], alphas[g], 1e-9)
                   for g in new_alphas if new_alphas[g] > 1e-9)
        alphas = new_alphas
        for g in gal_list:
            apply_gene_mutation(g[0], 'GAL-OE', alphas[g[0]], p_wt, p_mut, ovr)
        if conv and _round >= 1:
            converged = True
            break
    return converged, alphas

# =============================================================================
# Part 7: Structure-aware skeleton sampling (Scheme §3.1)
# =============================================================================

LEVEL_QUOTAS = [0.40, 0.30, 0.20, 0.10]
STRUCT_L2 = {'C-A': 0.30, 'C-B': 0.30, 'C-C': 0.15, 'C-D': 0.25}
STRUCT_L34 = {'cascade': 0.45, 'star': 0.25, 'antag_core': 0.20, 'orthogonal': 0.10}

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

def _sample_L2(boost=None):
    cls = QuotaPicker_picker_L2()
    if cls == 'C-C':
        center = str(np.random.choice(['CDC28', 'CLB2', 'MDT'], p=[0.4, 0.4, 0.2]))
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

def _sample_star(ki, boost=None):
    center = str(np.random.choice(['CDC28', 'CLB2', 'MDT'], p=[0.4, 0.4, 0.2]))
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
    """Returns (genes list, structure_class). Level quotas are exact."""
    if level == 1:
        return [_weighted_gene(boost)], 'L1'
    if level == 2:
        return _sample_L2(boost)
    ki = int(np.random.randint(3, 6)) if level == 3 else int(np.random.randint(4, 7))
    cls = _L34_PICKER.pick()
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

# =============================================================================
# Part 8: Candidate construction (Phase 1 + Phase 2 of the pseudocode)
# =============================================================================

PRIMARY_COORDS = {   # gene -> 1-2 primary D dims (1-indexed), Scheme §3.5-3
 'CLN3':[1],'BCK2':[1],'CLN2':[1],'SBF_MBF':[1],'CDC28':[1,2],'CLB5':[1,2],
 'ORI_BUD_SPN':[2,9],'Mcm1':[3],'CLB2':[3,5],'SIC1':[3,5],'CDC6':[3],'SWI5':[3],
 'CDC20':[4,5],'MAD2':[4],'APC_Core':[4,5],'CDH1':[5],'CDC14':[5,6],
 'ESP1':[10],'PDS1':[10],'SCC1':[10],'NET1':[6,7],'TEM1':[6],'CDC15':[6,7],
 'PPX':[7],'CDC5_Proxy':[7],'MDT':[8],'BUB2':[6],
}

def _sample_mode(gene):
    modes = list(GROUP_BY_NAME[gene]['modes'].keys())
    if MODE_FREQ == 's5':   # Ablation A: Table S5-fitted mode weights (leaks by design)
        w = np.array([GROUP_BY_NAME[gene]['modes'][m] for m in modes], dtype=float)
        return str(np.random.choice(modes, p=w / w.sum()))
    return str(np.random.choice(modes))   # flattened uniform (anti-leak, §3.1.4)

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
    # persistent conflict: drop the CONFLICTING pair itself (dropping the
    # last pair could leave the actual conflict unresolved, §3.1.6)
    if len(pairs) > 1:
        pairs.pop(conflict)
    return pairs

def build_candidate(level, samplers, boost=None, fold_bias_fn=None):
    """Phases 1-2: skeleton -> modes -> folds / GAL inversion -> ovr dict.
    Returns dict(ovr, desc, genes_modes, struct_class, mdt, gal_conv) or None."""
    genes, struct_class = sample_skeleton(level, boost)
    if level == 1:
        g0, m0 = next_L1_cell()
        if MODE_FREQ == 's5':
            # Ablation A: keep the grid's gene marginal, re-draw the mode from
            # S5 weights — only the mode-frequency channel changes
            m0 = _sample_mode(g0)
        genes, pairs = [g0], [(g0, m0)]
    else:
        pairs = [(g, _sample_mode(g)) for g in genes]
        # composite expression+structure mode, 25% (§3.1.6 legality)
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

    # edit order (§3.1.6): structural/KO -> scaling folds -> GAL absolute
    edits = []          # (gene, mode, val) for pre-simulative direction signs
    for g, m in pairs:
        desc_parts.append(f"{g}[{m}]")
        if m == 'GAL-OE':
            has_gal = True
            E = samplers['E'].sample(fold_bias_fn('E') if fold_bias_fn else None)
            gal_list.append((g, E))
            edits.append((g, m, E))
        elif (g, m) in FOLD_MODES:
            f = samplers['f_res' if m in RESIDUAL_MODES else 'f_mc'].sample(
                fold_bias_fn(m) if fold_bias_fn else None)
            apply_gene_mutation(g, m, f, p_wt, p_mut, ovr)
            edits.append((g, m, f))
        else:
            apply_gene_mutation(g, m, 1.0, p_wt, p_mut, ovr)
            if (g, m) in ALLELE_RESIDUAL:
                # allele-residual patch (§3.2.3): paper constant -> WT x f
                f = samplers['f_res'].sample(
                    fold_bias_fn(m) if fold_bias_fn else None)
                for pk in ALLELE_RESIDUAL[(g, m)]:
                    ovr[pk] = p_wt[pk] * f   # f=0 (zero-inflated) == paper allele
                edits.append((g, m, f))
            else:
                edits.append((g, m, None))

    gal_conv, _alphas = (True, {})
    if gal_list:
        gal_conv, _alphas = apply_gal_inversions(gal_list, p_wt, p_mut, ovr)

    # carbon source rule (§3.1.5): GAL -> Gal; else uniform Glc/Gal/Raff
    if 'mdt' not in ovr:
        if has_gal:
            ovr['mdt'] = 150.0
        else:
            ovr['mdt'] = float(np.random.choice([90.0, 150.0, 160.0]))

    return dict(ovr=ovr, desc="+".join(desc_parts), genes_modes=pairs,
                struct_class=struct_class, level=level, mdt=ovr['mdt'],
                gal_conv=gal_conv, edits=edits)

# =============================================================================
# Part 9: Coverage ledger (Scheme §3.5, M4')
# =============================================================================

class Ledger:
    def __init__(self, n_target, n_bins=N_BINS, nan_cap_frac=0.05):
        self.n_bins = n_bins
        self.edges = None                      # dim -> bin edges (pilot)
        self.counts = np.zeros((10, n_bins), dtype=int)
        self.cap = int(math.ceil(1.3 * n_target / n_bins))
        self.dir_counts = np.zeros((10, 3), dtype=int)   # up/down/neutral
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
                lo, hi = 0.0, 1.0
            else:
                lo, hi = float(np.min(fin)), float(np.max(fin))
            qs = np.quantile(fin, np.linspace(0, 1, self.n_bins + 1)) \
                 if len(fin) >= 20 else np.linspace(lo, hi, self.n_bins + 1)
            qs = np.maximum.accumulate(qs)
            # de-collapse: make interior edges strictly increasing so DISTINCT
            # values fall into distinct bins (identical values share a bin by
            # design — they are the same behavior, e.g. D1 = birth mass).
            for j in range(1, len(qs) - 1):
                if qs[j] <= qs[j - 1]:
                    base = qs[j - 1] if np.isfinite(qs[j - 1]) else 0.0
                    qs[j] = base + abs(base) * 1e-9 + 1e-12
            qs[0], qs[-1] = -np.inf, np.inf
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

# =============================================================================
# Part 10: Main generation pipeline
# =============================================================================

# Pre-simulative direction sign table (Scheme §3.5: "方向符号是参数的预仿真
# 函数"). Signs are ODE structural monotonicities (Tier 2): product/activity
# gain of the gene moves its primary D coordinates by the indicated sign.
# dim indices are 0-based (D1=0 ... D10=9). NO WT steady state is consulted.
SIGN = {
    ('CLN3','GAL-OE'):{0:-1}, ('CLN3','MC'):{0:-1}, ('CLN3','KO'):{0:1},
    ('BCK2','MC'):{0:-1}, ('BCK2','KO'):{0:1},
    ('CLN2','GAL-OE'):{0:-1}, ('CLN2','MC'):{0:-1}, ('CLN2','ts'):{0:1},
    ('CLN2','KO'):{0:1},
    ('SBF_MBF','KO'):{0:1}, ('SBF_MBF','CA'):{0:-1},
    ('CDC28','ts'):{0:1,1:-1}, ('CDC28','pLOF'):{0:1,1:-1},
    ('CLB5','GAL-OE'):{0:-1,1:1}, ('CLB5','MC'):{0:-1,1:1},
    ('CLB5','KO'):{0:1,1:-1}, ('CLB5','db'):{0:-1,1:1},
    ('ORI_BUD_SPN','cdc45Δ'):{1:-1}, ('ORI_BUD_SPN','spindle_defect'):{8:-1},
    ('ORI_BUD_SPN','cdc24Δ'):{}, ('ORI_BUD_SPN','perturb'):{},
    ('Mcm1','ts'):{2:-1}, ('Mcm1','KO'):{2:-1},
    ('CLB2','GAL-OE'):{2:1,4:1}, ('CLB2','MC'):{2:1,4:1},
    ('CLB2','KO'):{2:-1,4:1}, ('CLB2','ts'):{2:-1,4:1},
    ('CLB2','pLOF'):{2:-1,4:1}, ('CLB2','DAmP'):{2:-1,4:1},
    ('CLB2','db'):{4:1}, ('CLB2','CA'):{2:1,4:1},
    ('SIC1','GAL-OE'):{2:-1,4:-1}, ('SIC1','MC'):{2:-1,4:-1},
    ('SIC1','KO'):{2:1,4:1}, ('SIC1','db'):{2:-1,4:-1},
    ('SIC1','PM'):{2:-1,4:-1},
    ('CDC6','GAL-OE'):{2:-1}, ('CDC6','MC'):{2:-1},
    ('CDC6','KO'):{2:1}, ('CDC6','DAmP'):{2:1},
    ('SWI5','GAL-OE'):{2:-1}, ('SWI5','KO'):{2:1},
    ('CDC20','GAL-OE'):{3:1,4:-1}, ('CDC20','MC'):{3:1,4:-1},
    ('CDC20','KO'):{3:-1,4:1}, ('CDC20','ts'):{3:-1,4:1},
    ('CDC20','DAmP'):{3:-1,4:1}, ('CDC20','APC-A'):{3:-1,4:1},
    ('MAD2','CSL'):{3:1},
    ('BUB2','CKO'):{5:1},   # Tem1's GAP removed -> MEN permissive -> D6 up
    ('APC_Core','APC-A'):{3:-1,4:1}, ('APC_Core','ts'):{3:-1,4:1},
    ('CDH1','KO'):{4:1}, ('CDH1','CA'):{4:-1},
    ('CDC14','GAL-OE'):{4:-1,5:1}, ('CDC14','MC'):{4:-1,5:1},
    ('CDC14','ts'):{4:1,5:-1}, ('CDC14','DAmP'):{4:1,5:-1},
    ('ESP1','ts'):{9:-1}, ('ESP1','GAL-OE'):{9:1},
    ('PDS1','GAL-OE'):{9:1}, ('PDS1','KO'):{9:-1}, ('PDS1','db'):{9:1},
    ('SCC1','scc1-RR'):{9:-1},
    ('NET1','TAB6-1'):{5:1,6:1}, ('NET1','ts'):{5:1,6:1},
    ('NET1','DAmP'):{5:1,6:1}, ('NET1','GAL-OE'):{5:-1,6:-1},
    ('TEM1','GAL-OE'):{5:1}, ('TEM1','MC'):{5:1}, ('TEM1','KO'):{5:-1},
    ('TEM1','ts'):{5:-1}, ('TEM1','DAmP'):{5:-1},
    ('CDC15','GAL-OE'):{6:1}, ('CDC15','MC'):{6:1}, ('CDC15','KO'):{6:-1},
    ('CDC15','ts'):{6:-1}, ('CDC15','DAmP'):{6:-1},
    ('PPX','GAL-OE'):{6:-1}, ('PPX','KO'):{6:1}, ('PPX','ts'):{6:1},
    ('PPX','db'):{6:-1},
    ('CDC5_Proxy','KO'):{6:1}, ('CDC5_Proxy','db'):{6:-1},
    ('MDT','MDT'):{},      # mdt drawn from the same 3 carbon values -> neutral
}
NEUTRAL_LOG_TOL = 0.05   # |log10(f)| < 0.05 (or E) -> neutral direction

def direction_signs(edits):
    """edits: [(gene, mode, val)] where val = fold f (scaling) or target E
    (GAL-OE) or None (structural). Returns {dim: 0/1/2} with 0=neutral,
    1=up, 2=down, computed on primary coordinates only (§3.5-3)."""
    out = {}
    for g, m, val in edits:
        for d, s in SIGN.get((g, m), {}).items():
            if val is not None and val > 0 and \
               abs(math.log10(val)) < NEUTRAL_LOG_TOL:
                out[d] = 0
            else:
                out[d] = 1 if s > 0 else 2
    return out

def make_samplers():
    return {
        'f_res': LogBandSampler(-3.0, 0.0, zero_prob=0.20),
        'f_mc':  LogBandSampler(0.0, 2.0, zero_prob=0.0),
        'E':     LogBandSampler(-1.0, 1.0, zero_prob=0.0,
                                boundary=(BOUNDARY_LO, BOUNDARY_HI),
                                boundary_prob=0.30),
    }

def generate_wt_blind_dataset(n_synthetic=200, n_pilot=1000, seed=42,
                              num_workers=None, output_name=None,
                              output_dir=None, max_attempts_mult=30,
                              level_quotas=None, nan_cap_frac=0.05,
                              backlog_thresh=0.20, include_real=True,
                              mode_freq='flat'):
    global MODE_FREQ
    MODE_FREQ = mode_freq
    if level_quotas is None:
        level_quotas = LEVEL_QUOTAS
    np.random.seed(seed)
    t00 = time.time()
    if output_name is None:
        suffix = "_s5mode" if MODE_FREQ == 's5' else ""
        output_name = f"wb_mass_{n_synthetic}{suffix}_dataset_210min_500steps_dual_labels.npz"
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    dataset_path = os.path.join(output_dir, output_name)

    print("=" * 85)
    print(f"MASS v6.0 WB-MASS WT-Blind Sampler (Scheme v3.5) | target={n_synthetic} "
          f"| pilot={n_pilot} | seed={seed} | mode_freq={MODE_FREQ}"
          + ("  [ABLATION A: S5 mode freq = leak by design]" if MODE_FREQ == 's5' else ""))
    print("=" * 85)

    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])

    # ---- Phase 0: pilot (uniform skeleton, no WT run, no acceptance) ----
    print(f"\n[Phase 0] Pilot: {n_pilot} candidates -> D bin edges ...", flush=True)
    GAL_STATS['eff_guided'] = 0
    GAL_STATS['bio_draw'] = 0
    samplers = make_samplers()
    D_mat = np.full((n_pilot, 10), np.nan)
    pilot_nan = 0
    for i in range(n_pilot):
        level = int(np.random.choice([1, 2, 3, 4], p=level_quotas))
        cand = build_candidate(level, samplers)
        if cand is None:
            continue
        p_eff = defaults.copy(); p_eff.update(cand['ovr'])
        D = compute_D(p_eff, cand['mdt'])
        D_mat[i] = D
        # same criterion as the main-phase NaN bucket (L1165): D-NaN *or* GAL
        # inversion non-convergence. (v3.6 fix: previously counted D-NaN only,
        # which reported a misleading 0.0% while the bucket filled from the
        # gal_conv path.)
        if np.isnan(D).any() or not cand['gal_conv']:
            pilot_nan += 1
    ledger = Ledger(n_synthetic, nan_cap_frac=nan_cap_frac)
    ledger.fit_pilot(D_mat)
    # pilot draws consumed band-quota counters — reset so the MAIN phase
    # equalizes log bands on its own draws only
    samplers = make_samplers()
    print(f" -> pilot done in {time.time()-t00:.1f}s | fixed-point NaN rate: "
          f"{pilot_nan/max(1,n_pilot)*100:.1f}%", flush=True)

    # ---- Phases 1-3: acceptance loop ----
    print(f"\n[Phases 1-3] Sampling {n_synthetic} accepted candidates ...", flush=True)

    # exact Level quotas
    n_l = [int(round(q * n_synthetic)) for q in level_quotas[:3]]
    n_l.append(n_synthetic - sum(n_l))
    level_pool = [lv for lv, cnt in zip([1, 2, 3, 4], n_l) for _ in range(cnt)]
    np.random.shuffle(level_pool)

    ledger2 = Ledger(n_synthetic, nan_cap_frac=nan_cap_frac)
    ledger2.edges = ledger.edges
    ledger = ledger2

    accepted = []
    attempts = 0
    blocked_dup = 0
    dup_internal = 0
    seen_sigs = set()   # intra-dataset exact-duplicate ban (Scheme v3.5 §3.5-5)
    max_attempts = max_attempts_mult * n_synthetic
    accept_window = []
    backlog_mode = False
    stats = {'struct': {}, 'gal_nonconv': 0, 'nan_rejected': 0}
    # log ~25 lines regardless of scale (attempts run ~2-3x n_synthetic)
    log_every = max(500, n_synthetic // 10)

    while len(accepted) < n_synthetic and attempts < max_attempts:
        attempts += 1
        level = level_pool[len(accepted) % n_synthetic]

        # backlog directed generation (acceptance rate < 20%)
        boost, fold_bias_fn = None, None
        if backlog_mode:
            under = []
            fill = ledger.counts / max(1, ledger.cap)
            for d in range(10):
                for b in range(ledger.n_bins):
                    under.append((fill[d, b], d, b))
            under.sort()
            hot_dims = {d for _, d, _ in under[:12]}
            boost = {}
            for gname, dims in PRIMARY_COORDS.items():
                # PRIMARY_COORDS are 1-indexed, hot_dims 0-based
                if any((d - 1) in hot_dims for d in dims):
                    boost[gname] = 5.0
            low_D = any(ledger.counts[d, b] < ledger.cap / 2
                        for _, d, b in under[:6] if b < ledger.n_bins // 2)
            def fold_bias_fn(mode, _low=low_D):
                w = np.ones(N_LOG_BANDS)
                if mode == 'E':
                    return None
                if mode in RESIDUAL_MODES:
                    # low-D cell hungry -> stronger residual (smaller f)
                    w[:4] = 3.0 if _low else 1.0
                    w[4:] = 1.0 if _low else 3.0
                else:
                    w[4:] = 3.0 if _low else 1.0
                    w[:4] = 1.0 if _low else 3.0
                return w

        cand = build_candidate(level, samplers, boost=boost,
                               fold_bias_fn=fold_bias_fn)
        if cand is None:
            continue
        if not cand['gal_conv']:
            stats['gal_nonconv'] += 1

        # dedup vs D_real: the ONLY contact with the test set (pure exclusion)
        if is_duplicate_real_mutant(cand['ovr'], defaults):
            blocked_dup += 1
            continue

        # intra-dataset exact-duplicate ban: identical ovr vectors (esp. binary
        # structural modes at L1) carry zero new information -> reject (§3.5-5)
        sig = tuple(sorted((k, round(v, 10)) for k, v in cand['ovr'].items()))
        if sig in seen_sigs:
            dup_internal += 1
            continue
        cand['sig'] = sig

        p_eff = defaults.copy(); p_eff.update(cand['ovr'])
        D = compute_D(p_eff, cand['mdt'])

        # GAL inversion non-convergence -> NaN bucket (Scheme §3.2.2: iteration
        # <=3 rounds, non-convergent -> NaN bucket; audited by type below)
        nan_candidate = np.isnan(D).any() or (not cand['gal_conv'])
        if nan_candidate:
            # NaN bucket: cap 5%, per-type audit (§3.5-1)
            if ledger.nan_count < ledger.nan_cap:
                ledger.nan_count += 1
                for g, m in cand['genes_modes']:
                    key = f"{g}[{m}]"
                    ledger.nan_by_mode[key] = ledger.nan_by_mode.get(key, 0) + 1
                accepted.append(dict(cand, D=D, cells=[], dirs={}))
                seen_sigs.add(sig)
            else:
                stats['nan_rejected'] += 1
            accept_window.append(0)
        else:
            cells = ledger.cells(D)
            # pre-simulative direction signs (Tier 2 sign table, §3.5)
            dirs = direction_signs(cand['edits'])
            # accept if any cell under capacity AND direction quota satisfied.
            # Spec §3.5 TARGET is 45/45/10, but hard caps summing to 100%
            # deadlock: when a dim's natural supply is imbalanced (neutrals
            # are naturally rare — only ~8% of E draws), up and down both
            # hit their caps and the dim starves (verified: N=200 run
            # collapsed to 2% acceptance). Hard caps therefore carry slack
            # (50/50/20, sum 120%); the 45/45/10 target is pursued by the
            # backlog fold-bias mechanism instead. Measured distribution at
            # N=200: ~46/49/4 — close to the spec's intent.
            any_free = any(ledger.counts[d, b] < ledger.cap for d, b in cells)
            dir_ok = True
            for d, s in dirs.items():
                total_d = ledger.dir_counts[d].sum()
                if total_d > 20 and s in (1, 2) and \
                   ledger.dir_counts[d, s] >= 0.50 * total_d:
                    dir_ok = False
                if s == 0 and ledger.dir_counts[d, 0] >= 0.20 * max(1, total_d):
                    dir_ok = False
            if any_free and dir_ok and cells:
                for d, b in cells:
                    ledger.counts[d, b] += 1
                for d, s in dirs.items():
                    ledger.dir_counts[d, s] += 1
                for d, flag in enumerate(ledger.in_boundary(D)):
                    if flag:
                        ledger.boundary_count[d] += 1
                stats['struct'][cand['struct_class']] = \
                    stats['struct'].get(cand['struct_class'], 0) + 1
                accepted.append(dict(cand, D=D, cells=cells, dirs=dirs))
                seen_sigs.add(sig)
                accept_window.append(1)
            else:
                accept_window.append(0)

        if len(accept_window) > 200:
            accept_window.pop(0)
        if len(accept_window) == 200 and attempts % 50 == 0:
            rate = np.mean(accept_window)
            backlog_mode = rate < backlog_thresh
        if attempts % log_every == 0:
            print(f"  |-- attempts={attempts} accepted={len(accepted)} "
                  f"| dup_blocked={blocked_dup} | dup_internal={dup_internal} "
                  f"| nan={ledger.nan_count} "
                  f"| backlog={'ON' if backlog_mode else 'off'} "
                  f"| acc_rate={np.mean(accept_window[-100:]) if accept_window else 1:.2f}",
                  flush=True)

    print(f" -> accepted {len(accepted)} in {attempts} attempts "
          f"({blocked_dup} real-dup-blocked, {dup_internal} internal-dup-blocked, "
          f"{stats['nan_rejected']} NaN-capped, "
          f"{stats['gal_nonconv']} GAL non-converged)", flush=True)

    # ---- Phase 4: black-box ODE simulation + rollback ----
    # Real mutants (D_real) are simulated through the SAME black-box engine and
    # packaged into the final npz exactly like mass_dataset_generator v4.5 did
    # (real block first, then synthetic). This is a post-sampling EXPORT step:
    # no sampling decision ever reads it — the sampler stays WT-Blind.
    real_names, real_tasks = [], {}
    if include_real:
        for mid, (mname, movr) in REAL_MUTANTS.items():
            rn = f"{mid}_{mname}".replace(" ", "_")
            real_names.append(rn)
            real_tasks[rn] = movr
    n_real_q = len(real_tasks)
    print(f"\n[Phase 4] Simulating {len(accepted)} synthetic + {n_real_q} real "
          f"candidates ...", flush=True)
    if num_workers is None:
        num_workers = os.cpu_count() or 4   # no artificial cap; scale via --workers
    tasks = {}
    for rn in real_names:                   # real block first (engine convention)
        tasks[rn] = real_tasks[rn]
    for i, cand in enumerate(accepted):
        name = f"WB-MASS_{cand['desc']}_L{cand['level']}_{i+1}".replace(" ", "_")
        cand['name'] = name
        tasks[name] = cand['ovr']
    t_sim = time.time()
    # log ~50 progress lines regardless of dataset size
    sim_log_every = max(50, len(tasks) // 50)
    results, bio, pat = {}, {}, {}
    sim_fail = 0
    mass_disc = 0
    div_traj = 0
    real_fail = 0
    real_disc = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(simulate_mutant, n, tasks[n], 210, 500): n for n in tasks}
        for k, fut in enumerate(concurrent.futures.as_completed(futs)):
            name = futs[fut]
            is_real = name in real_tasks
            try:
                m_name, yu, bio_l, pat_l = fut.result(timeout=60.0)
            except Exception:
                if is_real: real_fail += 1
                else: sim_fail += 1
                continue
            if yu is not None:
                if is_real:
                    # D_real export block: stored as the engine returns it
                    # (engine already applies its own mass>10 discard).
                    results[name] = yu; bio[name] = bio_l; pat[name] = pat_l
                elif (not np.isfinite(yu).all()) or \
                        (np.nanmax(np.abs(yu)) > TRAJ_MAX):
                    # v3.7 numerical validity gate: combinatorial positive
                    # feedback can make the ODE itself diverge (|y| up to
                    # 1e268 observed). Such trajectories are integrator blow-up,
                    # not biology -> rolled back like mass divergence and
                    # replaced by a resample (keeps traj stats & training
                    # gradients sane at the dataset level).
                    div_traj += 1
                    _rollback(accepted, name, ledger, seen_sigs)
                else:
                    results[name] = yu; bio[name] = bio_l; pat[name] = pat_l
            elif bio_l == "Discarded_Mass":
                if is_real: real_disc += 1
                else:
                    mass_disc += 1
                    _rollback(accepted, name, ledger, seen_sigs)
            else:
                if is_real: real_fail += 1
                else:
                    sim_fail += 1
                    _rollback(accepted, name, ledger, seen_sigs)
            if (k + 1) % sim_log_every == 0:
                print(f"  |-- {k+1}/{len(tasks)} simulated "
                      f"({(k+1)/max(1e-9,time.time()-t_sim):.1f}/s)", flush=True)

    # ---- packaging (real block first, then synthetic — engine convention) ----
    keep = [c for c in accepted if c['name'] in results]
    ordered = sorted(keep, key=lambda c: c['name'])
    real_ok = [rn for rn in real_names if rn in results]

    names_all = real_ok + [c['name'] for c in ordered]
    ovr_by_name = {c['name']: c['ovr'] for c in ordered}
    ovr_by_name.update(real_tasks)
    tensors = np.array([results[n] for n in names_all]) \
        if names_all else np.zeros((0, 39, 500))
    params = np.array([[ovr_by_name[n].get(k, defaults[k]) for k in param_keys]
                       for n in names_all]) \
        if names_all else np.zeros((0, len(param_keys)))
    labels = np.array([bio[n] for n in names_all])
    patterns = np.array([pat[n] for n in names_all])
    is_real = np.array([n in real_tasks for n in names_all], dtype=bool)
    np.savez_compressed(dataset_path, data=tensors, params=params,
                        param_names=param_keys, mutant_names=names_all,
                        labels=labels, pattern_labels=patterns,
                        is_real=is_real)

    # ---- audit report ----
    print("\n" + "=" * 85)
    print(" WB-MASS v6.0 AUDIT REPORT")
    print("=" * 85)
    print(f"Dataset path                : {dataset_path}")
    print(f"Samples saved               : {len(names_all)} total = {len(real_ok)} real "
          f"+ {len(ordered)} synthetic / {len(accepted)} accepted "
          f"({sim_fail} numerical failures rolled back, {mass_disc} mass-discarded rolled back, "
          f"{div_traj} divergent-trajectory rolled back |TRAJ_MAX|={TRAJ_MAX:g})"
          f"\n                              real block: {real_fail} failed, "
          f"{real_disc} mass-discarded (not rolled back — D_real is an export block)")
    print(f"Real-mutant duplicates blocked (pure exclusion): {blocked_dup}")
    print(f"NaN bucket                  : {ledger.nan_count} "
          f"(cap {ledger.nan_cap}) | pilot NaN rate "
          f"{pilot_nan/max(1,n_pilot)*100:.1f}%")
    print(f"GAL inversion non-converged : {stats['gal_nonconv']}")
    _gt = GAL_STATS['eff_guided'] + GAL_STATS['bio_draw']
    if _gt:
        print(f"GAL alpha source            : effect-guided(in-range) "
              f"{GAL_STATS['eff_guided']}/{_gt} "
              f"({GAL_STATS['eff_guided']/_gt*100:.1f}%), "
              f"bio-range draw {GAL_STATS['bio_draw']}/{_gt}")
    if len(names_all):
        print(f"Sanity: max|params|         : {np.nanmax(np.abs(params)):.3g} "
              f"(GAL cap = {GAL_ALPHA_CAP_MULT:g}x native synthesis)")
        print(f"Sanity: max|traj| (synth)   : "
              f"{np.nanmax(np.abs(tensors[~is_real])):.3g} (gate {TRAJ_MAX:g})")
    print("\n[Structure class distribution]")
    for k, v in sorted(stats['struct'].items(), key=lambda x: -x[1]):
        print(f"  {k:<14}: {v}")
    print("\n[Level distribution]")
    lvls, cnts = np.unique([c['level'] for c in ordered], return_counts=True)
    for l, c in zip(lvls, cnts):
        print(f"  L{l}: {c}")
    print("\n[Ledger cell fill (per dim: min/median/max, boundary-band count)]")
    for d in range(10):
        row = ledger.counts[d]
        print(f"  {D_NAMES[d]:<10}: {row.min():4d}/{int(np.median(row)):4d}/{row.max():4d}"
              f"  | dir neutral/up/down = {ledger.dir_counts[d,0]}/"
              f"{ledger.dir_counts[d,1]}/{ledger.dir_counts[d,2]}"
              f"  | boundary={ledger.boundary_count[d]}")
    if ledger.nan_by_mode:
        top = sorted(ledger.nan_by_mode.items(), key=lambda x: -x[1])[:8]
        print("\n[NaN by mutation type (top 8, >10% triggers formula audit)]")
        for k, v in top:
            print(f"  {k:<18}: {v}")
    print("\n[Label distribution — synthetic block]")
    synth_labels = labels[~is_real] if len(labels) else labels
    for lb in sorted(set(synth_labels) if len(synth_labels) else []):
        print(f"  {lb:<34}: {int(np.sum(synth_labels == lb))}")
    if real_ok:
        real_labels = labels[is_real]
        print("[Label distribution — real block (D_real export)]")
        for lb in sorted(set(real_labels)):
            print(f"  {lb:<34}: {int(np.sum(real_labels == lb))}")
    print(f"\nTotal wall time: {time.time()-t00:.1f}s")
    print("=" * 85)
    return dataset_path

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
            if not cand.get('cells') and not cand.get('dirs'):
                ledger.nan_count = max(0, ledger.nan_count - 1)
            if seen_sigs is not None and cand.get('sig') is not None:
                seen_sigs.discard(cand['sig'])
            return

if __name__ == '__main__':
    # =========================================================================
    # 用户可调参数配置区 (随时在此处直接修改，或通过命令行参数指定；命令行优先)
    # =========================================================================
    NUM_SAMPLES = 100000                            # 生成的采样样本数量
    N_PILOT = 20000                                # Phase 0 pilot 候选数（决定账本分箱边界）
    NUM_WORKERS = 60                              # 并行调用的 CPU 核心数 (None = 自动取全部核心)
    SEED = 42                                     # 随机种子（同种子跨平台可复现）
    LEVEL_RATIOS = [0.40, 0.30, 0.20, 0.10]       # Level 1~4 目标采样占比 (机制化单调配额, §3.1.2)
    NAN_CAP_FRAC = 0.05                           # NaN 桶总量上限占目标样本数比例 (§3.5)
    BACKLOG_THRESH = 0.20                         # 接纳率低于此值时切换 backlog 定向生成 (§3.5)
    MAX_ATTEMPTS_MULT = 30                        # 最大尝试次数 = 该值 × 目标样本数
    INCLUDE_REAL = True                           # 是否把 126 株 real mutants 一并仿真并打包进最终 npz
                                                  # (与 v4.5 引擎一致；纯导出步骤，不影响采样决策)

    # 导出的数据集文件名 (默认自动携带样本数，也可直接在此自定义修改)
    DATASET_NAME = f"wb_mass_{NUM_SAMPLES}_dataset_210min_500steps_dual_labels.npz"

    # 导出的数据集目录路径 (自动适配系统环境，也可直接在此自定义写死绝对路径)
    if os.name == 'nt':
        OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
    else:
        OUTPUT_DIR = "/home/users/zsy/projects/budding_yeast_cell_cycle/mydata/budding_yeast"
    # =========================================================================

    import argparse
    parser = argparse.ArgumentParser(description="MASS v6.0 WB-MASS WT-Blind Sampler (Scheme v3.5)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES, help="Number of synthetic samples to generate")
    parser.add_argument("--pilot", type=int, default=N_PILOT, help="Phase-0 pilot size for ledger bin edges")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel CPU worker processes (0 = auto)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (reproducible across platforms)")
    parser.add_argument("--level_ratios", type=float, nargs=4, default=LEVEL_RATIOS, help="Proportions for Level 1-4 tiers (sum=1)")
    parser.add_argument("--nan_cap_frac", type=float, default=NAN_CAP_FRAC, help="NaN bucket cap as fraction of target samples")
    parser.add_argument("--backlog_thresh", type=float, default=BACKLOG_THRESH, help="Acceptance rate below which backlog directed generation turns on")
    parser.add_argument("--max_attempts_mult", type=int, default=MAX_ATTEMPTS_MULT, help="Max sampling attempts = this x target samples")
    parser.add_argument("--output_name", type=str, default=None, help="Export dataset .npz filename (default: auto with sample count)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Export directory path")
    parser.add_argument("--no_real", action="store_true", default=not INCLUDE_REAL, help="Skip simulating/packaging the 126 real mutants into the final npz")
    parser.add_argument("--mode_freq", choices=['flat', 's5'], default='s5',
                        help="Mode frequency source: 'flat' = uniform anti-leak (default) | "
                             "'s5' = Table S5-fitted weights (Ablation A: re-introduces the leak by design, "
                             "auto-suffixes output filename with _s5mode)")
    args = parser.parse_args()

    # 动态确定输出文件名：如果命令行显式指定了 --output_name 则使用指定的；否则自动使用基于 args.samples 的文件名
    if args.output_name is not None:
        final_output_name = args.output_name
    else:
        if args.samples == NUM_SAMPLES:
            final_output_name = DATASET_NAME
        else:
            final_output_name = f"wb_mass_{args.samples}_dataset_210min_500steps_dual_labels.npz"
        if args.mode_freq == 's5':
            final_output_name = final_output_name.replace('.npz', '_s5mode.npz')

    workers = args.workers if args.workers and args.workers > 0 else None
    print(f"Executing MASS v6.0 WB-MASS WT-Blind Sampler (Samples: {args.samples}, Pilot: {args.pilot}, "
          f"Workers: {workers or 'auto'}, Seed: {args.seed}, Output: {final_output_name})...", flush=True)
    generate_wt_blind_dataset(
        n_synthetic=args.samples,
        n_pilot=args.pilot,
        seed=args.seed,
        num_workers=workers,
        level_quotas=list(args.level_ratios),
        nan_cap_frac=args.nan_cap_frac,
        backlog_thresh=args.backlog_thresh,
        max_attempts_mult=args.max_attempts_mult,
        include_real=(not args.no_real),
        output_name=final_output_name,
        output_dir=args.output_dir,
        mode_freq=args.mode_freq
    )
    print("\n[SUCCESS] WB-MASS WT-Blind Sampler Execution Finished Cleanly!", flush=True)
