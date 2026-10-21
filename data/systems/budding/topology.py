"""Pruned MA module graph and derived distances/pair classes; separate from the 39-state graph."""
import numpy as np
from .catalog import GROUP_BY_NAME, GENE_NAMES

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
 ('PDS1', 'ESP1', '-'),
 ('APC_Core', 'CLB2', '-'),
 ('APC_Core', 'PDS1', '-'),
 ('CDC20', 'APC_Core', '+'),
 ('CDH1', 'APC_Core', '+'),
 ('CLB2', 'SWI5', '-'),
 ('CLB2', 'SIC1', '-'),
 ('CLB5', 'SIC1', '-'),
 ('CLN2', 'SIC1', '-')]

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

_NEG_PAIRS = {(a, b) for a, b, s in MODULE_EDGES if s == '-'}

_NEG_PAIRS |= {(b, a) for a, b, s in MODULE_EDGES if s == '-'}

_OUT = {n: set() for n in GENE_NAMES}

_OUT_NEG = {n: set() for n in GENE_NAMES}

_OUT_ALL = {n: set() for n in GENE_NAMES}

for a, b, s in MODULE_EDGES:
    if s == '+': _OUT[a].add(b)
    elif s == '-': _OUT_NEG[a].add(b)
    _OUT_ALL[a].add(b)

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

HUBS_DEG3 = [n for n in GENE_NAMES if len(ADJ[n]) >= 3]

STAR_LEAVES = {h: sorted(ADJ[h]) for h in HUBS_DEG3}

STAR_LEAVES['MDT'] = [n for n in GENE_NAMES if n != 'MDT']
