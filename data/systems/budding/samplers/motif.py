"""Original connected/distal motif tables and mode selection, with canonical shared topology."""
import numpy as np
import itertools
import networkx as nx
from ..topology import MODULE_EDGES

PARALLEL_DISTAL_PAIRS = [
    # 1. Spindle assembly checkpoint dual-branch
    ('MAD2', 'BUB2'),
    # 2. Mitotic exit & G1/S cross-phase couplings
    ('NET1', 'CLB2'),
    ('NET1', 'CLB5'),
    ('NET1', 'CDC20'),
    ('NET1', 'CDC14'),
    # 3. G1/S Starter-Brake direct couplings
    ('CLN3', 'SIC1'),
    ('BCK2', 'SIC1'),
    ('CLN3', 'CDH1'),
    ('BCK2', 'CDH1')
]

MASTER_SWITCH_CLUSTERS = [
    # G1/S Start Switch cluster (Cln3/Bck2/Cln2/Sic1/Cdh1)
    ['CLN2', 'CLN3', 'BCK2', 'SIC1', 'CDH1'],
    # Anaphase/Exit Switch cluster (Clb2/Cdc20/Pds1/Cdc6/Sic1/APC_Core)
    ['CLB2', 'CDC20', 'PDS1', 'CDC6', 'SIC1', 'APC_Core']
]
class MotifTopologySampler:
    """Extracts and manages topological connected motifs for zero-leakage generation."""

    def __init__(self, seed=42, *, ma):
        self.ma = ma
        self.rng = np.random.RandomState(seed)
        self.G = nx.Graph()
        for u, v, s in MODULE_EDGES:
            self.G.add_edge(u, v)
        self.active_nodes = [n for n in self.G.nodes() if n in self.ma.GENE_NAMES]
        self.l2_motifs = []
        for u, v in itertools.combinations(self.active_nodes, 2):
            if self.G.has_edge(u, v):
                self.l2_motifs.append((u, v))
        for u, v in PARALLEL_DISTAL_PAIRS:
            if u in self.ma.GENE_NAMES and v in self.ma.GENE_NAMES:
                self.l2_motifs.append((u, v))
        self.l2_motifs = list(set(self.l2_motifs))
        self.l3_motifs = []
        for triplet in itertools.combinations(self.active_nodes, 3):
            subG = self.G.subgraph(triplet)
            if nx.is_connected(subG):
                self.l3_motifs.append(tuple(sorted(triplet)))
        self.l3_motifs = list(set(self.l3_motifs))
        self.l4_motifs = []
        for quad in itertools.combinations(self.active_nodes, 4):
            subG = self.G.subgraph(quad)
            if nx.is_connected(subG):
                self.l4_motifs.append(tuple(sorted(quad)))
        self.l4_motifs = list(set(self.l4_motifs))
        self.l5_motifs = []
        for cluster in MASTER_SWITCH_CLUSTERS:
            for quint in itertools.combinations(cluster, 5):
                self.l5_motifs.append(tuple(sorted(quint)))
        self.l5_motifs = list(set(self.l5_motifs))
        self.l1_grid = [(g, m) for g in self.ma.GENE_NAMES for m in self.ma.GROUP_BY_NAME[g]['modes']]
        self.rng.shuffle(self.l1_grid)
        self.l1_idx = 0
        print('=' * 80)
        print(' [MotifTopologySampler] Initialized Topological Motifs:')
        print(f'  * Level 1 Cells (Gene x Mode)         : {len(self.l1_grid)}')
        print(f'  * Level 2 Pairs (Direct + Distal)     : {len(self.l2_motifs)}')
        print(f'  * Level 3 Triplets (Connected SubG)   : {len(self.l3_motifs)}')
        print(f'  * Level 4 Quadruplets (Connected SubG): {len(self.l4_motifs)}')
        print(f'  * Level 5 Master Switch Quints        : {len(self.l5_motifs)}')
        print(f'  * Total Discrete Motifs Available     : {len(self.l2_motifs) + len(self.l3_motifs) + len(self.l4_motifs) + len(self.l5_motifs)}')
        print('=' * 80)

    def sample_motif(self, level):
        """Samples a gene combination based on hierarchical level."""
        if level == 1:
            cell = self.l1_grid[self.l1_idx % len(self.l1_grid)]
            self.l1_idx += 1
            return ([cell[0]], [cell[1]])
        elif level == 2:
            idx = self.rng.randint(len(self.l2_motifs))
            genes = list(self.l2_motifs[idx])
        elif level == 3:
            idx = self.rng.randint(len(self.l3_motifs))
            genes = list(self.l3_motifs[idx])
        elif level == 4:
            idx = self.rng.randint(len(self.l4_motifs))
            genes = list(self.l4_motifs[idx])
        else:
            idx = self.rng.randint(len(self.l5_motifs))
            genes = list(self.l5_motifs[idx])
        modes = []
        for g in genes:
            avail = list(self.ma.GROUP_BY_NAME[g]['modes'].keys())
            ko_avail = [m for m in avail if m == 'KO']
            dec_avail = [m for m in avail if m in ('ts', 'db', 'pLOF', 'DAmP', 'CSL', 'CKO')]
            gof_avail = [m for m in avail if m in ('MC', 'GAL-OE', 'CA')]
            r = self.rng.rand()
            if r < 0.45 and ko_avail:
                mode = 'KO'
            elif r < 0.7 and dec_avail:
                mode = str(self.rng.choice(dec_avail))
            elif gof_avail:
                mode = str(self.rng.choice(gof_avail))
            else:
                mode = str(self.rng.choice(avail))
            modes.append(mode)
        return (genes, modes)

def build_perturbed_sample(genes, modes, rng, *, ma):
    """
    Constructs a complete parameter override dict for the chosen (genes, modes):
    - Strict Hard Zero (0.0) for KO mutations.
    - Continuous dynamic range scaling for ts, db, MC, GAL-OE.
    - Proper carbon source (mdt) decoupling.
    """
    p_wt = ma.get_default_params()
    p_wt.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    ovr = {}
    has_gal = False
    for g, m in zip(genes, modes):
        if m in ('GAL-OE','GAL-MC'):
            has_gal = True
        if m == 'KO':
            ma.apply_gene_mutation(g, 'KO', 0.0, p_wt, {}, ovr)
        else:
            alpha = ma.sample_scaling_factor(m, g)
            ma.apply_gene_mutation(g, m, alpha, p_wt, {}, ovr)
    if has_gal:
        ovr['mdt'] = 150.0
    else:
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    return ovr
