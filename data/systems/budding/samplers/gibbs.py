"""Shared exact Gibbs categorical graph sampler (formerly duplicated in two generators)."""
import numpy as np
import itertools
import time

class GibbsBoltzmannGraphSampler:
    """
    Exact Gibbs-Boltzmann sampler on the 27-gene budding yeast regulatory network.
    Precomputes normalized categorical distributions over all combinations of size k in {2, 3, 4, 5}.
    Sampling per candidate combination is exact and O(1).
    """

    def __init__(self, temperature=0.8, lambda_edge=1.0, lambda_fanin=1.5, seed=42, *, ma):
        self.ma = ma
        self.temperature = float(temperature)
        self.lambda_edge = float(lambda_edge)
        self.lambda_fanin = float(lambda_fanin)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.genes = list(self.ma.GENE_NAMES)
        self.N = len(self.genes)
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        adj = {g: set(self.ma.ADJ[g]) for g in self.genes}
        dist = np.full((self.N, self.N), 10.0, dtype=np.float32)
        np.fill_diagonal(dist, 0.0)
        for u in self.genes:
            for v in adj[u]:
                dist[self.gene_to_idx[u], self.gene_to_idx[v]] = 1.0
        for k in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        self.dist_matrix = dist
        self.edge_mat = np.zeros((self.N, self.N), dtype=np.float32)
        for u, v, _ in self.ma.MODULE_EDGES:
            if u in self.gene_to_idx and v in self.gene_to_idx:
                self.edge_mat[self.gene_to_idx[u], self.gene_to_idx[v]] = 1.0
                self.edge_mat[self.gene_to_idx[v], self.gene_to_idx[u]] = 1.0
        self.fanin_mat = np.zeros((self.N, self.N), dtype=np.float32)
        for a, b in self.ma.CONVERGENT_PAIRS:
            if a in self.gene_to_idx and b in self.gene_to_idx:
                self.fanin_mat[self.gene_to_idx[a], self.gene_to_idx[b]] = 1.0
                self.fanin_mat[self.gene_to_idx[b], self.gene_to_idx[a]] = 1.0
        self.W = 1.0 / np.maximum(dist, 0.5) + self.lambda_edge * self.edge_mat + self.lambda_fanin * self.fanin_mat
        np.fill_diagonal(self.W, 0.0)
        self.combinations = {}
        self.probabilities = {}
        self._precompute_gibbs_tables()

    def _precompute_gibbs_tables(self):
        t0 = time.time()
        for k in [2, 3, 4, 5]:
            combs = np.array(list(itertools.combinations(range(self.N), k)), dtype=np.int32)
            pairs = list(itertools.combinations(range(k), 2))
            num_pairs = len(pairs)
            affinities = np.zeros(len(combs), dtype=np.float32)
            for i, j in pairs:
                affinities += self.W[combs[:, i], combs[:, j]]
            affinities /= num_pairs
            scaled = (affinities - affinities.max()) / self.temperature
            unnorm = np.exp(scaled)
            probs = unnorm / unnorm.sum()
            self.combinations[k] = combs
            self.probabilities[k] = probs
        elapsed = time.time() - t0
        print(f'[GibbsSampler] Precomputed Gibbs-Boltzmann tables (L2..L5, {sum((len(c) for c in self.combinations.values())):,} combinations) in {elapsed * 1000:.1f} ms.', flush=True)

    def sample_genes(self, k, rng=None):
        """Draws exactly k gene names from the Gibbs-Boltzmann distribution in O(1)."""
        if rng is None:
            rng = self.rng
        if k == 1:
            idx = rng.choice(self.N)
            return [self.genes[idx]]
        if k not in self.combinations:
            return list(rng.choice(self.genes, size=k, replace=False))
        combs = self.combinations[k]
        probs = self.probabilities[k]
        comb_idx = rng.choice(len(combs), p=probs)
        chosen_indices = combs[comb_idx]
        return [self.genes[i] for i in chosen_indices]

def sample_gibbs_mutant(level, gibbs_sampler, rng=np.random, *, ma):
    """
    Constructs a valid cell-cycle mutant parameter override dictionary using Gibbs sampling.
    Fully maintains touch-set conflict resolution, physical scaling envelopes, and carbon rules.
    """
    p_wt = ma.get_default_params()
    p_mut = p_wt.copy()
    ovr = {}
    has_gal_oe = False
    mut_descriptions = []
    if level == 1:
        g0, m0 = ma.next_L1_cell()
        if ma.MODE_FREQ != 'flat':
            m0 = ma._sample_mode(g0)
        pairs = [(g0, m0)]
    else:
        genes = gibbs_sampler.sample_genes(level, rng=rng)
        pairs = [(g, ma._sample_mode(g)) for g in genes]
    expanded = []
    for g, m in pairs:
        expanded.append((g, m))
        gmode = ma.GROUP_BY_NAME.get(g, {}).get('modes', [])
        if g in ma.COMPOSITE_ELIGIBLE and rng.rand() < 0.25:
            synth = [x for x in ('GAL-OE', 'MC') if x in gmode]
            struc = [x for x in ('db', 'PM', 'CA') if x in gmode]
            if m in synth and struc:
                expanded.append((g, str(rng.choice(struc))))
            elif m in struc and synth:
                expanded.append((g, str(rng.choice(synth))))
    pairs = expanded
    pairs = ma._resolve_conflicts(pairs)
    if not pairs:
        return sample_gibbs_mutant(level, gibbs_sampler, rng, ma=ma)
    for g_name, m_chosen in pairs:
        mut_descriptions.append(f'{g_name}[{m_chosen}]')
        if m_chosen == 'GAL-OE':
            has_gal_oe = True
        alpha = ma.sample_scaling_factor(m_chosen, g_name) if m_chosen in ['GAL-OE', 'MC', 'ts', 'DAmP', 'pLOF', 'db'] else 1.0
        ma.apply_gene_mutation(g_name, m_chosen, alpha, p_wt, p_mut, ovr)
    if has_gal_oe:
        ovr['mdt'] = 150.0
    elif 'mdt' not in ovr:
        ovr['mdt'] = float(rng.choice([90.0, 150.0, 160.0], p=[0.75, 0.18, 0.07]))
    desc_str = '+'.join(mut_descriptions)
    return (ovr, desc_str)
