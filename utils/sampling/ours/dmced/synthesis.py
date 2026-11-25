"""Pillar 3: Hypergraph-Factorized Legal Query Synthesis (Track C).

Uses ONLY the canonical BuddingSystem mutation catalog (1,159 legal module
combinations), native strength distributions, native mdt probabilities, native
initial condition preparation, and native real-collision rescue.
Zero parameter interpolation; zero biological rule changes.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import math
import numpy as np

from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate
from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.distributions.ranges import GAL_MODES
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.budding.samplers.rescue import parse_pairs, rescue_budding
from ...contracts import PoolInputs
from .metrics import compute_pool_and_train_views
from .selector import epistemic_voronoi_select, round_epistemic_beta

DMCED_GEN_METHODS = ('ours_dmced_synth',)


def _build_legal_candidate(proposals, modules, allele, rng, index, *, paradigm='dmced_factorized_legal'):
    """Construct one legal Budding mutation candidate using unmodified canonical rules."""
    pairs = parse_pairs(allele)
    overrides = {}
    draws = []
    for gene, mode in pairs:
        alpha = proposals.backend.sample_scaling_factor(mode, gene, rng=rng)
        proposals.backend.apply_gene_mutation(
            gene, mode, alpha, proposals.defaults, proposals.defaults.copy(), overrides, rng=rng
        )
        draws.append(dict(gene=gene, mode=mode, alpha=float(alpha)))
    overrides['mdt'] = (
        150.0 if any(m in GAL_MODES for _, m in pairs)
        else float(rng.choice([90.0, 150.0, 160.0], p=[0.8, 0.15, 0.05]))
    )
    level = len(modules.split(';'))
    description = '+'.join(f'{g}[{m}]' for g, m in pairs)
    metadata = dict(
        description=description,
        pairs=pairs,
        proposal_modules=modules,
        proposal_alleles=allele,
        proposal_strength_draws=draws,
        original_overrides=dict(overrides),
        rescued=False,
        module_names=modules.split(';'),
        module_count=level,
        requested_level=level,
        source_candidate_id=1_000_000_000 + int(index),
        generation_proposal_index=int(index),
        sampling_paradigm=paradigm,
    )
    name = f'Generated-mutation-{int(index):09d}'
    vector = effective_inputs(overrides, proposals.defaults)
    if proposals.collides(vector):
        metadata['real_collision'] = True
        for attempt in range(30):
            proposed, new_desc = rescue_budding(overrides, metadata['description'], proposals.backend, rng)
            checked = valid_candidate(
                Candidate(int(index) + 1, name, proposed, level,
                          metadata=dict(metadata, description=new_desc, pairs=parse_pairs(new_desc))),
                'budding',
            )
            if checked is None:
                continue
            trial = effective_inputs(proposed, proposals.defaults)
            if not proposals.collides(trial):
                overrides = proposed
                vector = trial
                metadata = dict(checked.metadata, rescued=True, rescue_attempts=attempt + 1)
                break
        else:
            return None, None, None, None

    candidate = Candidate(int(index) + 1, name, overrides, level, metadata=metadata)
    candidate = valid_candidate(candidate, 'budding')
    if candidate is None:
        return None, None, None, None
    candidate.metadata['mutation_parameters'] = [
        dict(gene=g, mode=m, overrides={k: overrides[k] for k in sorted(touch_set(g, m)) if k in overrides})
        for g, m in candidate.metadata['pairs']
    ]
    signature = hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
    _, y0, _, _ = proposals.backend.prepare_initial_conditions(candidate.overrides)
    params = np.array(
        [candidate.overrides.get(k, proposals.defaults[k]) for k in proposals.backend.spec.param_names],
        dtype=np.float64,
    )
    return candidate, signature, params, np.asarray(y0, dtype=np.float64)


class DMCEDSynthesis:
    """Hypergraph-factorized legal proposal + Epistemic-Risk-Weighted Voronoi selection."""

    def __init__(self, name, initial, trajectories, seed=42, *,
                 pilot_size=12288, enriched_size=24000, eta=0.60, tau=1.0, shrinkage=10.0,
                 sketch_dim=512, gamma=0.5):
        if name not in DMCED_GEN_METHODS:
            raise ValueError(f'Unknown DMCED synthesis method: {name}')
        self.name = name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.pilot_size = int(pilot_size)
        self.enriched_size = int(enriched_size)
        self.eta = float(eta)
        self.tau = float(tau)
        self.shrinkage = float(shrinkage)
        self.sketch_dim = int(sketch_dim)
        self.gamma = float(gamma)

        self.predictor = None
        self.train_inputs = initial
        self.target = len(initial.ids)
        self.round_count = 0
        self._planned = []
        self._last_plan_diagnostics = {}

    def state_dict(self):
        return dict(
            name=self.name,
            seed=self.seed,
            rng_state=self.rng.bit_generator.state,
            round_count=self.round_count,
            last_plan_diagnostics=dict(self._last_plan_diagnostics),
        )

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng_state']
        self.round_count = int(state.get('round_count', 0))
        self._last_plan_diagnostics = dict(state.get('last_plan_diagnostics', {}))
        self._planned = []

    def refresh(self, inputs, truth, predictor, target=None):
        self.train_inputs = inputs
        self.predictor = predictor
        if target is not None:
            self.target = int(target)
        self._planned = []
        self.round_count += 1

    def observe(self, inputs, truth):
        return None

    def _estimate_factorized_proposal(self, domain):
        """Step 1: Unlabeled epistemic field over 1,159 legal module combinations (0 ODE calls)."""
        proposals = domain.proposals
        groups = proposals.groups
        n_groups = len(groups)
        probe_rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))

        probe_params = []
        probe_initials = []
        probe_atoms = []
        probe_idx = 0
        attempts = 0
        max_attempts = self.pilot_size * 5

        while len(probe_params) < self.pilot_size and attempts < max_attempts:
            attempts += 1
            g_idx = int(probe_rng.integers(n_groups))
            modules, options = groups[g_idx]
            allele = options[int(probe_rng.integers(len(options)))]
            cand, sig, p_vec, y0 = _build_legal_candidate(
                proposals, modules, allele, probe_rng, probe_idx, paradigm='dmced_unlabeled_probe'
            )
            if cand is None or sig in domain.forbidden:
                continue
            probe_idx += 1
            probe_params.append(p_vec)
            probe_initials.append(y0)
            atoms = [f'{g}:{m}' for g, m in cand.metadata['pairs']]
            probe_atoms.append(atoms)

        if len(probe_params) < 256:
            raise RuntimeError(f'Insufficient unlabeled legal probes: {len(probe_params)}')

        probe_view = PoolInputs(
            np.arange(len(probe_params), dtype=np.int64),
            np.asarray(probe_params, dtype=np.float64),
            np.asarray(probe_initials, dtype=np.float64),
        )
        views = compute_pool_and_train_views(
            self.predictor, probe_view, None, sketch_dim=self.sketch_dim, eps=0.01, return_latent=False
        )
        log_risk = np.log(np.maximum(views['risk'], 1e-6))
        global_mean = float(np.mean(log_risk))

        atom_sums = defaultdict(float)
        atom_counts = Counter()
        mod_sums = defaultdict(float)
        mod_counts = Counter()
        for atoms, lr in zip(probe_atoms, log_risk):
            for a in atoms:
                atom_sums[a] += float(lr)
                atom_counts[a] += 1
                mod = a.split(':', 1)[0]
                mod_sums[mod] += float(lr)
                mod_counts[mod] += 1

        # Empirical-Bayes shrunk atom & module epistemic risks
        atom_risk = {
            a: (atom_sums[a] + self.shrinkage * global_mean) / (atom_counts[a] + self.shrinkage)
            for a in atom_counts
        }
        mod_risk = {
            m: (mod_sums[m] + self.shrinkage * global_mean) / (mod_counts[m] + self.shrinkage)
            for m in mod_counts
        }

        # Score all 1,159 canonical legal module combinations in O(1159)
        group_scores = np.empty(n_groups, dtype=np.float64)
        for g_idx, (modules, _) in enumerate(groups):
            mods = modules.split(';')
            group_scores[g_idx] = float(np.mean([mod_risk.get(m, global_mean) for m in mods]))

        g_std = max(float(np.std(group_scores)), 1e-6)
        g_z = np.clip((group_scores - float(np.mean(group_scores))) / g_std, -2.0, 2.0)
        gibbs = np.exp(self.tau * g_z)
        gibbs /= gibbs.sum()
        # Exploration-protected mixture over all 1,159 canonical module combinations
        group_probs = (1.0 - self.eta) * (1.0 / n_groups) + self.eta * gibbs
        group_probs /= group_probs.sum()

        # Allele-level Empirical-Bayes tilting computed lazily & cached per sampled group
        a_std = max(float(np.std(list(atom_risk.values()))) if len(atom_risk) > 1 else g_std, 1e-6)
        allele_context = dict(atom_risk=atom_risk, global_mean=global_mean, scale=a_std)

        stats = dict(
            pilot_probed=len(probe_params),
            unique_atoms_observed=len(atom_risk),
            atom_log_risk_min=float(min(atom_risk.values())) if atom_risk else 0.0,
            atom_log_risk_max=float(max(atom_risk.values())) if atom_risk else 0.0,
            group_prob_ratio_max_min=float(group_probs.max() / max(group_probs.min(), 1e-12)),
        )
        return group_probs, allele_context, stats

    def _allele_cdf(self, options, allele_context):
        if len(options) <= 1:
            return None
        atom_risk = allele_context['atom_risk']
        global_mean = allele_context['global_mean']
        scale = allele_context['scale']
        scores = np.empty(len(options), dtype=np.float64)
        for idx, allele in enumerate(options):
            pairs = parse_pairs(allele)
            scores[idx] = float(np.mean([atom_risk.get(f'{g}:{m}', global_mean) for g, m in pairs])) if pairs else global_mean
        z = np.clip((scores - float(np.mean(scores))) / scale, -2.0, 2.0)
        ag = np.exp(self.tau * z)
        ag /= ag.sum()
        ap = (1.0 - self.eta) * (1.0 / len(options)) + self.eta * ag
        ap /= ap.sum()
        cdf = np.cumsum(ap)
        cdf[-1] = 1.0
        return cdf

    def _plan_batch(self, domain, plan_count):
        """Synthesize enriched legal bank and select ordered batch via Epistemic-Weighted Voronoi."""
        domain.verify()
        proposals = domain.proposals
        groups = proposals.groups
        group_probs, allele_context, prop_stats = self._estimate_factorized_proposal(domain)

        # Step 2: Sample fresh legal candidates from exploration-protected factorized proposal
        target_bank = max(self.enriched_size, plan_count * 4)
        group_cdf = np.cumsum(group_probs)
        group_cdf[-1] = 1.0
        allele_cdf_cache = {}

        candidates = []
        signatures = []
        params_list = []
        init_list = []
        local_seen = set()
        attempts = 0
        max_attempts = target_bank * 8

        while len(candidates) < target_bank and attempts < max_attempts:
            attempts += 1
            index = domain.next_index
            domain.next_index += 1
            domain.counts['proposals'] += 1
            g_idx = min(int(np.searchsorted(group_cdf, self.rng.random())), len(groups) - 1)
            modules, options = groups[g_idx]
            if len(options) <= 1:
                a_idx = 0
            else:
                if g_idx not in allele_cdf_cache:
                    # For very large allele tables (>256), sample uniformly from a random
                    # 64-allele slice tilted by atom risk so planning stays O(1) per draw.
                    if len(options) <= 256:
                        allele_cdf_cache[g_idx] = ('exact', self._allele_cdf(options, allele_context))
                    else:
                        allele_cdf_cache[g_idx] = ('sampled', None)
                kind, cdf = allele_cdf_cache[g_idx]
                if kind == 'exact':
                    a_idx = min(int(np.searchsorted(cdf, self.rng.random())), len(options) - 1)
                else:
                    sub = self.rng.integers(len(options), size=16)
                    sub_opts = [options[int(k)] for k in sub]
                    sub_cdf = self._allele_cdf(sub_opts, allele_context)
                    a_idx = int(sub[min(int(np.searchsorted(sub_cdf, self.rng.random())), len(sub) - 1)])
            allele = options[a_idx]
            cand, sig, p_vec, y0 = _build_legal_candidate(
                proposals, modules, allele, self.rng, index, paradigm='dmced_factorized_legal'
            )
            if cand is None:
                domain.counts['real_collision_exhausted'] += 1
                continue
            if sig in domain.forbidden:
                domain.counts['holdout_collision'] += 1
                continue
            if sig in domain.seen or sig in local_seen:
                domain.counts['duplicate_proposal'] += 1
                continue
            local_seen.add(sig)
            domain.counts['rescued'] += int(cand.metadata.get('rescued', False))
            candidates.append(cand)
            signatures.append(sig)
            params_list.append(p_vec)
            init_list.append(y0)

        if len(candidates) < plan_count:
            raise RuntimeError(f'DMCED synthesis generated insufficient legal bank: {len(candidates)}/{plan_count}')

        bank_view = PoolInputs(
            np.arange(len(candidates), dtype=np.int64),
            np.asarray(params_list, dtype=np.float64),
            np.asarray(init_list, dtype=np.float64),
        )

        # Step 3: Exact Pillar 1 + Pillar 2 Epistemic-Risk-Weighted Voronoi selection
        views = compute_pool_and_train_views(
            self.predictor,
            bank_view,
            self.train_inputs,
            sketch_dim=self.sketch_dim,
            eps=0.01,
            gamma=self.gamma,
            return_latent=True,
        )
        round_budget = max(1, self.target - len(self.train_inputs.ids))
        beta, stage = round_epistemic_beta(len(self.train_inputs.ids), round_budget)
        chosen_idx = epistemic_voronoi_select(
            views['pool_embed'],
            views['train_embed'],
            views['risk'],
            plan_count,
            beta=beta,
            device=self.predictor.device,
        )

        self._planned = [
            (candidates[i], signatures[i], params_list[i], init_list[i])
            for i in chosen_idx
        ]
        level_counts = Counter(int(candidates[i].level) for i in chosen_idx)
        self._last_plan_diagnostics = dict(
            method=self.name,
            round_stage=stage,
            epistemic_beta=beta,
            eta=self.eta,
            tau=self.tau,
            enriched_bank_size=len(candidates),
            planned_queries=len(self._planned),
            planned_levels=dict(sorted(level_counts.items())),
            risk_selected_mean=float(np.mean(views['risk'][chosen_idx])),
            risk_bank_mean=float(np.mean(views['risk'])),
            **prop_stats,
        )

    def propose_batch(self, domain, count):
        """Serve the next `count` pre-planned legal candidates to generation/runner.py."""
        remaining = max(count, self.target - len(self.train_inputs.ids))
        if len(self._planned) < count:
            # Plan entire remaining round + 12% reserve for rare ODE failures in one shot
            plan_count = max(count, int(math.ceil(remaining * 1.12)))
            self._plan_batch(domain, plan_count)

        # Filter out any signatures that may already have been consumed
        chosen_cands, chosen_sigs, chosen_params, chosen_inits = [], [], [], []
        while self._planned and len(chosen_cands) < count:
            cand, sig, p_vec, y0 = self._planned.pop(0)
            if sig in domain.seen or sig in domain.forbidden:
                continue
            chosen_cands.append(cand)
            chosen_sigs.append(sig)
            chosen_params.append(p_vec)
            chosen_inits.append(y0)

        if len(chosen_cands) < count:
            self._plan_batch(domain, max(count * 2, 128))
            while self._planned and len(chosen_cands) < count:
                cand, sig, p_vec, y0 = self._planned.pop(0)
                if sig in domain.seen or sig in domain.forbidden:
                    continue
                chosen_cands.append(cand)
                chosen_sigs.append(sig)
                chosen_params.append(p_vec)
                chosen_inits.append(y0)

        bank = PoolInputs(
            np.arange(len(chosen_cands), dtype=np.int64),
            np.asarray(chosen_params, dtype=np.float64),
            np.asarray(chosen_inits, dtype=np.float64),
        )
        return chosen_cands, chosen_sigs, bank

    def select(self, bank, count):
        n = min(int(count), len(bank.ids))
        return np.arange(n, dtype=np.int64), dict(self._last_plan_diagnostics, batch_served=n)
