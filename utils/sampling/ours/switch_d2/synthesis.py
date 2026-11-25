"""Pillar 3: Level-Marginal-Preserving Legal Query Synthesis + Stochastic Weighted D^2 Selection.

Strictly preserves canonical L1/L2/L3 marginal shares over the 1,159 legal Budding
module combinations while gently tilting WITHIN each level using Empirical-Bayes
epistemic relative variance, then selecting queries via Stochastic Weighted D^2.
"""
from collections import Counter, defaultdict
import math
import numpy as np

from budding_yeast_v2.data.systems.budding.samplers.rescue import parse_pairs
from ...contracts import PoolInputs
from ..dmced.synthesis import _build_legal_candidate
from .metrics import compute_epistemic_relative_views
from .selector import stochastic_d2_select

SWITCH_D2_GEN_METHODS = ('ours_switch_d2_synth',)


class SwitchD2Synthesis:
    """Level-marginal-preserving legal query synthesizer paired with Stochastic Weighted D^2."""

    def __init__(self, name, initial, trajectories, seed=42, *,
                 pilot_size=12288, enriched_size=24000, eta=0.50, tau=0.75, shrinkage=10.0,
                 sketch_dim=512, gamma=0.5):
        if name not in SWITCH_D2_GEN_METHODS:
            raise ValueError(f'Unknown Switch-D2 synthesis method: {name}')
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

    def _estimate_level_preserving_proposal(self, domain):
        """Estimate within-level Empirical-Bayes epistemic tilting with exact native level marginals."""
        proposals = domain.proposals
        groups = proposals.groups
        n_groups = len(groups)
        probe_rng = np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))

        probe_params, probe_initials, probe_atoms = [], [], []
        probe_idx = 0
        attempts = 0
        max_attempts = self.pilot_size * 5

        while len(probe_params) < self.pilot_size and attempts < max_attempts:
            attempts += 1
            g_idx = int(probe_rng.integers(n_groups))
            modules, options = groups[g_idx]
            allele = options[int(probe_rng.integers(len(options)))]
            cand, sig, p_vec, y0 = _build_legal_candidate(
                proposals, modules, allele, probe_rng, probe_idx, paradigm='switch_d2_unlabeled_probe'
            )
            if cand is None or sig in domain.forbidden:
                continue
            probe_idx += 1
            probe_params.append(p_vec)
            probe_initials.append(y0)
            probe_atoms.append([f'{g}:{m}' for g, m in cand.metadata['pairs']])

        if len(probe_params) < 256:
            raise RuntimeError(f'Insufficient unlabeled legal probes: {len(probe_params)}')

        probe_view = PoolInputs(
            np.arange(len(probe_params), dtype=np.int64),
            np.asarray(probe_params, dtype=np.float64),
            np.asarray(probe_initials, dtype=np.float64),
        )
        views = compute_epistemic_relative_views(
            self.predictor, probe_view, None, sketch_dim=self.sketch_dim, eps=0.01, return_latent=False
        )
        log_e = np.log(np.maximum(views['energy'], 1e-12))
        global_mean = float(np.mean(log_e))

        atom_sums, mod_sums = defaultdict(float), defaultdict(float)
        atom_counts, mod_counts = Counter(), Counter()
        for atoms, le in zip(probe_atoms, log_e):
            for a in atoms:
                atom_sums[a] += float(le)
                atom_counts[a] += 1
                mod = a.split(':', 1)[0]
                mod_sums[mod] += float(le)
                mod_counts[mod] += 1

        atom_risk = {
            a: (atom_sums[a] + self.shrinkage * global_mean) / (atom_counts[a] + self.shrinkage)
            for a in atom_counts
        }
        mod_risk = {
            m: (mod_sums[m] + self.shrinkage * global_mean) / (mod_counts[m] + self.shrinkage)
            for m in mod_counts
        }

        # Partition 1,159 legal module combinations strictly by level so marginal L1/L2/L3 shares never drift!
        by_level = defaultdict(list)
        for g_idx, (modules, _) in enumerate(groups):
            mods = modules.split(';')
            by_level[len(mods)].append((g_idx, float(np.mean([mod_risk.get(m, global_mean) for m in mods]))))

        group_probs = np.zeros(n_groups, dtype=np.float64)
        for lv, items in by_level.items():
            # Exact native marginal share of this level (= count_in_level / 1159)
            level_mass = len(items) / float(n_groups)
            g_ids = [x[0] for x in items]
            g_scores = np.array([x[1] for x in items], dtype=np.float64)
            if len(items) > 1:
                sd = max(float(np.std(g_scores)), 1e-6)
                gz = np.clip((g_scores - float(np.mean(g_scores))) / sd, -2.0, 2.0)
                gibbs = np.exp(self.tau * gz)
                gibbs /= gibbs.sum()
                local_p = (1.0 - self.eta) * (1.0 / len(items)) + self.eta * gibbs
                local_p /= local_p.sum()
            else:
                local_p = np.array([1.0], dtype=np.float64)
            for gid, lp in zip(g_ids, local_p):
                group_probs[gid] = level_mass * lp

        group_probs /= group_probs.sum()
        a_std = max(float(np.std(list(atom_risk.values()))) if len(atom_risk) > 1 else 1.0, 1e-6)
        allele_context = dict(atom_risk=atom_risk, global_mean=global_mean, scale=a_std)

        level_shares = {
            int(lv): float(sum(group_probs[x[0]] for x in items))
            for lv, items in sorted(by_level.items())
        }
        stats = dict(
            pilot_probed=len(probe_params),
            unique_atoms_observed=len(atom_risk),
            proposal_level_shares=level_shares,
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
        domain.verify()
        proposals = domain.proposals
        groups = proposals.groups
        group_probs, allele_context, prop_stats = self._estimate_level_preserving_proposal(domain)

        target_bank = max(self.enriched_size, plan_count * 4)
        group_cdf = np.cumsum(group_probs)
        group_cdf[-1] = 1.0
        allele_cdf_cache = {}

        candidates, signatures, params_list, init_list = [], [], [], []
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
                proposals, modules, allele, self.rng, index, paradigm='switch_d2_level_preserving'
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
            raise RuntimeError(f'Switch-D2 synthesis generated insufficient legal bank: {len(candidates)}/{plan_count}')

        bank_view = PoolInputs(
            np.arange(len(candidates), dtype=np.int64),
            np.asarray(params_list, dtype=np.float64),
            np.asarray(init_list, dtype=np.float64),
        )
        views = compute_epistemic_relative_views(
            self.predictor,
            bank_view,
            self.train_inputs,
            sketch_dim=self.sketch_dim,
            eps=0.01,
            gamma=self.gamma,
            return_latent=True,
        )
        sel_seed = int(self.rng.integers(0, 2**31 - 1))
        chosen_idx = stochastic_d2_select(
            views['pool_embed'],
            views['train_embed'],
            views['energy'],
            plan_count,
            seed=sel_seed,
            device=self.predictor.device,
        )
        self._planned = [
            (candidates[i], signatures[i], params_list[i], init_list[i])
            for i in chosen_idx
        ]
        level_counts = Counter(int(candidates[i].level) for i in chosen_idx)
        bank_levels = Counter(int(c.level) for c in candidates)
        self._last_plan_diagnostics = dict(
            method=self.name,
            eta=self.eta,
            tau=self.tau,
            enriched_bank_size=len(candidates),
            bank_levels=dict(sorted(bank_levels.items())),
            planned_queries=len(self._planned),
            planned_levels=dict(sorted(level_counts.items())),
            energy_selected_mean=float(np.mean(views['energy'][chosen_idx])),
            energy_bank_mean=float(np.mean(views['energy'])),
            **prop_stats,
        )

    def propose_batch(self, domain, count):
        remaining = max(count, self.target - len(self.train_inputs.ids))
        if len(self._planned) < count:
            plan_count = max(count, int(math.ceil(remaining * 1.12)))
            self._plan_batch(domain, plan_count)

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
