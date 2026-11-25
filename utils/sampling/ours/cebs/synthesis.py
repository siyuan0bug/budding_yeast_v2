"""Track C CEBS Query Synthesis: Compositional Epistasis Selection + 0-ODE Bifurcation Dose Refinement."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import numpy as np

from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate
from ...contracts import PoolInputs
from .metrics import compute_cebs_scores, epistasis_amplified_scores, _forward_ensemble_stats
from .selector import order_stratified_power_select

CEBS_GEN_METHODS = ('ours_epibif_synth',)
_CONTINUOUS_MODES = frozenset({'ts', 'MC', 'GAL-MC', 'GAL-OE', 'pLOF', 'db'})
_DOSE_QUANTILES = (0.06, 0.17, 0.28, 0.39, 0.50, 0.61, 0.72, 0.83, 0.94)


def _find_continuous_axes(candidate, backend):
    """Detect non-degenerate, non-overlapping continuous mutation axes without hardcoding gene names."""
    pairs = list(candidate.metadata.get('pairs', ()))
    if not pairs:
        return []
    pair_touches = [touch_set(str(g), str(m)) for g, m in pairs]
    axes = []
    for idx, (gene, mode) in enumerate(pairs):
        gene, mode = str(gene), str(mode)
        if mode not in _CONTINUOUS_MODES:
            continue
        # Ensure this axis's touched parameter footprint doesn't collide with another mutation in the same mutant
        my_touch = pair_touches[idx]
        if any(my_touch & pair_touches[j] for j in range(len(pairs)) if j != idx):
            continue
        try:
            a_lo = float(backend.sample_scaling_factor_sobol(mode, gene, _DOSE_QUANTILES[0]))
            a_hi = float(backend.sample_scaling_factor_sobol(mode, gene, _DOSE_QUANTILES[-1]))
        except Exception:
            continue
        if abs(a_hi - a_lo) <= 1e-9:
            continue
        axes.append((gene, mode, my_touch))
    return axes


def _build_dose_variant(base_cand, gene, mode, my_touch, alpha, backend, defaults, param_keys, rng):
    """Create a strictly legal, validated candidate differing only in (gene, mode)'s continuous strength alpha."""
    changes = {}
    backend.apply_gene_mutation(gene, mode, float(alpha), defaults, defaults.copy(), changes, rng=rng)
    overrides = {k: v for k, v in base_cand.overrides.items() if k not in my_touch}
    overrides.update(changes)

    meta = deepcopy(base_cand.metadata)
    meta['original_overrides'] = dict(overrides)
    draws = []
    updated_draw = False
    for rec in meta.get('proposal_strength_draws', []):
        if str(rec.get('gene')) == gene and str(rec.get('mode')) == mode and not updated_draw:
            draws.append(dict(gene=gene, mode=mode, alpha=float(alpha)))
            updated_draw = True
        else:
            draws.append(dict(rec))
    if not updated_draw:
        draws.append(dict(gene=gene, mode=mode, alpha=float(alpha)))
    meta['proposal_strength_draws'] = draws
    pairs = list(meta.get('pairs', ()))
    meta['mutation_parameters'] = [
        dict(gene=str(g), mode=str(m),
             overrides={k: overrides[k] for k in sorted(touch_set(str(g), str(m))) if k in overrides})
        for g, m in pairs
    ]
    meta.update(
        sampling_paradigm='cebs_bifurcation_dose_synthesis',
        cebs_refined_axis=[gene, mode],
        cebs_refined_alpha=float(alpha),
        rescued=False,
        real_collision=False,
    )
    meta.pop('rescue_attempts', None)

    cand = valid_candidate(replace(base_cand, overrides=overrides, metadata=meta), 'budding')
    if cand is None:
        return None
    vector = effective_inputs(cand.overrides, defaults)
    signature = hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
    _, y0, _, _ = backend.prepare_initial_conditions(cand.overrides)
    params = np.array([cand.overrides.get(k, defaults[k]) for k in param_keys], dtype=np.float64)
    return cand, signature, vector, params, np.asarray(y0, dtype=np.float64)


class CEBSSynthesis:
    """Track C CEBS Synthesizer:
    1. Unbiased fresh legal bank (24k out-of-pool combinations, 0 ODE)
    2. Order-stratified 0-ODE virtual epistasis-amplified selection (Modules 1 & 2)
    3. 0-ODE 9-point continuous bifurcation-boundary dose synthesis (Module 3)
    """

    def __init__(self, name, initial, trajectories, seed=42, *,
                 bank_size=24000, lam=1.0, dose_tau=2.0, beta=1.0):
        if name not in CEBS_GEN_METHODS:
            raise ValueError(f'Unknown CEBS synthesis method: {name}')
        self.name = name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.bank_size = int(bank_size)
        self.lam = float(lam)
        self.dose_tau = float(dose_tau)
        self.beta = float(beta)

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

    def _plan_batch(self, domain, plan_count):
        domain.verify()
        backend = domain.proposals.backend
        defaults = domain.proposals.defaults
        param_keys = backend.PARAM_KEYS

        target_bank = max(self.bank_size, plan_count * 4)
        # Step 1: 100% canonical unbiased legal out-of-pool bank (0 ODE)
        candidates, signatures, raw_view = domain.bank(target_bank)
        meta_tuple = tuple(dict(c.metadata, _effective_overrides=dict(c.overrides)) for c in candidates)
        bank_view = PoolInputs(raw_view.ids, raw_view.params, raw_view.initial, meta_tuple)

        # Step 2: Modules 1 & 2 — native order-stratified virtual epistasis-amplified selection (0 ODE)
        views = compute_cebs_scores(
            self.predictor,
            bank_view,
            compute_epistasis=True,
            backend=backend,
        )
        scores, epi_medians = epistasis_amplified_scores(
            views['u_power'],
            views['epi_mag'],
            views['orders'],
            lam=self.lam,
        )
        sel_seed = int(self.rng.integers(0, 2**31 - 1))
        chosen_idx, alloc = order_stratified_power_select(
            scores,
            views['orders'],
            plan_count,
            seed=sel_seed,
            beta=self.beta,
        )

        # Step 3: Module 3 — 0-ODE 9-point continuous bifurcation-boundary dose synthesis
        # Each selected combination keeps slot 0 = original candidate, plus up to 9 legal quantile doses.
        per_target_options = []
        eval_params = []
        eval_y0 = []

        for idx in chosen_idx:
            base_cand = candidates[idx]
            base_sig = signatures[idx]
            base_p = np.asarray(bank_view.params[idx], dtype=np.float64)
            base_y0 = np.asarray(bank_view.initial[idx], dtype=np.float64)
            base_u = float(views['u_power'][idx])

            # Slot 0 is always the original canonical candidate (already scored!)
            options = [(False, base_cand, base_sig, base_p, base_y0, None, base_u)]
            local_sigs = {base_sig}
            axes = _find_continuous_axes(base_cand, backend)
            if axes:
                axis_i = int(self.rng.integers(0, len(axes)))
                gene, mode, my_touch = axes[axis_i]
                for u in _DOSE_QUANTILES:
                    alpha = float(backend.sample_scaling_factor_sobol(mode, gene, float(u)))
                    built = _build_dose_variant(
                        base_cand, gene, mode, my_touch, alpha, backend, defaults, param_keys, self.rng
                    )
                    if built is None:
                        continue
                    cand_v, sig_v, vec_v, p_v, y0_v = built
                    if (sig_v in local_sigs or sig_v in domain.seen or sig_v in domain.forbidden
                            or domain.proposals.collides(vec_v)):
                        continue
                    local_sigs.add(sig_v)
                    flat_idx = len(eval_params)
                    eval_params.append(p_v)
                    eval_y0.append(y0_v)
                    options.append((True, cand_v, sig_v, p_v, y0_v, flat_idx, None))
            per_target_options.append(options)

        # Single batched GPU 2-member forward pass for all synthesized continuous doses (0 ODE!)
        if eval_params:
            dose_u, _ = _forward_ensemble_stats(
                self.predictor,
                np.asarray(eval_params, dtype=np.float64),
                np.asarray(eval_y0, dtype=np.float64),
                need_mean=False,
            )
        else:
            dose_u = np.empty(0, dtype=np.float64)

        planned = []
        planned_sigs = set()
        refined_count = 0
        has_cont_axis_count = 0
        u_ratios = []

        for options in per_target_options:
            if len(options) > 1:
                has_cont_axis_count += 1
            valid_opts = []
            valid_us = []
            base_u = max(float(options[0][6]), 1e-12)
            for is_new, cand, sig, p_v, y0_v, flat_idx, orig_u in options:
                if sig in planned_sigs or sig in domain.seen or sig in domain.forbidden:
                    continue
                u_val = float(dose_u[flat_idx]) if flat_idx is not None else float(orig_u)
                valid_opts.append((is_new, cand, sig, p_v, y0_v, max(u_val, 1e-12)))
                valid_us.append(max(u_val, 1e-12))
            if not valid_opts:
                continue

            if len(valid_opts) == 1:
                pick = 0
            else:
                log_w = self.dose_tau * np.log(np.asarray(valid_us, dtype=np.float64))
                log_w -= np.max(log_w)
                probs = np.exp(log_w)
                probs /= np.sum(probs)
                pick = int(self.rng.choice(len(valid_opts), p=probs))

            is_new, cand, sig, p_v, y0_v, chosen_u = valid_opts[pick]
            if is_new:
                new_index = domain.next_index
                domain.next_index += 1
                cand = replace(
                    cand,
                    name=f'CEBS-bifurcation-{new_index:09d}',
                    metadata=dict(
                        cand.metadata,
                        generation_proposal_index=new_index,
                        source_candidate_id=1_000_000_000 + new_index,
                    ),
                )
                domain.counts['cebs_bifurcation_refined'] += 1
                refined_count += 1
            u_ratios.append(chosen_u / base_u)
            planned_sigs.add(sig)
            planned.append((cand, sig, p_v, y0_v))

        self._planned = planned
        bank_orders = Counter(int(k) for k in views['orders'])
        planned_orders = Counter(int(c.level) for c, _, _, _ in planned)
        self._last_plan_diagnostics = dict(
            method=self.name,
            unbiased_bank_size=len(candidates),
            epistasis_lambda=self.lam,
            dose_tau=self.dose_tau,
            dose_grid_size=len(_DOSE_QUANTILES),
            bank_orders=dict(sorted(bank_orders.items())),
            order_budgets=dict(sorted(alloc.items())),
            planned_queries=len(planned),
            planned_orders=dict(sorted(planned_orders.items())),
            continuous_axis_targets=has_cont_axis_count,
            bifurcation_refined_count=refined_count,
            mean_dose_u_gain_ratio=float(np.mean(u_ratios)) if u_ratios else 1.0,
            epistasis_median_by_order={str(k): float(v) for k, v in sorted(epi_medians.items())},
            u_power_selected_mean=float(np.mean(views['u_power'][chosen_idx])),
            u_power_bank_mean=float(np.mean(views['u_power'])),
        )

    def propose_batch(self, domain, count):
        remaining = max(count, self.target - len(self.train_inputs.ids))
        if len(self._planned) < count:
            plan_count = max(count, int(math.ceil(remaining * 1.08)))
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
