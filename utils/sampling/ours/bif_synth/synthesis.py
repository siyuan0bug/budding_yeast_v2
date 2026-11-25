"""Topology-Preserving 0-ODE Continuous Bifurcation-Boundary Query Synthesis (R26)."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import numpy as np
import torch

from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate
from ...baseline.uncertainty import _power_sample_indices
from ...contracts import PoolInputs
from ..cebs.metrics import _forward_ensemble_stats

BIF_SYNTH_GEN_METHODS = (
    'ours_power_synth',
    'ours_bif_power_synth',
    'ours_grad_bif_synth',
)

_CONTINUOUS_MODES = frozenset({'ts', 'MC', 'GAL-MC', 'GAL-OE', 'pLOF', 'db'})
_SINGLE_AXIS_QUANTILES = (0.06, 0.17, 0.28, 0.39, 0.50, 0.61, 0.72, 0.83, 0.94)
_COARSE_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)


def find_continuous_axes(candidate, backend):
    """Detect all non-degenerate, non-overlapping continuous mutation axes (zero gene hardcoding)."""
    pairs = list(candidate.metadata.get('pairs', ()))
    if not pairs:
        return []
    pair_touches = [touch_set(str(g), str(m)) for g, m in pairs]
    axes = []
    for idx, (gene, mode) in enumerate(pairs):
        gene, mode = str(gene), str(mode)
        if mode not in _CONTINUOUS_MODES:
            continue
        my_touch = pair_touches[idx]
        if any(my_touch & pair_touches[j] for j in range(len(pairs)) if j != idx):
            continue
        try:
            a_lo = float(backend.sample_scaling_factor_sobol(mode, gene, 0.05))
            a_hi = float(backend.sample_scaling_factor_sobol(mode, gene, 0.95))
        except Exception:
            continue
        if abs(a_hi - a_lo) <= 1e-9:
            continue
        axes.append((gene, mode, my_touch))
    return axes


def build_legal_dose_variant(base_cand, gene, mode, my_touch, alpha, backend, defaults, param_keys, rng, *, paradigm):
    """Build a strictly legal, validated candidate modifying only (gene, mode)'s continuous strength alpha."""
    changes = {}
    backend.apply_gene_mutation(gene, mode, float(alpha), defaults, defaults.copy(), changes, rng=rng)
    overrides = {k: v for k, v in base_cand.overrides.items() if k not in my_touch}
    overrides.update(changes)

    meta = deepcopy(base_cand.metadata)
    meta['original_overrides'] = dict(overrides)
    draws = []
    updated = False
    for rec in meta.get('proposal_strength_draws', []):
        if str(rec.get('gene')) == gene and str(rec.get('mode')) == mode and not updated:
            draws.append(dict(gene=gene, mode=mode, alpha=float(alpha)))
            updated = True
        else:
            draws.append(dict(rec))
    if not updated:
        draws.append(dict(gene=gene, mode=mode, alpha=float(alpha)))
    meta['proposal_strength_draws'] = draws
    pairs = list(meta.get('pairs', ()))
    meta['mutation_parameters'] = [
        dict(gene=str(g), mode=str(m),
             overrides={k: overrides[k] for k in sorted(touch_set(str(g), str(m))) if k in overrides})
        for g, m in pairs
    ]
    meta.update(
        sampling_paradigm=paradigm,
        bif_refined_axis=[gene, mode],
        bif_refined_alpha=float(alpha),
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


def _eval_fiber_uncertainties_and_jumps(predictor, curves_params, curves_y0, curves_u, *, chunk_curves=256):
    """Evaluate exact Power uncertainty and adjacent finite-difference trajectory sensitivity ||dy/du||^2."""
    n_curves = len(curves_params)
    curve_u_vals = [None] * n_curves
    curve_slopes = [None] * n_curves
    curve_interval_jumps = [None] * n_curves

    for start in range(0, n_curves, int(chunk_curves)):
        stop = min(start + int(chunk_curves), n_curves)
        flat_p, flat_y0, offsets = [], [], []
        for idx in range(start, stop):
            ps, y0s = curves_params[idx], curves_y0[idx]
            offsets.append((len(flat_p), len(flat_p) + len(ps)))
            flat_p.extend(ps)
            flat_y0.extend(y0s)

        if not flat_p:
            for idx in range(start, stop):
                curve_u_vals[idx] = np.empty(0, dtype=np.float64)
                curve_slopes[idx] = np.empty(0, dtype=np.float64)
                curve_interval_jumps[idx] = np.empty(0, dtype=np.float64)
            continue

        u_flat, mean_flat = _forward_ensemble_stats(
            predictor,
            np.asarray(flat_p, dtype=np.float64),
            np.asarray(flat_y0, dtype=np.float64),
            need_mean=True,
        )

        for local_i, idx in enumerate(range(start, stop)):
            a, b = offsets[local_i]
            u_arr = np.maximum(u_flat[a:b], 1e-12)
            m_arr = mean_flat[a:b]
            q_arr = np.asarray(curves_u[idx], dtype=np.float64)
            L = len(u_arr)
            curve_u_vals[idx] = u_arr
            if L <= 1:
                curve_slopes[idx] = np.zeros(L, dtype=np.float64)
                curve_interval_jumps[idx] = np.empty(0, dtype=np.float64)
                continue
            du = np.maximum(np.diff(q_arr), 1e-3)
            diffs = m_arr[1:] - m_arr[:-1]
            jumps = np.mean(diffs.astype(np.float64) ** 2, axis=(1, 2)) / (du ** 2)
            slopes = np.zeros(L, dtype=np.float64)
            slopes[:-1] = np.maximum(slopes[:-1], jumps)
            slopes[1:] = np.maximum(slopes[1:], jumps)
            curve_slopes[idx] = slopes
            curve_interval_jumps[idx] = jumps

    return curve_u_vals, curve_slopes, curve_interval_jumps


class BifurcationSynthesis:
    """R26 Track C Synthesizer:
    - ours_power_synth: 24k unbiased legal bank + pure Power (strict control, 0 dose synthesis)
    - ours_bif_power_synth: 24k unbiased legal bank + pure Power + 0-ODE 9-pt single-axis bifurcation (tau=2)
    - ours_grad_bif_synth: 24k unbiased legal bank + pure Power + 0-ODE multi-axis steepest-bifurcation
      coarse-to-fine localization weighted by U_power * (1 + gamma * log1p(J / med_J))
    """

    def __init__(self, name, initial, trajectories, seed=42, *,
                 bank_size=24000, beta=1.0, dose_tau=2.0, slope_gamma=0.35):
        if name not in BIF_SYNTH_GEN_METHODS:
            raise ValueError(f'Unknown R26 bifurcation synthesis method: {name}')
        self.name = name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.bank_size = int(bank_size)
        self.beta = float(beta)
        self.dose_tau = float(dose_tau)
        self.slope_gamma = float(slope_gamma)

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

    def _select_topologies_power(self, bank_view, plan_count):
        u_power = np.maximum(np.asarray(self.predictor.uncertainty(bank_view), dtype=np.float64), 1e-12)
        sel_seed = int(self.rng.integers(0, 2**31 - 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sel_seed)
            chosen_idx = _power_sample_indices(u_power, min(plan_count, len(u_power)), beta=self.beta)
        return np.asarray(chosen_idx, dtype=np.int64), u_power

    def _plan_power_only(self, candidates, signatures, bank_view, chosen_idx, u_power):
        planned = [
            (candidates[i], signatures[i], np.asarray(bank_view.params[i], dtype=np.float64),
             np.asarray(bank_view.initial[i], dtype=np.float64))
            for i in chosen_idx
        ]
        return planned, dict(
            continuous_axis_targets=0,
            bifurcation_refined_count=0,
            mean_dose_u_gain_ratio=1.0,
        )

    def _plan_single_axis_bif(self, domain, candidates, signatures, bank_view, chosen_idx, u_power):
        backend = domain.proposals.backend
        defaults = domain.proposals.defaults
        param_keys = backend.PARAM_KEYS

        per_target_options = []
        eval_params, eval_y0 = [], []

        for idx in chosen_idx:
            base_cand = candidates[idx]
            base_sig = signatures[idx]
            base_p = np.asarray(bank_view.params[idx], dtype=np.float64)
            base_y0 = np.asarray(bank_view.initial[idx], dtype=np.float64)
            base_u = float(u_power[idx])

            options = [(False, base_cand, base_sig, base_p, base_y0, None, base_u)]
            local_sigs = {base_sig}
            axes = find_continuous_axes(base_cand, backend)
            if axes:
                axis_i = int(self.rng.integers(0, len(axes)))
                gene, mode, my_touch = axes[axis_i]
                for u in _SINGLE_AXIS_QUANTILES:
                    alpha = float(backend.sample_scaling_factor_sobol(mode, gene, float(u)))
                    built = build_legal_dose_variant(
                        base_cand, gene, mode, my_touch, alpha, backend, defaults, param_keys, self.rng,
                        paradigm='bif_power_single_axis_synthesis',
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

        if eval_params:
            dose_u, _ = _forward_ensemble_stats(
                self.predictor,
                np.asarray(eval_params, dtype=np.float64),
                np.asarray(eval_y0, dtype=np.float64),
                need_mean=False,
            )
        else:
            dose_u = np.empty(0, dtype=np.float64)

        planned, planned_sigs = [], set()
        refined_count = 0
        has_cont = 0
        u_ratios = []

        for options in per_target_options:
            if len(options) > 1:
                has_cont += 1
            valid_opts, valid_us = [], []
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
                    name=f'BifPower-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['bif_power_refined'] += 1
                refined_count += 1
            u_ratios.append(chosen_u / base_u)
            planned_sigs.add(sig)
            planned.append((cand, sig, p_v, y0_v))

        return planned, dict(
            continuous_axis_targets=has_cont,
            bifurcation_refined_count=refined_count,
            mean_dose_u_gain_ratio=float(np.mean(u_ratios)) if u_ratios else 1.0,
        )

    def _plan_multiaxis_coarse_fine_bif(self, domain, candidates, signatures, bank_view, chosen_idx, u_power):
        backend = domain.proposals.backend
        defaults = domain.proposals.defaults
        param_keys = backend.PARAM_KEYS

        target_bases = []
        coarse_axis_records = []
        curves_p, curves_y0, curves_u = [], [], []
        target_to_curve_ids = [[] for _ in range(len(chosen_idx))]

        for t_i, idx in enumerate(chosen_idx):
            base_cand = candidates[idx]
            base_sig = signatures[idx]
            base_p = np.asarray(bank_view.params[idx], dtype=np.float64)
            base_y0 = np.asarray(bank_view.initial[idx], dtype=np.float64)
            base_u = max(float(u_power[idx]), 1e-12)
            target_bases.append((base_cand, base_sig, base_p, base_y0, base_u))

            axes = find_continuous_axes(base_cand, backend)
            for gene, mode, my_touch in axes:
                pts = []
                local_sigs = {base_sig}
                for u in _COARSE_QUANTILES:
                    alpha = float(backend.sample_scaling_factor_sobol(mode, gene, float(u)))
                    built = build_legal_dose_variant(
                        base_cand, gene, mode, my_touch, alpha, backend, defaults, param_keys, self.rng,
                        paradigm='grad_bif_coarse_fine_synthesis',
                    )
                    if built is None:
                        continue
                    cand_v, sig_v, vec_v, p_v, y0_v = built
                    if (sig_v in local_sigs or sig_v in domain.seen or sig_v in domain.forbidden
                            or domain.proposals.collides(vec_v)):
                        continue
                    local_sigs.add(sig_v)
                    pts.append((float(u), cand_v, sig_v, p_v, y0_v))
                if len(pts) >= 2:
                    cid = len(coarse_axis_records)
                    target_to_curve_ids[t_i].append(cid)
                    coarse_axis_records.append((t_i, (gene, mode, my_touch), pts))
                    curves_p.append([x[3] for x in pts])
                    curves_y0.append([x[4] for x in pts])
                    curves_u.append([x[0] for x in pts])

        c_u_vals, _, c_jumps = _eval_fiber_uncertainties_and_jumps(
            self.predictor, curves_p, curves_y0, curves_u, chunk_curves=256
        )

        all_pos_j = np.concatenate([j[j > 0] for j in c_jumps if len(j) > 0]) if c_jumps else np.empty(0)
        med_j_coarse = max(float(np.median(all_pos_j)) if len(all_pos_j) > 0 else 1.0, 1e-8)

        fine_records = []
        fine_curves_p, fine_curves_y0, fine_curves_u = [], [], []
        target_to_fine_id = {}
        multiaxis_targets = 0

        for t_i in range(len(chosen_idx)):
            cids = target_to_curve_ids[t_i]
            if not cids:
                continue
            if len(cids) >= 2:
                multiaxis_targets += 1

            best_cid = None
            best_axis_score = -1.0
            best_bracket = None

            for cid in cids:
                u_arr = c_u_vals[cid]
                j_arr = c_jumps[cid]
                pts = coarse_axis_records[cid][2]
                q_arr = [x[0] for x in pts]
                iv_scores = 0.5 * (u_arr[:-1] + u_arr[1:]) * (1.0 + self.slope_gamma * np.log1p(j_arr / med_j_coarse))
                q_best = int(np.argmax(iv_scores))
                axis_peak = float(np.max(iv_scores))
                if axis_peak > best_axis_score:
                    best_axis_score = axis_peak
                    best_cid = cid
                    u_lo, u_hi = q_arr[q_best], q_arr[q_best + 1]
                    u_max_i = int(np.argmax(u_arr))
                    if u_max_i == 0 and u_arr[0] > 1.2 * u_arr[1]:
                        u_lo, u_hi = 0.02, q_arr[1]
                    elif u_max_i == len(u_arr) - 1 and u_arr[-1] > 1.2 * u_arr[-2]:
                        u_lo, u_hi = q_arr[-2], 0.98
                    best_bracket = (u_lo, u_hi)

            base_cand, base_sig, _, _, _ = target_bases[t_i]
            _, (gene, mode, my_touch), coarse_pts = coarse_axis_records[best_cid]
            existing_sigs = {base_sig} | {x[2] for x in coarse_pts}
            existing_us = {round(x[0], 4) for x in coarse_pts}

            u_lo, u_hi = best_bracket
            fine_us = np.linspace(u_lo, u_hi, 7)[1:-1]
            merged = list(coarse_pts)
            for u in fine_us:
                u = float(np.clip(u, 0.02, 0.98))
                if round(u, 4) in existing_us:
                    continue
                alpha = float(backend.sample_scaling_factor_sobol(mode, gene, u))
                built = build_legal_dose_variant(
                    base_cand, gene, mode, my_touch, alpha, backend, defaults, param_keys, self.rng,
                    paradigm='grad_bif_coarse_fine_synthesis',
                )
                if built is None:
                    continue
                cand_v, sig_v, vec_v, p_v, y0_v = built
                if (sig_v in existing_sigs or sig_v in domain.seen or sig_v in domain.forbidden
                        or domain.proposals.collides(vec_v)):
                    continue
                existing_sigs.add(sig_v)
                existing_us.add(round(u, 4))
                merged.append((u, cand_v, sig_v, p_v, y0_v))

            merged.sort(key=lambda x: x[0])
            fid = len(fine_records)
            target_to_fine_id[t_i] = fid
            fine_records.append((t_i, merged))
            fine_curves_p.append([x[3] for x in merged])
            fine_curves_y0.append([x[4] for x in merged])
            fine_curves_u.append([x[0] for x in merged])

        f_u_vals, f_slopes, _ = _eval_fiber_uncertainties_and_jumps(
            self.predictor, fine_curves_p, fine_curves_y0, fine_curves_u, chunk_curves=256
        )
        all_pos_s = np.concatenate([s[s > 0] for s in f_slopes if len(s) > 0]) if f_slopes else np.empty(0)
        med_s_fine = max(float(np.median(all_pos_s)) if len(all_pos_s) > 0 else 1.0, 1e-8)

        planned, planned_sigs = [], set()
        refined_count = 0
        has_cont = len(fine_records)
        u_ratios = []

        for t_i in range(len(chosen_idx)):
            base_cand, base_sig, base_p, base_y0, base_u = target_bases[t_i]
            fid = target_to_fine_id.get(t_i)
            options, weights = [], []
            if base_sig not in planned_sigs and base_sig not in domain.seen and base_sig not in domain.forbidden:
                options.append((False, base_cand, base_sig, base_p, base_y0, base_u))
                weights.append(base_u)

            if fid is not None:
                merged = fine_records[fid][1]
                u_arr = f_u_vals[fid]
                s_arr = f_slopes[fid]
                for g_i, (_, cand_v, sig_v, p_v, y0_v) in enumerate(merged):
                    if sig_v in planned_sigs or sig_v in domain.seen or sig_v in domain.forbidden:
                        continue
                    uv = max(float(u_arr[g_i]), 1e-12)
                    sv = max(float(s_arr[g_i]), 0.0)
                    b_score = uv * (1.0 + self.slope_gamma * math.log1p(sv / med_s_fine))
                    options.append((True, cand_v, sig_v, p_v, y0_v, uv))
                    weights.append(b_score)

            if not options:
                continue
            if len(options) == 1:
                pick = 0
            else:
                log_w = self.dose_tau * np.log(np.maximum(np.asarray(weights, dtype=np.float64), 1e-12))
                log_w -= np.max(log_w)
                probs = np.exp(log_w)
                probs /= np.sum(probs)
                pick = int(self.rng.choice(len(options), p=probs))

            is_new, cand, sig, p_v, y0_v, chosen_u = options[pick]
            if is_new:
                new_index = domain.next_index
                domain.next_index += 1
                cand = replace(
                    cand,
                    name=f'GradBif-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['grad_bif_refined'] += 1
                refined_count += 1
            u_ratios.append(chosen_u / base_u)
            planned_sigs.add(sig)
            planned.append((cand, sig, p_v, y0_v))

        return planned, dict(
            continuous_axis_targets=has_cont,
            multiaxis_targets=multiaxis_targets,
            coarse_fibers_evaluated=len(coarse_axis_records),
            fine_fibers_evaluated=len(fine_records),
            bifurcation_refined_count=refined_count,
            mean_dose_u_gain_ratio=float(np.mean(u_ratios)) if u_ratios else 1.0,
            median_coarse_jump=med_j_coarse,
            median_fine_slope=med_s_fine,
        )

    def _plan_batch(self, domain, plan_count):
        domain.verify()
        target_bank = max(self.bank_size, plan_count * 4)
        candidates, signatures, bank_view = domain.bank(target_bank)
        chosen_idx, u_power = self._select_topologies_power(bank_view, plan_count)

        if self.name == 'ours_power_synth':
            planned, extra = self._plan_power_only(candidates, signatures, bank_view, chosen_idx, u_power)
        elif self.name == 'ours_bif_power_synth':
            planned, extra = self._plan_single_axis_bif(domain, candidates, signatures, bank_view, chosen_idx, u_power)
        else:
            planned, extra = self._plan_multiaxis_coarse_fine_bif(domain, candidates, signatures, bank_view, chosen_idx, u_power)

        self._planned = planned
        bank_levels = Counter(int(c.level) for c in candidates)
        planned_levels = Counter(int(c.level) for c, _, _, _ in planned)
        self._last_plan_diagnostics = dict(
            method=self.name,
            unbiased_bank_size=len(candidates),
            power_beta=self.beta,
            dose_tau=self.dose_tau,
            slope_gamma=self.slope_gamma if self.name == 'ours_grad_bif_synth' else 0.0,
            bank_levels=dict(sorted(bank_levels.items())),
            planned_queries=len(planned),
            planned_levels=dict(sorted(planned_levels.items())),
            u_power_selected_mean=float(np.mean(u_power[chosen_idx])),
            u_power_bank_mean=float(np.mean(u_power)),
            **extra,
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

