"""Pool-Anchored Continuous Bifurcation & Hybrid Allelic-Series Mode-Switching Query Synthesis.

Zero pool labels read; zero external random topology generator.
Uses strictly unlabelled pool inputs (candidates \\ initial) as genetic-module anchors, then runs
0-ODE GPU batched synthesis under canonical biological legality and collision gates:
  - ours_pool_bif_synth: Pure pool-anchored 9-pt continuous bifurcation synthesis (preserves exact
    genetic topology, discrete modes, and COMPOSITE_REAL_EVIDENCE composites; 1 ODE/query).
  - ours_pool_mode_bif_v2_synth: Composite-safe, scale-normalized gain-gated allelic-series mode
    switching (KO / Suppression / Overexpression) + continuous bifurcation synthesis.
  - ours_pool_mode_bif_synth: R26 v1 unconstrained two-level mode+dose synthesis (retained verbatim).
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import numpy as np
import torch

from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.systems.budding.catalog import (
    GROUP_BY_NAME,
    MODE_CLASS,
    first_mutation_conflict,
    touch_set,
)
from budding_yeast_v2.data.systems.budding.distributions.ranges import (
    AMPLITUDE_MODES,
    EXACT_ZERO_TARGETS,
    GAL_MODES,
)
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.sampling_policy import ENVIRONMENT_MODULES, valid_candidate
from ...baseline.uncertainty import _power_sample_indices
from ...contracts import PoolInputs
from ..cebs.metrics import _forward_ensemble_stats
from .synthesis import (
    _SINGLE_AXIS_QUANTILES,
    build_legal_dose_variant,
    find_continuous_axes,
)

POOL_MODE_GEN_METHODS = (
    'ours_pool_mode_bif_synth',
    'ours_pool_bif_synth',
    'ours_pool_mode_bif_v2_synth',
)
_MODE_DOSE_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)


def _is_continuous_mode(gene, mode, backend):
    try:
        a_lo = float(backend.sample_scaling_factor_sobol(mode, gene, 0.05))
        a_hi = float(backend.sample_scaling_factor_sobol(mode, gene, 0.95))
    except Exception:
        return False
    return abs(a_hi - a_lo) > 1e-9


def _forward_ensemble_power_and_rel(predictor, params_arr, y0_arr, *, eps_rel=0.5):
    """Single batched 2-member forward pass returning both exact Power variance and scale-normalized relative variance."""
    n = len(params_arr)
    device = predictor.device
    steps = predictor.steps
    bs = predictor.batch_size
    u_power = np.empty(n, dtype=np.float64)
    u_rel = np.empty(n, dtype=np.float64)
    eps_sq = float(eps_rel) ** 2

    with torch.no_grad():
        for start in range(0, n, bs):
            stop = min(start + bs, n)
            x, p = predictor.normalization.encode(params_arr[start:stop], y0_arr[start:stop], steps)
            x_d, p_d = x.to(device), p.to(device)
            preds = []
            for model in predictor.members:
                model.to(device).eval()
                forward = getattr(model, 'forward_ic_time', model)
                out = forward(x_d, p_d).float()
                if not torch.isfinite(out).all():
                    raise FloatingPointError(f'Nonfinite ensemble prediction at batch {start}')
                preds.append(out)
            stack = torch.stack(preds, dim=0)                         # (M, B, V, T)
            mean = stack.mean(dim=0)                                  # (B, V, T)
            sq_err = (stack - mean.unsqueeze(0)).square().mean(dim=0) # (B, V, T)
            var_ch = sq_err.mean(dim=-1)                              # (B, V)
            scale_ch = mean.square().mean(dim=-1) + eps_sq            # (B, V)
            u_power[start:stop] = var_ch.mean(dim=-1).cpu().numpy()
            u_rel[start:stop] = (var_ch / scale_ch).mean(dim=-1).cpu().numpy()

    return u_power, u_rel


def _candidate_from_pool_row(meta, ovr, backend, defaults, param_keys):
    """Build a canonical Candidate from an unlabelled pool row's metadata and effective overrides."""
    pairs = [(str(g), str(m)) for g, m in meta.get('pairs', ())]
    level = int(meta.get('module_count', meta.get('requested_level', max(1, len({g for g, _ in pairs})))))
    desc = str(meta.get('description') or '+'.join(f'{g}[{m}]' for g, m in pairs))
    clean_meta = deepcopy(meta)
    clean_meta.pop('_effective_overrides', None)
    clean_meta['description'] = desc
    clean_meta['pairs'] = [[g, m] for g, m in pairs]
    cand = Candidate(
        int(meta.get('source_candidate_id', 0)),
        f'PoolAnchor-{int(meta.get("source_candidate_id", 0)):09d}',
        dict(ovr),
        level,
        metadata=clean_meta,
    )
    vector = effective_inputs(cand.overrides, defaults)
    sig = hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
    _, y0, _, _ = backend.prepare_initial_conditions(cand.overrides)
    params = np.array([cand.overrides.get(k, defaults[k]) for k in param_keys], dtype=np.float64)
    return cand, sig, vector, params, np.asarray(y0, dtype=np.float64)


def _build_hybrid_mode_variant(anchor_meta, anchor_ovr, target_gene, new_mode, alpha,
                               backend, defaults, param_keys, rng, *,
                               paradigm='pool_anchored_hybrid_mode_bifurcation_synthesis'):
    """Switch single-mode target_gene to new_mode (with strength alpha) under canonical Budding rules."""
    old_pairs = [(str(g), str(m)) for g, m in anchor_meta.get('pairs', ())]
    other_pairs = [(g, m) for g, m in old_pairs if g != target_gene]
    old_target_modes = [m for g, m in old_pairs if g == target_gene]
    new_pairs = sorted(other_pairs + [(target_gene, new_mode)])
    if first_mutation_conflict(new_pairs) is not None:
        return None

    old_touch = set()
    for om in old_target_modes:
        old_touch |= touch_set(target_gene, om)
    other_touch = set()
    for og, om in other_pairs:
        other_touch |= touch_set(og, om)
    strip_keys = old_touch - other_touch
    overrides = {k: v for k, v in anchor_ovr.items() if k not in strip_keys}

    changes = {}
    backend.apply_gene_mutation(target_gene, new_mode, float(alpha), defaults, defaults.copy(), changes, rng=rng)
    if set(changes.keys()) & other_touch:
        return None
    overrides.update(changes)

    has_new_gal = any(m in GAL_MODES for _, m in new_pairs)
    had_old_gal = any(m in GAL_MODES for _, m in old_pairs)
    if has_new_gal:
        overrides['mdt'] = 150.0
    elif had_old_gal and float(overrides.get('mdt', 90.0)) == 150.0:
        overrides['mdt'] = 90.0

    level = len({g for g, _ in new_pairs})
    desc = '+'.join(f'{g}[{m}]' for g, m in new_pairs)

    meta = deepcopy(anchor_meta)
    meta.pop('_effective_overrides', None)
    meta['description'] = desc
    meta['pairs'] = [[g, m] for g, m in new_pairs]
    meta['original_overrides'] = dict(overrides)
    draws = [dict(r) for r in meta.get('proposal_strength_draws', []) if str(r.get('gene')) != target_gene]
    draws.append(dict(gene=target_gene, mode=new_mode, alpha=float(alpha)))
    meta['proposal_strength_draws'] = draws
    meta['mutation_parameters'] = [
        dict(gene=g, mode=m,
             overrides={k: overrides[k] for k in sorted(touch_set(g, m)) if k in overrides})
        for g, m in new_pairs
    ]
    old_mode_label = '+'.join(old_target_modes) if old_target_modes else 'none'
    meta.update(
        sampling_paradigm=paradigm,
        anchor_source_candidate_id=int(anchor_meta.get('source_candidate_id', -1)),
        hybrid_target_gene=target_gene,
        hybrid_old_mode=old_mode_label,
        hybrid_new_mode=new_mode,
        hybrid_new_alpha=float(alpha),
        rescued=False,
        real_collision=False,
    )
    meta.pop('rescue_attempts', None)

    cand = Candidate(0, f'Hybrid-{desc}', overrides, level, metadata=meta)
    cand = valid_candidate(cand, 'budding')
    if cand is None:
        return None

    vector = effective_inputs(cand.overrides, defaults)
    sig = hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
    _, y0, _, _ = backend.prepare_initial_conditions(cand.overrides)
    params = np.array([cand.overrides.get(k, defaults[k]) for k in param_keys], dtype=np.float64)
    return cand, sig, vector, params, np.asarray(y0, dtype=np.float64)


def _sample_power_weighted(weights, tau, rng):
    arr = np.maximum(np.asarray(weights, dtype=np.float64), 1e-12)
    if len(arr) == 1:
        return 0
    lw = float(tau) * np.log(arr)
    lw -= np.max(lw)
    pr = np.exp(lw)
    pr /= np.sum(pr)
    return int(rng.choice(len(arr), p=pr))


class PoolModeBifSynthesis:
    """Pool-Anchored Continuous Bifurcation & Hybrid Allelic-Series Synthesis."""

    def __init__(self, name, initial, trajectories, seed=42, *,
                 beta=1.0, tau_mode=0.75, tau_dose=2.0,
                 mode_gain_cont=1.35, mode_gain_disc=1.15, eps_rel=0.5):
        if name not in POOL_MODE_GEN_METHODS:
            raise ValueError(f'Unknown pool-anchored synthesis method: {name}')
        self.name = name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.beta = float(beta)
        self.tau_mode = float(tau_mode)
        self.tau_dose = float(tau_dose)
        self.mode_gain_cont = float(mode_gain_cont)
        self.mode_gain_disc = float(mode_gain_disc)
        self.eps_rel = float(eps_rel)

        self.predictor = None
        self.train_inputs = initial
        self.target = len(initial.ids)
        self.round_count = 0
        self._pool_anchors = None
        self._used_anchor_positions = set()
        self._planned = []
        self._last_plan_diagnostics = {}

    def set_pool_anchors(self, pool_anchors):
        """Attach strictly unlabelled pool candidate inputs (params, y_init, metadata; NO trajectories)."""
        if pool_anchors is None or not len(pool_anchors.ids) or not pool_anchors.metadata:
            raise ValueError('PoolModeBifSynthesis requires unlabelled pool anchors with metadata')
        self._pool_anchors = pool_anchors

    def state_dict(self):
        return dict(
            name=self.name,
            seed=self.seed,
            rng_state=self.rng.bit_generator.state,
            round_count=self.round_count,
            used_anchor_positions=sorted(self._used_anchor_positions),
            last_plan_diagnostics=dict(self._last_plan_diagnostics),
        )

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng_state']
        self.round_count = int(state.get('round_count', 0))
        self._used_anchor_positions = set(int(i) for i in state.get('used_anchor_positions', ()))
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

    def _select_pool_anchors(self, plan_count):
        if self._pool_anchors is None:
            raise RuntimeError('PoolModeBifSynthesis missing unlabelled pool anchors; call set_pool_anchors first')
        n_pool = len(self._pool_anchors.ids)
        avail_pos = np.array([i for i in range(n_pool) if i not in self._used_anchor_positions], dtype=np.int64)
        if len(avail_pos) < plan_count:
            self._used_anchor_positions.clear()
            avail_pos = np.arange(n_pool, dtype=np.int64)

        avail_view = PoolInputs(
            self._pool_anchors.ids[avail_pos],
            self._pool_anchors.params[avail_pos],
            self._pool_anchors.initial[avail_pos],
        )
        u_pool = np.maximum(np.asarray(self.predictor.uncertainty(avail_view), dtype=np.float64), 1e-12)
        sel_seed = int(self.rng.integers(0, 2**31 - 1))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sel_seed)
            local_chosen = _power_sample_indices(u_pool, min(plan_count, len(u_pool)), beta=self.beta)
        local_chosen = np.asarray(local_chosen, dtype=np.int64)
        chosen_pos = avail_pos[local_chosen]
        self._used_anchor_positions.update(int(p) for p in chosen_pos)
        return avail_pos, u_pool, local_chosen, chosen_pos

    def _plan_pool_bif_only(self, domain, plan_count):
        """ours_pool_bif_synth: Pure pool-anchored 9-pt continuous bifurcation synthesis (preserves modes & composites)."""
        backend = domain.proposals.backend
        defaults = domain.proposals.defaults
        param_keys = backend.PARAM_KEYS
        avail_pos, u_pool, local_chosen, chosen_pos = self._select_pool_anchors(plan_count)

        per_anchor_fibers = []
        eval_params, eval_y0 = [], []
        cont_targets = 0

        for l_idx, pos in zip(local_chosen, chosen_pos):
            anchor_meta = self._pool_anchors.metadata[int(pos)]
            anchor_ovr = anchor_meta.get('_effective_overrides', anchor_meta.get('original_overrides', {}))
            anchor_u = float(u_pool[int(l_idx)])
            base_cand, base_sig, base_vec, base_p, base_y0 = _candidate_from_pool_row(
                anchor_meta, anchor_ovr, backend, defaults, param_keys
            )
            local_sigs = {base_sig}
            entries = []
            if (base_sig not in domain.seen and base_sig not in domain.forbidden
                    and not domain.proposals.collides(base_vec)):
                entries.append((False, base_cand, base_sig, base_p, base_y0, None, anchor_u))

            axes = find_continuous_axes(base_cand, backend)
            if axes:
                cont_targets += 1
                gene, mode, my_touch = axes[int(self.rng.integers(0, len(axes)))]
                for u in _SINGLE_AXIS_QUANTILES:
                    try:
                        alpha = float(backend.sample_scaling_factor_sobol(mode, gene, float(u)))
                    except Exception:
                        continue
                    built = build_legal_dose_variant(
                        base_cand, gene, mode, my_touch, alpha,
                        backend, defaults, param_keys, self.rng,
                        paradigm='pool_anchored_continuous_bifurcation_synthesis',
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
                    entries.append((True, cand_v, sig_v, p_v, y0_v, flat_idx, None))

            per_anchor_fibers.append((anchor_u, entries))

        if eval_params:
            var_u, _ = _forward_ensemble_stats(
                self.predictor,
                np.asarray(eval_params, dtype=np.float64),
                np.asarray(eval_y0, dtype=np.float64),
                need_mean=False,
            )
        else:
            var_u = np.empty(0, dtype=np.float64)

        planned, planned_sigs = [], set()
        dose_refined = 0
        kept_anchor = 0
        u_ratios = []

        for anchor_u, entries in per_anchor_fibers:
            valid_e, valid_u = [], []
            for is_new, cand, sig, p_v, y0_v, flat_idx, orig_u in entries:
                if sig in planned_sigs or sig in domain.seen or sig in domain.forbidden:
                    continue
                uv = max(float(var_u[flat_idx]) if flat_idx is not None else float(orig_u), 1e-12)
                valid_e.append((is_new, cand, sig, p_v, y0_v, uv))
                valid_u.append(uv)
            if not valid_e:
                continue
            pick = _sample_power_weighted(valid_u, self.tau_dose, self.rng)
            is_new, cand, sig, p_v, y0_v, chosen_u = valid_e[pick]
            new_index = domain.next_index
            domain.next_index += 1
            if is_new:
                dose_refined += 1
                cand = replace(
                    cand,
                    name=f'PoolBif-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['pool_bif_dose_refined'] += 1
            else:
                kept_anchor += 1
                cand = replace(
                    cand,
                    name=f'PoolAnchor-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['pool_bif_kept_anchor'] += 1
            u_ratios.append(chosen_u / max(anchor_u, 1e-12))
            planned_sigs.add(sig)
            planned.append((cand, sig, p_v, y0_v))

        self._planned = planned
        planned_levels = Counter(int(c.level) for c, _, _, _ in planned)
        self._last_plan_diagnostics = dict(
            method=self.name,
            anchor_source='unlabelled_mutation_pool_only',
            available_pool_anchors=len(avail_pos),
            continuous_axis_targets=cont_targets,
            total_dose_variants_scored=len(eval_params),
            tau_dose=self.tau_dose,
            planned_queries=len(planned),
            planned_levels=dict(sorted(planned_levels.items())),
            bifurcation_refined_count=dose_refined,
            kept_anchor_count=kept_anchor,
            mean_dose_u_gain_ratio=float(np.mean(u_ratios)) if u_ratios else 1.0,
            anchor_u_selected_mean=float(np.mean(u_pool[local_chosen])),
            anchor_u_pool_mean=float(np.mean(u_pool)),
        )

    def _plan_pool_mode_v2(self, domain, plan_count):
        """ours_pool_mode_bif_v2_synth: Composite-safe, scale-normalized gain-gated hybrid mode + bifurcation synthesis."""
        backend = domain.proposals.backend
        defaults = domain.proposals.defaults
        param_keys = backend.PARAM_KEYS
        avail_pos, u_pool, local_chosen, chosen_pos = self._select_pool_anchors(plan_count)

        per_anchor_records = []
        eval_params, eval_y0 = [], []
        protected_composites = 0

        for l_idx, pos in zip(local_chosen, chosen_pos):
            anchor_meta = self._pool_anchors.metadata[int(pos)]
            anchor_ovr = anchor_meta.get('_effective_overrides', anchor_meta.get('original_overrides', {}))
            anchor_u = float(u_pool[int(l_idx)])
            pairs = [(str(g), str(m)) for g, m in anchor_meta.get('pairs', ())]

            base_cand, base_sig, base_vec, base_p, base_y0 = _candidate_from_pool_row(
                anchor_meta, anchor_ovr, backend, defaults, param_keys
            )
            local_sigs = {base_sig}

            # Group 1: Original mode (anchor + full 9-pt continuous bifurcation along an existing continuous axis)
            orig_entries = []
            if (base_sig not in domain.seen and base_sig not in domain.forbidden
                    and not domain.proposals.collides(base_vec)):
                base_idx = len(eval_params)
                eval_params.append(base_p)
                eval_y0.append(base_y0)
                orig_entries.append((False, '__orig__', base_cand, base_sig, base_p, base_y0, base_idx))

            axes = find_continuous_axes(base_cand, backend)
            has_cont_axis = bool(axes)
            if axes:
                a_gene, a_mode, a_touch = axes[int(self.rng.integers(0, len(axes)))]
                for u in _SINGLE_AXIS_QUANTILES:
                    try:
                        alpha = float(backend.sample_scaling_factor_sobol(a_mode, a_gene, float(u)))
                    except Exception:
                        continue
                    built = build_legal_dose_variant(
                        base_cand, a_gene, a_mode, a_touch, alpha,
                        backend, defaults, param_keys, self.rng,
                        paradigm='pool_anchored_hybrid_v2_dose_bifurcation',
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
                    orig_entries.append((True, '__orig_dose__', cand_v, sig_v, p_v, y0_v, flat_idx))

            # Group 2: Composite-safe alternative modes ONLY for single-mode multi-mode genes
            modes_by_gene = {}
            for g, m in pairs:
                modes_by_gene.setdefault(g, []).append(m)
            if any(len(ms) > 1 for ms in modes_by_gene.values()):
                protected_composites += 1

            switchable_genes = sorted(
                g for g, ms in modes_by_gene.items()
                if len(ms) == 1
                and g in GROUP_BY_NAME
                and g not in ENVIRONMENT_MODULES
                and len(GROUP_BY_NAME[g]['modes']) >= 2
            )

            alt_mode_groups = {}
            target_gene = ''
            old_mode_str = ''
            if switchable_genes:
                target_gene = switchable_genes[int(self.rng.integers(0, len(switchable_genes)))]
                old_mode_str = modes_by_gene[target_gene][0]
                alt_modes = sorted(m for m in GROUP_BY_NAME[target_gene]['modes'].keys() if m != old_mode_str)

                for new_mode in alt_modes:
                    if (target_gene, new_mode) in EXACT_ZERO_TARGETS:
                        alphas = (0.0,)
                    elif new_mode not in AMPLITUDE_MODES:
                        alphas = (1.0,)
                    else:
                        is_cont = _is_continuous_mode(target_gene, new_mode, backend)
                        quantiles = _MODE_DOSE_QUANTILES if is_cont else (0.5,)
                        alphas = []
                        for u in quantiles:
                            try:
                                alphas.append(float(backend.sample_scaling_factor_sobol(new_mode, target_gene, float(u))))
                            except Exception:
                                continue
                    group_entries = []
                    for alpha in alphas:
                        built = _build_hybrid_mode_variant(
                            anchor_meta, anchor_ovr, target_gene, new_mode, alpha,
                            backend, defaults, param_keys, self.rng,
                            paradigm='pool_anchored_hybrid_v2_mode_switch',
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
                        group_entries.append((True, new_mode, cand_v, sig_v, p_v, y0_v, flat_idx))
                    if group_entries:
                        alt_mode_groups[new_mode] = group_entries

            per_anchor_records.append((anchor_u, has_cont_axis, target_gene, old_mode_str, orig_entries, alt_mode_groups))

        # Single batched GPU pass computing BOTH exact Power variance & scale-normalized relative variance (0 ODE!)
        if eval_params:
            var_u, rel_u = _forward_ensemble_power_and_rel(
                self.predictor,
                np.asarray(eval_params, dtype=np.float64),
                np.asarray(eval_y0, dtype=np.float64),
                eps_rel=self.eps_rel,
            )
        else:
            var_u = np.empty(0, dtype=np.float64)
            rel_u = np.empty(0, dtype=np.float64)

        planned, planned_sigs = [], set()
        mode_switched = 0
        dose_only_refined = 0
        kept_anchor = 0
        gate_passed_anchors = 0
        u_ratios = []
        rel_ratios = []
        class_transitions = Counter()

        for anchor_u, has_cont_axis, target_gene, old_mode_str, orig_entries, alt_mode_groups in per_anchor_records:
            # Step 1: Pick best original-mode champion via exact Power variance (tau_dose=2.0)
            valid_orig = []
            for is_new, tag, cand, sig, p_v, y0_v, idx in orig_entries:
                if sig in planned_sigs or sig in domain.seen or sig in domain.forbidden:
                    continue
                valid_orig.append((is_new, tag, cand, sig, p_v, y0_v, max(float(var_u[idx]), 1e-12), max(float(rel_u[idx]), 1e-12)))

            orig_champ = None
            if valid_orig:
                o_pick = _sample_power_weighted([e[6] for e in valid_orig], self.tau_dose, self.rng)
                orig_champ = valid_orig[o_pick]

            # Step 2: Pick within-mode bifurcation champion for each alternative mode & apply scale-normalized gain gate
            gate_mult = self.mode_gain_cont if has_cont_axis else self.mode_gain_disc
            orig_rel = orig_champ[7] if orig_champ is not None else 0.0
            qualifying_alts = []
            for alt_mode, entries in alt_mode_groups.items():
                valid_a = []
                for is_new, m_name, cand, sig, p_v, y0_v, idx in entries:
                    if sig in planned_sigs or sig in domain.seen or sig in domain.forbidden:
                        continue
                    valid_a.append((is_new, m_name, cand, sig, p_v, y0_v, max(float(var_u[idx]), 1e-12), max(float(rel_u[idx]), 1e-12)))
                if not valid_a:
                    continue
                a_pick = _sample_power_weighted([e[6] for e in valid_a], self.tau_dose, self.rng)
                a_champ = valid_a[a_pick]
                if orig_champ is None or a_champ[7] >= gate_mult * orig_rel:
                    qualifying_alts.append(a_champ)

            # Step 3: Decide between original-mode champion and qualifying alternative-mode champions
            if orig_champ is None and not qualifying_alts:
                continue
            if qualifying_alts:
                gate_passed_anchors += 1
                pool_champs = ([orig_champ] if orig_champ is not None else []) + qualifying_alts
                m_pick = _sample_power_weighted([c[7] for c in pool_champs], self.tau_mode, self.rng)
                chosen = pool_champs[m_pick]
            else:
                chosen = orig_champ

            is_new, chosen_tag, cand, sig, p_v, y0_v, chosen_u, chosen_rel = chosen
            new_index = domain.next_index
            domain.next_index += 1
            if not is_new:
                kept_anchor += 1
                cand = replace(
                    cand,
                    name=f'PoolAnchor-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['pool_v2_kept_anchor'] += 1
            elif chosen_tag == '__orig_dose__':
                dose_only_refined += 1
                cand = replace(
                    cand,
                    name=f'PoolBifV2-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['pool_v2_dose_refined'] += 1
            else:
                mode_switched += 1
                cand = replace(
                    cand,
                    name=f'PoolModeV2-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['pool_v2_mode_switched'] += 1
                old_cls = MODE_CLASS.get(old_mode_str, old_mode_str)
                new_cls = MODE_CLASS.get(chosen_tag, chosen_tag)
                class_transitions[f'{old_cls}->{new_cls}'] += 1
                if orig_rel > 0:
                    rel_ratios.append(chosen_rel / orig_rel)

            u_ratios.append(chosen_u / max(anchor_u, 1e-12))
            planned_sigs.add(sig)
            planned.append((cand, sig, p_v, y0_v))

        self._planned = planned
        planned_levels = Counter(int(c.level) for c, _, _, _ in planned)
        self._last_plan_diagnostics = dict(
            method=self.name,
            anchor_source='unlabelled_mutation_pool_only',
            available_pool_anchors=len(avail_pos),
            protected_composite_anchors=protected_composites,
            total_variants_scored=len(eval_params),
            tau_mode=self.tau_mode,
            tau_dose=self.tau_dose,
            mode_gain_cont=self.mode_gain_cont,
            mode_gain_disc=self.mode_gain_disc,
            eps_rel=self.eps_rel,
            planned_queries=len(planned),
            planned_levels=dict(sorted(planned_levels.items())),
            gate_passed_anchors=gate_passed_anchors,
            mode_switched_count=mode_switched,
            dose_only_refined_count=dose_only_refined,
            kept_anchor_count=kept_anchor,
            top_class_transitions=dict(class_transitions.most_common(8)),
            mean_hybrid_u_gain_ratio=float(np.mean(u_ratios)) if u_ratios else 1.0,
            mean_mode_switch_rel_gain=float(np.mean(rel_ratios)) if rel_ratios else 1.0,
            anchor_u_selected_mean=float(np.mean(u_pool[local_chosen])),
            anchor_u_pool_mean=float(np.mean(u_pool)),
        )

    def _plan_pool_mode_v1(self, domain, plan_count):
        """R26 v1 unconstrained two-level mode+dose synthesis (retained verbatim)."""
        backend = domain.proposals.backend
        defaults = domain.proposals.defaults
        param_keys = backend.PARAM_KEYS
        avail_pos, u_pool, local_chosen, chosen_pos = self._select_pool_anchors(plan_count)

        per_anchor_mode_groups = []
        eval_params, eval_y0 = [], []

        for l_idx, pos in zip(local_chosen, chosen_pos):
            anchor_meta = self._pool_anchors.metadata[int(pos)]
            anchor_ovr = anchor_meta.get('_effective_overrides', anchor_meta.get('original_overrides', {}))
            anchor_u = float(u_pool[int(l_idx)])
            pairs = [(str(g), str(m)) for g, m in anchor_meta.get('pairs', ())]

            base_cand, base_sig, base_vec, base_p, base_y0 = _candidate_from_pool_row(
                anchor_meta, anchor_ovr, backend, defaults, param_keys
            )
            local_sigs = {base_sig}
            anchor_entry = None
            if (base_sig not in domain.seen and base_sig not in domain.forbidden
                    and not domain.proposals.collides(base_vec)):
                anchor_entry = (False, '__anchor__', base_cand, base_sig, base_p, base_y0, None, anchor_u)

            genes = sorted({g for g, _ in pairs if g in GROUP_BY_NAME and g not in ENVIRONMENT_MODULES})
            multi_mode_genes = [g for g in genes if len(GROUP_BY_NAME[g]['modes']) >= 2]
            candidates_genes = multi_mode_genes if multi_mode_genes else genes
            old_mode_str = ''
            target_gene = ''
            mode_groups = {}

            if candidates_genes:
                target_gene = candidates_genes[int(self.rng.integers(0, len(candidates_genes)))]
                old_modes = [m for g, m in pairs if g == target_gene]
                old_mode_str = '+'.join(old_modes)
                legal_modes = sorted(GROUP_BY_NAME[target_gene]['modes'].keys())

                if anchor_entry is not None:
                    anchor_bucket = old_mode_str if (len(old_modes) == 1 and old_mode_str in legal_modes) else '__anchor__'
                    mode_groups.setdefault(anchor_bucket, []).append(anchor_entry)

                for new_mode in legal_modes:
                    if (target_gene, new_mode) in EXACT_ZERO_TARGETS:
                        alphas = (0.0,)
                    elif new_mode not in AMPLITUDE_MODES:
                        alphas = (1.0,)
                    else:
                        is_cont = _is_continuous_mode(target_gene, new_mode, backend)
                        quantiles = _MODE_DOSE_QUANTILES if is_cont else (0.5,)
                        alphas = []
                        for u in quantiles:
                            try:
                                alphas.append(float(backend.sample_scaling_factor_sobol(new_mode, target_gene, float(u))))
                            except Exception:
                                continue
                    group_entries = mode_groups.setdefault(new_mode, [])
                    for alpha in alphas:
                        built = _build_hybrid_mode_variant(
                            anchor_meta, anchor_ovr, target_gene, new_mode, alpha,
                            backend, defaults, param_keys, self.rng,
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
                        group_entries.append((True, new_mode, cand_v, sig_v, p_v, y0_v, flat_idx, None))
                    if not group_entries:
                        mode_groups.pop(new_mode, None)
            elif anchor_entry is not None:
                mode_groups['__anchor__'] = [anchor_entry]

            per_anchor_mode_groups.append((anchor_u, target_gene, old_mode_str, mode_groups))

        if eval_params:
            var_u, _ = _forward_ensemble_stats(
                self.predictor,
                np.asarray(eval_params, dtype=np.float64),
                np.asarray(eval_y0, dtype=np.float64),
                need_mean=False,
            )
        else:
            var_u = np.empty(0, dtype=np.float64)

        planned, planned_sigs = [], set()
        mode_switched = 0
        dose_only_refined = 0
        kept_anchor = 0
        u_ratios = []
        class_transitions = Counter()

        for anchor_u, target_gene, old_mode_str, mode_groups in per_anchor_mode_groups:
            mode_champions = []
            mode_weights = []
            for mode_key, entries in mode_groups.items():
                valid_e, valid_u = [], []
                for is_new, m_name, cand, sig, p_v, y0_v, flat_idx, orig_u in entries:
                    if sig in planned_sigs or sig in domain.seen or sig in domain.forbidden:
                        continue
                    uv = max(float(var_u[flat_idx]) if flat_idx is not None else float(orig_u), 1e-12)
                    valid_e.append((is_new, m_name, cand, sig, p_v, y0_v, uv))
                    valid_u.append(uv)
                if not valid_e:
                    continue
                d_pick = _sample_power_weighted(valid_u, self.tau_dose, self.rng)
                champ = valid_e[d_pick]
                mode_champions.append(champ)
                mode_weights.append(champ[6])

            if not mode_champions:
                continue
            m_pick = _sample_power_weighted(mode_weights, self.tau_mode, self.rng)
            is_new, chosen_mode, cand, sig, p_v, y0_v, chosen_u = mode_champions[m_pick]
            new_index = domain.next_index
            domain.next_index += 1
            if is_new:
                cand = replace(
                    cand,
                    name=f'PoolHybrid-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                if chosen_mode != old_mode_str:
                    mode_switched += 1
                    domain.counts['pool_hybrid_mode_switched'] += 1
                    old_cls = MODE_CLASS.get(old_mode_str, old_mode_str)
                    new_cls = MODE_CLASS.get(chosen_mode, chosen_mode)
                    class_transitions[f'{old_cls}->{new_cls}'] += 1
                else:
                    dose_only_refined += 1
                    domain.counts['pool_hybrid_dose_refined'] += 1
            else:
                kept_anchor += 1
                cand = replace(
                    cand,
                    name=f'PoolAnchor-{new_index:09d}',
                    metadata=dict(cand.metadata, generation_proposal_index=new_index,
                                  source_candidate_id=1_000_000_000 + new_index),
                )
                domain.counts['pool_hybrid_kept_anchor'] += 1

            u_ratios.append(chosen_u / max(anchor_u, 1e-12))
            planned_sigs.add(sig)
            planned.append((cand, sig, p_v, y0_v))

        self._planned = planned
        planned_levels = Counter(int(c.level) for c, _, _, _ in planned)
        self._last_plan_diagnostics = dict(
            method=self.name,
            anchor_source='unlabelled_mutation_pool_only',
            available_pool_anchors=len(avail_pos),
            total_hybrid_variants_scored=len(eval_params),
            tau_mode=self.tau_mode,
            tau_dose=self.tau_dose,
            planned_queries=len(planned),
            planned_levels=dict(sorted(planned_levels.items())),
            mode_switched_count=mode_switched,
            dose_only_refined_count=dose_only_refined,
            kept_anchor_count=kept_anchor,
            top_class_transitions=dict(class_transitions.most_common(8)),
            mean_hybrid_u_gain_ratio=float(np.mean(u_ratios)) if u_ratios else 1.0,
            anchor_u_selected_mean=float(np.mean(u_pool[local_chosen])),
            anchor_u_pool_mean=float(np.mean(u_pool)),
        )

    def _plan_batch(self, domain, plan_count):
        domain.verify()
        if self.name == 'ours_pool_bif_synth':
            self._plan_pool_bif_only(domain, plan_count)
        elif self.name == 'ours_pool_mode_bif_v2_synth':
            self._plan_pool_mode_v2(domain, plan_count)
        else:
            self._plan_pool_mode_v1(domain, plan_count)

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
