"""0-ODE virtual single-module epistasis scoring & native compositional order accounting."""
from collections import Counter, defaultdict
import math
import numpy as np
import torch

from budding_yeast_v2.data.systems.budding.specification import BuddingSystem


def parse_module_blocks(meta, full_ovr=None):
    """Split a mutant's overrides into per-module dict blocks + shared environment overrides (e.g. mdt)."""
    if full_ovr is None:
        full_ovr = meta.get('_effective_overrides', meta.get('original_overrides', {}))
    by_mod = defaultdict(dict)
    touched = set()
    for entry in meta.get('mutation_parameters', ()):
        gene = str(entry['gene'])
        ov = entry.get('overrides', {})
        by_mod[gene].update(ov)
        touched.update(ov.keys())
    mods = sorted(by_mod.keys())
    if not mods:
        # Fallback if mutation_parameters is absent
        pairs = meta.get('pairs', ())
        mods = sorted({str(g) for g, _ in pairs})
    env_ovr = {k: v for k, v in full_ovr.items() if k not in touched}
    return mods, by_mod, env_ovr


def candidate_orders(metadata):
    """Extract compositional mutation order k >= 1 (number of distinct mutated genetic modules)."""
    orders = np.empty(len(metadata), dtype=np.int64)
    for i, meta in enumerate(metadata):
        mods = meta.get('module_names')
        if mods:
            k = len(set(mods))
        else:
            pairs = meta.get('pairs', ())
            if pairs:
                k = len({str(g) for g, _ in pairs})
            else:
                k = int(meta.get('module_count', meta.get('requested_level', 1)))
        orders[i] = max(1, int(k))
    return orders


def allocate_order_budgets(orders, budget):
    """Allocate integer budgets matching exact native pool order proportions (Hamilton largest remainder)."""
    orders = np.asarray(orders, dtype=np.int64)
    n_total = len(orders)
    budget = min(int(budget), n_total)
    if budget <= 0 or n_total == 0:
        return {}
    counts = Counter(int(k) for k in orders)
    keys = sorted(counts.keys())
    quotas = {k: budget * counts[k] / float(n_total) for k in keys}
    alloc = {k: min(counts[k], int(math.floor(quotas[k]))) for k in keys}
    remaining = budget - sum(alloc.values())
    if remaining > 0:
        remainders = sorted(
            keys,
            key=lambda k: (-(quotas[k] - math.floor(quotas[k])), -counts[k], k),
        )
        while remaining > 0:
            progressed = False
            for k in remainders:
                if remaining == 0:
                    break
                if alloc[k] < counts[k]:
                    alloc[k] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
    return alloc


def _canonical_ovr_key(ovr):
    return tuple(sorted((str(k), round(float(v), 12) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in ovr.items()))


def _build_params_y0(backend, defaults, param_keys, ovr):
    _, y0, _, _ = backend.prepare_initial_conditions(ovr)
    params = np.array([ovr.get(k, defaults[k]) for k in param_keys], dtype=np.float64)
    return params, np.asarray(y0, dtype=np.float64)


def _forward_ensemble_stats(predictor, params_arr, y0_arr, *, need_mean=True):
    """Batched 2-member forward pass returning exact Power MSE variance and optional ensemble mean."""
    n = len(params_arr)
    device = predictor.device
    steps = predictor.steps
    bs = predictor.batch_size
    u_power = np.empty(n, dtype=np.float64)
    mean_out = None

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
            stack = torch.stack(preds, dim=0)               # (M, B, V, T)
            mean = stack.mean(dim=0)                        # (B, V, T)
            var = (stack - mean.unsqueeze(0)).square().mean(dim=(0, 2, 3))  # (B,)
            u_power[start:stop] = var.cpu().numpy()
            if need_mean:
                if mean_out is None:
                    mean_out = np.empty((n, mean.shape[1], mean.shape[2]), dtype=np.float32)
                mean_out[start:stop] = mean.cpu().numpy()

    return u_power, mean_out


def compute_cebs_scores(predictor, pool_inputs, *, compute_epistasis=True, chunk_size=1024, backend=None):
    """Compute exact Power uncertainty u_power, 0-ODE nonlinear epistasis magnitude epi_mag, and orders.

    Memory-bounded via global caching of repeated discrete virtual sub-mutants (WT_e, KO, db, CA...)
    and streaming candidate chunks (< 300 MB peak RAM even on an 88,000-sample pool).
    """
    if len(predictor.members) < 2:
        raise ValueError('CEBS requires an ensemble with at least 2 independently trained members')
    n = len(pool_inputs.ids)
    metadata = pool_inputs.metadata
    if not metadata or len(metadata) != n:
        raise ValueError('CEBS requires structured candidate metadata for compositional order/epistasis')

    orders = candidate_orders(metadata)
    if not compute_epistasis:
        u_power = np.maximum(np.asarray(predictor.uncertainty(pool_inputs), dtype=np.float64), 1e-12)
        return dict(u_power=u_power, epi_mag=np.zeros(n, dtype=np.float64), orders=orders)

    if backend is None:
        backend = BuddingSystem()
    defaults = backend.defaults()
    param_keys = backend.PARAM_KEYS

    # Parse virtual sub-mutant keys for every multi-module candidate (k >= 2)
    sample_vkeys = [None] * n
    vkey_to_ovr = {}
    vkey_freq = Counter()

    for i in range(n):
        if orders[i] <= 1:
            continue
        mods, by_mod, env_ovr = parse_module_blocks(metadata[i])
        if len(mods) <= 1 or not all(len(by_mod[g]) > 0 for g in mods):
            continue
        wt_key = _canonical_ovr_key(env_ovr)
        if wt_key not in vkey_to_ovr:
            vkey_to_ovr[wt_key] = dict(env_ovr)
        vkey_freq[wt_key] += 1

        single_keys = []
        for g in mods:
            sub = dict(env_ovr)
            sub.update(by_mod[g])
            sk = _canonical_ovr_key(sub)
            if sk not in vkey_to_ovr:
                vkey_to_ovr[sk] = sub
            vkey_freq[sk] += 1
            single_keys.append(sk)
        sample_vkeys[i] = (wt_key, single_keys)

    # Pre-evaluate and cache all repeated virtual sub-mutants (WT_e + discrete singletons, typically <300 entries!)
    hot_keys = [k for k, cnt in vkey_freq.items() if cnt >= 2]
    # Cap hot cache at 2048 entries (~150 MB max) to guarantee strict memory bound
    if len(hot_keys) > 2048:
        hot_keys = [k for k, _ in vkey_freq.most_common(2048)]
    hot_cache = {}
    if hot_keys:
        hp = np.empty((len(hot_keys), len(param_keys)), dtype=np.float64)
        hy = np.empty((len(hot_keys), pool_inputs.initial.shape[1]), dtype=np.float64)
        for idx, vk in enumerate(hot_keys):
            hp[idx], hy[idx] = _build_params_y0(backend, defaults, param_keys, vkey_to_ovr[vk])
        _, hot_means = _forward_ensemble_stats(predictor, hp, hy, need_mean=True)
        for idx, vk in enumerate(hot_keys):
            hot_cache[vk] = hot_means[idx]

    u_power = np.empty(n, dtype=np.float64)
    epi_mag = np.zeros(n, dtype=np.float64)

    # Process pool in bounded chunks so peak RAM stays < 300 MB
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        c_u, c_mean = _forward_ensemble_stats(
            predictor,
            np.asarray(pool_inputs.params[start:stop], dtype=np.float64),
            np.asarray(pool_inputs.initial[start:stop], dtype=np.float64),
            need_mean=True,
        )
        u_power[start:stop] = np.maximum(c_u, 1e-12)

        # Gather uncached virtual keys needed by this chunk
        cold_keys = []
        cold_seen = set()
        for i in range(start, stop):
            entry = sample_vkeys[i]
            if entry is None:
                continue
            wt_k, s_keys = entry
            for vk in (wt_k, *s_keys):
                if vk not in hot_cache and vk not in cold_seen:
                    cold_seen.add(vk)
                    cold_keys.append(vk)

        cold_cache = {}
        if cold_keys:
            cp = np.empty((len(cold_keys), len(param_keys)), dtype=np.float64)
            cy = np.empty((len(cold_keys), pool_inputs.initial.shape[1]), dtype=np.float64)
            for idx, vk in enumerate(cold_keys):
                cp[idx], cy[idx] = _build_params_y0(backend, defaults, param_keys, vkey_to_ovr[vk])
            _, cold_means = _forward_ensemble_stats(predictor, cp, cy, need_mean=True)
            for idx, vk in enumerate(cold_keys):
                cold_cache[vk] = cold_means[idx]

        # Compute exact nonlinear epistasis magnitude M_epi(S) = mean((y(S) - sum y(M_j) + (k-1) y(WT_e))^2)
        for local_i, i in enumerate(range(start, stop)):
            entry = sample_vkeys[i]
            if entry is None:
                continue
            wt_k, s_keys = entry
            mw = hot_cache[wt_k] if wt_k in hot_cache else cold_cache[wt_k]
            m_add = mw.copy()
            for sk in s_keys:
                ms = hot_cache[sk] if sk in hot_cache else cold_cache[sk]
                m_add += (ms - mw)
            diff = c_mean[local_i] - m_add
            epi_mag[i] = float(np.mean(diff * diff))

    return dict(u_power=u_power, epi_mag=epi_mag, orders=orders)


def epistasis_amplified_scores(u_power, epi_mag, orders, *, lam=1.0):
    """Within each order k >= 2: u_power * (1 + lam * log1p(epi_mag / med_k)); for k=1: u_power."""
    u_power = np.maximum(np.asarray(u_power, dtype=np.float64), 1e-12)
    epi_mag = np.maximum(np.asarray(epi_mag, dtype=np.float64), 0.0)
    orders = np.asarray(orders, dtype=np.int64)
    scores = u_power.copy()
    medians = {}

    if float(lam) <= 0.0:
        return scores, medians

    for k in sorted(set(int(x) for x in orders)):
        if k <= 1:
            continue
        mask = (orders == k)
        vals = epi_mag[mask]
        pos = vals[vals > 0]
        med_k = max(float(np.median(pos)) if len(pos) > 0 else 1.0, 1e-8)
        medians[k] = med_k
        scores[mask] = u_power[mask] * (1.0 + float(lam) * np.log1p(vals / med_k))

    return scores, medians
