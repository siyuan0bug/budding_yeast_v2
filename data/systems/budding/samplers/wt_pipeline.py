"""Original WT-Blind pilot and ledger acceptance; common runner owns simulation/export."""
import time
import math
import numpy as np
from .wt_blind import *
from . import wt_blind as w
from ...contracts import Candidate
from .rescue import parse_pairs


def rescue_wt(cand, sampler, report):
    from ...generation import ensure_candidate
    candidate=Candidate(1,'wt_candidate',dict(cand['ovr']),cand['level'],metadata={'description':cand['desc']})
    fixed=ensure_candidate(candidate,sampler,report)
    if fixed.overrides==cand['ovr']:
        return cand
    result=dict(cand,ovr=fixed.overrides,desc=fixed.metadata['description'],mdt=fixed.overrides['mdt'])
    pairs=parse_pairs(result['desc'])
    result['genes_modes']=pairs
    defaults=sampler.backend.defaults()
    edits=[]
    previous={(g,m):value for g,m,value in cand['edits']}
    for gene,mode in pairs:
        value=previous.get((gene,mode))
        if (gene,mode) not in previous:
            keys=w.touch_set(gene,mode)
            ratios=[result['ovr'][k]/defaults[k] for k in sorted(keys) if defaults.get(k,0)>0 and k in result['ovr']]
            value=ratios[0] if ratios else None
        edits.append((gene,mode,value))
    result['edits']=edits
    return result


def prepare_wt_candidates(sampler, config, report):
    n_synthetic=config.n_samples
    if n_synthetic==0:
        from ..real_mutants import real_tasks
        return [Candidate(-i-1,n,o,0,True) for i,(n,o) in enumerate(real_tasks())] if config.include_real else []
    n_pilot=config.pilot_samples
    level_quotas=config.level_ratios or sampler.recipe.ratios
    level_quotas=np.asarray(level_quotas,dtype=float)
    level_quotas=level_quotas/level_quotas.sum()
    nan_cap_frac=config.nan_cap_frac
    backlog_thresh=config.backlog_thresh
    max_attempts_mult=config.max_attempts_mult
    t00=time.time()
    w.MODE_FREQ=config.mode_freq
    w.FOLD_DIST=config.wt_fold_dist
    w._L1_IDX['i']=0
    w._L2_PICKER=w.QuotaPicker(list(w.STRUCT_L2),list(w.STRUCT_L2.values()))
    w._L34_PICKER=w.QuotaPicker(list(w.STRUCT_L34),list(w.STRUCT_L34.values()))
    defaults = get_default_params()
    defaults.update({'init_CDH1T': 1.0, 'init_CDH1': 0.9304992, 'init_BUB2': 0.2})
    param_keys = sorted([k for k in defaults.keys() if k != 'is_mutant_104'])
    print(f'\n[Phase 0] Pilot: {n_pilot} candidates -> D bin edges ...', flush=True)
    GAL_STATS['eff_guided'] = 0
    GAL_STATS['bio_draw'] = 0
    samplers = make_samplers()
    D_mat = np.full((n_pilot, 10), np.nan)
    pilot_nan = 0
    for i in range(n_pilot):
        level = int(np.random.choice([1, 2, 3, 4], p=level_quotas))
        cand = build_candidate(level, samplers, ma=sampler.backend)
        if cand is None:
            continue
        p_eff = defaults.copy()
        p_eff.update(cand['ovr'])
        D = compute_D(p_eff, cand['mdt'])
        D_mat[i] = D
        if np.isnan(D).any() or not cand['gal_conv']:
            pilot_nan += 1
    ledger = Ledger(n_synthetic, nan_cap_frac=nan_cap_frac)
    ledger.fit_pilot(D_mat)
    samplers = make_samplers()
    print(f' -> pilot done in {time.time() - t00:.1f}s | fixed-point NaN rate: {pilot_nan / max(1, n_pilot) * 100:.1f}%', flush=True)
    print(f'\n[Phases 1-3] Sampling {n_synthetic} accepted candidates ...', flush=True)
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
    seen_sigs = set()
    max_attempts = config.max_attempts or max_attempts_mult * n_synthetic
    accept_window = []
    backlog_mode = False
    stats = {'struct': {}, 'gal_nonconv': 0, 'nan_rejected': 0}
    log_every = max(500, n_synthetic // 10)
    while len(accepted) < n_synthetic and attempts < max_attempts:
        attempts += 1
        level = level_pool[len(accepted) % n_synthetic]
        boost, fold_bias_fn = (None, None)
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
                if any((d - 1 in hot_dims for d in dims)):
                    boost[gname] = 5.0
            low_D = any((ledger.counts[d, b] < ledger.cap / 2 for _, d, b in under[:6] if b < ledger.n_bins // 2))
    
            def fold_bias_fn(mode, _low=low_D):
                w = np.ones(N_LOG_BANDS)
                if mode == 'E':
                    return None
                if mode in RESIDUAL_MODES:
                    w[:4] = 3.0 if _low else 1.0
                    w[4:] = 1.0 if _low else 3.0
                else:
                    w[4:] = 3.0 if _low else 1.0
                    w[:4] = 1.0 if _low else 3.0
                return w
        cand = build_candidate(level, samplers, boost=boost, fold_bias_fn=fold_bias_fn, ma=sampler.backend)
        if cand is None:
            continue
        if not cand['gal_conv']:
            stats['gal_nonconv'] += 1
        cand = rescue_wt(cand, sampler, report)
        sig = tuple(sorted(((k, round(v, 10)) for k, v in cand['ovr'].items())))
        if sig in seen_sigs:
            dup_internal += 1
            continue
        cand['sig'] = sig
        p_eff = defaults.copy()
        p_eff.update(cand['ovr'])
        D = compute_D(p_eff, cand['mdt'])
        nan_candidate = np.isnan(D).any() or not cand['gal_conv']
        if nan_candidate:
            if ledger.nan_count < ledger.nan_cap:
                ledger.nan_count += 1
                for g, m in cand['genes_modes']:
                    key = f'{g}[{m}]'
                    ledger.nan_by_mode[key] = ledger.nan_by_mode.get(key, 0) + 1
                accepted.append(dict(cand, D=D, cells=[], dirs={}))
                seen_sigs.add(sig)
            else:
                stats['nan_rejected'] += 1
            accept_window.append(0)
        else:
            cells = ledger.cells(D)
            dirs = direction_signs(cand['edits'])
            any_free = any((ledger.counts[d, b] < ledger.cap for d, b in cells))
            dir_ok = True
            for d, s in dirs.items():
                total_d = ledger.dir_counts[d].sum()
                if total_d > 20 and s in (1, 2) and (ledger.dir_counts[d, s] >= 0.5 * total_d):
                    dir_ok = False
                if s == 0 and ledger.dir_counts[d, 0] >= 0.2 * max(1, total_d):
                    dir_ok = False
            if any_free and dir_ok and cells:
                for d, b in cells:
                    ledger.counts[d, b] += 1
                for d, s in dirs.items():
                    ledger.dir_counts[d, s] += 1
                for d, flag in enumerate(ledger.in_boundary(D)):
                    if flag:
                        ledger.boundary_count[d] += 1
                stats['struct'][cand['struct_class']] = stats['struct'].get(cand['struct_class'], 0) + 1
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
            print(f"  |-- attempts={attempts} accepted={len(accepted)} | dup_blocked={blocked_dup} | dup_internal={dup_internal} | nan={ledger.nan_count} | backlog={('ON' if backlog_mode else 'off')} | acc_rate={(np.mean(accept_window[-100:]) if accept_window else 1):.2f}", flush=True)
    print(f" -> accepted {len(accepted)} in {attempts} attempts ({blocked_dup} real-dup-blocked, {dup_internal} internal-dup-blocked, {stats['nan_rejected']} NaN-capped, {stats['gal_nonconv']} GAL non-converged)", flush=True)

    from ..real_mutants import real_tasks
    from ...factory import candidate_name
    candidates=[Candidate(-i-1,n,o,0,True) for i,(n,o) in enumerate(real_tasks())] if config.include_real else []
    for i,cand in enumerate(accepted,1):
        name=candidate_name('wt_blind',cand['desc'],cand['level'],i)
        candidates.append(Candidate(i,name,cand['ovr'],cand['level'],metadata={'description':cand['desc']}))
    report['sampled_synthetic']=len(accepted)
    report['wt_attempts']=attempts
    report['wt_pilot_nan']=pilot_nan
    report['wt_internal_duplicates']=dup_internal
    # Final ledger counts are rebuilt from exported rows by the runner if needed;
    # failed simulations never enter the saved dataset.
    return candidates
