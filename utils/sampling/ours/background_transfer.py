"""Experimental cross-background intervention groups, using queried anchors only.

This prototype is deliberately not registered as a production sampler. It must
pass an actual learning-gain probe before any full benchmark is scheduled.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib

import numpy as np
import torch

from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME, touch_set
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.contracts import Candidate
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate
from budding_yeast_v2.utils.sampling.contracts import PoolInputs
from budding_yeast_v2.utils.sampling.baseline.physics_pre import continuous_mask
from budding_yeast_v2.utils.sampling.ours.mechanistic.systems import BiologicalODE
from budding_yeast_v2.utils.sampling.benchmark.diagnostics import context_key


def group_bank(domain, anchors, *, groups=128, backgrounds=8, seed=42, excluded_contexts=(), focal_genes=None):
    """Apply one canonically drawn intervention to several distinct backgrounds.

    Remove the old focal intervention's entire footprint before replacing it.
    Other modules, their actual strengths, environment, flags and native initial
    preparation stay unchanged. No parameter-space interpolation is used.
    """
    if domain.kind != 'mutation' or backgrounds < 3 or groups < 1:
        raise ValueError('This prototype needs mutation inputs and >=3 backgrounds')
    domain.verify()
    rng = np.random.default_rng(seed)
    backend, defaults = domain.proposals.backend, domain.proposals.defaults
    excluded_contexts = set(excluded_contexts)
    by_gene = defaultdict(list)
    for i, meta in enumerate(anchors.metadata):
        pairs = meta.get('pairs', meta.get('genes', ()))
        for gene in sorted({g for g, _ in pairs}):
            by_gene[gene].append(i)
    if not by_gene:
        raise ValueError('Queried anchors need canonical mutation metadata')
    if focal_genes is not None:
        by_gene = {g: v for g, v in by_gene.items() if g in set(focal_genes)}
        if not by_gene:
            raise ValueError('No anchors for declared focal mechanisms')
    candidates, signatures, vectors, local = [], [], [], set()
    for _ in range(groups * 100):
        if len(candidates) == groups * backgrounds:
            break
        group_id = len(candidates) // backgrounds
        gene = str(rng.choice(sorted(by_gene)))
        mode = str(rng.choice(sorted(GROUP_BY_NAME[gene]['modes'])))
        alpha = backend.sample_scaling_factor(mode, gene, rng=rng)
        edits = {}
        backend.apply_gene_mutation(gene, mode, alpha, defaults, defaults.copy(), edits, rng=rng)
        target_footprint = touch_set(gene, mode)
        if not set(edits) <= target_footprint:
            raise ValueError('Canonical target mutation has an undeclared footprint')
        pending, contexts = [], set()
        for j in rng.permutation(by_gene[gene]):
            meta = anchors.metadata[int(j)]
            pairs = [tuple(p) for p in meta.get('pairs', meta.get('genes', ()))]
            old = [(g, m) for g, m in pairs if g == gene]
            touched = target_footprint | set().union(*(touch_set(g, m) for g, m in old))
            if any(touched & touch_set(g, m) for g, m in pairs if g != gene):
                continue
            original = meta.get('_effective_overrides', meta.get('original_overrides'))
            if original is None:
                raise ValueError('Missing actual anchor overrides')
            mdt = float(original.get('mdt', defaults['mdt']))
            if mode.startswith('GAL') and mdt != 150.:
                continue
            anchor_context = context_key(meta, mdt)
            if anchor_context in contexts:
                continue
            overrides = {k: v for k, v in original.items() if k not in touched}
            overrides.update(edits)
            if float(overrides.get('mdt', defaults['mdt'])) != mdt:
                raise ValueError('Environment must never be edited for intervention generation')
            new_pairs = sorted([(g, m) for g, m in pairs if g != gene] + [(gene, mode)])
            metadata = deepcopy(meta)
            for key in ('_effective_overrides', 'adaptive_path_id', 'adaptive_axis', 'adaptive_anchor_id'):
                metadata.pop(key, None)
            metadata.update(pairs=new_pairs, genes=new_pairs, original_overrides=dict(overrides),
                description='+'.join(f'{g}[{m}]' for g, m in new_pairs),
                rescued=False, transfer_group=group_id, transfer_anchor=int(j),
                transfer_gene=gene, transfer_mode=mode, transfer_alpha=float(alpha),
                transfer_source_modes=[m for _, m in old],
                sampling_paradigm='cross_background_canonical_intervention')
            for key in ('proposal_modules', 'proposal_alleles'):
                metadata.pop(key, None)
            if context_key(metadata, mdt) in excluded_contexts:
                continue
            metadata['proposal_strength_draws'] = [r for r in meta.get('proposal_strength_draws', [])
                if r['gene'] != gene] + [dict(gene=gene, mode=mode, alpha=float(alpha))]
            metadata['mutation_parameters'] = [dict(gene=g, mode=m, overrides={k: overrides[k]
                for k in sorted(touch_set(g, m)) if k in overrides}) for g, m in new_pairs]
            vector = effective_inputs(overrides, defaults)
            signature = hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
            if (np.array_equal(vector, effective_inputs(original, defaults))
                    or domain.proposals.collides(vector) or signature in domain.forbidden
                    or signature in domain.seen or signature in local
                    or any(signature == item[1] for item in pending)):
                continue
            index = domain.next_index
            domain.next_index += 1
            candidate = valid_candidate(Candidate(index + 1, f'Transfer-{index:09d}', overrides,
                len({g for g, _ in new_pairs}), metadata=metadata), 'budding')
            if candidate is None:
                continue
            _, initial, _, _ = backend.prepare_initial_conditions(overrides)
            params = np.array([overrides.get(k, defaults[k]) for k in backend.PARAM_KEYS])
            pending.append((candidate, signature, params, initial))
            contexts.add(anchor_context)
            if len(pending) == backgrounds:
                break
        if len(pending) != backgrounds:
            continue
        for candidate, signature, params, initial in pending:
            candidates.append(candidate)
            signatures.append(signature)
            vectors.append((params, initial))
            local.add(signature)
    if len(candidates) != groups * backgrounds:
        raise RuntimeError(f'Bounded group planner exhausted: {len(candidates)}/{groups*backgrounds}')
    view = PoolInputs(np.arange(len(candidates)), np.array([p for p, _ in vectors]),
        np.array([h for _, h in vectors]), tuple(dict(c.metadata,
        _effective_overrides=dict(c.overrides)) for c in candidates))
    return candidates, signatures, view


@torch.no_grad()
def response_defects(bank, anchors, anchor_truth, predictor):
    """Anchored delta dynamics; no unqueried candidate solution is accepted.

    For each background: q = known anchor + (predicted intervention - predicted
    anchor). Check the local integral identity for that change using native RHS.
    Epistasis is allowed: no equality of responses across backgrounds is assumed.
    """
    system = BiologicalODE('budding')
    anchor_predictions = predictor.physical(anchors)
    scale = np.sqrt(np.mean(np.square(anchor_truth.astype(np.float64)), axis=(0, 2)))
    scale = np.maximum(scale, .01)
    scores = np.full(len(bank.ids), np.nan)
    audit = []
    dt = system.horizon / (anchor_truth.shape[-1] - 1)
    for start, values in predictor.batches(bank, physical=True):
        stop = start + len(values)
        meta = bank.metadata[start:stop]
        ids = np.array([m['transfer_anchor'] for m in meta])
        device = predictor.device
        base = torch.as_tensor(anchor_truth[ids], device=device, dtype=torch.float64)
        delta = values.to(device=device, dtype=torch.float64) - torch.as_tensor(
            anchor_predictions[ids], device=device, dtype=torch.float64)
        counterfactual = base + delta
        p = torch.as_tensor(bank.params[start:stop].copy(), device=device, dtype=torch.float64)
        pa = torch.as_tensor(anchors.params[ids].copy(), device=device, dtype=torch.float64)
        ameta = tuple(anchors.metadata[i] for i in ids)
        times = torch.linspace(0, system.horizon, base.shape[-1], device=device, dtype=torch.float64)
        flows = system.rhs(counterfactual.transpose(1, 2), p, times, meta).transpose(1, 2)
        flows = flows - system.rhs(base.transpose(1, 2), pa, times, ameta).transpose(1, 2)
        valid = continuous_mask(base, window=2) & continuous_mask(counterfactual, window=2)
        valid = valid[:, :-1] & valid[:, 1:]
        defect = delta[:, :, 1:] - delta[:, :, :-1] - dt * .5 * (flows[:, :, 1:] + flows[:, :, :-1])
        finite = torch.isfinite(defect).all(1)
        for i in range(len(values)):
            mask = valid[i]
            # An unscorable sample is logged, never silently assigned zero.
            reason = None
            if int(mask.sum()) < 8:
                reason = 'insufficient_continuous_intervals'
            elif not finite[i, mask].all():
                reason = 'nonfinite_anchored_rhs'
            if reason is None:
                expected_ic_delta = torch.as_tensor(bank.initial[start+i]-anchors.initial[ids[i]],
                    device=device, dtype=torch.float64)
                ic_defect = delta[i, :, 0] - expected_ic_delta
                channel = ((defect[i, :, mask].square().mean(-1)+ic_defect.square()).sqrt()
                    / torch.as_tensor(scale, device=device)).mean()
                scores[start+i] = float(channel)
            audit.append(dict(candidate=int(start+i), continuous_intervals=int(mask.sum()),
                invalid_reason=reason, score=None if reason else float(scores[start+i])))
    return scores, audit


def select_groups(bank, scores, count, *, seed=42, random_control=False):
    """Same builder/control, median background score, bounded context repetition."""
    if count < 1 or len(scores) != len(bank.ids):
        raise ValueError('Invalid group selection inputs')
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, row in enumerate(bank.metadata):
        groups[int(row['transfer_group'])].append(i)
    valid = {g: float(np.median(scores[ids])) for g, ids in groups.items()
        if np.isfinite(scores[ids]).all()}
    if not valid:
        raise ValueError('No fully scoreable cross-background group')
    order = list(valid)
    rng.shuffle(order)
    if not random_control:
        order.sort(key=lambda g: -valid[g])
    selected, contexts = [], defaultdict(int)
    mdt = BiologicalODE('budding').param_names.index('mdt')
    for cap in (1, 2, 4):
        for group in order:
            for i in groups[group]:
                key = context_key(bank.metadata[i], bank.params[i, mdt])
                if i not in selected and contexts[key] < cap:
                    selected.append(i)
                    contexts[key] += 1
                    if len(selected) == count:
                        return np.array(selected, dtype=np.int64), dict(
                            scoreable_groups=len(valid), total_groups=len(groups),
                            selected_contexts=len(contexts), context_cap=cap,
                            selected_groups=len({bank.metadata[j]['transfer_group'] for j in selected}),
                            random_group_control=bool(random_control))
    raise RuntimeError('Not enough unique grouped candidates for requested count')
