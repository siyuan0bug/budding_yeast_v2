"""Canonical mutation compiler for factorial proposal banks, not a new biology.

Exclusion identities stay inside this generation service. Strategies only see
accepted inputs. Every corner reapplies the original mutation assignment and
initial-state rules, then passes the same validity/collision/deduplication gates.
"""
from copy import deepcopy
from dataclasses import replace
from itertools import combinations, product

import numpy as np

from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.real_mutants import effective_inputs
from budding_yeast_v2.data.systems.sampling_policy import valid_candidate
from ..contracts import PoolInputs


def controlled_bank(domain, count):
    import hashlib
    if domain.kind != 'mutation':
        raise ValueError('Factorial controls require canonical mutation proposals')
    domain.verify()
    output, signatures, vectors, local = [], [], [], set()
    backend, defaults = domain.proposals.backend, domain.proposals.defaults
    for _ in range(max(100, count * 30)):
        if len(output) >= count:
            break
        seeds, _, _ = domain.bank(1)
        base = seeds[0]
        pairs = list(base.metadata['pairs'])
        modules = sorted({g for g, _ in pairs})
        entropy = [domain.proposals.seed, domain.next_index, 4242]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        options = list(combinations(modules, 2))
        rng.shuffle(options)
        family = None
        for ga, gb in options:
            controls = {g: [(gene, mode) for gene, mode in pairs if gene == g] for g in (ga, gb)}
            touches = {g: set().union(*(touch_set(*pair) for pair in controls[g])) for g in controls}
            if touches[ga] & touches[gb]:
                continue
            edits, draw_records = {}, {}
            for gene in (ga, gb):
                for branch in (0, 1):
                    changes, draws = {}, []
                    for g, mode in controls[gene]:
                        alpha = backend.sample_scaling_factor(mode, g, rng=rng)
                        backend.apply_gene_mutation(g, mode, alpha, defaults, defaults.copy(), changes, rng=rng)
                        draws.append(dict(gene=g, mode=mode, alpha=float(alpha)))
                    edits[gene, branch], draw_records[gene, branch] = changes, draws
            if any(edits[g, 0] == edits[g, 1] for g in (ga, gb)):
                continue
            family = []
            fixed = {k: v for k, v in base.overrides.items() if k not in touches[ga] | touches[gb]}
            for a, b in product((0, 1), repeat=2):
                overrides = dict(fixed, **edits[ga, a], **edits[gb, b])
                meta = deepcopy(base.metadata)
                meta['original_overrides'] = dict(overrides)
                meta['proposal_strength_draws'] = ([r for r in meta.get('proposal_strength_draws', [])
                                                   if r['gene'] not in (ga, gb)]
                                                  + draw_records[ga, a] + draw_records[gb, b])
                meta.update(controlled_axes=[ga, gb], controlled_corner=[a, b],
                            sampling_paradigm='canonical_mutation_factorial_controls', rescued=False,
                            real_collision=False, controlled_seed_entropy=entropy,
                            source_base_candidate_id=int(base.metadata['source_candidate_id']),
                            source_base_rescued=bool(base.metadata.get('rescued',False)))
                meta.pop('rescue_attempts',None)
                meta['mutation_parameters'] = [dict(gene=g, mode=m,
                    overrides={k: overrides[k] for k in sorted(touch_set(g, m)) if k in overrides}) for g, m in pairs]
                candidate = valid_candidate(replace(base, overrides=overrides, metadata=meta), 'budding')
                if candidate is None:
                    family = None
                    break
                vector = effective_inputs(candidate.overrides, defaults)
                signature = hashlib.sha256(vector.astype('<f8').tobytes()).hexdigest()
                if (domain.proposals.collides(vector) or signature in domain.forbidden
                        or signature in domain.seen or signature in local
                        or signature in {r[1] for r in family}):
                    family = None
                    break
                _, y0, _, _ = backend.prepare_initial_conditions(candidate.overrides)
                params = np.array([candidate.overrides.get(k, defaults[k]) for k in backend.PARAM_KEYS])
                family.append((candidate, signature, params, y0))
            if family:
                break
        if family is None:
            continue
        for candidate, signature, params, y0 in family:
            if len(output) >= count:
                break
            index = domain.next_index
            domain.next_index += 1
            candidate = replace(candidate, name=f'Controlled-mutation-{index:09d}',
                metadata=dict(candidate.metadata, generation_proposal_index=index,
                              source_candidate_id=1_000_000_000 + index))
            output.append(candidate)
            signatures.append(signature)
            vectors.append((params, y0))
            local.add(signature)
            domain.counts['controlled_proposals'] += 1
    if len(output) != count:
        raise RuntimeError(f'Canonical factorial bank exhausted: {len(output)}/{count}')
    return output, signatures, PoolInputs(np.arange(count), np.array([p for p, _ in vectors]),
        np.array([y for _, y in vectors]), tuple(dict(c.metadata, _effective_overrides=c.overrides) for c in output))
