"""Budding collision rescue with fixed modules, environment and other doses."""
import re

import numpy as np


def parse_pairs(description):
    from ..catalog import GENE_NAMES
    pairs=[]
    for gene,modes in re.findall(r'([A-Za-z0-9_]+)\[([^\]]+)\]',description):
        if gene.lstrip('_')=='MDT' and modes!='MDT':
            continue
        if gene not in GENE_NAMES:
            gene=next((g for g in sorted(GENE_NAMES,key=len,reverse=True) if gene.endswith('_'+g)),gene)
        pairs.extend((gene,mode) for mode in modes.split('+'))
    return pairs


def rescue_budding(overrides, description, backend, rng):
    """70% KO softening; otherwise redraw one mutation, never the environment.

    Only conflict-free replacements are proposed. Unchanged results mean there
    is no proposal on this attempt; the shared bounded caller retries and then
    redraws a candidate in the same L stratum. Fixed event modes are not jittered.
    """
    from ..real_mutants import effective_inputs

    pairs = parse_pairs(description)
    defaults = backend.defaults()
    original = effective_inputs(overrides, defaults)

    def allowed(index, mode):
        gene = pairs[index][0]
        if gene == 'MDT' or gene not in backend.GROUP_BY_NAME:
            return False
        if mode not in backend.GROUP_BY_NAME[gene]['modes']:
            return False
        if mode in ('GAL-OE','GAL-MC') and overrides.get('mdt', defaults['mdt']) != 150.:
            return False
        if any(g==gene and (mode=='GAL-MC' or m=='GAL-MC') for j,(g,m) in enumerate(pairs) if j!=index):
            return False
        others = set().union(*(backend.touch_set(g,m) for j,(g,m) in enumerate(pairs) if j != index))
        return not (backend.touch_set(gene, mode) & others)

    def redraw(index, mode):
        gene, old_mode = pairs[index]
        # Remove the complete old allele footprint (including ICs/ts branches),
        # then apply the selected allele from WT. All other entries stay exact.
        old_keys = backend.touch_set(gene, old_mode)
        others = set().union(*(backend.touch_set(g,m) for j,(g,m) in enumerate(pairs) if j != index))
        if old_keys & others:
            return None  # Ambiguous ownership: never alter the other mutation.
        new = {key:value for key,value in overrides.items() if key not in old_keys}
        alpha = (backend.sample_scaling_factor(mode,gene,rng=rng)
                 if mode in ('GAL-OE','GAL-MC','MC','ts','DAmP','pLOF','db') else 1.)
        backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),new,rng=rng)
        if np.array_equal(original, effective_inputs(new, defaults)):
            return None
        updated = list(pairs)
        updated[index] = (gene, mode)
        return new, '+'.join(f'{g}[{m}]' for g,m in updated)

    if rng.random() < .70:
        eligible = [(i,[mode for mode in ('DAmP','ts','pLOF') if allowed(i,mode)])
                    for i,(_,mode) in enumerate(pairs) if mode == 'KO']
        eligible = [(i,modes) for i,modes in eligible if modes]
        if eligible:
            index, modes = eligible[int(rng.choice(len(eligible)))]
            result = redraw(index, str(rng.choice(modes)))
            if result is not None:
                return result

    # The 30% arm (also the fallback when no KO can be softened): try the
    # existing allele first, then a legal alternative on that same module.
    indices = [i for i,(gene,_) in enumerate(pairs) if gene != 'MDT']
    for index in rng.permutation(indices):
        gene, mode = pairs[index]
        if gene not in backend.GROUP_BY_NAME:
            continue
        if allowed(index, mode):
            result = redraw(index, mode)
            if result is not None:
                return result
        alternatives = [m for m in backend.GROUP_BY_NAME[gene]['modes']
                        if m != mode and allowed(index,m)]
        for replacement in rng.permutation(alternatives):
            result = redraw(index, str(replacement))
            if result is not None:
                return result
    return dict(overrides), description
