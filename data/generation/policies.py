"""Quotas and one collision-rescue policy, independent of ODE implementation."""
import re

import numpy as np


def level_schedule(n, ratios, rng, *, rounding='floor', shuffle=True):
    r = np.asarray(ratios, dtype=float)
    if n < 0 or r.ndim != 1 or len(r) < 1 or not np.isfinite(r).all() or (r < 0).any() or r.sum() <= 0:
        raise ValueError('Sample count and level ratios must be finite, nonnegative and nonempty')
    r = r / r.sum()
    if rounding == 'motif':
        if len(r) != 5:
            raise ValueError('Motif requires five effective level ratios')
        counts = [round(n*x) for x in r[:3]]
        rest = n - sum(counts)
        # Original last 30% was split 70/30 after rounding the first three tiers.
        fourth = round(rest * r[3] / (r[3] + r[4])) if r[3]+r[4] else 0
        counts += [fourth, rest-fourth]
    elif rounding in ('first_remainder','second_remainder'):
        counts = [round(n*x) for x in r]
        counts[0 if rounding=='first_remainder' else min(1,len(counts)-1)] += n-sum(counts)
    else:
        counts = [int(round(n*x) if rounding == 'round' else np.floor(n*x)) for x in r[:-1]]
        counts.append(n-sum(counts))
    if min(counts) < 0:
        raise ValueError('Rounded quotas are negative for this small budget; increase samples or change ratios')
    schedule = np.repeat(np.arange(1,len(counts)+1), counts)
    if shuffle:
        rng.shuffle(schedule)
    return schedule


def parse_pairs(description):
    from ..systems.budding.catalog import GENE_NAMES
    pairs=[]
    for gene,modes in re.findall(r'([A-Za-z0-9_]+)\[([^\]]+)\]',description):
        if gene.lstrip('_')=='MDT' and modes!='MDT':
            continue
        if gene not in GENE_NAMES:
            gene=next((g for g in sorted(GENE_NAMES,key=len,reverse=True) if gene.endswith('_'+g)),gene)
        pairs.extend((gene,mode) for mode in modes.split('+'))
    return pairs


def rescue_budding(overrides, description, backend, rng):
    """Original MA 70/30 rescue; rebuild through the selected assignment/distribution.

    This preserves the established same-profile draw sequence. The caller checks
    the result and retries under a finite overall attempt budget.
    """
    pairs = parse_pairs(description)
    defaults = backend.defaults()
    if rng.random() < .70:
        ko_genes = [g for g,m in pairs if m=='KO' and g in backend.GROUP_BY_NAME]
        if ko_genes:
            target = str(rng.choice(ko_genes))
            modes = [m for m in ('DAmP','ts','pLOF') if m in backend.GROUP_BY_NAME[target]['modes']]
            if modes:
                replacement = str(rng.choice(modes))
                pairs = [(g,replacement if g==target else m) for g,m in pairs]
                new = {}
                for gene,mode in pairs:
                    alpha = backend.sample_scaling_factor(mode,gene,rng=rng) if mode in ('GAL-OE','MC','ts','DAmP','pLOF','db') else 1.
                    backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),new,rng=rng)
                new['mdt'] = overrides.get('mdt',90.)
                return new, '+'.join(f'{g}[{m}]' for g,m in pairs)
    new = dict(overrides)
    new['mdt'] = float(rng.choice([150.,160.]))
    return new, description + f'_MDT[{int(new["mdt"])}]'


def rescue_fission(overrides, description, rng):
    """Change one legal module mode, retaining the Fission strength semantics."""
    from ..systems.fission.catalog import SAMPLING_MODULES
    from ..systems.fission.distributions import _strength
    pairs = [tuple(part.split(':',1)) for part in description.split('+')]
    eligible = [i for i,(module,mode) in enumerate(pairs) if any(m!=mode for m in SAMPLING_MODULES[module]['modes'])]
    if not eligible:
        # Single-mode modules are continuously resampled in their existing band.
        eligible = list(range(len(pairs)))
    if not eligible:
        raise ValueError('No module available for collision rescue')
    i = int(rng.choice(eligible))
    module, old = pairs[i]
    modes = [m for m in SAMPLING_MODULES[module]['modes'] if m != old]
    mode = str(rng.choice(modes)) if modes else old
    pairs[i] = (module,mode)
    new = dict(overrides)
    for key in SAMPLING_MODULES[module]['params']:
        new[key] = _strength(mode,key,rng)
    return new, '+'.join(f'{g}:{m}' for g,m in pairs)
