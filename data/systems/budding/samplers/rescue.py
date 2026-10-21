"""Budding collision description parsing and the original 70/30 rescue policy."""
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
