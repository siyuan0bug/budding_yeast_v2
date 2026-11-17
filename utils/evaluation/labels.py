"""Canonical report identities and descriptive labels, independent of prediction."""
import re
import numpy as np


def get_real_mutations_str(mutant_name,system_id='budding'):
    from budding_yeast_v2.data.systems.registry import real_mutation_rules
    rules=real_mutation_rules(system_id)
    name=str(mutant_name)
    matched=rules.get(name)
    if matched is None and system_id=='budding':
        prefix=re.match(r'^(\d+(?:\.\d+)?)_',name)
        if prefix:
            candidates=[v for k,v in rules.items() if k.split('_',1)[0]==prefix.group(1)]
            if len(candidates)==1:matched=candidates[0]
    if matched is None:
        normal=lambda value:re.sub(r'[\s_]+',' ',value).strip()
        candidates=[v for k,v in rules.items() if normal(k)==normal(name)]
        if len(candidates)==1:matched=candidates[0]
    if matched is None:return 'Canonical mutation definition unavailable'
    if not matched:return 'WT (Default)'
    return ', '.join(f'{key}={value:.4g}' for key,value in matched.items())


def trajectory_labels(targets):
    if np.asarray(targets).shape[1]!=39:return [{} for _ in targets]
    from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
    from budding_yeast_v2.data.systems.budding.phenotypes import analyze_mass_pattern
    groups=pattern_group_metadata(targets)
    return [dict(pattern_label=analyze_mass_pattern(y[0]),
                 pattern_group=str(groups['pattern_group_codes'][i]),
                 observed_division_count=int(groups['observed_division_counts'][i]))
            for i,y in enumerate(targets)]
