"""Current synthetic sampling policy; biological equations and real rows are exempt."""
from dataclasses import replace
import numpy as np

VERSION = 'exact_genetic_modules_1to3_v1'
STATIC_RATIOS = (.10, .40, .50)
POOL_RATIOS = (.05, .40, .55)
ENVIRONMENT_MODULES = {'MDT', 'Growth'}


def ratios_for(values, *, pool=False):
    ratios = tuple(POOL_RATIOS if pool else STATIC_RATIOS) if values is None else tuple(values)
    array = np.asarray(ratios, dtype=float)
    if (len(array)<3 or not np.isfinite(array).all() or (array<0).any()
            or array[:3].sum()<=0 or array[3:].sum()!=0):
        raise ValueError('Use L1/L2/L3 ratios; all higher-level weights must be zero')
    return tuple(array[:3]/array[:3].sum())


def module_metadata(candidate, system):
    if candidate.is_real:
        return candidate  # Includes the original four-module #42.
    description = candidate.metadata.get('description', candidate.name)
    if system == 'budding':
        from .budding.samplers.rescue import parse_pairs
        from .budding.catalog import GENE_NAMES, require_mutation
        pairs = parse_pairs(description)
        legal = set(GENE_NAMES)
    else:
        from .fission.catalog import SAMPLING_MODULES
        pairs = [tuple(part.split(':',1)) for part in description.split('+') if ':' in part]
        legal = set(SAMPLING_MODULES)
    if not pairs and candidate.metadata.get('pairs'):
        pairs = candidate.metadata['pairs']
    invalid = []
    for gene, mode in pairs:
        if gene not in legal:
            invalid.append((gene, mode))
        elif system == 'budding':
            try:
                require_mutation(gene, mode)
            except ValueError:
                invalid.append((gene, mode))
    if system=='budding':
        from .budding.distributions.ranges import GAL_MC_BASE,GAL_MC_MIN,GAL_MC_MAX
        components=[]
        for gene,mode in pairs:
            if mode!='GAL-MC':continue
            if sum(g==gene for g,_ in pairs)!=1:
                invalid.append((gene,mode))
            copies=float(candidate.overrides.get('ksb2p',float('nan')))/GAL_MC_BASE
            if not np.isfinite(copies) or not np.isclose(copies,round(copies),atol=1e-10,rtol=0) or not GAL_MC_MIN<=copies<=GAL_MC_MAX:
                invalid.append((gene,mode));continue
            if candidate.overrides.get('mdt',90.)!=150.:
                invalid.append((gene,mode))
            components.append(dict(gene=gene,mode=mode,gal_base_strength=GAL_MC_BASE,
                                   copy_number=int(round(copies)),effective_parameter='ksb2p',
                                   effective_strength=float(candidate.overrides['ksb2p'])))
        if components or 'mutation_components' in candidate.metadata:
            metadata=dict(candidate.metadata);metadata.pop('mutation_components',None)
            if components:metadata['mutation_components']=components
            candidate=replace(candidate,metadata=metadata)
    modules = sorted({gene for gene,mode in pairs if gene in legal and gene not in ENVIRONMENT_MODULES})
    environment = sorted({gene for gene,mode in pairs if gene in ENVIRONMENT_MODULES})
    return replace(candidate,metadata=dict(candidate.metadata,pairs=pairs,
        module_names=modules,module_count=len(modules),environment_modules=environment,
        invalid_mutations=invalid,
        requested_level=int(candidate.level),sampling_policy=VERSION))


def valid_candidate(candidate, system):
    row = module_metadata(candidate,system)
    return row if row.is_real or (1<=row.level<=3 and row.metadata['module_count']==row.level
                                and not row.metadata['invalid_mutations']) else None


def draw_candidate(sampler, level, number, report, references=None):
    """Bounded rejection after assignment, conflict resolution AND collision rescue."""
    from .contracts import Candidate
    from .generation import ensure_candidate
    for _ in range(sampler.config.max_attempts or 100):
        overrides,description,refinement,pairs = sampler.sample(int(level))
        candidate=Candidate(number,sampler.candidate_name(description,int(level),number),
            overrides,int(level),refinement=refinement,metadata={'description':description,'pairs':pairs})
        candidate=valid_candidate(candidate,sampler.recipe.system)
        if candidate is None:
            report['module_count_rejected']+=1
            continue
        try:
            candidate=ensure_candidate(candidate,sampler,report,references)
        except RuntimeError as exc:
            if 'Collision rescue exhausted' not in str(exc):raise
            report['collision_exhausted']+=1
            continue
        candidate=valid_candidate(candidate,sampler.recipe.system)
        if candidate is not None:return candidate
        report['module_count_rejected']+=1
    raise RuntimeError(f'Cannot draw an exact L{level} candidate within the attempt budget')
