"""Inventory canonical inputs and marginal mutation supports for LHS planning.

This compiles an approved marginal domain, not a production joint sampler. Uses the
production random-distribution and mutation functions without Sobol or any ODE.
Marginal envelopes do not make their independent Cartesian product biological.
"""
import argparse
import ast
from collections import Counter
import csv
import inspect
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME
from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
from budding_yeast_v2.data.systems.budding.distributions.ranges import copy_support
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.data.systems.budding.quotas import enumerate_combinations
from budding_yeast_v2.data.systems.budding import ode, simulation

UNAFFECTED_WINDOW = (0.5, 1.5)


def write_csv(path,records):
    with path.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def parameter_reads():
    """Literal p[key]/p.get(key) reads in the actual RHS and hybrid solver."""
    keys=set()
    for module in (ode,simulation):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if (isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
                and node.value.id=='p' and isinstance(node.slice,ast.Constant)):
                keys.add(node.slice.value)
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='p'
                and node.func.attr=='get' and node.args and isinstance(node.args[0],ast.Constant)):
                keys.add(node.args[0].value)
    return keys


def support_components(backend):
    """Keep branches and discrete atoms instead of merging gaps into one box."""
    defaults=backend.defaults();rows=[]
    for key in backend.PARAM_KEYS:
        rows.append(dict(parameter=key,gene='',mode='WT',branch='',support_kind='point',
            lower=float(defaults[key]),upper=float(defaults[key]),discrete_values_json=json.dumps([float(defaults[key])]),
            distribution_note='WT point; mixture weight not assigned',lhs_weight=''))
    for gene,group in GROUP_BY_NAME.items():
        if gene=='MDT':continue
        for mode in group['modes']:
            branches=(('synthesis',.25),('activity',.75)) if (gene,mode)==('CDC14','ts') else (('',.75),)
            for branch,branch_value in branches:
                assignments=[]
                if mode in ('MC','GAL-MC'):
                    for alpha in copy_support(gene,mode):
                        o={};backend.apply_gene_mutation(gene,mode,float(alpha),defaults,defaults.copy(),o,
                            rng=EndpointRNG(False,branch_value));assignments.append(o)
                else:
                    for upper in (False,True):
                        rng=EndpointRNG(upper,branch_value);alpha=backend.sample_scaling_factor(mode,gene,rng=rng)
                        o={};backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),o,rng=rng);assignments.append(o)
                for key in sorted({key for value in assignments for key in value}):
                    values=sorted({float(row[key]) for row in assignments if key in row})
                    kind='point' if len(values)==1 else 'discrete' if mode in ('MC','GAL-MC') else 'continuous'
                    note=('Integer dosage atoms; P(N) proportional to 1/N' if mode in ('MC','GAL-MC') else
                          'Clipped Normal; endpoint atoms also present' if mode=='DAmP' else
                          'See production mutation/distribution rule; branch preserved')
                    rows.append(dict(parameter=key,gene=gene,mode=mode,branch=branch,support_kind=kind,
                        lower=min(values),upper=max(values),discrete_values_json=json.dumps(values) if kind!='continuous' else '',
                        distribution_note=note,lhs_weight=''))
    return rows


def random_proposal():
    """Exact PRE-REJECTION probabilities for the proposed uniform-set law."""
    cells,levels=enumerate_combinations();sizes=Counter(c['modules'] for c in cells);n=len(sizes)
    inclusion=Counter();alleles=Counter();modes=Counter();gal_mass=Counter()
    for cell in cells:
        probability=1./n/sizes[cell['modules']]
        if any(mode in cell['alleles'] for mode in ('GAL-OE','GAL-MC')):
            gal_mass[cell['level']]+=probability
        for item in cell['alleles'].split(';'):
            gene,state=item.split('[',1);state=state[:-1]
            inclusion[gene]+=probability;alleles[gene,state]+=probability
            for mode in state.split('+'):modes[gene,mode]+=probability
    module_rows=[dict(gene=gene,probability_present=inclusion[gene],
        percent_present=100*inclusion[gene],note='Proposed law; before rescue, deduplication and numerical rejection') for gene in sorted(inclusion)]
    allele_rows=[dict(gene=g,allele=state,probability_given_gene=probability/inclusion[g],
        probability_in_candidate=probability) for (g,state),probability in sorted(alleles.items())]
    mode_rows=[dict(gene=g,mode=mode,probability_given_gene=probability/inclusion[g],probability_in_candidate=probability,
        catalog_reference_weight=GROUP_BY_NAME[g]['modes'][mode],catalog_weight_used=False,
        note='Component presence; sums may exceed one because composites contain multiple modes')
        for (g,mode),probability in sorted(modes.items())]
    assert abs(sum(c['feasible_module_sets']/n for c in levels)-1)<1e-12
    for gene in inclusion:
        assert abs(sum(r['probability_given_gene'] for r in allele_rows if r['gene']==gene)-1)<1e-10
    for row in levels:
        row['proposal_probability']=row['feasible_module_sets']/n
        row['joint_gal_proposal_probability']=gal_mass[row['level']]
    return module_rows,allele_rows,mode_rows,levels


def zero_review(row):
    """Equation interpretation only; never equate all zero parameters with KO."""
    key=row['name']
    if not row['read_by_rhs_or_solver']:
        return 'no_dynamical_effect','Current RHS/reset does not consume this coordinate'
    if row['kind']=='environment':return 'strictly_positive','mdt=0 is undefined in log(2)/mdt'
    if key.startswith(('J','j')):
        return 'retain_positive','Threshold/denominator parameter; zero is not reaction deletion and may be singular'
    if key.startswith('e'):
        return 'regulatory_edge_closure_candidate','Zero removes one weighted regulatory contribution, not the regulator gene'
    if key=='Dn3':
        return 'CLN3_input_closure_candidate','With positive Jn3, Dn3=0 makes the CLN3 auxiliary signal zero'
    if key.startswith('kd'):
        return 'degradation_or_conversion_closure_candidate','May remove degradation, conversion or dissociation; not protein deletion'
    return 'reaction_channel_closure_candidate','May remove an activation/production/inactivation contribution; gene identity not inferred'


def mixed_marginals(backend,components,mode_probabilities,unaffected,level_counts):
    """Approved marginal recipe; no joint sampling or additional zero mass yet."""
    probabilities={(r['gene'],r['mode']):r['probability_in_candidate'] for r in mode_probabilities}
    defaults=backend.defaults();marginals={};unaffected_names={r['name'] for r in unaffected}
    for key in backend.PARAM_KEYS:
        if key in unaffected_names:
            marginals[key]=[dict(family='uniform',weight=1.,lower=UNAFFECTED_WINDOW[0]*defaults[key],upper=UNAFFECTED_WINDOW[1]*defaults[key],
                source='approved_0.5_to_1.5_WT',extra_zero_mass=0.)]
            continue
        if key=='mdt':
            gal=sum(r['joint_gal_proposal_probability'] for r in level_counts)
            marginals[key]=[dict(family='discrete',weight=1.,values=[90.,150.,160.],
                probabilities=[.8*(1-gal),gal+.15*(1-gal),.05*(1-gal)],source='mutation_proposal_environment_marginal')]
            continue
        parts=[]
        for c in components:
            if c['parameter']!=key or c['mode']=='WT':continue
            weight=probabilities[c['gene'],c['mode']]*(.5 if c['branch'] else 1.)
            item=dict(weight=float(weight),lower=c['lower'],upper=c['upper'],
                source_gene=c['gene'],source_mode=c['mode'],source_branch=c['branch'])
            if c['support_kind']=='point':item.update(family='point',value=c['lower'])
            elif c['support_kind']=='discrete':
                copies=copy_support(c['gene'],c['mode']);weights=1./copies;weights/=weights.sum()
                values=json.loads(c['discrete_values_json'])
                if len(values)!=len(weights):raise ValueError('Discrete support collapsed; review marginal')
                item.update(family='discrete',values=values,probabilities=weights.tolist())
            elif c['mode']=='DAmP':
                item.update(family='clipped_normal_scaled',alpha_mean=.2,alpha_std=.05,
                            alpha_clip=[.05,.4],parameter_scale=float(defaults[key]))
            elif (c['mode'] in ('GAL-OE','TAB6-1') or (c['mode']=='db' and c['gene']=='CLB2')
                  or (c['mode']=='ts' and c['gene'] in ('TEM1','ESP1','NET1'))):
                item['family']='log_uniform'
            elif c['mode']=='ts' and c['gene'] not in ('CDC14','CDC20','CDC15','APC_Core'):
                item.update(family='scaled_beta',a=.5,b=2.,parameter_scale=c['upper'])
            else:item['family']='uniform'
            parts.append(item)
        changed=sum(p['weight'] for p in parts)
        if changed>1.+1e-10:raise ValueError(f'Overlapping parameter ownership: {key}')
        if changed<1.:parts.insert(0,dict(family='point',value=float(defaults[key]),weight=1.-changed,source='unmodified_WT'))
        assert abs(sum(p['weight'] for p in parts)-1.)<1e-10
        marginals[key]=parts
    return dict(version='budding_mixed_lhs_marginals_v1',
        status='marginals_configured_joint_sampler_not_implemented',
        positive_window_approved=list(UNAFFECTED_WINDOW),affected_weight_law='pre-rejection random-mutation proposal marginals',
        extra_zero_mass_status='under_review_no_new_zero_mass_added',
        constraints=['Use a common quantile coordinate for init_CDH1T and init_CDH1 to preserve active<=total',
                     'Use a common zero indicator for BUB2_active and init_BUB2',
                     'Prepare actual initial states using canonical prepare_initial_conditions',
                     'Keep thresholds and mdt positive; derive no new genotype labels from zero coordinates',
                     'Invert the full mixture CDF; component concatenation is not a monotone mixture quantile when supports overlap'],
        marginals=marginals)


class EndpointRNG:
    """Reach support endpoints; branch choices are audited independently."""
    def __init__(self,upper,branch):self.upper=upper;self.branch=branch
    def random(self):return self.branch
    def uniform(self,low=0.,high=1.):return high if self.upper else low
    def beta(self,a,b):return float(self.upper)
    def normal(self,loc=0.,scale=1.):return np.inf if self.upper else -np.inf
    def choice(self,values,**kwargs):return values[-1] if self.upper else values[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    backend=BuddingSystem();defaults=backend.defaults()
    values={k:[float(defaults[k])] for k in backend.PARAM_KEYS};origins={k:set() for k in values}
    extra={};alleles=[]
    for gene,group in GROUP_BY_NAME.items():
        if gene=='MDT':continue
        for mode in group['modes']:
            sampled=[];assignments=[]
            for upper in (False,True):
                for branch in (.25,.75):
                    rng=EndpointRNG(upper,branch)
                    alpha=backend.sample_scaling_factor(mode,gene,rng=rng)
                    overrides={};backend.apply_gene_mutation(gene,mode,alpha,defaults,defaults.copy(),overrides,rng=rng)
                    sampled.append(float(alpha));assignments.append(overrides)
                    for key,value in overrides.items():
                        if key in values:
                            values[key].append(float(value));origins[key].add(f'{gene}[{mode}]')
                        else:extra.setdefault(key,set()).add(str(value))
            keys=sorted({key for row in assignments for key in row})
            alleles.append(dict(gene=gene,mode=mode,scaling_support_min=min(sampled),scaling_support_max=max(sampled),
                assigned_keys=';'.join(keys),
                override_envelope_json=json.dumps({key:[min(float(row[key]) for row in assignments if key in row),
                    max(float(row[key]) for row in assignments if key in row)] for key in keys},sort_keys=True)))
    rows=[]
    for key in backend.PARAM_KEYS:
        domain=values[key]
        kind=('initial_override' if key.startswith('init_') else 'environment' if key=='mdt'
              else 'event_activity' if key.endswith('_active') else 'model_parameter')
        rows.append(dict(name=key,wt=defaults[key],kind=kind,mutation_envelope_min=min(domain),mutation_envelope_max=max(domain),
            has_exact_zero=0. in domain,mutations=';'.join(sorted(origins[key])),lhs_lower='',lhs_upper='',lhs_status='pending_review'))
    components=support_components(backend);used=parameter_reads()
    for row in rows:
        key=row['name'];parts=[r for r in components if r['parameter']==key]
        row['has_zero_atom']=any(r['support_kind']!='continuous' and r['lower']==0 for r in parts)
        row['continuous_support_touches_zero']=any(r['support_kind']=='continuous' and r['lower']==0 for r in parts)
        row['read_by_rhs_or_solver']=key in used or row['kind']=='initial_override'
    unchanged=[r for r in rows if r['mutation_envelope_min']==r['mutation_envelope_max']]
    unaffected=[]
    for r in unchanged:
        if r['kind']!='model_parameter':continue
        key=r['name'];value=float(r['wt'])
        family=('threshold_like' if key.startswith(('J','j')) else 'coupling_weight' if key.startswith('e')
                else 'Dn3_factor' if key=='Dn3' else 'kinetic_rate')
        unaffected.append(dict(r,discussion_group=family,
            previous_lhs_lower=.8*value,previous_lhs_upper=1.2*value,
            lhs_lower=UNAFFECTED_WINDOW[0]*value,lhs_upper=UNAFFECTED_WINDOW[1]*value,lhs_status='positive_window_approved_extra_zero_under_review',
            proposal_status='0.5..1.5 WT positive branch approved; extra zero mass under review',
            issue='' if r['read_by_rhs_or_solver'] else 'Exported coordinate not read by current RHS/reset; varying it alone has no direct dynamical effect'))
    for r in unaffected:
        category,meaning=zero_review(r)
        r.update(zero_review_category=category,zero_interpretation=meaning,new_zero_mass=0.,zero_mass_status='not_assigned_under_review')
        original=next(row for row in rows if row['name']==r['name'])
        original.update(lhs_lower=r['lhs_lower'],lhs_upper=r['lhs_upper'],lhs_status=r['lhs_status'])
    modules,allele_probabilities,mode_probabilities,level_counts=random_proposal()
    mixture=mixed_marginals(backend,components,mode_probabilities,unaffected,level_counts)
    (a.output/'mixed_lhs_domain.json').write_text(json.dumps(mixture,indent=2,ensure_ascii=False))
    selected_components=[]
    for key,parts in mixture['marginals'].items():
        for i,part in enumerate(parts):
            selected_components.append(dict(parameter=key,component=i,family=part['family'],weight=part['weight'],
                lower=part.get('lower',part.get('value','')),upper=part.get('upper',part.get('value','')),
                values_json=json.dumps(part.get('values',[])),probabilities_json=json.dumps(part.get('probabilities',[])),
                details_json=json.dumps(part,sort_keys=True)))
    exports=[('parameter_domain_inventory.csv',rows),('mutation_support_inventory.csv',alleles),
        ('variable_parameter_inventory.csv',[r for r in rows if r not in unchanged]),
        ('unaffected_coordinates.csv',unchanged),('unaffected_model_parameters.csv',unaffected),
        ('unaffected_zero_review.csv',unaffected),
        ('parameter_support_components.csv',components),('proposed_gene_probabilities.csv',modules),
        ('selected_mixed_lhs_components.csv',selected_components),
        ('proposed_allele_probabilities.csv',allele_probabilities),('proposed_mode_probabilities.csv',mode_probabilities)]
    for filename,records in exports:write_csv(a.output/filename,records)
    metadata=dict(parameters=len(rows),allele_rules=len(alleles),range_fingerprint=RANGE_FINGERPRINT,
        variable_inputs=sum(r['mutation_envelope_min']!=r['mutation_envelope_max'] for r in rows),
        extra_override_fields={key:sorted(value) for key,value in extra.items()},
        unaffected_coordinates=len(unchanged),unaffected_model_parameters=len(unaffected),
        unaffected_families=dict(Counter(r['discussion_group'] for r in unaffected)),
        unconsumed_model_coordinates=[r['name'] for r in unaffected if not r['read_by_rhs_or_solver']],
        selected_lhs=dict(kind='mixed_marginal_lhs',unaffected_positive_window=list(UNAFFECTED_WINDOW),
            affected_weights='match_random_mutation_proposal_marginals',new_zero_mass='not_assigned_under_review'),
        proposed_random_law=dict(status='proposal_not_implemented',module_sets=sum(r['feasible_module_sets'] for r in level_counts),
            module_set_law='uniform over legal module sets',allele_law='uniform over complete legal assignments within selected set',
            level_counts=level_counts,catalog_weights_used=False,hard_quotas=False),
        environment_rule=dict(with_GAL=150,without_GAL={90:.8,150:.15,160:.05}),
        warnings=[f'{len(unaffected)} unaffected positive bounds approved; added zero masses remain under review',
                  'Mixed marginal components configured; joint constrained LHS sampler not implemented',
                  'Each marginal envelope includes WT and single-rule endpoints; joint independent combinations not validated',
                  'mdt inventory starts at WT; categorical environment rule is recorded separately',
                  'Fixed mutation modes may return alpha=1 yet set rates to zero; use override_envelope_json',
                  'Use parameter_support_components.csv for disjoint intervals, branch supports, atoms and integer values',
                  'Gene/mode probabilities describe a proposed law before collision rescue, uniqueness and solver rejection',
                  'Scaled Beta/clipped Normal/log-uniform/discrete weights are not encoded by bounds alone'],
        ode_calls=0,sobol_calls=0)
    (a.output/'inventory.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print(json.dumps({key:metadata[key] for key in ('parameters','allele_rules','variable_inputs','ode_calls','sobol_calls')}))


if __name__=='__main__':main()
