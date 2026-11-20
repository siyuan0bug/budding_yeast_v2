"""Audit zero-rate meanings without changing mutation catalog or the ODE.

All simulations are diagnostic canonical WT-background controls, not a generated
training pool. Numerical success alone is not biological validation of an allele.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from budding_yeast_v2.data.systems.budding.ode import eqns
from budding_yeast_v2.data.systems.budding.simulation import prepare_initial_conditions,simulate_mutant
from budding_yeast_v2.data.systems.budding.pattern_groups import pattern_group_metadata
from budding_yeast_v2.data.systems.budding.states import initial_conds,VARIABLE_NAMES


CASES=(('WT',{},'Reference'),('Dn3_zero',{'Dn3':0.},'Remove effective CLN3 input in this equation'),
    ('CLN3_KO_reference',{'C0':0.},'Canonical CLN3 KO parameter control'),
    ('kdn2_zero',{'kdn2':0.},'Remove CLN2 degradation term, not CLN2 synthesis'),
    ('ksori_zero',{'ksori':0.},'Remove ORI production term'),
    ('ksbud_zero',{'ksbud':0.},'Remove BUD production term'),
    ('ksspn_zero',{'ksspn':0.},'Remove SPN production term'),
    ('eorib5_zero',{'eorib5':0.},'Remove CLB5 contribution to ORI production only'),
    ('ksppx_zero',{'ksppx':0.},'Remove PPX synthesis; canonical initial PPX remains nonzero'),
    ('kdppxpp_zero',{'kdppxpp':0.},'Remove the regulated PPX degradation contribution; not PPX KO'),
    ('kppnetpp_zero',{'kppnetpp':0.},'Remove PPX contribution to NET1 dephosphorylation'),
    ('kasbf_zero',{'kasbf':0.},'Disable the SBF activation contribution; not an observed real allele'),
    ('kamcm_zero',{'kamcm':0.},'Disable the Mcm1 activation contribution; not an observed real allele'),
    ('ki20p_zero',{'ki20p':0.},'Exported input not consumed by the current canonical dynamics'),
    ('ki20pp_zero',{'ki20pp':0.},'Exported input not consumed by the current canonical dynamics'))


def run(task):
    name,override,meaning,horizon,steps=task;started=time.monotonic()
    _,y0,_,_=prepare_initial_conditions(override)
    _,y,bio,pattern=simulate_mutant(name,override,horizon,steps)
    row=dict(name=name,overrides_json=json.dumps(override),equation_interpretation=meaning,
        success=y is not None,seconds=time.monotonic()-started,biological_label=bio,abc_label=pattern)
    if y is not None:
        group=pattern_group_metadata(np.asarray(y)[None])
        row.update(group=str(group['pattern_group_codes'][0]),mass_peak=float(y[0].max()),
            cln2_peak=float(y[1].max()),ori_peak=float(y[33].max()),spn_peak=float(y[35].max()),minimum=float(y.min()))
    return row,y,y0


def boundary_examples():
    examples=[]
    for name,override,zero_states in (
        ('Jspn_zero_with_CLB2_zero',{'Jspn':0.},('CLB2',)),
        ('Jd2c1_zero_with_SIC1_total_zero',{'Jd2c1':0.},('SIC1','C2','C5','SIC1P','C2P','C5P')),
        ('Jn3_and_Dn3_zero',{'Jn3':0.,'Dn3':0.},())):
        p,y,_,_=prepare_initial_conditions(override)
        for state in zero_states:y[VARIABLE_NAMES.index(state)]=0.
        with np.errstate(all='ignore'):derivative=eqns(0.,y,p)
        examples.append(dict(name=name,overrides=override,zero_states=zero_states,
            nonfinite_derivative_indices=np.flatnonzero(~np.isfinite(derivative)).tolist(),
            scope='Equation boundary counterexample, not a simulated trajectory'))
    return examples


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--horizon',type=float,default=210.)
    p.add_argument('--steps',type=int,default=500);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tasks=[(*case,a.horizon,a.steps) for case in CASES]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
        results=list(pool.map(run,tasks))
    rows=[r[0] for r in results];values={r[0]['name']:r[1] for r in results if r[1] is not None}
    reference=values.get('WT');checks={}
    for other in ('ki20p_zero','ki20pp_zero'):
        if reference is not None and other in values:
            checks[other+'_max_abs_difference_from_WT']=float(np.max(np.abs(reference-values[other])))
    if 'Dn3_zero' in values and 'CLN3_KO_reference' in values:
        checks['Dn3_zero_vs_CLN3_KO_max_abs_difference']=float(np.max(np.abs(values['Dn3_zero']-values['CLN3_KO_reference'])))
    payload=dict(cases=rows,equivalent_input_checks=checks,boundary_examples=boundary_examples(),
        solver='Current canonical BDF, native tolerance and timeout',horizon=a.horizon,steps=a.steps,
        numerical_success_is_not_biological_validation=True,training_pool_generated=False)
    (a.output/'zero_audit.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (a.output/'zero_audit.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    good=[r for r in results if r[1] is not None]
    np.savez_compressed(a.output/'diagnostic_trajectories.npz',
        artifact_kind='diagnostic_zero_audit',names=np.asarray([r[0]['name'] for r in good]),
        data=np.asarray([r[1] for r in good]),y_init=np.asarray([r[2] for r in good]),
        time_grid=np.linspace(0,a.horizon,a.steps),variable_names=VARIABLE_NAMES)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
