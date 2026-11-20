"""Post-hoc synthetic acquisition diagnostics; no oracle calls or model changes.

Joins saved pre-query predictions to accepted IDs and their subsequently observed
synthetic groups. These outcomes are analysis only, never acquisition inputs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS=('P1','P2','P3','P4','P5a','P5b')


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--comparison-status',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    runs={t['strategy']:Path(t['experiment_dir']) for t in read(a.manifest)['tasks']}
    if a.comparison_status:
        sources=read(a.comparison_status)['complete']
        for name in ('al4pde_power','al4pde_random','al4pde_lcmd','ours_global_tail'):
            runs[name]=Path(sources[name])
    with np.load(a.initial,allow_pickle=False) as z:
        initial=Counter(z['pattern_group_codes'][z['split_train']].astype(str))
    counts=[];diagnostics=[];confusions=[];physics=[];directions=[]
    for strategy,run in runs.items():
        cumulative=initial.copy()
        for directory in sorted((run/'acquisitions').glob('after_round_*')):
            r=int(directory.name.rsplit('_',1)[1])
            with np.load(directory/'added_samples.npz',allow_pickle=False) as z:
                ids=z['candidate_ids'];codes=z['pattern_group_codes'].astype(str)
                levels=z['module_counts'];actual=np.array([GROUPS.index(c) for c in codes])
            cumulative.update(codes)
            counts.append(dict(strategy=strategy,acquisition=r,accepted=len(ids),
                **{g:int((codes==g).sum()) for g in GROUPS},
                **{f'cumulative_{g}':cumulative[g] for g in GROUPS},
                **{f'L{l}':int((levels==l).sum()) for l in (1,2,3)}))
            if not (directory/'mechanism_predictions.npz').exists():continue
            with np.load(directory/'mechanism_predictions.npz',allow_pickle=False) as z:
                lookup={int(cid):i for i,cid in enumerate(z['candidate_ids'])}
                selected=np.array([lookup[int(cid)] for cid in ids])
                if 'pattern_probability' in z:
                    probability=z['pattern_probability'][selected];votes=z['trajectory_votes'][selected]
                    predicted=probability.argmax(1);voted=votes.argmax(1)
                    entropy=-(probability*np.log(np.maximum(probability,1e-12))).sum(1)
                    confident=probability.max(1)>=.8;unanimous=votes.max(1)==1.
                    meta=read(directory/'pattern_classifier.json')
                    diagnostics.append(dict(strategy=strategy,acquisition=r,accepted=len(ids),
                        classifier_accuracy=float((predicted==actual).mean()),
                        trajectory_vote_accuracy=float((voted==actual).mean()),
                        ensemble_vote_tie_fraction=float((~unanimous).mean()),
                        unanimous_vote_accuracy=float((voted[unanimous]==actual[unanimous]).mean()) if unanimous.any() else None,
                        classifier_high_confidence_count=int(confident.sum()),
                        classifier_high_confidence_accuracy=float((predicted[confident]==actual[confident]).mean()) if confident.any() else None,
                        mean_probability_entropy=float(entropy.mean()),
                        train_group_oof_accuracy=meta['calibration'].get('accuracy'),
                        train_group_oof_nll=meta['calibration'].get('nll'),
                        **{f'expected_{g}':float(probability[:,i].sum()) for i,g in enumerate(GROUPS)}))
                    for x,g in enumerate(GROUPS):
                        for y,h in enumerate(GROUPS):
                            confusions.append(dict(strategy=strategy,acquisition=r,predicted=g,actual=h,
                                count=int(((predicted==x)&(actual==y)).sum())))
                else:
                    scores=z['score'][selected];ic=z['initial_condition_error'][selected]
                    valid=z['valid_time_fraction'][selected];positive=scores>0
                    fraction=np.clip(ic[positive]/scores[positive]**2,0,1)
                    physics.append(dict(strategy=strategy,acquisition=r,accepted=len(ids),
                        positive_score_count=int(positive.sum()),median_score=float(np.median(scores)),
                        median_valid_time=float(np.median(valid)),
                        median_initial_term_fraction=float(np.median(fraction)) if len(fraction) else None,
                        initial_term_above_half_fraction=float((fraction>.5).mean()) if len(fraction) else None,
                        q90_initial_term_fraction=float(np.quantile(fraction,.9)) if len(fraction) else None))
                    for field in ('gene','mode','direction_kind'):
                        for value,count in Counter(z[field][selected].astype(str)).items():
                            directions.append(dict(strategy=strategy,acquisition=r,field=field,value=value,count=count))
    for name,rows in [('synthetic_groups',counts),('classifier_diagnostics',diagnostics),
                      ('classifier_confusion',confusions),('physics_diagnostics',physics),('physics_directions',directions)]:
        pd.DataFrame(rows).to_csv(a.output/(name+'.csv'),index=False)
    (a.output/'interpretation.json').write_text(json.dumps(dict(
        scope='Accepted synthetic candidates only; selected-set accuracy is not unbiased pool accuracy',
        vote_ties='Two-member ties use argmax for the simple accuracy; tie fraction and unanimous accuracy reported separately',
        real_targets_used=False,initial_counts=dict(initial)),indent=2))
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print(pd.DataFrame(physics).to_string(index=False))


if __name__=='__main__':main()
