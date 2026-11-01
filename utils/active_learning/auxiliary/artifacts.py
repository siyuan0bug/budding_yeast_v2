"""Atomic persistence of acquisition-only estimators and pre-query evidence."""
from pathlib import Path
import json
import joblib
import numpy as np


def write_artifacts(engine,directory,index):
    rows=getattr(engine,'_mechanism_artifacts',{}).pop(index,[])
    if not rows:return
    directory=Path(directory)
    classifier=next((r['classifier'] for r in rows if r.get('classifier') is not None),None)
    if classifier is not None:
        temporary=directory/'pattern_classifier.joblib.tmp'
        joblib.dump(classifier,temporary,compress=3);temporary.replace(directory/'pattern_classifier.joblib')
        temporary=directory/'pattern_classifier.json.tmp'
        temporary.write_text(json.dumps(classifier.report,indent=2));temporary.replace(directory/'pattern_classifier.json')
    keys=set(rows[0]['predictions'])
    fields={k:np.concatenate([r['predictions'][k] for r in rows]) for k in keys}
    temporary=directory/'mechanism_predictions.npz.tmp'
    with temporary.open('wb') as f:np.savez_compressed(f,**fields)
    temporary.replace(directory/'mechanism_predictions.npz')
    temporary=directory/'mechanism_predictions.json.tmp'
    temporary.write_text(json.dumps(dict(version='mechanism_query_v1',
        rows=len(fields['candidate_ids']),fields=sorted(keys),
        interpretation='Pre-query predictions for attempted candidates; join oracle outcomes by candidate_ids',
        sources='Synthetic training labels and unlabelled pool inputs only'),indent=2))
    temporary.replace(directory/'mechanism_predictions.json')
    # Cache is only useful inside one acquisition's failure-refill loop.
    if hasattr(engine,'_pattern_classifier_cache'):del engine._pattern_classifier_cache
