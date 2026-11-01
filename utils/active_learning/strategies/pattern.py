"""Train-only Pattern balancing and conditional dynamical evidence acquisition."""
from collections import defaultdict
import hashlib
import time
import numpy as np
from ..auxiliary.views import build_view
from ..auxiliary.pattern_classifier import PatternClassifier
from ..auxiliary.phenotype import training_descriptors,predicted_descriptors,DESCRIPTOR_NAMES
from ..auxiliary.coverage import balanced_select,sparse_geometry,coverage_select

STRATEGIES=('ours_pattern_balance','ours_pattern_context','ours_pattern_support')


def choose(view,members,strategy,budget,seed,device,classifier=None):
    """Pure acquisition boundary: no engine/datamodule/benchmark/file paths."""
    true_phi,labels=training_descriptors(view)
    if classifier is None:classifier=PatternClassifier(seed).fit(view.train.inputs,labels)
    probabilities=classifier.predict(view.pool)
    predicted,votes=predicted_descriptors(view,members,device)
    if strategy=='ours_pattern_balance':
        selected,gains=balanced_select(probabilities,labels,budget,seed)
        geometry=dict(objective='sum predicted class probability / current expected class count',
                      approximation='refresh global 1024-point shortlist every 128 sequential picks')
    else:
        with_dynamics=strategy=='ours_pattern_support'
        kernel,support,geometry=sparse_geometry(view,classifier.features(view.pool),classifier.features(view.train.inputs),
            probabilities,labels,predicted if with_dynamics else None,true_phi if with_dynamics else None)
        selected,gains=coverage_select(kernel,support,budget,seed)
        geometry['objective']='sum log(1 + selected kernel support / actual queried training support)'
    report=dict(geometry=geometry,classifier=classifier.report,
        voted_classifier_agreement=float((votes.argmax(1)==probabilities.argmax(1)).mean()),
        observed_train_counts=np.bincount(labels,minlength=6).tolist(),
        predicted_pool_counts=np.bincount(probabilities.argmax(1),minlength=6).tolist(),
        descriptors=DESCRIPTOR_NAMES,no_oracle_before_selection=True)
    predictions=dict(candidate_ids=view.pool.ids[selected],pattern_probability=probabilities[selected],
        trajectory_votes=votes[selected],physical_descriptor=predicted[selected],marginal_gain=gains)
    return selected,report,predictions,classifier


def _pattern_sample(self,trainer,pl_module,dm,device,current_epoch):
    if not self._pool_size:return None
    started=time.monotonic();view=build_view(self,dm)
    seed=int(self.random_seed+1009*self._al4pde_al_iter)
    digest=hashlib.sha256(view.train.inputs.ids.astype('<i8').tobytes()).hexdigest()
    cached=getattr(self,'_pattern_classifier_cache',None)
    classifier=cached[1] if cached and cached[0]==digest else None
    selected,report,predictions,classifier=choose(view,self._ensemble_members,self.strategy,
        self._al4pde_scheduled_num_add(),seed,device,classifier)
    self._pattern_classifier_cache=(digest,classifier)
    if not hasattr(self,'_experimental_audits'):self._experimental_audits=defaultdict(list)
    index=int(self._al4pde_al_iter)
    self._experimental_audits[index].append(dict(strategy=self.strategy,version='conditional_pattern_v1',
        seconds=time.monotonic()-started,feedback=report,budget=len(selected),pool_size=self._pool_size,
        choices=[dict(kind=self.strategy,endpoints=[dict(kind='pool',candidate_id=int(view.pool.ids[i]))],
                      new_queries=1,score=float(gain)) for i,gain in zip(selected,predictions['marginal_gain'])]))
    if not hasattr(self,'_mechanism_artifacts'):self._mechanism_artifacts={}
    self._mechanism_artifacts.setdefault(index,[]).append(dict(classifier=classifier,predictions=predictions))
    print(f'[{self.strategy}] selected={len(selected)} elapsed_s={time.monotonic()-started:.1f}',flush=True)
    return self.make_selection_plan(selected,dm)
