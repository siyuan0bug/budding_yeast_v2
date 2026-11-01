"""Train-only auxiliary Pattern estimator with grouped out-of-fold calibration.

An ExtraTrees estimator for acquisition, not a replacement for the FNO and not
registered in models/. Full fitted estimator and preprocessing are persisted.
"""
import hashlib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from .views import contexts,feature_values

SETTINGS=dict(n_estimators=32,max_depth=14,min_samples_leaf=8,n_jobs=2,
              folds=3,temperatures=(.5,.75,1.,1.5,2.,3.),probability_pseudocount=1.)


class PatternClassifier:
    def __init__(self,seed):self.seed=int(seed)

    @staticmethod
    def _temper(p,temperature):
        z=np.log(np.maximum(p,1e-12))/temperature;z-=z.max(1,keepdims=True)
        z=np.exp(z);return (z/z.sum(1,keepdims=True)).astype(np.float32)

    def _probability(self,model,x):
        output=np.ones((len(x),6),dtype=np.float64)
        output[:,np.asarray(model.classes_,dtype=int)]+=model.predict_proba(x)*self.training_rows
        return (output/output.sum(1,keepdims=True)).astype(np.float32)

    def fit(self,inputs,labels):
        from budding_yeast_v2.data.systems.budding.catalog import GROUP_BY_NAME
        self.tokens=tuple(sorted((str(g),str(mode)) for g,group in GROUP_BY_NAME.items() for mode in group['modes']))
        raw=feature_values(inputs,self.tokens);self.mean=raw.mean(0,dtype=np.float64).astype(np.float32)
        self.scale=raw.std(0,dtype=np.float64).astype(np.float32);self.scale[self.scale<1e-6]=1.
        x=(raw-self.mean)/self.scale;self.training_rows=len(x)
        labels=np.asarray(labels,dtype=np.int64)
        if labels.shape!=(len(x),) or not np.isin(labels,np.arange(6)).all():raise ValueError('Invalid train Pattern labels')
        keys=contexts(inputs);unique={key:i for i,key in enumerate(dict.fromkeys(keys))}
        groups=np.asarray([unique[k] for k in keys]);oof=np.empty((len(x),6),dtype=np.float32)
        def model():return ExtraTreesClassifier(n_estimators=SETTINGS['n_estimators'],max_depth=SETTINGS['max_depth'],
            min_samples_leaf=SETTINGS['min_samples_leaf'],class_weight='balanced',n_jobs=SETTINGS['n_jobs'],random_state=self.seed)
        folds=min(SETTINGS['folds'],len(unique));self.temperature=1.;calibration=dict(status='insufficient_groups')
        if folds>=2:
            for train,test in GroupKFold(folds).split(x,labels,groups):
                estimator=model().fit(x[train],labels[train]);oof[test]=self._probability(estimator,x[test])
            losses=[]
            for t in SETTINGS['temperatures']:
                probability=self._temper(oof,t)
                losses.append(float(-np.log(np.maximum(probability[np.arange(len(x)),labels],1e-12)).mean()))
            self.temperature=float(SETTINGS['temperatures'][int(np.argmin(losses))])
            calibration=dict(status='train_group_oof',folds=folds,groups=len(unique),
                temperature=self.temperature,nll=float(min(losses)),accuracy=float((oof.argmax(1)==labels).mean()),
                note='Grouped training folds only; not held-out benchmark accuracy')
        self.estimator=model().fit(x,labels)
        self.report=dict(estimator='ExtraTreesClassifier',training_rows=len(x),seed=self.seed,
            class_counts=np.bincount(labels,minlength=6).tolist(),calibration=calibration,settings=SETTINGS,
            training_ids_sha256=hashlib.sha256(inputs.ids.astype('<i8').tobytes()).hexdigest(),
            input_fields='physical parameters, actual initial states, legal mutation tokens; train-only Z-score')
        return self

    def features(self,inputs):return (feature_values(inputs,self.tokens)-self.mean)/self.scale

    def predict(self,inputs):
        x=self.features(inputs)
        return np.concatenate([self._temper(self._probability(self.estimator,x[i:i+8192]),self.temperature)
                               for i in range(0,len(x),8192)])
