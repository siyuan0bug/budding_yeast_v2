"""Frozen shared validation/test/reference loading for generated-input studies."""
import json
from pathlib import Path
import numpy as np

from budding_yeast_v2.data.dataset_archive import TrajectoryStore,file_sha256
from ..benchmark.runner import input_view
from ..benchmark.protocol import effective_signatures
from ..contracts import PoolInputs
from budding_yeast_v2.utils.evaluation.metrics import summarize
from budding_yeast_v2.utils.evaluation.reporting import (save_real_evaluation,export_sample_metrics,save_global_metrics)


def verify_frozen_rules(source,kind,lhs_config,*,diagnostic_only=False):
    """A new generator cannot quietly use different rules than its initial pool."""
    from budding_yeast_v2.data.systems.budding.catalog import CATALOG_VERSION
    from budding_yeast_v2.data.systems.budding.distributions.ranges import RANGE_FINGERPRINT
    required={'sampling_range_fingerprint','catalog_version','generation_config'}
    if diagnostic_only and not required<=source.fields.keys():return
    if not required<=source.fields.keys():raise ValueError('Source pool lacks frozen biological rule metadata')
    for key,expected in [('sampling_range_fingerprint',RANGE_FINGERPRINT),('catalog_version',CATALOG_VERSION)]:
        if str(source.scalar(key).item())!=expected:raise ValueError(f'Frozen biological rule mismatch: {key}')
    if kind=='lhs':
        specification=json.loads(str(source.scalar('generation_config').item()))
        expected=specification.get('lhs_spec_sha256')
        if not expected or lhs_config is None or file_sha256(lhs_config)!=expected:
            raise ValueError('LHS specification must exactly match the frozen source pool')


class FrozenCohorts:
    def __init__(self,root,pool,config,*,diagnostic_only=False,lhs_config=None,verify_generation=True):
        root=Path(root);self.protocol=json.loads((root/'protocol.json').read_text());self.pool=pool
        for name,digest in self.protocol['artifacts'].items():
            if file_sha256(root/name)!=digest:raise ValueError('Shared protocol artifact changed')
        self.validation=[];self.tests=[];self.forbidden=set();self.initial_inputs=None;self.initial_rows=None
        self.source=None
        for name,definition in sorted(self.protocol['pools'].items()):
            source=TrajectoryStore(definition['path'])
            if source.digest!=definition['sha256']:raise ValueError('Frozen source pool changed')
            with np.load(root/f'{name}_inputs.npz',allow_pickle=False) as z:inputs={k:z[k] for k in z.files}
            with np.load(root/f'{name}_split.npz') as z:split={k:z[k] for k in z.files}
            held=np.r_[split['validation'],split['test']]
            self.forbidden.update(effective_signatures(inputs['params'][held],inputs['y_init'][held]))
            roles=['test']
            if name==pool or config.validation_scope=='both':roles.insert(0,'validation')
            values=source.read(np.concatenate([split[r] for r in roles]))['data'];offset=0
            for role in roles:
                ids=split[role];truth=values[offset:offset+len(ids)];offset+=len(ids)
                if role=='validation':self.validation.append((input_view(inputs,ids),truth))
                else:self.tests.append((name,input_view(inputs,ids),truth,
                    inputs.get('mutant_names',np.arange(source.count))[ids]))
            if name==pool:
                if verify_generation:verify_frozen_rules(source,pool,lhs_config,diagnostic_only=diagnostic_only)
                required={'params','param_names','y_init','variable_names','data','time_grid','labels',
                          'pattern_labels','mutation_descriptions','mutation_pairs_json','overrides_json',
                          'requested_levels','module_counts','pattern_group_ids','observed_division_counts'}
                if not diagnostic_only and not required<=source.fields.keys():
                    raise ValueError(f'Formal generation needs complete canonical metadata: {required-source.fields.keys()}')
                self.initial_rows=split['initial'];self.initial_inputs=input_view(inputs,self.initial_rows)
                self.source=source
        if self.source is None:raise ValueError('Unknown source pool')
        self.reference=None;self.monitoring={}
        for name,view,truth,_ in self.tests:
            if name==pool:self.monitoring['synth']=(view,truth)
        if not diagnostic_only:
            ref=self.protocol.get('real_reference')
            if ref is None or file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Need the frozen real reference')
            with np.load(ref['path'],allow_pickle=False) as z:self.reference={k:z[k] for k in z.files}
            ref=self.reference
            self.monitoring['real']=(PoolInputs(np.arange(len(ref['data'])),ref['params'],ref['y_init']),ref['data'])

    def evaluate(self,predictor,directory,round_index,count,*,plots=False,context=None):
        directory=Path(directory);metrics={}
        for name,view,truth,names in self.tests:
            prediction=predictor.physical(view);metrics[name+'_test']=summarize(prediction,truth)
            if name==self.pool:
                export_sample_metrics(prediction,truth,names,directory/'per_synth_test_metrics.csv',
                                      source=name,pool_rows=view.ids,candidate_ids=view.ids)
        if self.reference is not None:
            ref=self.reference;view=PoolInputs(np.arange(len(ref['data'])),ref['params'],ref['y_init'])
            metrics['real_mutants']=save_real_evaluation(predictor.physical(view),ref['data'],ref['mutant_names'],
                directory,plots=plots,write_summary=False,plot_workers=2,context=context)
        save_global_metrics(directory,metrics,pool_name=self.pool,round_index=round_index,training_rows=count)
        return metrics
