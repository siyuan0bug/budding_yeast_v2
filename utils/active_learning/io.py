"""io services extracted from the pre-package implementation."""
from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL
import os
import json
import numpy as np
from budding_yeast_v2.data.label_metadata import trajectory_label_metadata
from budding_yeast_v2.data.systems.sampling_policy import VERSION
from budding_yeast_v2.data.sample_metadata import mutation_metadata


def source_provenance(archive):
    """Read exact saved inputs; explicitly mark absent historical metadata."""
    n = len(archive['mutant_names'])
    keys = set(archive.files if hasattr(archive, 'files') else archive)
    has_initial = 'y_init' in keys
    initial = archive['y_init'] if has_initial else archive['data'][:, :, 0]
    columns = {key: archive[key] for key in (
        'initial_state_source','overrides_json','overrides_available',
        'candidate_metadata_json','simulation_diagnostics_json','simulation_status',
        'candidate_ids','pair_id') if key in keys}
    def value(key, i, default):
        return columns[key][i] if key in columns else default
    return [dict(y_init=np.asarray(initial[i], dtype=np.float64),
        initial_state_source=str(value('initial_state_source', i,
                                      'saved_y_init' if has_initial else 'trajectory_t0')),
        overrides_json=str(value('overrides_json', i, '{}')),
        overrides_available=bool(value('overrides_available', i, 'overrides_json' in columns)),
        candidate_metadata_json=str(value('candidate_metadata_json', i, '{}')),
        simulation_diagnostics_json=str(value('simulation_diagnostics_json', i, '{}')),
        simulation_status=str(value('simulation_status', i, 'success')),
        candidate_id=int(value('candidate_ids', i, -1)),
        pair_id=int(value('pair_id', i, -1))) for i in range(n)]


def pool_provenance(engine, indices, initials=None):
    """Capture accepted rows before the pool arrays are pruned/reindexed."""
    ids = engine.pool.ensure_ids()
    rows = []
    for offset, i in enumerate(indices):
        meta = dict(engine._pool_mass_meta[i])
        meta.setdefault('source_candidate_id', int(ids[i]))
        meta['source_pool'] = engine.al_pool_dataset
        if initials is None:
            initial = engine._pool_initial_physical[i]
            if hasattr(initial, 'numpy'):
                initial = initial.cpu().numpy()
        else:
            initial = initials[offset]
        rows.append(dict(y_init=np.asarray(initial, dtype=np.float64).copy(),
            initial_state_source='source_pool' if initials is None else 'native_simulation',
            overrides_json=json.dumps(engine._pool_ovr[i], sort_keys=True),
            overrides_available=True, candidate_metadata_json=json.dumps(meta, sort_keys=True),
            simulation_diagnostics_json='{}', simulation_status='success',
            candidate_id=int(ids[i]), pair_id=-1))
    return rows


def aligned_provenance(data, provenance):
    """Explicitly fill absent legacy records, separately for each split."""
    rows = list(provenance)
    if len(rows) > len(data):
        raise ValueError('More provenance records than samples')
    # Compatibility with older in-memory callers. Missing overrides stay marked
    # unavailable; never fabricate a biological mutation from a parameter vector.
    for i in range(len(rows), len(data)):
        rows.append(dict(y_init=data[i, :, 0], initial_state_source='trajectory_t0',
                         overrides_available=False))
    return rows


def batch_provenance(batch):
    return aligned_provenance(np.asarray(batch['Y_real']), batch.get('provenance', []))


def provenance_fields(names, data, provenance, flags, system_id):
    """NPZ-ready aligned provenance, including exact ICs independent of float32 Y."""
    if data.ndim != 3 or len(names) != len(data) or len(flags) != len(data):
        raise ValueError('Sample provenance must align with trajectory rows')
    rows = aligned_provenance(data, provenance)
    fields = dict(y_init=np.asarray([r['y_init'] for r in rows], dtype=np.float64),
        initial_state_source=np.asarray([r['initial_state_source'] for r in rows]),
        overrides_available=np.asarray([r.get('overrides_available', False) for r in rows]),
        candidate_ids=np.asarray([r.get('candidate_id', -1) for r in rows], dtype=np.int64),
        pair_id=np.asarray([r.get('pair_id', -1) for r in rows], dtype=np.int64),
        sample_id=np.arange(1, len(names)+1, dtype=np.int64),
        simulation_status=np.asarray([r.get('simulation_status', 'success') for r in rows]))
    for key in ('overrides_json', 'candidate_metadata_json', 'simulation_diagnostics_json'):
        fields[key] = np.asarray([r.get(key, '{}') for r in rows], dtype=str)
    fields.update(mutation_metadata(names, fields['candidate_metadata_json'], flags, system_id))
    if fields['y_init'].shape != data.shape[:2] or not np.isfinite(fields['y_init']).all():
        raise ValueError('Malformed initial states in AL provenance')
    return fields


def source_level_fields(archive):
    """Preserve current metadata, without auditing or rewriting historical files."""
    import re
    names = archive['mutant_names']
    flags = archive.get('is_real',np.array([bool(re.match(r'^\d',str(n))) for n in names]))
    levels = np.asarray(archive.get('levels',[0 if real else int(re.search(r'_L(\d+)',str(name)).group(1))
                        if re.search(r'_L(\d+)',str(name)) else -1 for name,real in zip(names,flags)]),dtype=np.int16)
    requested = np.asarray(archive.get('requested_levels',levels),dtype=np.int16)
    counts = np.asarray(archive.get('module_counts',np.where(flags,-1,levels)),dtype=np.int16)
    if any(values.shape!=(len(names),) for values in (levels,requested,counts)):
        raise ValueError('L metadata must align with dataset rows')
    return levels,requested,counts


def cache_source_metadata(self, dm, archive):
    self._initial_consumed_names=set(map(str,archive.get('initial_consumed_names',[])))
    levels,requested,counts = source_level_fields(archive)
    idx = np.asarray(dm._train_idx_synth,dtype=int)
    rows = source_provenance(archive)
    self._cum_synth_provenance = [rows[i] for i in idx]
    self._benchmark_real_provenance = [rows[i] for i in dm._test_idx_real]
    self._source_dataset_metadata = {key:archive[key] for key in (
        'model_version','benchmark_profile','mutation_profile','distribution_profile',
        'catalog_version','sampling_range_fingerprint','sampling_rng_contract','mass_output_scale') if key in archive}
    self._cum_synth_levels = levels[idx].tolist()
    self._cum_synth_requested_levels = requested[idx].tolist()
    self._cum_synth_module_counts = counts[idx].tolist()
    val = list(getattr(dm,'_val_idx_synth',[]));test = list(getattr(dm,'_test_idx_synth',[]))
    indices = np.asarray(val+test,dtype=int)
    self._heldout_val_count = len(val)
    self._heldout = dict(data=list(archive['data'][indices]),params=list(archive['params'][indices]),
        names=list(archive['mutant_names'][indices]),patterns=list(archive['pattern_labels'][indices]),
        biology=list(archive.get('labels',np.full(len(levels),UNKNOWN_BIOLOGY_LABEL))[indices]),
        provenance=[rows[i] for i in indices],
        levels=levels[indices].tolist(),requested_levels=requested[indices].tolist(),module_counts=counts[indices].tolist())


def configure_snapshot_storage(self, dm, exp_dir):
    from budding_yeast_v2.data.dataset_archive import shared_initial_archive
    self._snapshot_initial_path = (shared_initial_archive(dm.dataset_path, exp_dir) if exp_dir
                                   else os.path.abspath(dm.dataset_path))
    self._snapshot_real_rows = np.asarray(dm._test_idx_real, dtype=np.int64)
    self._snapshot_train_rows = np.asarray(dm._train_idx_synth, dtype=np.int64)
    self._snapshot_heldout_rows = np.asarray(list(getattr(dm, '_val_idx_synth', []))
        + list(getattr(dm, '_test_idx_synth', [])), dtype=np.int64)
    self._snapshot_added_paths = []


def round_record(metrics, round_index, train_count):
    """Empty evaluation splits stay null rather than crashing or becoming zero."""
    record = dict(round=round_index, train_samples=train_count)
    for source, prefix in (('real_mutants', 'real_mutants'), ('synth_test', 'synth_test')):
        block = metrics[source]
        key = next((key for key in block if key.startswith('all_') and key.endswith('_vars')), 'all_variables')
        values = block.get(key) or {}
        record[prefix + '_count'] = int(block.get('count', 0))
        for metric, suffix in (('MAE', 'mae'), ('MSE', 'mse'),
                               ('Relative L2', 'rel_l2'), ('Correlation', 'corr')):
            value = values.get(metric)
            record[prefix + '_' + suffix] = None if value is None else float(value)
    return record


def metric_text(value):
    return 'N/A' if value is None else f'{value:.4f}'


def _save_al_snapshot(self, path, dm):
    """Preserve viability, ABC and neutral groups without inferring missing biology."""
    held = getattr(self,'_heldout',{})
    data = np.asarray(self._benchmark_real_data + self._cum_synth_data + held.get('data',[]), dtype=np.float32)
    biology = self._benchmark_real_biological_labels + self._cum_synth_biological_labels + held.get('biology',[])
    system_id = getattr(dm, 'system_id', 'budding')
    fields = trajectory_label_metadata(data, biology, system_id)
    nr,nt = len(self._benchmark_real_data),len(self._cum_synth_data)
    nv = getattr(self,'_heldout_val_count',0)
    names = np.asarray(self._benchmark_real_names + self._cum_synth_names+held.get('names',[]))
    flags = np.r_[np.ones(nr, dtype=bool), np.zeros(len(data)-nr, dtype=bool)]
    provenance = (aligned_provenance(data[:nr], getattr(self,'_benchmark_real_provenance',[]))
        + aligned_provenance(data[nr:nr+nt], getattr(self,'_cum_synth_provenance',[]))
        + aligned_provenance(data[nr+nt:], held.get('provenance',[])))
    fields.update(provenance_fields(names, data, provenance, flags, system_id))
    fields.update(getattr(self, '_source_dataset_metadata', {}))
    levels = np.asarray([0]*nr+self._cum_synth_levels+held.get('levels',[]),dtype=np.int16)
    requested = np.asarray([0]*nr+self._cum_synth_requested_levels+held.get('requested_levels',[]),dtype=np.int16)
    counts = np.asarray([-1]*nr+self._cum_synth_module_counts+held.get('module_counts',[]),dtype=np.int16)
    if (len(levels)!=len(data) or len(requested)!=len(data) or len(counts)!=len(data)
            or not np.isin(levels[nr:],[1,2,3]).all() or not np.array_equal(levels[nr:],counts[nr:])
            or not np.array_equal(levels[nr:],requested[nr:])):
        raise ValueError('AL snapshot lacks valid aligned exact-L metadata; use a current initial dataset')
    arrays = dict(data=data,
        params=np.asarray(self._benchmark_real_params + self._cum_synth_params+held.get('params',[]), dtype=np.float64),
        mutant_names=names,
        pattern_labels=np.asarray(self._benchmark_real_labels + self._cum_synth_labels+held.get('patterns',[])),
        param_names=self._param_names_cached, system_id=np.array(system_id),
        variable_names=np.asarray(dm.variable_names),var_names=np.asarray(dm.variable_names),
        artifact_kind=np.array('trajectory_dataset'),sampling_policy=np.array(VERSION),levels=levels,requested_levels=requested,module_counts=counts,
        split_train=np.arange(nr,nr+nt),split_val=np.arange(nr+nt,nr+nt+nv),
        split_test=np.arange(nr+nt+nv,len(data)),split_real=np.arange(nr),
        split_seed=np.int64(self.random_seed),split_policy=np.array('al_fixed_holdouts_v1'),
        time_grid=np.linspace(0, float(getattr(dm, 't_max', 210)), data.shape[2]),
        is_real=flags, **fields)
    from budding_yeast_v2.data.dataset_archive import write_dataset_manifest
    if not hasattr(self, '_snapshot_initial_path'):
        # The legacy on-demand initial-simulation path also gets one base block.
        base = os.path.join(os.path.dirname(path), 'initial_block.npz')
        np.savez_compressed(base, **arrays)
        self._snapshot_initial_path = base
        self._snapshot_real_rows = np.arange(nr)
        self._snapshot_train_rows = np.arange(nr, nr+nt)
        self._snapshot_heldout_rows = np.arange(nr+nt, len(data))
        self._snapshot_added_paths = []
    initial = self._snapshot_initial_path
    segments = [(initial, self._snapshot_real_rows), (initial, self._snapshot_train_rows)]
    segments.extend((block, None) for block in self._snapshot_added_paths)
    segments.append((initial, self._snapshot_heldout_rows))
    return write_dataset_manifest(path, arrays, segments,
        provenance=dict(layout='shared_initial_append_only', system_id=system_id))



def save_round(self, dm, bd, exp_dir):
    # 持久化落盘当轮更新后的数据集 .npz 与选拔明细 CSV
    if exp_dir and 'Y_real' in bd and len(bd['Y_real']) > 0:
        ds_dir = os.path.join(exp_dir, 'rounds',f'round_{self._al4pde_al_iter:02d}')
        os.makedirs(ds_dir, exist_ok=True)
        self._cum_synth_data.extend(bd['Y_real'])
        self._cum_synth_params.extend(bd['P_real'])
        self._cum_synth_names.extend(bd['names'])
        self._cum_synth_labels.extend(bd['labels'])
        self._cum_synth_biological_labels.extend(bd.get('biological_labels', [UNKNOWN_BIOLOGY_LABEL] * len(bd['names'])))
        self._cum_synth_provenance = aligned_provenance(
            np.asarray(self._cum_synth_data[:-len(bd['names'])]),
            getattr(self, '_cum_synth_provenance', [])) + batch_provenance(bd)
        for key in ('levels','requested_levels','module_counts'):
            getattr(self,'_cum_synth_'+key).extend(bd[key])

        round_ds_path = os.path.join(ds_dir, 'dataset.json')

        # 保存选样明细清单 CSV
        import csv
        acquisition_dir = os.path.join(exp_dir,'acquisitions',f'after_round_{self._al4pde_al_iter-1:02d}')
        sample_csv_path = os.path.join(acquisition_dir,'selected.csv')
        os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
        if self.strategy == 'ours_response':
            from .strategies.response import write_response_audit
            write_response_audit(self, acquisition_dir, bd)
        if hasattr(self, '_experimental_audits'):
            from .strategies.experimental import write_audit
            write_audit(self, acquisition_dir, bd)
        with open(sample_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Selection_Index','Mutant_Name','Pattern_Label','AL_Iteration','Strategy','levels','requested_levels','module_counts'])
            for s_i, (s_name, s_lbl) in enumerate(zip(bd['names'], bd['labels'])):
                writer.writerow([s_i+1,s_name,s_lbl,self._al4pde_al_iter,self.strategy,bd['levels'][s_i],bd['requested_levels'][s_i],bd['module_counts'][s_i]])
        added_data = np.asarray(bd['Y_real'],dtype=np.float32)
        added_fields = trajectory_label_metadata(added_data,bd.get('biological_labels'),dm.system_id)
        added_fields.update(provenance_fields(bd['names'],added_data,batch_provenance(bd),
            np.zeros(len(bd['names']),dtype=bool),dm.system_id))
        added_fields.update(getattr(self, '_source_dataset_metadata', {}))
        np.savez_compressed(os.path.join(acquisition_dir,'added_samples.npz'),data=added_data,
            params=np.asarray(bd['P_real']),mutant_names=np.asarray(bd['names']),param_names=self._param_names_cached,
            system_id=np.array(dm.system_id),time_grid=np.linspace(0,dm.t_max,np.asarray(bd['Y_real']).shape[-1]),
            variable_names=np.asarray(dm.variable_names),var_names=np.asarray(dm.variable_names),
            sampling_policy=np.array(VERSION),artifact_kind=np.array('trajectory_dataset'),
            is_real=np.zeros(len(bd['names']),dtype=bool),pattern_labels=np.asarray(bd['labels']),
            levels=np.asarray(bd['levels'],dtype=np.int16),requested_levels=np.asarray(bd['requested_levels'],dtype=np.int16),
            module_counts=np.asarray(bd['module_counts'],dtype=np.int16),
            **added_fields)
        self._snapshot_added_paths.append(os.path.abspath(os.path.join(acquisition_dir,'added_samples.npz')))
        self._latest_dataset_path = self._save_al_snapshot(round_ds_path, dm)
        print(f"💾 [AL] 已保存新增样本及第 {self._al4pde_al_iter} 轮完整数据清单: {self._latest_dataset_path}")
        with open(os.path.join(acquisition_dir,'report.json'),'w') as handle:
            json.dump(dict(strategy=self.strategy,accepted=len(bd['names']),
                consumed=len(bd.get('consumed_ids',bd['names'])),candidate_ids=list(map(int,bd.get('candidate_ids',[]))),
                levels={str(l):bd['levels'].count(l) for l in (1,2,3)}),handle,indent=2)
        print(f"📝 [al4pde v2.3] 已将当轮选拔样本清单保存至: {sample_csv_path}")

