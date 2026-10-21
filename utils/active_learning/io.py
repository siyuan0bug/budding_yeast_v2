"""io services extracted from the pre-package implementation."""
from budding_yeast_v2.data.label_metadata import UNKNOWN_BIOLOGY_LABEL
import os
import numpy as np
from budding_yeast_v2.data.label_metadata import trajectory_label_metadata


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
    data = np.asarray(self._benchmark_real_data + self._cum_synth_data, dtype=np.float32)
    biology = self._benchmark_real_biological_labels + self._cum_synth_biological_labels
    system_id = getattr(dm, 'system_id', 'budding')
    fields = trajectory_label_metadata(data, biology, system_id)
    np.savez_compressed(path, data=data,
        params=np.asarray(self._benchmark_real_params + self._cum_synth_params, dtype=np.float64),
        mutant_names=np.asarray(self._benchmark_real_names + self._cum_synth_names),
        pattern_labels=np.asarray(self._benchmark_real_labels + self._cum_synth_labels),
        param_names=self._param_names_cached, system_id=np.array(system_id),
        time_grid=np.linspace(0, float(getattr(dm, 't_max', 210)), data.shape[2]),
        is_real=np.r_[np.ones(len(self._benchmark_real_data), dtype=bool),
                      np.zeros(len(self._cum_synth_data), dtype=bool)], **fields)



def save_round(self, dm, bd, exp_dir):
    # 持久化落盘当轮更新后的数据集 .npz 与选拔明细 CSV
    if exp_dir and 'Y_real' in bd and len(bd['Y_real']) > 0:
        ds_dir = os.path.join(exp_dir, 'datasets')
        os.makedirs(ds_dir, exist_ok=True)
        self._cum_synth_data.extend(bd['Y_real'])
        self._cum_synth_params.extend(bd['P_real'])
        self._cum_synth_names.extend(bd['names'])
        self._cum_synth_labels.extend(bd['labels'])
        self._cum_synth_biological_labels.extend(bd.get('biological_labels', [UNKNOWN_BIOLOGY_LABEL] * len(bd['names'])))

        round_ds_path = os.path.join(ds_dir, f'al_iter{self._al4pde_al_iter}_train{len(self._cum_synth_data)}.npz')
        self._save_al_snapshot(round_ds_path, dm)
        print(f"💾 [al4pde v2.3] 已将 Iteration {self._al4pde_al_iter} 扩充数据集完整保存至: {round_ds_path}")

        # 保存选样明细清单 CSV
        import csv
        sample_csv_path = os.path.join(exp_dir, 'eval', f'selected_samples_round_{self._al4pde_al_iter}.csv')
        os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
        with open(sample_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Selection_Index', 'Mutant_Name', 'Pattern_Label', 'AL_Iteration', 'Strategy'])
            for s_i, (s_name, s_lbl) in enumerate(zip(bd['names'], bd['labels'])):
                writer.writerow([s_i + 1, s_name, s_lbl, self._al4pde_al_iter, self.strategy])
        print(f"📝 [al4pde v2.3] 已将当轮选拔样本清单保存至: {sample_csv_path}")

