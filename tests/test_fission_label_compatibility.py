"""Keep Fission's original labels independent of Budding's new window groups."""
import ast
from dataclasses import replace
import inspect

import numpy as np

from budding_yeast_v2.data import fission_random_sampler as original
from budding_yeast_v2.data.dataset_utils import load_yeast_dataset_universal
from budding_yeast_v2.data.systems.contracts import Candidate, GenerationConfig, SimulationResult
from budding_yeast_v2.data.systems.dataset_io import write_dataset
from budding_yeast_v2.data.systems.fission import constants, phenotypes
from budding_yeast_v2.data.systems.fission.specification import SPEC


def test_fission_label_function_and_threshold_are_identical_to_original():
    def syntax(function):
        return ast.dump(ast.parse(inspect.getsource(function)), include_attributes=False)
    assert syntax(phenotypes.assign_labels) == syntax(original.assign_labels)
    assert constants.TOO_SMALL_M == original.TOO_SMALL_M


def test_fission_labels_survive_export_and_old_new_loading_without_reclassification(tmp_path):
    # Include absent patterns: original export stores None as the string 'None'.
    cases = [(0, .5, [1.]), (0, .50001, [1.]), (2, .8, [.6249]),
             (2, .8, [.625]), (1, .8, [1.])]
    y = np.ones((16, 60))
    records = [original.assign_labels(dict(div_times=list(range(n)), max_MPF=mpf,
        birth_masses=births, status='End'), y) for n, mpf, births in cases]
    labels = np.array([row[0] for row in records])
    patterns = np.array([str(row[1]) for row in records])
    np.testing.assert_array_equal(labels, ['Inviable: G2 Arrest', 'Inviable: M-phase Arrest',
        'Inviable: Too Small', 'Viable', 'Viable'])
    np.testing.assert_array_equal(patterns, ['None', 'None', 'None', 'Pattern_A', 'Pattern_B'])
    params = original.param_vector(original.WT_parameters())
    rows = [SimulationResult(Candidate(i, f'FISSION_{i}', {}, 1), y.copy(), params.copy(),
        y[:, 0].copy(), label, record[1], 'success') for i, (label, record) in enumerate(zip(labels, records))]
    path = tmp_path/'new_fission.npz'
    write_dataset(path, replace(SPEC, steps=60), rows, GenerationConfig('fission_random'),
        dict(mutation_profile='fission', distribution_profile='fission', rng_contract='test'))
    with np.load(path) as archive:
        np.testing.assert_array_equal(archive['labels'], labels)
        np.testing.assert_array_equal(archive['pattern_labels'], patterns)
        assert not any(key.startswith('pattern_group_') for key in archive.files)
        old = tmp_path/'original_format.npz'
        np.savez_compressed(old, **{key:archive[key] for key in ('data', 'params', 'param_names',
            'mutant_names', 'labels', 'pattern_labels', 'levels', 'var_names', 'time_grid')})
    for candidate in (old, path):
        before = candidate.read_bytes()
        loaded = load_yeast_dataset_universal(candidate, return_metadata=True)
        np.testing.assert_array_equal(loaded[4], patterns)
        np.testing.assert_array_equal(loaded[5]['labels'], labels)
        assert not any(key.startswith('pattern_group_') for key in loaded[5])
        assert candidate.read_bytes() == before
