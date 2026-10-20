"""Descriptive window groups supplementing viability and ABC.

The partition matches the audited user six-pattern executable tree. Its labels
report finite-window observations, without claiming exhaustion, stable limit
cycles, or bifurcations. Input must be in physical, unnormalised units.
"""
import json

import numpy as np

PATTERN_GROUP_VERSION = 'budding_window_groups_v1'
PATTERN_GROUP_CODES = ('P1', 'P2', 'P3', 'P4', 'P5a', 'P5b')
PATTERN_GROUP_LABELS = (
    'P1: Multiple observed divisions',
    'P2: Single observed division',
    'P3: No observed division; low ORI signal',
    'P4: No observed division; elevated ORI signal',
    'P5a: No observed division; terminal Clb2/Esp1 threshold group',
    'P5b: No observed division; high Clb2',
)
PATTERN_GROUP_TO_ABC = (
    'Pattern_A', 'Pattern_B', 'Pattern_C', 'Pattern_C', 'Pattern_C', 'Pattern_C',
)
PATTERN_GROUP_FIELDS = (
    'pattern_group_codes', 'pattern_group_labels', 'pattern_group_ids',
    'observed_division_counts', 'pattern_group_version', 'pattern_group_rules_json',
)
_RULES = dict(
    version=PATTERN_GROUP_VERSION,
    channels=dict(MASS=0, CLB2=2, ESP1=32, ORI=33),
    mass_drop_strictly_greater_than=.1,
    drop_cluster_gap_steps_strictly_greater_than=10,
    drop_reference='previous detected drop, including suppressed drops',
    ori_max_split=.5,
    terminal_clb2_lower_bound=.35,
    terminal_esp1_strict_upper_bound=.1,
    hyper_clb2_max_strict_lower_bound=10.,
    hyper_clb2_end_strict_lower_bound=5.,
    priority=[
        'divisions>=2', 'divisions==1',
        'zero-division Clb2/Esp1 branch, then hyper split',
        'remaining zero-division rows, split by ORI maximum',
    ],
    interpretation='finite-window descriptive groups; no viability or bifurcation claim',
)


def pattern_group_metadata(trajectories, *, batch_size=256):
    """Return NPZ-ready row labels for (N,39,T), (N,T,39), or one trajectory.

    Finite rows are required. Sampling-grid drop detection and float promotion
    reproduce the audited definition, including chain-style ten-step de-jitter.
    """
    y = np.asarray(trajectories)
    if y.ndim == 2:
        y = y[None]
    if y.ndim != 3 or (y.shape[1] == 39) == (y.shape[2] == 39):
        raise ValueError(f'Expected unambiguous Budding N,39,T or N,T,39, got {y.shape}')
    if y.shape[2] == 39:
        y = y.swapaxes(1, 2)
    if y.shape[2] < 2 or batch_size < 1:
        raise ValueError('Need at least two time points and a positive batch_size')

    ids = np.empty(len(y), dtype=np.int8)
    counts = np.empty(len(y), dtype=np.int64)
    for offset in range(0, len(y), batch_size):
        block = y[offset:offset + batch_size]
        if not np.isfinite(block).all():
            raise ValueError('Cannot label nonfinite physical trajectories')
        drops = np.diff(block[:, 0], axis=1) < -.1
        positions = np.where(drops, np.arange(drops.shape[1]), -100000)
        last = np.maximum.accumulate(positions, axis=1)
        previous = np.concatenate(
            (np.full((len(block), 1), -100000), last[:, :-1]), axis=1,
        )
        divisions = (drops & (positions - previous > 10)).sum(1)

        # The audited reference converts extrema to Python floats first.
        bmax = block[:, 2].max(1).astype(np.float64)
        bend = block[:, 2, -1].astype(np.float64)
        eend = block[:, 32, -1].astype(np.float64)
        omax = block[:, 33].max(1).astype(np.float64)
        group = np.where(omax < .5, 2, 3).astype(np.int8)
        terminal = (bend >= .35) | (eend < .1) | (bmax > 10.)
        hyper = (bmax > 10.) | (bend > 5.)
        group[terminal] = 4
        group[terminal & hyper] = 5
        group[divisions == 1] = 1
        group[divisions >= 2] = 0
        ids[offset:offset + len(block)] = group
        counts[offset:offset + len(block)] = divisions

    return dict(
        pattern_group_codes=np.asarray(PATTERN_GROUP_CODES)[ids],
        pattern_group_labels=np.asarray(PATTERN_GROUP_LABELS)[ids],
        pattern_group_ids=ids,
        observed_division_counts=counts,
        pattern_group_version=np.array(PATTERN_GROUP_VERSION),
        pattern_group_rules_json=np.array(json.dumps(_RULES, sort_keys=True)),
    )


def validate_stored_pattern_groups(stored, calculated):
    """Reject incomplete or stale supplementary group fields."""
    present = set(PATTERN_GROUP_FIELDS) & set(stored)
    if not present:
        return
    if present != set(PATTERN_GROUP_FIELDS):
        raise ValueError('Incomplete supplementary pattern-group metadata')
    for key in PATTERN_GROUP_FIELDS:
        if key == 'pattern_group_rules_json':
            if json.loads(str(stored[key])) != _RULES:
                raise ValueError('Unsupported pattern-group rules')
        elif not np.array_equal(stored[key], calculated[key]):
            raise ValueError(f'Stored {key} disagrees with physical trajectories or supported version')


def abc_pattern_ids(pattern_labels):
    """Map actual legacy ABC labels to their existing 0/1/2 training targets."""
    prefixes = [str(label).split(':', 1)[0] for label in pattern_labels]
    lookup = {'Pattern_A': 0, 'Pattern_B': 1, 'Pattern_C': 2}
    try:
        return np.asarray([lookup[p] for p in prefixes], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f'Cannot encode an unknown ABC pattern: {exc.args[0]}') from exc
