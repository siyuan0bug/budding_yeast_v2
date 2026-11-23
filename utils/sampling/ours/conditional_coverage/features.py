"""Read-only canonical intervention contrasts along supplied physical states.

These are vector-field contrasts, NOT counterfactual solution trajectories.
No finite differences cross events. Initial changes and reset contrasts have
separate coordinates. Each same-gene composite is ablated as one intervention.
"""
from collections import defaultdict
import json

import numpy as np
import torch

from budding_yeast_v2.data.systems.budding.catalog import touch_set
from budding_yeast_v2.data.systems.budding.specification import BuddingSystem
from budding_yeast_v2.utils.sampling.ours.mechanistic.systems import BiologicalODE
from budding_yeast_v2.utils.sampling.baseline.geometry import gaussian_projection


def parse_metadata(arrays):
    result = [json.loads(str(x)) for x in arrays['candidate_metadata_json']]
    for row, value in zip(result, arrays['overrides_json']):
        row['_effective_overrides'] = json.loads(str(value))
    return tuple(result)


def intervention_groups(metadata):
    """Merge overlapping footprints, avoiding ambiguous individual ablations."""
    by_gene = defaultdict(list)
    for gene, mode in metadata.get('pairs', metadata.get('genes', ())):
        if gene == 'MDT':
            continue
        footprint = touch_set(gene, mode)
        if not footprint:
            raise ValueError(f'Undeclared intervention footprint: {gene}/{mode}')
        by_gene[gene].append(mode)
    groups = []
    for gene, modes in sorted(by_gene.items()):
        modes = sorted(set(modes))
        groups.append(([f'{gene}[{"+".join(modes)}]'], set().union(*(touch_set(gene, m) for m in modes))))
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                if groups[i][1] & groups[j][1]:
                    groups[i] = (sorted(groups[i][0]+groups[j][0]), groups[i][1] | groups[j][1])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [('&'.join(tokens), footprint) for tokens, footprint in groups]


def ablate(metadata, footprint, backend=None):
    """Restore only this footprint to defaults; retain actual environment."""
    backend = backend or BuddingSystem()
    defaults = backend.defaults()
    original = metadata['_effective_overrides']
    overrides = {k: v for k, v in original.items() if k not in footprint}
    if overrides.get('mdt', defaults['mdt']) != original.get('mdt', defaults['mdt']):
        raise ValueError('Environmental condition must survive ablation')
    _, initial, _, _ = backend.prepare_initial_conditions(overrides)
    params = np.array([overrides.get(k, defaults[k]) for k in backend.PARAM_KEYS])
    return params, initial, dict(_effective_overrides=overrides)


class MechanismFeatures:
    """Portable feature construction with a currently Budding-specific compiler.

    Scales come from the common revealed initial cohort. Selection features use
    signed log compression, independent of the unchanged model normalization.
    """
    def __init__(self, initial_truth, *, dimension=48, seed=42, device='cpu'):
        self.system = BiologicalODE('budding')
        self.backend = BuddingSystem()
        self.scale = np.maximum(np.sqrt(np.mean(np.square(np.asarray(initial_truth, dtype=np.float64)), (0, 2))), .01)
        self.dimension, self.seed, self.device = dimension, seed, device
        self.projections = {}
        self.reset_cache = {}

    def project(self, values, key):
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError(f'Nonfinite {key} features')
        if key not in self.projections:
            self.projections[key] = gaussian_projection(values.shape[1], self.dimension, self.seed).numpy()
        return values @ self.projections[key]

    def trajectory(self, truth):
        # Four temporal bins, signed mean and RMS; no named biological pattern.
        value = np.sign(truth)*np.log1p(np.abs(truth)/self.scale[None, :, None])
        moments = np.concatenate([np.concatenate((b.mean(-1), np.sqrt(np.mean(b*b, -1))), 1)
                                  for b in np.array_split(value, 4, axis=-1)], 1)
        return self.project(moments, 'trajectory')

    @torch.no_grad()
    def mechanisms(self, params, initial, states, metadata):
        """Return occurrence owner, action token, and conditional effect features.

        Supplied states can be true trajectories (supervised pool selection), or
        explicitly declared anchor-state proxies (generative screening only).
        """
        if not len(params) == len(initial) == len(states) == len(metadata):
            raise ValueError('Unaligned mechanism inputs')
        owners, tokens, reference, ablated_p, ablated_h, ablated_meta = [], [], [], [], [], []
        for i, meta in enumerate(metadata):
            for token, footprint in intervention_groups(meta):
                p, h, m = ablate(meta, footprint, self.backend)
                owners.append(i); tokens.append(token); reference.append(states[i])
                ablated_p.append(p); ablated_h.append(h); ablated_meta.append(m)
        if not owners:
            return np.array([], dtype=int), (), np.empty((0, self.dimension), np.float32)
        owners = np.asarray(owners)
        y = torch.as_tensor(np.asarray(reference).transpose(0, 2, 1), device=self.device, dtype=torch.float64)
        p = torch.as_tensor(np.asarray(params)[owners].copy(), device=self.device, dtype=torch.float64)
        pa = torch.as_tensor(np.asarray(ablated_p), device=self.device, dtype=torch.float64)
        times = torch.linspace(0, self.system.horizon, y.shape[1], device=self.device, dtype=y.dtype)
        meta = tuple(metadata[i] for i in owners)
        effect = self.system.rhs(y, p, times, meta)-self.system.rhs(y, pa, times, tuple(ablated_meta))
        effect = effect.transpose(1, 2).cpu().numpy()*self.system.horizon/self.scale[None, :, None]
        effect = np.sign(effect)*np.log1p(np.abs(effect))
        moments = np.concatenate([np.concatenate((b.mean(-1), np.sqrt(np.mean(b*b, -1))), 1)
                                  for b in np.array_split(effect, 4, axis=-1)], 1)
        delta_ic = (np.asarray(initial)[owners]-np.asarray(ablated_h))/self.scale
        # Native reset maps evaluated at an observed representative state;
        # these are algebraic reset contrasts, not an event-occurrence claim.
        resets = []
        # With mdt fixed, every parameter-dependent difference of the native
        # Budding reset maps is state independent. Cache these differences on
        # CPU; this avoids hundreds of thousands of small CUDA synchronizations.
        reset_keys = ('mdt', 'Vi20_active', 'BUB2_active', 'init_BUB2')
        columns = [self.system.param_names.index(k) for k in reset_keys]
        for j, owner in enumerate(owners):
            pp = np.asarray(params)[owner]
            aa = ablated_p[j]
            key = (tuple(pp[columns]), tuple(aa[columns]))
            if key not in self.reset_cache:
                center = torch.ones(self.system.n, dtype=torch.float64)
                rows = [(self.system.reset(center, torch.tensor(pp), event, meta[j])-
                         self.system.reset(center, torch.tensor(aa), event, ablated_meta[j])).numpy()/self.scale
                        for event in range(4)]
                self.reset_cache[key] = np.concatenate(rows)
            resets.append(self.reset_cache[key])
        raw = np.concatenate((moments, np.sign(delta_ic)*np.log1p(np.abs(delta_ic)),
                              np.sign(resets)*np.log1p(np.abs(resets))), 1)
        return owners, tuple(tokens), self.project(raw, 'mechanism')
