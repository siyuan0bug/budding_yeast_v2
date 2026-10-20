"""Adapters for extracted algorithms; algorithms do not own the export loop."""
import importlib
from dataclasses import replace

import numpy as np
from scipy.stats import qmc

from ..systems.budding.specification import BuddingSystem
from . import common
from ..recipes import resolve_config


class Sampler:
    def __init__(self, recipe, config):
        config=resolve_config(config)
        self.recipe, self.config = recipe, config
        self.rng = np.random.RandomState(config.seed)
        self.backend = None
        self.kernel = None
        self.sobol = None
        common.MODE_FREQ = config.mode_freq
        common._L1_IDX['i'] = 0
        if recipe.system == 'fission':
            from . import fission_random as fission
            self.module = fission
            fission.MODE_FREQ = config.mode_freq
            # Rebuild from the immutable catalog, not an already shuffled list.
            fission._L1_GRID[:] = [(m,mode) for m in fission.SAMPLING_MODULES for mode in fission.SAMPLING_MODULES[m]['modes']]
            np.random.RandomState(config.seed).shuffle(fission._L1_GRID)
            fission._L1_IDX['i'] = 0
            self.rng = np.random.default_rng(config.seed)
            return
        self.backend = BuddingSystem(config.mutation_profile or recipe.mutation,
                                     config.distribution_profile or recipe.distribution)
        kind = recipe.sampler
        module_name = {'random_ko_interval':'random','mass_random':'random','mass_random_dist':'random',
                       'mass_motifs':'mass_motifs','gibbs_sobol':'sobol','pool':'sobol',
                       'kdpp_curvature':'kdpp'}.get(kind, kind)
        self.module = importlib.import_module(f'{__package__}.{module_name}')
        if kind in ('gibbs','gibbs_sobol'):
            self.kernel = self.module.GibbsBoltzmannGraphSampler(temperature=config.temperature,seed=config.seed,ma=self.backend)
        elif kind == 'kdpp':
            self.kernel = self.module.GraphDiffusionKDPP(seed=config.seed,ma=self.backend)
        elif kind == 'kdpp_curvature':
            self.kernel = self.module.SignedDirectedGraphKDPP(seed=config.seed,ma=self.backend)
        elif kind == 'motif':
            self.kernel = self.module.MotifTopologySampler(seed=config.seed,ma=self.backend)
        elif kind == 'mass':
            self.module.MODE_FREQ = config.mode_freq
            self.module._L2_PICKER = common.QuotaPicker(list(self.module.STRUCT_L2),list(self.module.STRUCT_L2.values()))
            self.module._L34_PICKER = common.QuotaPicker(list(self.module.STRUCT_L34),list(self.module.STRUCT_L34.values()))
        elif kind == 'bams':
            self.module._L1_IDX['i'] = 0
        if kind in ('gibbs_sobol','pool'):
            self.sobol = qmc.Sobol(d=16 if kind=='gibbs_sobol' else 8,scramble=True,seed=config.seed)
            self._sobol_points = None
            self._sobol_index = 0

    def _coordinates(self):
        if self._sobol_points is None or self._sobol_index == len(self._sobol_points):
            # Preserve sequential Sobol points; batch size affects throughput only.
            self._sobol_points = self.sobol.random(65536)
            self._sobol_index = 0
        row = self._sobol_points[self._sobol_index]
        self._sobol_index += 1
        return row

    def sample(self, level):
        m, kind, rng, backend = self.module, self.recipe.sampler, self.rng, self.backend
        refinement = None
        if kind in ('random','random_ko_interval'):
            overrides,desc = m.sample_pure_random_mutant(level,rng,
                enable_ko_interval=kind=='random_ko_interval',ko_interval_max=self.config.ko_interval_max,ma=backend)
        elif kind=='mass_random':
            overrides,desc=m.sample_mass_random(level,ma=backend)
        elif kind=='mass_random_dist':
            overrides,desc=m.sample_mass_random_dist(level,ma=backend)
        elif kind in ('mass','mass_motifs'):
            overrides,desc=m.sample_mass_mutant(level,ma=backend)
        elif kind=='gibbs':
            overrides,desc=m.sample_gibbs_mutant(level,self.kernel,rng,ma=backend)
        elif kind=='gibbs_sobol':
            overrides,desc=m.sample_gibbs_sobol_mutant(level,self.kernel,self._coordinates(),rng,ma=backend)
        elif kind=='pool':
            overrides,desc,_=m.sample_pure_random_candidate(level,self._coordinates(),rng,ma=backend)
        elif kind=='kdpp':
            overrides,desc=m.sample_kdpp_mutant(level,self.kernel,rng,ma=backend)
        elif kind=='kdpp_curvature':
            overrides,desc=m.sample_signed_kdpp_mutant(level,self.kernel,rng,ma=backend)
        elif kind=='motif':
            genes,modes=self.kernel.sample_motif(level)
            overrides=m.build_perturbed_sample(genes,modes,rng,ma=backend)
            desc='+'.join(f'{g}_{mode}' for g,mode in zip(genes,modes))
            # Preserve the legacy external name, retain machine-readable edits.
            refinement=None
            return overrides,desc,None,[(g,mode) for g,mode in zip(genes,modes)]
        elif kind=='bams':
            overrides,desc,refinement=m.sample_bams_mutant_task(level,ma=backend)
            if not self.config.enable_bisection:
                refinement=None
        elif kind=='fission_random':
            overrides,desc=m.sample_random_mutant(level,rng)
        else:
            raise ValueError(f'Unsupported candidate adapter: {kind}')
        return overrides,desc,refinement,None


def candidate_name(kind, description, level, number):
    if kind=='fission_random':
        return f'Rand_{description}_L{level}_{number}'.replace(' ','_')
    if kind=='motif':
        return f'Synth_L{level}_{number:05d}_{description}'
    prefix={'random':'RANDOM','random_ko_interval':'RANDOM','pool':'MASS_RAND','gibbs':'GIBBS',
            'gibbs_sobol':'SOBOL','kdpp':'KDPP','bams':'BAMS','wt_blind':'WB-MASS'}.get(kind,'MASS')
    return f'{prefix}_{description}_L{level}_{number}'.replace(' ','_')
