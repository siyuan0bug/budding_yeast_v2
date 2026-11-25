"""Unbiased fresh legal bank + Power-exact MSE Stochastic Weighted D^2 Query Synthesis."""
from collections import Counter
import math
import numpy as np

from ...contracts import PoolInputs
from .metrics import compute_epistemic_mse_views
from .selector import stochastic_dp_select

EPISTEMIC_D2_GEN_METHODS = ('ours_epistemic_d2_synth', 'ours_epistemic_d1_synth')
_DEFAULT_DIST_POWER = {
    'ours_epistemic_d2_synth': 2.0,
    'ours_epistemic_d1_synth': 1.0,
}


class EpistemicD2Synthesis:
    """Zero-covariate-shift legal synthesizer: unbiased 24k fresh bank + Epistemic-MSE D^p."""

    def __init__(self, name, initial, trajectories, seed=42, *,
                 bank_size=24000, dist_power=None, sketch_dim=512, gamma=0.5):
        if name not in EPISTEMIC_D2_GEN_METHODS:
            raise ValueError(f'Unknown Epistemic-D2 synthesis method: {name}')
        self.name = name
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.bank_size = int(bank_size)
        self.dist_power = float(_DEFAULT_DIST_POWER[name] if dist_power is None else dist_power)
        self.sketch_dim = int(sketch_dim)
        self.gamma = float(gamma)

        self.predictor = None
        self.train_inputs = initial
        self.target = len(initial.ids)
        self.round_count = 0
        self._planned = []
        self._last_plan_diagnostics = {}

    def state_dict(self):
        return dict(
            name=self.name,
            seed=self.seed,
            rng_state=self.rng.bit_generator.state,
            round_count=self.round_count,
            last_plan_diagnostics=dict(self._last_plan_diagnostics),
        )

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng_state']
        self.round_count = int(state.get('round_count', 0))
        self._last_plan_diagnostics = dict(state.get('last_plan_diagnostics', {}))
        self._planned = []

    def refresh(self, inputs, truth, predictor, target=None):
        self.train_inputs = inputs
        self.predictor = predictor
        if target is not None:
            self.target = int(target)
        self._planned = []
        self.round_count += 1

    def observe(self, inputs, truth):
        return None

    def _plan_batch(self, domain, plan_count):
        target_bank = max(self.bank_size, plan_count * 4)
        # 100% canonical unbiased legal proposal bank outside the fixed pool (0 ODE calls!)
        candidates, signatures, bank_view = domain.bank(target_bank)
        views = compute_epistemic_mse_views(
            self.predictor,
            bank_view,
            self.train_inputs,
            sketch_dim=self.sketch_dim,
            gamma=self.gamma,
            return_latent=True,
        )
        sel_seed = int(self.rng.integers(0, 2**31 - 1))
        chosen_idx = stochastic_dp_select(
            views['pool_embed'],
            views['train_embed'],
            views['energy'],
            plan_count,
            seed=sel_seed,
            dist_power=self.dist_power,
            device=self.predictor.device,
        )
        self._planned = [
            (candidates[i], signatures[i], bank_view.params[i], bank_view.initial[i])
            for i in chosen_idx
        ]
        bank_levels = Counter(int(c.level) for c in candidates)
        planned_levels = Counter(int(candidates[i].level) for i in chosen_idx)
        energy = views['energy']
        self._last_plan_diagnostics = dict(
            method=self.name,
            dist_power=self.dist_power,
            unbiased_bank_size=len(candidates),
            bank_levels=dict(sorted(bank_levels.items())),
            planned_queries=len(self._planned),
            planned_levels=dict(sorted(planned_levels.items())),
            energy_median=float(np.median(energy)),
            energy_p99_over_p50=float(np.quantile(energy, 0.99) / max(np.median(energy), 1e-12)),
            energy_selected_mean=float(np.mean(energy[chosen_idx])),
            energy_bank_mean=float(np.mean(energy)),
        )

    def propose_batch(self, domain, count):
        remaining = max(count, self.target - len(self.train_inputs.ids))
        if len(self._planned) < count:
            plan_count = max(count, int(math.ceil(remaining * 1.12)))
            self._plan_batch(domain, plan_count)

        chosen_cands, chosen_sigs, chosen_params, chosen_inits = [], [], [], []
        while self._planned and len(chosen_cands) < count:
            cand, sig, p_vec, y0 = self._planned.pop(0)
            if sig in domain.seen or sig in domain.forbidden:
                continue
            chosen_cands.append(cand)
            chosen_sigs.append(sig)
            chosen_params.append(p_vec)
            chosen_inits.append(y0)

        if len(chosen_cands) < count:
            self._plan_batch(domain, max(count * 2, 128))
            while self._planned and len(chosen_cands) < count:
                cand, sig, p_vec, y0 = self._planned.pop(0)
                if sig in domain.seen or sig in domain.forbidden:
                    continue
                chosen_cands.append(cand)
                chosen_sigs.append(sig)
                chosen_params.append(p_vec)
                chosen_inits.append(y0)

        bank = PoolInputs(
            np.arange(len(chosen_cands), dtype=np.int64),
            np.asarray(chosen_params, dtype=np.float64),
            np.asarray(chosen_inits, dtype=np.float64),
        )
        return chosen_cands, chosen_sigs, bank

    def select(self, bank, count):
        n = min(int(count), len(bank.ids))
        return np.arange(n, dtype=np.int64), dict(self._last_plan_diagnostics, batch_served=n)
