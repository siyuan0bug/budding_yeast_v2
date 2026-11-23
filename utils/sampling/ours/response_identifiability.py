"""Experimental analytic-response omission score, not a validated sampler.

Three explicitly audited independent scalar laws; no claim of automatic or
cross-system extraction. Candidate full trajectories are never accepted here.
"""
import numpy as np
import torch
from budding_yeast_v2.data.normalization import normalize_y
from budding_yeast_v2.data.systems.budding.specification import SPEC

LAWS = {'CDH1': ('CDH1T', 'kscdh', 'kdcdh'),
        'CDC14': ('CDC14T', 'ks14', 'kd14'),
        'NET1': ('NET1T', 'ksnet', 'kdnet')}


def analytic_flow(inputs, gene, times):
    variable, synthesis, degradation = LAWS[gene]
    j = SPEC.variable_names.index(variable)
    a = np.asarray(inputs.params[:, SPEC.param_names.index(synthesis)], float)[:, None]
    b = np.asarray(inputs.params[:, SPEC.param_names.index(degradation)], float)[:, None]
    y0 = np.asarray(inputs.initial[:, j], float)[:, None]
    t = np.asarray(times, float)[None, :]
    integral = np.divide(-np.expm1(-b*t), b,
        out=np.broadcast_to(t, (len(a), t.shape[1])).copy(), where=b != 0)
    return y0*np.exp(-b*t) + a*integral


@torch.no_grad()
def response_omissions(bank, anchors, predictor):
    """Error in predicted change, in checkpoint coordinates, without ratio caps.

    Zero-effect actions are explicitly unscorable. An invariant wrong constant
    predictor has positive omission when a real intervention changes the law.
    Absolute offset errors common to both endpoints cancel by design; this is
    a response diagnostic, not an absolute-accuracy certificate.
    """
    steps = predictor.steps
    times = np.linspace(0., SPEC.t_max if hasattr(SPEC, 't_max') else 210., steps)
    # Budding-only horizon is explicit; do not masquerade as generic ODE code.
    if steps < 2: raise ValueError('Need a full time axis')
    anchor_raw = torch.cat([v for _, v in predictor.batches(anchors)])
    norm = predictor.normalization
    expected_anchors = {g: analytic_flow(anchors, g, times) for g in LAWS}
    scores = np.full(len(bank.ids), np.nan); audit = []
    for start, raw in predictor.batches(bank):
        for local, row in enumerate(bank.metadata[start:start+len(raw)]):
            i = start+local; gene = row['transfer_gene']; anchor = int(row['transfer_anchor'])
            if gene not in LAWS: raise ValueError('Unaudited mechanism')
            j = SPEC.variable_names.index(LAWS[gene][0])
            # Compute only the declared law, using the public encoding function.
            from budding_yeast_v2.utils.sampling.contracts import PoolInputs
            item = PoolInputs(bank.ids[i:i+1], bank.params[i:i+1], bank.initial[i:i+1])
            target = analytic_flow(item, gene, times)
            origin = expected_anchors[gene][anchor:anchor+1]
            z = normalize_y(torch.as_tensor(target, dtype=torch.float32),
                norm.mean_y[:, j, :], norm.std_y[:, j, :], norm.norm_type)
            za = normalize_y(torch.as_tensor(origin, dtype=torch.float32),
                norm.mean_y[:, j, :], norm.std_y[:, j, :], norm.norm_type)
            expected = z[0]-za[0]; effect = float(expected.square().mean().sqrt())
            reason = 'exact_zero_effect' if np.array_equal(target, origin) else None
            if reason is None:
                value = (raw[local, j]-anchor_raw[anchor, j]-expected).square().mean().sqrt()
                if not torch.isfinite(value): raise FloatingPointError('Nonfinite response omission')
                scores[i] = float(value)
            audit.append(dict(candidate=i, gene=gene, effect_RMS_normalized=effect,
                score=None if reason else float(scores[i]), invalid_reason=reason))
    return scores, audit


def matched_mechanism_selection(bank, scores, count, *, seed=42, random_control=False):
    """Same per-mechanism budgets for score/random; no outcome-based mixing."""
    from budding_yeast_v2.utils.sampling.contracts import PoolInputs
    from .background_transfer import select_groups
    selections = []; audit = {}
    for index, gene in enumerate(sorted(LAWS)):
        quota = count//len(LAWS) + int(index < count % len(LAWS))
        rows = np.array([i for i, m in enumerate(bank.metadata) if m['transfer_gene'] == gene])
        view = PoolInputs(bank.ids[rows], bank.params[rows], bank.initial[rows],
            tuple(bank.metadata[i] for i in rows))
        chosen, detail = select_groups(view, np.asarray(scores)[rows], quota,
            seed=seed+index, random_control=random_control)
        selections.append(rows[chosen].tolist()); audit[gene] = dict(quota=quota, **detail)
    # Query reserve consumers take prefixes; concatenating by gene would spend
    # the entire first batch on one mechanism and invalidate the matched control.
    selected = [values[i] for i in range(max(map(len, selections)))
                for values in selections if i < len(values)]
    return np.asarray(selected, dtype=np.int64), audit
