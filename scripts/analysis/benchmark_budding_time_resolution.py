"""Reproducible sampling audit and FNO pilot; never changes production defaults."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.interpolate import interp1d

from budding_yeast_v2.data.systems.budding import simulation as sim
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.budding.parameters import get_default_params, PARAM_KEYS, _defaults


def trajectory(name, overrides):
    segments = []
    original = sim.solve_ivp

    def capture(*args, **kwargs):
        result = original(*args, **kwargs, dense_output=True)
        segments.append(result)
        return result

    start = time.perf_counter()
    with patch.object(sim, 'solve_ivp', capture):
        result = sim.simulate_mutant(name, overrides, ns=21001)
    elapsed = time.perf_counter() - start
    if result[1] is None:
        return None
    return result[1], segments, elapsed


def piecewise(segments, times):
    out = np.empty((39, len(times)))
    # Overwrite shared endpoints with the following segment's post-reset value.
    for s in segments:
        mask = (times >= s.t[0]) & (times <= s.t[-1])
        out[:, mask] = s.sol(times[mask])
    return out


def summarize(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), p95=float(np.quantile(a, .95)), max=float(a.max()))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--synthetic', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--device', default='cuda:2')
    ap.add_argument('--event-only', action='store_true', help='Audit brief LTE1 pulses and solver invariance only')
    ap.add_argument('--benchmark-only', action='store_true', help='Reuse output/synthetic.npz for FNO runs')
    ap.add_argument('--pilot-lengths', type=int, nargs='+', default=[128, 256, 500])
    ap.add_argument('--report-name', default='fno.json', help='JSON filename within output directory')
    ap.add_argument('--lengths', type=int, nargs='+', default=[64, 128, 211, 256, 500, 1000])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.event_only:
        event_audit(args)
        return
    if args.benchmark_only:
        with np.load(args.output/'synthetic.npz') as data:
            samples = list(zip(data['params'], data['initial'], data['y']))
        benchmark(args, samples, np.linspace(0, 210, 21001))
        return
    rng = np.random.default_rng(args.seed)
    grid = np.linspace(0, 210, 21001)
    stats = {n: {'log_rel_l2': [], 'core_log_rel_l2': [], 'event_nrmse': [],
                 'smooth_nrmse': [], 'event_midpoint_error_min': []} for n in args.lengths}
    failures, solve_times, native_steps, biases, event_widths = [], [], [], [], []
    core = [0, 1, 2, 3, 4, 20, 22, 33, 35]
    defaults = get_default_params()
    keys = ['mdt', 'ksn2pp', 'ksb2pp', 'ksb5pp', 'ksc1pp', 'ks20pp', 'kscdh', 'kdb2pp']
    synthetic = []
    cases = [(f'real_{key}', value[1], None) for key, value in REAL_MUTANTS.items()]
    for i in range(args.synthetic):
        factors = np.exp(rng.uniform(np.log(.7), np.log(1.3), len(keys)))
        overrides = {k: defaults[k] * f for k, f in zip(keys, factors)}
        cases.append((f'synthetic_{i}', overrides, factors))
    for index, (name, overrides, factors) in enumerate(cases):
        run = trajectory(name, overrides)
        if run is None:
            failures.append(name)
            continue
        y, segments, elapsed = run
        solve_times.append(elapsed)
        native_steps.append(sum(len(s.t)-1 for s in segments))
        events = np.asarray([s.t[-1] for s in segments[:-1]])
        event_mask = np.zeros(len(grid), dtype=bool)
        for t in events:
            event_mask |= np.abs(grid-t) <= 1
        scale = np.maximum(np.ptp(y, axis=1), .1)[:, None]
        logy = np.log1p(np.maximum(y, 0))
        for n in args.lengths:
            t = np.linspace(0, 210, n)
            # Sample original native interpolation, not an already downsampled grid.
            native_t = np.concatenate([segments[0].t] + [s.t[1:] for s in segments[1:]])
            native_y = np.concatenate([segments[0].y] + [s.y[:, 1:] for s in segments[1:]], axis=1)
            coarse = interp1d(native_t, native_y, axis=1)(t)
            recon = interp1d(t, coarse, axis=1)(grid)
            err = np.log1p(np.maximum(recon, 0))-logy
            stats[n]['log_rel_l2'].append(float(np.linalg.norm(err)/np.linalg.norm(logy)))
            stats[n]['core_log_rel_l2'].append(float(np.linalg.norm(err[core])/np.linalg.norm(logy[core])))
            squared = ((recon-y)/scale)**2
            if event_mask.any():
                stats[n]['event_nrmse'].append(float(np.sqrt(squared[:, event_mask].mean())))
            stats[n]['smooth_nrmse'].append(float(np.sqrt(squared[:, ~event_mask].mean())))
            for te in events:
                j = np.clip(np.searchsorted(t, te), 1, n-1)
                stats[n]['event_midpoint_error_min'].append(float(abs((t[j-1]+t[j])/2-te)))
        exact = piecewise(segments, grid)
        biases.append(float(np.linalg.norm(np.log1p(np.maximum(exact, 0))-logy)/np.linalg.norm(logy)))
        for before, after in zip(segments[:-1], segments[1:]):
            event_widths.append(float(after.t[1]-after.t[0]))
        if factors is not None:
            # Parameter input retains the actual 141-dimensional physical contract.
            p, initial, *_ = sim.prepare_initial_conditions(overrides)
            synthetic.append((np.array([p.get(k, _defaults[k]) for k in PARAM_KEYS]), initial, y))
        if (index+1) % 20 == 0:
            print(f'solved {index+1}/{len(cases)}', flush=True)
    report = {'seed': args.seed, 'cases': len(cases), 'successful': len(solve_times),
              'failures': failures, 'reference': 'production BDF rtol=1e-4 atol=1e-6; 0.01min evaluation grid',
              'solve_seconds': summarize(solve_times), 'native_steps': summarize(native_steps),
              'production_vs_piecewise_dense_log_rel_l2': summarize(biases),
              'post_event_first_step_min': summarize(event_widths),
              'sampling': {n: {k: summarize(v) for k, v in d.items() if v} for n, d in stats.items()}}
    (args.output/'sampling.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    np.savez(args.output/'synthetic.npz', params=np.stack([x[0] for x in synthetic]),
             initial=np.stack([x[1] for x in synthetic]), y=np.stack([x[2] for x in synthetic]))
    benchmark(args, synthetic, grid)


def benchmark(args, samples, grid):
    import torch
    from budding_yeast_v2.models.baselines.fno import FNO
    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    params = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32, device=device)
    initial = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32, device=device)
    dense = np.log1p(np.maximum(np.stack([s[2] for s in samples]), 0))
    split = int(len(samples)*.75)
    pm, ps = params[:split].mean(0), params[:split].std(0).clamp_min(1e-5)
    params = (params-pm)/ps
    ym = torch.tensor(dense[:split].mean((0, 2)), dtype=torch.float32, device=device)[None, :, None]
    ys = torch.tensor(dense[:split].std((0, 2)), dtype=torch.float32, device=device)[None, :, None].clamp_min(.01)
    report = {'device': str(device), 'gpu': torch.cuda.get_device_name(device),
              'train': split, 'test': len(samples)-split, 'epochs': args.epochs,
              'pilot': [], 'training_step': []}

    def inputs(n):
        t = torch.linspace(0, 1, n, device=device)
        x = torch.empty((len(samples), 39, 2, n), device=device)
        x[:, :, 0] = ((torch.log1p(initial.clamp_min(0))[:, :, None]-ym)/ys)
        x[:, :, 1] = t
        raw = interp1d(grid, dense, axis=2)(np.linspace(0, 210, n))
        target = (torch.tensor(raw, dtype=torch.float32, device=device)-ym)/ys
        return x, target

    for n in args.pilot_lengths:
        torch.manual_seed(args.seed)
        model = FNO(width=64, modes=24).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x, target = inputs(n)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for epoch in range(args.epochs):
            for batch in torch.arange(split, device=device).split(32):
                opt.zero_grad(set_to_none=True)
                pred = model(x[batch], params[batch])
                loss = (pred-target[batch]).square().mean()
                loss.backward()
                opt.step()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter()-start
        with torch.no_grad():
            pred = (model(x[split:], params[split:])*ys+ym).cpu().numpy()
        recon = interp1d(np.linspace(0, 210, n), pred, axis=2)(grid)
        rel = np.linalg.norm((recon-dense[split:]).reshape(len(samples)-split, -1), axis=1)/np.linalg.norm(dense[split:].reshape(len(samples)-split, -1), axis=1)
        report['pilot'].append(dict(points=n, seconds=seconds, test_log_rel_l2=summarize(rel)))
        print('pilot', report['pilot'][-1], flush=True)
        del model, opt, x, target

    for n in args.lengths:
        torch.manual_seed(args.seed)
        model = FNO(width=192, modes=24).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = 128
        x = torch.zeros(batch, 39, 2, n, device=device)
        x[:, :, 1] = torch.linspace(0, 1, n, device=device)
        p = torch.randn(batch, 141, device=device)
        target = torch.randn(batch, 39, n, device=device)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x, p)
                loss = (pred-target).square().mean()
            loss.backward()
            opt.step()

        for _ in range(5):
            step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(20):
            start = time.perf_counter()
            step()
            torch.cuda.synchronize(device)
            times.append((time.perf_counter()-start)*1000)
        report['training_step'].append(dict(points=n, batch=batch, width=192, modes=24,
            precision='bf16 mixed; FFT float32', median_ms=float(np.median(times)),
            peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20))
        print('step', report['training_step'][-1], flush=True)
        del model, opt, x, p, target
    (args.output/args.report_name).write_text(json.dumps(report, indent=2))


def event_audit(args):
    """Explicit diagnostic subset, distinct from the full reconstruction cohort."""
    keys = ['1', '2.1', '2.2', '8', '9', '10', '11', '27', '34', '36', '38',
            '39', '49', '55', '63', '65', '73', '77', '80', '81', '82', '104']
    pulses, failures, pairs = [], [], []
    original = sim.solve_ivp
    for key in keys:
        segments = []

        def capture(*a, **kw):
            result = original(*a, **kw)
            segments.append(result)
            return result

        with patch.object(sim, 'solve_ivp', capture):
            result = sim.simulate_mutant(key, REAL_MUTANTS[key][1])
        if result[1] is None:
            failures.append(key)
            continue
        event_times = [s.t[-1] for s in segments[:-1]]
        for a, b in zip(event_times[:-1], event_times[1:]):
            pairs.append(dict(mutant=key, first=float(a), second=float(b),
                gap=float(b-a), same_cell={n: bool(np.floor(a/(210/(n-1)))
                    == np.floor(b/(210/(n-1)))) for n in args.lengths}))
        for i in range(1, len(segments)-1):
            s = segments[i]
            if (s.y[37, 0] > .9 and segments[i-1].y[37, 0] < .9
                    and segments[i+1].y[37, 0] < .9):
                a, b = s.t[0], s.t[-1]
                counts = {n: int(np.count_nonzero((np.linspace(0, 210, n) > a)
                          & (np.linspace(0, 210, n) < b))) for n in args.lengths}
                pulses.append(dict(mutant=key, start=float(a), stop=float(b),
                                   duration=float(b-a), sample_counts=counts))
    report = dict(cases_requested=len(keys), failures=failures, pulses=pulses, adjacent_events=pairs,
        summary={n: dict(missed=sum(p['sample_counts'][n] == 0 for p in pulses),
                        fewer_than_two=sum(p['sample_counts'][n] < 2 for p in pulses),
                        adjacent_event_same_cell=sum(p['same_cell'][n] for p in pairs),
                        adjacent_event_pairs=len(pairs),
                        total=len(pulses)) for n in args.lengths})
    (args.output/'event_pulses.json').write_text(json.dumps(report, indent=2))
    invariance = []
    for n in [128, 256, 500]:
        segments = []

        def capture(*a, **kw):
            result = original(*a, **kw)
            segments.append(result)
            return result

        start = time.perf_counter()
        with patch.object(sim, 'solve_ivp', capture):
            result = sim.simulate_mutant('WT', {'mdt': 90.}, ns=n)
        invariance.append(dict(n=n, seconds=time.perf_counter()-start,
            nfev=sum(s.nfev for s in segments), steps=sum(len(s.t)-1 for s in segments),
            events=[float(s.t[-1]) for s in segments[:-1]], success=result[1] is not None))
    (args.output/'solver_invariance.json').write_text(json.dumps(invariance, indent=2))
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
