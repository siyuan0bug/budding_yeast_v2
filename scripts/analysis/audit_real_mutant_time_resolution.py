"""Audit every canonical real-mutant configuration without changing training data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from budding_yeast_v2.scripts.analysis.benchmark_budding_time_resolution import trajectory, summarize
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES

CORE = [0, 1, 2, 3, 4, 20, 22, 33, 35]
SMOOTH_PEAK_CHANNELS = [1, 2, 3, 4, 20, 22]
COLORS = {256: '#8b55b0', 300: '#08916b', 350: '#d63052', 400: '#ea7800', 500: '#2563eb'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Deterministic canonical cases; no random sampling')
    parser.add_argument('--lengths', type=int, nargs='+', default=[300, 350, 400, 500])
    parser.add_argument('--all-pages', action='store_true', help='Save all successful cases as a multipage PDF')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 210, 21001)
    rows, failures, cache = [], [], {}
    for index, (key, (name, overrides)) in enumerate(REAL_MUTANTS.items()):
        run = trajectory(key, overrides)
        if run is None:
            failures.append(key)
            continue
        ref, segments, _ = run
        native_t = np.concatenate([segments[0].t] + [s.t[1:] for s in segments[1:]])
        native_y = np.concatenate([segments[0].y] + [s.y[:, 1:] for s in segments[1:]], axis=1)
        sampling = interp1d(native_t, native_y, axis=1)
        event_times = [float(s.t[-1]) for s in segments[:-1]]
        # #104 also changes a synthesis coefficient at t=180min.
        if overrides.get('is_mutant_104'):
            event_times.append(180.)
        event_mask = np.zeros(len(grid), dtype=bool)
        for te in event_times:
            event_mask |= abs(grid-te) <= 1
        scale = np.maximum(np.ptp(ref, axis=1), .1)
        logref = np.log1p(np.maximum(ref, 0))
        cache[key] = dict(t=native_t, y=native_y, events=event_times)
        for n in args.lengths:
            t = np.linspace(0, 210, n)
            coarse = sampling(t)
            recon = interp1d(t, coarse, axis=1)(grid)
            err = np.log1p(np.maximum(recon, 0))-logref
            normerr = (recon-ref)/scale[:, None]
            peak_miss = np.maximum(0, ref.max(1)-coarse.max(1))/scale
            smooth_channel_rmse = np.sqrt(np.mean(normerr[:, ~event_mask]**2, axis=1))
            peak_channel = CORE[int(np.argmax(peak_miss[CORE]))]
            smooth_channel = CORE[int(np.argmax(smooth_channel_rmse[CORE]))]
            peak_count_diff = []
            for channel in SMOOTH_PEAK_CHANNELS:
                if np.ptp(ref[channel]) < .1:
                    continue
                prominence = max(.1*np.ptp(ref[channel]), .01)
                a = len(find_peaks(ref[channel], prominence=prominence, distance=100)[0])
                b = len(find_peaks(coarse[channel], prominence=prominence,
                                   distance=max(1, round((n-1)/210)))[0])
                if a != b:
                    peak_count_diff.append(dict(channel=VARIABLE_NAMES[channel], reference=a, sampled=b))
            row = dict(mutant=key, name=name, points=n,
                log_rel_l2=float(np.linalg.norm(err)/np.linalg.norm(logref)),
                core_log_rel_l2=float(np.linalg.norm(err[CORE])/np.linalg.norm(logref[CORE])),
                core_mae_range=float(np.mean(np.abs(normerr[CORE]))),
                core_smooth_nrmse=float(np.sqrt(np.mean(normerr[CORE][:, ~event_mask]**2))),
                core_event_nrmse=float(np.sqrt(np.mean(normerr[CORE][:, event_mask]**2))) if event_mask.any() else None,
                core_peak_miss_max=float(peak_miss[CORE].max()),
                peak_worst_channel=VARIABLE_NAMES[peak_channel],
                smooth_worst_channel=VARIABLE_NAMES[smooth_channel],
                smooth_worst_channel_nrmse=float(smooth_channel_rmse[smooth_channel]),
                core_fraction_over_5pct=float(np.mean(np.abs(normerr[CORE]) > .05)),
                peak_count_changes=peak_count_diff)
            rows.append(row)
        if (index+1) % 15 == 0:
            print(f'audited {index+1}/{len(REAL_MUTANTS)}', flush=True)
    summary = {}
    metrics = ['log_rel_l2', 'core_log_rel_l2', 'core_mae_range', 'core_smooth_nrmse',
               'core_event_nrmse', 'core_peak_miss_max', 'core_fraction_over_5pct']
    for n in args.lengths:
        group = [r for r in rows if r['points'] == n]
        summary[n] = {metric: summarize([r[metric] for r in group if r[metric] is not None]) for metric in metrics}
        summary[n]['mutants_with_peak_count_change'] = sum(bool(r['peak_count_changes']) for r in group)
        summary[n]['top_peak_miss'] = sorted(group, key=lambda r: r['core_peak_miss_max'], reverse=True)[:8]
        summary[n]['top_smooth_error'] = sorted(group, key=lambda r: r['core_smooth_nrmse'], reverse=True)[:8]
    report = dict(requested=len(REAL_MUTANTS), successful=len(cache), failures=failures,
        reference='Same production BDF solve, native linear interpolation; 0.01min evaluation grid',
        scope='Sampling reconstruction of canonical real-mutant simulations, not measured biological trajectories or model predictions',
        rows=rows, summary=summary)
    (args.output/'audit.json').write_text(json.dumps(report, indent=2))
    plots(args, cache, rows, summary)
    print(json.dumps({n: {k: v for k, v in s.items() if not k.startswith('top_')} for n, s in summary.items()}, indent=2), flush=True)


def draw(ax, case, channel, lengths, lo=0, hi=210, dots=False):
    sample = interp1d(case['t'], case['y'], axis=1)
    ax.plot(case['t'], case['y'][channel], color='.65', lw=1.8, label='Native reference')
    for n in reversed(lengths):
        t = np.linspace(0, 210, n)
        ax.plot(t, sample(t)[channel], color=COLORS.get(n),
                ls='-' if n == 500 else '--' if n == 400 else ':' if n == 350 else '-.',
                lw=1.1, marker='o' if dots else None, markersize=3, label=f'{n} points')
    ax.set(xlim=(lo, hi), xlabel='Time (min)', ylabel=VARIABLE_NAMES[channel])
    # Zoom y limits must be based on local values, not the full trajectory.
    mask = (case['t'] >= lo) & (case['t'] <= hi)
    if mask.any():
        y = case['y'][channel, mask]
        pad = max(float(np.ptp(y))*.08, .005)
        ax.set_ylim(y.min()-pad, y.max()+pad)
    ax.grid(alpha=.2)


def plots(args, cache, rows, summary):
    lowest = min(args.lengths)
    compact = []
    for ranking, field in [('top_smooth_error', 'smooth_worst_channel'), ('top_peak_miss', 'peak_worst_channel')]:
        r = summary[lowest][ranking][0]
        compact.append((r['mutant'], VARIABLE_NAMES.index(r[field])))
    save_examples(args, cache, compact)
    selected = []
    for metric, field in [('top_smooth_error', 'smooth_worst_channel'), ('top_peak_miss', 'peak_worst_channel')]:
        for r in summary[lowest][metric][:4]:
            pair = (r['mutant'], VARIABLE_NAMES.index(r[field]))
            if pair not in selected:
                selected.append(pair)
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 2.5*len(selected)), constrained_layout=True)
    for row, (key, channel) in enumerate(selected):
        case = cache[key]
        draw(axes[row, 0], case, channel, args.lengths)
        sample = interp1d(case['t'], case['y'], axis=1)
        dense = np.linspace(0, 210, 21001)
        t = np.linspace(0, 210, lowest)
        err = abs(interp1d(t, sample(t)[channel])(dense)-sample(dense)[channel])
        center = dense[np.argmax(err)]
        lo, hi = max(0, center-3), min(210, center+3)
        draw(axes[row, 1], case, channel, args.lengths, lo, hi, dots=True)
        for ax in axes[row]:
            ax.set_title(f'#{key}: {REAL_MUTANTS[key][0]}', fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Cases selected by largest smooth-region errors / peak losses at lowest resolution')
    fig.savefig(args.output/'worst_cases.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(axes,
            ['core_mae_range', 'core_smooth_nrmse', 'core_peak_miss_max'],
            ['Core waveform MAE / range', 'Core smooth-region NRMSE', 'Largest core peak loss / range']):
        values = [100*np.array([r[metric] for r in rows if r['points'] == n]) for n in args.lengths]
        ax.boxplot(values, tick_labels=args.lengths, showfliers=True)
        ax.set(xlabel='Output points', ylabel='Error (%)', title=title)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle(f'All {len(cache)} canonical real-mutant configurations')
    fig.savefig(args.output/'distributions.png', dpi=180)
    plt.close(fig)

    if args.all_pages:
        with PdfPages(args.output/'all_real_mutants.pdf') as pdf:
            for index, (key, case) in enumerate(cache.items()):
                fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
                for ax, channel in zip(axes.flat, CORE):
                    draw(ax, case, channel, args.lengths)
                axes[0, 0].legend(fontsize=6)
                fig.suptitle(f'#{key}: {REAL_MUTANTS[key][0]}', fontsize=12)
                pdf.savefig(fig)
                plt.close(fig)
                if (index+1) % 30 == 0:
                    print(f'plotted {index+1}/{len(cache)} PDF pages', flush=True)


def save_examples(args, cache, selected):
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.2*len(selected)), constrained_layout=True)
    for row, (key, channel) in enumerate(selected):
        case = cache[key]
        draw(axes[row, 0], case, channel, args.lengths)
        if row == 0:
            lo, hi = 0, 1.5
        else:
            center = case['t'][np.argmax(case['y'][channel])]
            lo, hi = max(0, center-3), min(210, center+3)
        draw(axes[row, 1], case, channel, args.lengths, lo, hi, dots=True)
        for ax in axes[row]:
            ax.set_title(f'#{key}: {REAL_MUTANTS[key][0]}', fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Most sensitive cases selected from all 126 configurations')
    fig.savefig(args.output/'sensitive_examples.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
