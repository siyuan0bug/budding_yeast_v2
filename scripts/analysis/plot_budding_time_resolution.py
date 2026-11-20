"""Plot the visible effect of uniform output sampling, using physical units."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from budding_yeast_v2.scripts.analysis.benchmark_budding_time_resolution import trajectory
from budding_yeast_v2.data.systems.budding.real_mutants import REAL_MUTANTS
from budding_yeast_v2.data.systems.budding.states import VARIABLE_NAMES


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Reserved; these canonical trajectories are deterministic')
    parser.add_argument('--lengths', type=int, nargs='+', default=[500, 256, 128])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    for key in ['1', '36', '80', '104']:
        run = trajectory(key, REAL_MUTANTS[key][1])
        if run is None:
            raise RuntimeError(f'Simulation failed: {key}')
        y, segments, _ = run
        t = np.concatenate([segments[0].t] + [s.t[1:] for s in segments[1:]])
        native = np.concatenate([segments[0].y] + [s.y[:, 1:] for s in segments[1:]], axis=1)
        sample = interp1d(t, native, axis=1)
        results[key] = (sample, segments)

    def curves(ax, sample, channel, lo=0, hi=210, markers=False):
        dense = np.linspace(lo, hi, 10001)
        ax.plot(dense, sample(dense)[channel], color='.75', lw=2, label='Native reference', zorder=1)
        colors = ['#2563eb', '#ea7800', '#d63052', '#08916b', '#8b55b0']
        styles = ['-', '--', ':', '-.', '--']
        for i, n in enumerate(args.lengths):
            color, style = colors[i % len(colors)], styles[i % len(styles)]
            t = np.linspace(0, 210, n)
            y = sample(t)[channel]
            ax.plot(t, y, color=color, linestyle=style, lw=1.3,
                    marker='o' if markers else None, markersize=4, label=f'{n} points', zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(VARIABLE_NAMES[channel])
        ax.grid(alpha=.2)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for ax, channel in zip(axes, [0, 2, 35]):
        curves(ax, results['1'][0], channel)
    axes[0].legend(fontsize=7)
    fig.suptitle('Wild type: identical simulation, different output sampling grids')
    fig.savefig(args.output/'wt_overview.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for ax, channel in zip(axes.flat, [0, 1, 2, 3, 4, 20, 22, 33, 35]):
        curves(ax, results['1'][0], channel)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Wild type: 0-210 min, identical simulation, different output grids')
    fig.savefig(args.output/'wt_full.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True)
    for row, key in enumerate(results):
        for ax, channel in zip(axes[row], [0, 2, 35]):
            curves(ax, results[key][0], channel)
            ax.set_title(f'Mutant #{key}: {REAL_MUTANTS[key][0]}', fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Physical trajectories at full simulation scale')
    fig.savefig(args.output/'mutants_full.png', dpi=180)
    plt.close(fig)

    sample, segments = results['1']
    event = next(s.t[-1] for s in segments if s.status == 1 and len(s.t_events[0]))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, channel in zip(axes.flat, [0, 33, 35, 2]):
        curves(ax, sample, channel, event-3, event+3, markers=True)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Wild type: six-minute zoom around first division; dots are output samples')
    fig.savefig(args.output/'wt_event_zoom.png', dpi=180)


if __name__ == '__main__':
    main()
