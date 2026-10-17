"""Build report figures from an existing summary; never run model inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('reports/lowdim_fp16/summary.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/lowdim_fp16'))
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    results = json.loads(raw)['results']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'Description': 'Input summary SHA-256: ' + hashlib.sha256(raw).hexdigest(), 'Date': None}
    plt.rcParams.update({'font.size': 10, 'svg.hashsalt': 'lowdim-fp16-report'})

    def save(fig, name):
        svg_path = args.output_dir / (name + '.svg')
        fig.savefig(svg_path, metadata=metadata)
        # Matplotlib emits trailing spaces in SVG path data; keep Git checks tidy.
        svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / (name + '.png'), dpi=140)
        plt.close(fig)

    colors = ('#31688e', '#e1812c')
    labels = ['Original', 'Threshold 5', 'ACS 5', 'ACS 20']
    suffixes = ['original', 'threshold5', 'acs5_fp32', 'acs20_fp32']
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
    for i, precision in enumerate(('fp32', 'fp16_amp')):
        values = [results[f'{precision}_{s}']['benchmark'] for s in suffixes]
        bars = ax.bar(x + (i - .5) * .36, [v['mean_ms'] for v in values], .36,
                      yerr=[v['std_ms'] for v in values], capsize=3,
                      label=('FP32', 'FP16 AMP')[i], color=colors[i])
        ax.bar_label(bars, fmt='%.1f', padding=7, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylabel='Complete predict_action latency (ms)',
           title='Same FP32 schedules, batch 3, 100 denoising steps', ylim=(0, 760))
    ax.legend()
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    save(fig, 'latency_comparison')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    labels = ['Original', 'Th. 5', 'ACS 5', 'ACS 5\n+BUA', 'ACS 20', 'ACS 20\n+BUA']
    suffixes = ['original', 'threshold5', 'acs5_fp32', 'acs5_fp32_bua', 'acs20_fp32', 'acs20_fp32_bua']
    bottom = np.zeros(len(labels))
    for suffix, color, name in zip(('dropout1', 'dropout2', 'dropout3'),
                                    ('#31688e', '#35b779', '#e5b337'), ('SA', 'CA', 'FFN')):
        counts = [sum(v for k, v in results['fp32_' + s]['dtype_audit']['block_compute_counts'].items()
                      if k.endswith(suffix)) for s in suffixes]
        axes[0].bar(np.arange(len(labels)), counts, bottom=bottom, color=color, label=name)
        bottom += counts
    for i, count in enumerate(bottom):
        axes[0].text(i, count + 35, str(int(count)), ha='center', fontsize=9)
    axes[0].set(xticks=np.arange(len(labels)), xticklabels=labels, ylim=(0, 2800),
                ylabel='Actual branch executions per prediction', title='Caching skips branches; model calls stay at 100')
    axes[0].legend(ncol=3)
    x = np.arange(4)
    configs = [('fp32', 'fp32', 5), ('fp16_amp', 'fp16_amp', 5),
               ('fp32', 'fp32', 20), ('fp16_amp', 'fp16_amp', 20)]
    for i, bu in enumerate((False, True)):
        values = []
        for precision, schedule, budget in configs:
            label = f'{precision}_acs{budget}_{schedule}' + ('_bua' if bu else '')
            values.append(statistics.mean(v['vs_same_precision_original']['action']['mae']
                                          for v in results[label]['validation']))
        bars = axes[1].bar(x + (i - .5) * .36, values, .36,
                          label=('ACS', 'ACS + BUA')[i], color=colors[i])
        axes[1].bar_label(bars, labels=[f'{v:.2e}' for v in values], padding=4, fontsize=8)
    axes[1].set(xticks=x, xticklabels=['FP32\nK=5', 'AMP\nK=5', 'FP32\nK=20', 'AMP\nK=20'],
                yscale='log', ylim=(1e-4, 2e-2), ylabel='Action MAE vs same-precision Original (log scale)',
                title='BUA is not guaranteed to reduce action error')
    axes[1].legend()
    axes[1].yaxis.grid(True, alpha=.2)
    axes[1].set_axisbelow(True)
    save(fig, 'cache_accuracy_tradeoff')
    print('Generated two SVG figures from SHA-256', hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
