#!/usr/bin/env python3
"""Generate paper figures from reproduced COSIMS-Auction experiment results.

Outputs are written to results/figures by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.env import Env
from policies.rr import RoundRobin
from policies.greedy_latency import GreedyLatency
from policies.sqlf import ShortestQueueFirst
from policies.cap_greedy import CapAwareGreedy
from policies.cosims_sinkhorn import COSIMSSinkhorn
from policies.cosims_auction_B import COSIMSAuctionB

POLICY_KEYS = ['rr', 'greedy_latency', 'sqlf', 'cap_greedy', 'cosims_sinkhorn', 'cosims_auction_B']
POLICY_SHOW = {
    'rr': 'RR',
    'greedy_latency': 'Greedy-Latency',
    'sqlf': 'SQLF',
    'cap_greedy': 'Cap-Greedy',
    'cosims_sinkhorn': 'COSIMS-Sinkhorn',
    'cosims_auction_B': 'COSIMS-Auction',
}
COLORS = {
    'rr': (0.1216, 0.4667, 0.7059),
    'greedy_latency': (1.0000, 0.4980, 0.0549),
    'sqlf': (0.1725, 0.6275, 0.1725),
    'cap_greedy': (0.8392, 0.1529, 0.1569),
    'cosims_sinkhorn': (0.5804, 0.4039, 0.7412),
    'cosims_auction_B': (0.0, 0.0, 0.0),
}
LINESTYLES = {'rr': '-', 'greedy_latency': '--', 'sqlf': '-.', 'cap_greedy': '-', 'cosims_sinkhorn': '--', 'cosims_auction_B': '-'}
MARKERS = {'rr': 'o', 'greedy_latency': 's', 'sqlf': '^', 'cap_greedy': 'D', 'cosims_sinkhorn': 'v', 'cosims_auction_B': '*'}

BOX_POLICIES = [
    ('rr', 'RR', RoundRobin),
    ('greedy_latency', 'Greedy-Latency', GreedyLatency),
    ('sqlf', 'SQLF', ShortestQueueFirst),
    ('cosims_sinkhorn', 'COSIMS-Sinkhorn', COSIMSSinkhorn),
    ('cap_greedy', 'Cap-Greedy', CapAwareGreedy),
    ('cosims_auction_B', 'COSIMS-Auction', COSIMSAuctionB),
]
BOX_FACE = {
    'rr': '#D9E2EC', 'greedy_latency': '#FDE3C2', 'sqlf': '#DDEEDB',
    'cosims_sinkhorn': '#E7DDF1', 'cap_greedy': '#F6D8C8', 'cosims_auction_B': '#F2F2F2',
}
BOX_EDGE = {
    'rr': '#6B8FB3', 'greedy_latency': '#D47A1F', 'sqlf': '#5F9E6E',
    'cosims_sinkhorn': '#8B6BB1', 'cap_greedy': '#C85C3D', 'cosims_auction_B': '#111111',
}


def setup_matplotlib() -> None:
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] wrote {out_dir / (name + ".png")}')


def plot_latency_two_panel(df: pd.DataFrame, x_col: str, out_dir: Path, fig_name: str, trace: bool) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), sharex=True)
    if trace:
        panels = [
            (axes[0], 'avg_latency_mean', 'avg_latency_std', '(a) Average latency', 'Average latency', (5000, 11200)),
            (axes[1], 'p95_latency_mean', 'p95_latency_std', '(b) P95 latency', 'P95 latency', (7500, 23500)),
        ]
        linewidths = {'rr': 1.45, 'greedy_latency': 1.45, 'sqlf': 1.45, 'cap_greedy': 1.85, 'cosims_sinkhorn': 1.75, 'cosims_auction_B': 2.95}
        y_scale = 1.0
        xlabel = 'Trace Load Factor'
    else:
        panels = [
            (axes[0], 'avg_latency_mean', 'avg_latency_std', '(a) Average latency', 'Average latency', (170, 610)),
            (axes[1], 'p95_latency_mean', 'p95_latency_std', '(b) P95 latency', 'P95 latency', (280, 1450)),
        ]
        linewidths = {'rr': 1.30, 'greedy_latency': 1.30, 'sqlf': 1.30, 'cap_greedy': 2.75, 'cosims_sinkhorn': 1.55, 'cosims_auction_B': 2.35}
        y_scale = 1e9
        xlabel = 'Load Factor'
    zorders = {'rr': 2, 'greedy_latency': 2, 'sqlf': 2, 'cap_greedy': 4, 'cosims_sinkhorn': 3, 'cosims_auction_B': 6}
    handles, labels = [], []
    for ax, metric, std_col, title, ylabel, ylim in panels:
        for key in POLICY_KEYS:
            sub = df[df.policy == key].sort_values(x_col)
            x = sub[x_col].to_numpy()
            y = sub[metric].to_numpy() / y_scale
            ystd = sub[std_col].fillna(0).to_numpy() / y_scale
            if key in {'cap_greedy', 'cosims_auction_B'}:
                ax.fill_between(x, y - ystd, y + ystd, color=COLORS[key], alpha=0.16 if key == 'cap_greedy' else 0.18, linewidth=0, zorder=zorders[key] - 1)
            line = ax.plot(
                x, y, label=POLICY_SHOW[key], color=COLORS[key], linestyle=LINESTYLES[key], marker=MARKERS[key],
                linewidth=linewidths[key], markersize=7.0 if key == 'cosims_auction_B' else 4.8,
                markerfacecolor=COLORS[key], markeredgecolor='white' if key == 'cosims_auction_B' else COLORS[key],
                markeredgewidth=0.85 if key == 'cosims_auction_B' else 0.55, zorder=zorders[key],
            )[0]
            if ax is axes[0]:
                handles.append(line); labels.append(POLICY_SHOW[key])
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.30)
        ax.set_xlim(0.75, 1.65)
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
        ax.set_ylim(*ylim)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.045), frameon=False, fontsize=8.7, handlelength=2.35, columnspacing=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.905), w_pad=2.3)
    save(fig, out_dir, fig_name)


def collect_delays(config: dict, policy_cls, seeds: list[int], load: float) -> np.ndarray:
    delays = []
    warmup = float(config['warmup_time'])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        policy = policy_cls(config, rng)
        env = Env(config, rng=rng, policy=policy)
        if hasattr(policy, 'on_env_init'):
            policy.on_env_init(env)
        tasks = env.run(load_scale=load)
        for task in tasks:
            if task.arrival_t >= warmup:
                delays.append(float(task.finish_t - task.arrival_t))
    return np.asarray(delays, dtype=float)


def plot_fig5_2(config_path: Path, out_dir: Path, tables_dir: Path, seeds: list[int]) -> None:
    config = json.loads(config_path.read_text())
    sample_path = tables_dir / 'fig5_2_task_delay_distribution_sample.csv'
    labels = [label for _, label, _ in BOX_POLICIES]
    if sample_path.exists():
        sample_df = pd.read_csv(sample_path)
        data = [np.sort(sample_df[sample_df.method == label].completion_delay.to_numpy()) for label in labels]
    else:
        sample_rng = np.random.default_rng(42)
        rows, data = [], []
        for key, label, cls in BOX_POLICIES:
            values = collect_delays(config, cls, seeds, load=1.2) / 1e9
            if values.size > 10000:
                values = values[sample_rng.choice(values.size, size=10000, replace=False)]
            values = np.sort(values)
            data.append(values)
            rows.extend({'method': label, 'completion_delay': float(v)} for v in values)
        tables_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(sample_path, index=False)
    setup_matplotlib()
    positions = np.arange(len(labels)) * 0.56 + 1
    fig, ax = plt.subplots(figsize=(5.95, 3.45))
    bp = ax.boxplot(
        data, positions=positions, labels=labels, patch_artist=True, whis=(5, 95), showfliers=False, widths=0.30,
        medianprops={'color': 'black', 'linewidth': 1.45}, boxprops={'linewidth': 1.05},
        whiskerprops={'color': '#555555', 'linewidth': 0.95}, capprops={'color': '#555555', 'linewidth': 0.95},
    )
    for box, (key, _, _) in zip(bp['boxes'], BOX_POLICIES):
        box.set_facecolor(BOX_FACE[key]); box.set_edgecolor(BOX_EDGE[key])
        box.set_alpha(0.72 if key != 'cosims_auction_B' else 0.84)
        box.set_linewidth(1.05 if key != 'cosims_auction_B' else 1.45)
    for median, (key, _, _) in zip(bp['medians'], BOX_POLICIES):
        median.set_color(BOX_EDGE[key] if key == 'cosims_auction_B' else 'black')
        median.set_linewidth(1.75 if key == 'cosims_auction_B' else 1.45)
    ax.set_ylabel('Task completion delay', fontsize=10)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.24)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8.0, rotation=10, pad=2)
    for tick in ax.get_xticklabels():
        tick.set_ha('center')
        tick.set_rotation_mode('anchor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(positions[0] - 0.34, positions[-1] + 0.34)
    ax.set_ylim(0.0, float(np.percentile(np.concatenate(data), 99)) * 1.05)
    fig.tight_layout()
    save(fig, out_dir, 'fig5_2_task_delay_distribution')


def plot_fig5_4(trace_df: pd.DataFrame, out_dir: Path, tables_dir: Path) -> None:
    cap = trace_df[trace_df.policy == 'cap_greedy'].sort_values('trace_load_factor')
    auc = trace_df[trace_df.policy == 'cosims_auction_B'].sort_values('trace_load_factor')
    merged = cap[['trace_load_factor', 'avg_latency_mean', 'cloud_rate_mean']].merge(
        auc[['trace_load_factor', 'avg_latency_mean', 'cloud_rate_mean']], on='trace_load_factor', suffixes=('_cap', '_auction')
    )
    merged['avg_latency_reduction_pct'] = (merged.avg_latency_mean_cap - merged.avg_latency_mean_auction) / merged.avg_latency_mean_cap * 100.0
    merged['cloud_util_reduction_pct'] = (merged.cloud_rate_mean_cap - merged.cloud_rate_mean_auction) / merged.cloud_rate_mean_cap * 100.0
    tables_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(tables_dir / 'fig5_4_trace_tradeoff_relative.csv', index=False)
    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45), sharex=True)
    panels = [
        (axes[0], merged.avg_latency_reduction_pct, '(a) Average-latency reduction', 'Average latency reduction vs Cap-Greedy (%)', (-0.5, 6.4)),
        (axes[1], merged.cloud_util_reduction_pct, '(b) Cloud-utilization reduction', 'Cloud-utilization reduction vs Cap-Greedy (%)', (-3.3, 5.5)),
    ]
    x = merged.trace_load_factor
    for ax, values, title, ylabel, ylim in panels:
        ax.axhline(0.0, color='#555555', linestyle='--', linewidth=0.85, zorder=1)
        ax.plot(x, values, color='black', marker='o', markersize=4.8, linewidth=1.9, markerfacecolor='white', markeredgecolor='black', markeredgewidth=1.0, zorder=4)
        ax.fill_between(x.to_numpy(), 0.0, values.to_numpy(), color='#D9D9D9', alpha=0.15, zorder=2)
        for xi, yi in zip(x, values):
            if float(xi) in {0.8, 1.0, 1.2, 1.4, 1.6}:
                va = 'bottom' if yi >= 0 else 'top'
                off = 0.18 if yi >= 0 else -0.18
                x_shift = 0.0
                if abs(float(yi)) < 0.35:
                    off = -0.48 if yi < 0 else 0.40
                    x_shift = 0.025
                ax.text(xi + x_shift, yi + off, f'{yi:+.1f}', ha='center', va=va, fontsize=7.1, color='#333333')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Trace Load Factor', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9.3)
        ax.set_xlim(0.77, 1.63)
        ax.set_xticks([0.8, 1.0, 1.2, 1.4, 1.6])
        ax.set_ylim(*ylim)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.30, zorder=0)
        ax.tick_params(axis='both', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.suptitle('COSIMS-Auction vs. Cap-Greedy under trace replay', fontsize=10.2, y=1.02)
    fig.tight_layout(w_pad=2.1)
    save(fig, out_dir, 'fig5_4_trace_tradeoff')


def parse_seeds(text: str) -> list[int]:
    if '-' in text:
        a, b = text.split('-', 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/auction_B_recommended_sqlf_cloud.json')
    ap.add_argument('--seeds', type=str, default='0-4')
    args = ap.parse_args()
    out_dir = args.results_dir / 'figures'
    tables_dir = args.results_dir / 'tables'
    fig5_1 = args.results_dir / 'fig5_1_synthetic_step01/main_results/summary_mean_std.csv'
    fig5_3 = args.results_dir / 'fig5_3_trace_step01/main_results/summary_mean_std.csv'
    if fig5_1.exists():
        plot_latency_two_panel(pd.read_csv(fig5_1), 'load', out_dir, 'fig5_1_synthetic_latency', trace=False)
    else:
        print(f'[WARN] missing {fig5_1}, skip Fig. 5.1')
    plot_fig5_2(args.config, out_dir, tables_dir, parse_seeds(args.seeds))
    if fig5_3.exists():
        trace_df = pd.read_csv(fig5_3)
        plot_latency_two_panel(trace_df, 'trace_load_factor', out_dir, 'fig5_3_trace_latency', trace=True)
        plot_fig5_4(trace_df, out_dir, tables_dir)
    else:
        print(f'[WARN] missing {fig5_3}, skip Fig. 5.3/Fig. 5.4')


if __name__ == '__main__':
    main()
