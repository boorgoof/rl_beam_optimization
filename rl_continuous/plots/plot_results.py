"""
Generate comparison plots from CSV metrics files.

Usage:
    python plots/plot_results.py --log_dir runs --env Pendulum-v1
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

ALGO_COLORS = {
    'ddpg': '#e41a1c',  # red
    'td3':  '#377eb8',  # blue
    'sac':  '#4daf4a',  # green
    'ppo':  '#ff7f00',  # orange
    'a2c':  '#984ea3',  # purple
}


def load_csv(path):
    episodes, rewards, moving_avgs = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            moving_avgs.append(float(row['moving_avg_reward']))
    return np.array(episodes), np.array(rewards), np.array(moving_avgs)


def plot_comparison(csv_paths: dict, env_name: str, out_path: str):
    """
    csv_paths: dict mapping algo_name -> path to metrics.csv
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Algorithm Comparison — {env_name}', fontsize=14)

    ax_raw, ax_avg = axes

    for algo, path in csv_paths.items():
        if not os.path.exists(path):
            print(f'Warning: {path} not found, skipping {algo}')
            continue
        episodes, rewards, moving_avgs = load_csv(path)
        color = ALGO_COLORS.get(algo, 'black')
        label = algo.upper()

        # Raw episode reward (faint)
        ax_raw.plot(episodes, rewards, color=color, alpha=0.3, linewidth=0.8)
        ax_raw.plot(episodes, moving_avgs, color=color, linewidth=2.0, label=label)

        # Moving average only
        ax_avg.plot(episodes, moving_avgs, color=color, linewidth=2.0, label=label)

    ax_raw.set_title('Episode Reward (raw + moving avg 100)')
    ax_raw.set_xlabel('Episode')
    ax_raw.set_ylabel('Reward')
    ax_raw.legend()
    ax_raw.grid(alpha=0.3)

    ax_avg.set_title('Moving Average Reward (window=100)')
    ax_avg.set_xlabel('Episode')
    ax_avg.set_ylabel('Avg Reward')
    ax_avg.legend()
    ax_avg.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--env',     type=str, default='Pendulum-v1')
    parser.add_argument('--out',     type=str, default='plots/comparison.png')
    args = parser.parse_args()

    algos = ['ddpg', 'td3', 'sac', 'ppo', 'a2c']
    csv_paths = {}
    for algo in algos:
        p = os.path.join(args.log_dir, f'{algo}_{args.env}', 'metrics.csv')
        if os.path.exists(p):
            csv_paths[algo] = p

    if not csv_paths:
        print('No CSV files found. Run compare_all.py first.')
    else:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        plot_comparison(csv_paths, args.env, args.out)
