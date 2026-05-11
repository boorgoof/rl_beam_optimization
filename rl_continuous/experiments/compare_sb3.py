"""
Compare custom implementations vs Stable Baselines3 on the same environment.

Usage:
    python experiments/compare_sb3.py --env Pendulum-v1 --episodes 300
    python experiments/compare_sb3.py --env Pendulum-v1 --episodes 300 --algos ddpg sac ppo
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG, TD3, SAC, PPO, A2C as SB3_A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl_continuous.algorithms.ddpg import DDPG as CustomDDPG
from rl_continuous.algorithms.td3  import TD3  as CustomTD3
from rl_continuous.algorithms.sac  import SAC  as CustomSAC
from rl_continuous.algorithms.ppo  import PPO  as CustomPPO
from rl_continuous.algorithms.a2c  import A2C  as CustomA2C
from rl_continuous.training.trainer import Trainer


ALGO_COLORS = {
    'ddpg': '#e41a1c',
    'td3':  '#377eb8',
    'sac':  '#4daf4a',
    'ppo':  '#ff7f00',
    'a2c':  '#984ea3',
}

SB3_CLASSES = {
    'ddpg': DDPG,
    'td3':  TD3,
    'sac':  SAC,
    'ppo':  PPO,
    'a2c':  SB3_A2C,
}

CUSTOM_CLASSES = {
    'ddpg': CustomDDPG,
    'td3':  CustomTD3,
    'sac':  CustomSAC,
    'ppo':  CustomPPO,
    'a2c':  CustomA2C,
}


# ------------------------------------------------------------------
# SB3 episode-reward logger callback
# ------------------------------------------------------------------

class EpisodeCSVCallback(BaseCallback):
    """
    Records episode rewards after each episode and writes them to a CSV
    in the same format used by our custom Logger, so plot_comparison can
    read both.
    """
    def __init__(self, csv_path, verbose=0):
        super().__init__(verbose)
        self.csv_path      = csv_path
        self._episode      = 0
        self._total_steps  = 0
        self._reward_window = deque(maxlen=100)
        self._csv_file     = None
        self._csv_writer   = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._csv_file   = open(self.csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ['episode', 'reward', 'moving_avg_reward',
             'value_loss', 'policy_loss', 'alpha_loss', 'total_steps'])

    def _on_step(self):
        self._total_steps += 1
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._episode += 1
                reward = float(info['episode']['r'])
                self._reward_window.append(reward)
                moving_avg = float(np.mean(self._reward_window))
                self._csv_writer.writerow([
                    self._episode, reward, moving_avg,
                    None, None, None, self._total_steps])
                self._csv_file.flush()
        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()


# ------------------------------------------------------------------
# Training helpers
# ------------------------------------------------------------------

def train_custom(algo, env_name, n_episodes, log_dir, ckpt_dir):
    """Train a custom algorithm and save metrics to log_dir/custom_{algo}_{env}/metrics.csv."""
    env = gym.make(env_name)
    obs_dim       = env.observation_space.shape[0]
    act_dim       = env.action_space.shape[0]
    action_bounds = (env.action_space.low.tolist(), env.action_space.high.tolist())
    agent = CUSTOM_CLASSES[algo](obs_dim, act_dim, action_bounds)
    env.close()

    algo_tag = f'custom_{algo}'
    trainer  = Trainer(
        agent          = agent,
        env_name       = env_name,
        algo_name      = algo_tag,
        log_dir        = log_dir,
        checkpoint_dir = ckpt_dir,
        save_every     = max(1, n_episodes // 5),
        eval_every     = max(1, n_episodes // 10),
    )
    trainer.train(n_episodes=n_episodes)

    return os.path.join(log_dir, f'{algo_tag}_{env_name}', 'metrics.csv')


def train_sb3(algo, env_name, n_episodes, log_dir):
    """Train an SB3 algorithm and save metrics to log_dir/sb3_{algo}_{env}/metrics.csv."""
    # max_episode_steps=200 for Pendulum; total steps ≈ n_episodes * 200
    env        = Monitor(gym.make(env_name))
    total_steps = n_episodes * 200

    csv_dir  = os.path.join(log_dir, f'sb3_{algo}_{env_name}')
    csv_path = os.path.join(csv_dir, 'metrics.csv')

    model    = SB3_CLASSES[algo]('MlpPolicy', env, verbose=0)
    callback = EpisodeCSVCallback(csv_path)
    model.learn(total_timesteps=total_steps, callback=callback)
    env.close()

    return csv_path


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def load_csv(path):
    episodes, rewards, moving_avgs = [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row['episode']))
            rewards.append(float(row['reward']))
            moving_avgs.append(float(row['moving_avg_reward']))
    return np.array(episodes), np.array(rewards), np.array(moving_avgs)


def plot_sb3_comparison(results, env_name, out_path):
    """
    results: dict  algo -> {'custom': csv_path, 'sb3': csv_path}
    Produces one subplot per algorithm, custom vs SB3 on the same axes.
    """
    n_algos = len(results)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5), squeeze=False)
    fig.suptitle(f'Custom vs Stable-Baselines3 — {env_name}', fontsize=14)

    for ax, (algo, paths) in zip(axes[0], results.items()):
        color = ALGO_COLORS.get(algo, 'black')
        ax.set_title(algo.upper())

        for label, path, ls in [('Custom', paths['custom'], '-'),
                                 ('SB3',    paths['sb3'],    '--')]:
            if path and os.path.exists(path):
                eps, rews, avgs = load_csv(path)
                ax.plot(eps, rews,  color=color, alpha=0.2, linewidth=0.7, linestyle=ls)
                ax.plot(eps, avgs,  color=color, linewidth=2.0, linestyle=ls, label=label)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',      type=str, default='Pendulum-v1')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--algos',    type=str, nargs='+',
                        default=['ddpg', 'td3', 'sac', 'ppo', 'a2c'])
    parser.add_argument('--log_dir',  type=str, default='runs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--plot_dir', type=str, default='plots')
    args = parser.parse_args()

    results = {}

    for algo in args.algos:
        print(f'\n{"="*60}')
        print(f'  Custom {algo.upper()} on {args.env}')
        print(f'{"="*60}')
        custom_csv = train_custom(algo, args.env, args.episodes,
                                  args.log_dir, args.ckpt_dir)

        print(f'\n{"="*60}')
        print(f'  SB3 {algo.upper()} on {args.env}')
        print(f'{"="*60}')
        sb3_csv = train_sb3(algo, args.env, args.episodes, args.log_dir)

        results[algo] = {'custom': custom_csv, 'sb3': sb3_csv}

    os.makedirs(args.plot_dir, exist_ok=True)
    out_path = os.path.join(args.plot_dir, f'vs_sb3_{args.env}.png')
    plot_sb3_comparison(results, args.env, out_path)
    print(f'\nDone. Plot saved to: {out_path}')


if __name__ == '__main__':
    main()
