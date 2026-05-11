"""
Train all algorithms on the same environment and save comparison plots.

Usage:
    python experiments/compare_all.py --env Pendulum-v1 --episodes 300
    python experiments/compare_all.py --env HalfCheetah-v4 --episodes 1000
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym

from rl_continuous.algorithms.ddpg import DDPG
from rl_continuous.algorithms.td3 import TD3
from rl_continuous.algorithms.sac import SAC
from rl_continuous.algorithms.ppo import PPO
from rl_continuous.algorithms.a2c import A2C
from rl_continuous.training.trainer import Trainer
from plots.plot_results import plot_comparison


ALGOS = ['ddpg', 'td3', 'sac', 'ppo', 'a2c']


def make_agent(algo, env):
    obs_dim       = env.observation_space.shape[0]
    act_dim       = env.action_space.shape[0]
    action_bounds = (env.action_space.low.tolist(), env.action_space.high.tolist())
    if algo == 'ddpg':
        return DDPG(obs_dim, act_dim, action_bounds)
    elif algo == 'td3':
        return TD3(obs_dim, act_dim, action_bounds)
    elif algo == 'sac':
        return SAC(obs_dim, act_dim, action_bounds)
    elif algo == 'ppo':
        return PPO(obs_dim, act_dim, action_bounds)
    elif algo == 'a2c':
        return A2C(obs_dim, act_dim, action_bounds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',      type=str, default='Pendulum-v1')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--log_dir',  type=str, default='runs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--plot_dir', type=str, default='plots')
    args = parser.parse_args()

    csv_paths = {}

    for algo in ALGOS:
        print(f'\n{"="*60}')
        print(f'  Training {algo.upper()} on {args.env}')
        print(f'{"="*60}')

        tmp_env = gym.make(args.env)
        agent   = make_agent(algo, tmp_env)
        tmp_env.close()

        trainer = Trainer(
            agent          = agent,
            env_name       = args.env,
            algo_name      = algo,
            log_dir        = args.log_dir,
            checkpoint_dir = args.ckpt_dir,
            save_every     = max(1, args.episodes // 5),
            eval_every     = max(1, args.episodes // 10),
        )
        trainer.train(n_episodes=args.episodes)

        run_dir = os.path.join(args.log_dir, f'{algo}_{args.env}')
        csv_paths[algo] = os.path.join(run_dir, 'metrics.csv')

    # After all runs, generate the comparison plot.
    os.makedirs(args.plot_dir, exist_ok=True)
    out_path = os.path.join(args.plot_dir, f'comparison_{args.env}.png')
    plot_comparison(csv_paths, args.env, out_path)
    print(f'\nComparison plot saved to: {out_path}')


if __name__ == '__main__':
    main()
