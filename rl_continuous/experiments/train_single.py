"""
Train a single RL algorithm on a Gym environment.

Usage:
    python experiments/train_single.py --algo sac --env Pendulum-v1 --episodes 300
    python experiments/train_single.py --algo td3 --env HalfCheetah-v4 --episodes 1000
    python experiments/train_single.py --algo ppo --env Hopper-v4 --episodes 2000
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


def make_agent(algo, env):
    obs_dim      = env.observation_space.shape[0]
    act_dim      = env.action_space.shape[0]
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
    else:
        raise ValueError(f'Unknown algorithm: {algo}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',     type=str, default='sac',
                        choices=['ddpg', 'td3', 'sac', 'ppo', 'a2c'])
    parser.add_argument('--env',      type=str, default='Pendulum-v1')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--log_dir',  type=str, default='runs')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--seed',     type=int, default=42)
    args = parser.parse_args()

    # Build a temporary env just to read dimensions.
    tmp_env = gym.make(args.env)
    agent   = make_agent(args.algo, tmp_env)
    tmp_env.close()

    trainer = Trainer(
        agent        = agent,
        env_name     = args.env,
        algo_name    = args.algo,
        log_dir      = args.log_dir,
        checkpoint_dir = args.ckpt_dir,
        save_every   = max(1, args.episodes // 10),
        eval_every   = max(1, args.episodes // 20),
    )
    trainer.train(n_episodes=args.episodes)


if __name__ == '__main__':
    main()
