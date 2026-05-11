import os
import time

import gymnasium as gym
import numpy as np

from rl_continuous.training.logger import Logger


class Trainer:
    """
    Trainer provides a unified training loop for all four algorithms: DDPG, TD3, SAC, PPO.
    The loop follows Morales' structure:
      collect experience -> optimize -> log -> (soft update is inside the agent)
    Off-policy agents (DDPG, TD3, SAC) optimize at every step.
    On-policy agents (PPO) optimize at the end of each episode.
    """
    def __init__(self,
                 agent,
                 env_name,
                 algo_name,
                 log_dir='runs',
                 checkpoint_dir='checkpoints',
                 save_every=50,
                 eval_every=10,
                 n_eval_episodes=5,
                 max_episode_steps=None):
        # (2) Store the agent and environment configuration.
        self.agent           = agent
        self.env_name        = env_name
        self.algo_name       = algo_name
        self.save_every      = save_every
        self.eval_every      = eval_every
        self.n_eval_episodes = n_eval_episodes

        # (3) Build the training and evaluation environments.
        self.train_env = gym.make(env_name, max_episode_steps=max_episode_steps)
        self.eval_env  = gym.make(env_name, max_episode_steps=max_episode_steps)

        # (4) Build the logger for TensorBoard and CSV output.
        self.logger = Logger(log_dir, algo_name, env_name)

        # (5) Prepare checkpoint directory.
        self.checkpoint_dir = os.path.join(checkpoint_dir, f'{algo_name}_{env_name}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # (6) Detect whether the agent is on-policy (PPO) or off-policy (DDPG/TD3/SAC).
        self.is_on_policy = hasattr(agent, 'episode_buffer')

    def train(self, n_episodes=1000):
        """Main training loop: one iteration per episode."""
        total_steps = 0
        best_moving_avg = -float('inf')
        t0 = time.time()

        for episode in range(1, n_episodes + 1):

            if self.is_on_policy:
                ep_reward, ep_steps, value_loss, policy_loss = \
                    self._run_episode_on_policy()
                alpha_loss = None
            else:
                ep_reward, ep_steps, value_loss, policy_loss, alpha_loss = \
                    self._run_episode_off_policy()

            total_steps += ep_steps

            # (8) Log metrics and compute the moving average reward.
            moving_avg = self.logger.log_episode(
                episode, ep_reward, total_steps,
                value_loss, policy_loss, alpha_loss)

            # (9) Print a progress summary every 10 episodes.
            if episode % 10 == 0:
                elapsed = time.time() - t0
                print(f'[{self.algo_name} | {self.env_name}] '
                      f'ep={episode:4d}  reward={ep_reward:8.1f}  '
                      f'avg100={moving_avg:8.1f}  steps={total_steps:7d}  '
                      f'time={elapsed:.0f}s')

            # (10) Periodically evaluate the policy in greedy mode (no exploration noise).
            if episode % self.eval_every == 0:
                eval_reward = self._evaluate()
                self.logger.writer.add_scalar('Reward/eval', eval_reward, episode)

            # (11) Save a checkpoint every save_every episodes and whenever we beat the best score.
            if episode % self.save_every == 0:
                path = os.path.join(self.checkpoint_dir, f'ep_{episode}.pt')
                self.agent.save(path)

            if moving_avg > best_moving_avg:
                best_moving_avg = moving_avg
                path = os.path.join(self.checkpoint_dir, 'best.pt')
                self.agent.save(path)

        self.logger.close()
        self.train_env.close()
        self.eval_env.close()
        print(f'Training complete. Best moving avg reward: {best_moving_avg:.1f}')

    def _run_episode_off_policy(self):
        """
        Off-policy episode loop (DDPG, TD3, SAC):
        collect one transition -> store -> optimize -> repeat.
        """
        state, _ = self.train_env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        value_loss  = None
        policy_loss = None
        alpha_loss  = None

        while True:
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated

            # (13) Store the transition in the replay buffer.
            self.agent.store(state, action, reward, next_state, float(terminated))

            # (14) Optimize the agent at every step (off-policy).
            result = self.agent.optimize()
            if result is not None and result[0] is not None:
                if len(result) == 3:
                    value_loss, policy_loss, alpha_loss = result
                else:
                    value_loss, policy_loss = result

            ep_reward += reward
            ep_steps  += 1
            state = next_state

            if done:
                break

        return ep_reward, ep_steps, value_loss, policy_loss, alpha_loss

    def _run_episode_on_policy(self):
        """
        On-policy episode loop (PPO):
        collect the full episode -> compute GAE -> optimize multiple epochs.
        """
        state, _ = self.train_env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        last_value = 0.0

        while True:
            # (16) PPO's select_action returns (action, logpa, value) during training.
            action, logpa, value = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated

            # (17) Store the full transition including the value estimate and log-probability.
            self.agent.store(state, action, reward, float(value), float(logpa), float(done))

            ep_reward += reward
            ep_steps  += 1
            state = next_state

            if done:
                # (18) If the episode was truncated (not a true terminal), bootstrap V(s_T).
                if truncated and not terminated:
                    import torch
                    with torch.no_grad():
                        _, _, last_value_t = self.agent.select_action(next_state, training=True)
                    last_value = float(last_value_t)
                break

        # (19) Optimize PPO on the collected episode data.
        value_loss, policy_loss = self.agent.optimize(last_value)
        return ep_reward, ep_steps, value_loss, policy_loss

    def _evaluate(self):
        """
        Evaluate the current policy in greedy mode for n_eval_episodes episodes.
        Returns the mean reward across all evaluation episodes.
        """
        total_reward = 0.0
        for _ in range(self.n_eval_episodes):
            state, _ = self.eval_env.reset()
            while True:
                if self.is_on_policy:
                    action = self.agent.select_action(state, training=False)
                else:
                    action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        return total_reward / self.n_eval_episodes
