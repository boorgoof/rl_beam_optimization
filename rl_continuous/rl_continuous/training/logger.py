import csv
import os
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger handles all metric tracking for a training run.
    It writes to two backends in parallel:
      - TensorBoard (SummaryWriter) for interactive visualizations
      - CSV file for offline matplotlib plotting and comparisons
    """
    def __init__(self, log_dir, algo_name, env_name):
        # (2) Create the TensorBoard writer pointing to the run-specific directory.
        run_dir = os.path.join(log_dir, f'{algo_name}_{env_name}')
        self.writer = SummaryWriter(log_dir=run_dir)
        # (3) Open the CSV file for writing tabular metrics.
        csv_path = os.path.join(run_dir, 'metrics.csv')
        os.makedirs(run_dir, exist_ok=True)
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ['episode', 'reward', 'moving_avg_reward',
             'value_loss', 'policy_loss', 'alpha_loss', 'total_steps'])
        # (4) Maintain a sliding window of the last 100 episode rewards for the moving average.
        self.reward_window = deque(maxlen=100)
        self.algo_name = algo_name
        self.env_name  = env_name

    def log_episode(self, episode, reward, total_steps,
                    value_loss=None, policy_loss=None, alpha_loss=None):
        """Update the reward sliding window, compute the moving average, and write to TensorBoard and CSV."""
        self.reward_window.append(reward)
        moving_avg = np.mean(self.reward_window)

        # (6) Write scalar metrics to TensorBoard for real-time monitoring.
        self.writer.add_scalar('Reward/episode',     reward,     episode)
        self.writer.add_scalar('Reward/moving_avg',  moving_avg, episode)
        self.writer.add_scalar('Steps/total',        total_steps, episode)
        if value_loss  is not None:
            self.writer.add_scalar('Loss/value',  value_loss,  episode)
        if policy_loss is not None:
            self.writer.add_scalar('Loss/policy', policy_loss, episode)
        if alpha_loss  is not None:
            self.writer.add_scalar('Loss/alpha',  alpha_loss,  episode)

        # (7) Append the same metrics to the CSV file for offline analysis.
        self.csv_writer.writerow([
            episode, reward, moving_avg,
            value_loss, policy_loss, alpha_loss, total_steps])
        self.csv_file.flush()

        return moving_avg

    def close(self):
        """Close TensorBoard writer and CSV file cleanly when training ends."""
        self.writer.close()
        self.csv_file.close()
