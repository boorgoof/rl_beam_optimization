"""Unified logger: TensorBoard + CSV for all algorithms."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Log scalar metrics to TensorBoard and a CSV file simultaneously.

    Usage:
        logger = Logger("runs/sac_run1")
        logger.log({"episode_reward": 42.3, "score": 80.1}, step=100)
        logger.close()
    """

    def __init__(self, run_dir: str, algorithm: str = ""):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.run_dir))
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._start_time = time.time()
        self.algorithm = algorithm

    def log(self, metrics: Dict[str, float], step: int):
        """Write metrics to TensorBoard and CSV."""
        # TensorBoard
        for key, val in metrics.items():
            self.tb.add_scalar(key, val, global_step=step)

        # CSV — write header on first call
        metrics_with_meta = {"step": step, "elapsed_s": time.time() - self._start_time,
                              **metrics}
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(metrics_with_meta.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics_with_meta)
        self._csv_file.flush()

    def close(self):
        self.tb.close()
        self._csv_file.close()
