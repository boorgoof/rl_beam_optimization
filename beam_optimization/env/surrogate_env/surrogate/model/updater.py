"""
SurrogateDatasetUpdater centralizes TraceWin data ingestion and surrogate fine-tuning.

It is the online component that accepts BeamSimulationResult objects from
TraceWin during MBPO/model-update workflows. It uses the common dataset utility
to convert them to BeamDataset's flat format, stores new samples in an online
dataset, and fine-tunes the surrogate ensemble on online data alone or on a mix
of offline + online data.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from beam_optimization.config.adige import N_OUTPUT_STAGES
from beam_optimization.env.dataset import BeamDataset, tracewin_result_to_flat_sample
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


class SurrogateDatasetUpdater:
    """Collect TraceWin samples, update BeamDataset, and fine-tune surrogates.

    Args:
        surrogates: One ModularMLP or an ensemble of ModularMLPs.
        offline_dataset: Optional seed dataset used to mix old and new samples
            during fine-tuning.
        model_dir: Optional output directory for fine-tuned surrogate weights.
        lr: Adam learning rate.
        batch_size: Batch size for each fine-tuning call.
        epochs: Gradient steps per fine-tuning call.
        min_samples: Minimum online samples required before update().
        online_mix_ratio: Target fraction of each batch drawn from online data
            when an offline dataset is available.
        device: Torch device string or None for auto-detect.
        online_dataset_save_path: Optional default path for save_online_dataset().
        merged_dataset_save_path: Optional default path for save_merged_dataset().
    """

    def __init__(
        self,
        surrogates: Union[ModularMLP, List[ModularMLP]],
        offline_dataset: Optional[BeamDataset] = None,
        model_dir: Optional[str | Path] = None,
        lr: float = 3e-5,
        batch_size: int = 64,
        epochs: int = 20,
        min_samples: int = 20,
        online_mix_ratio: float = 1.0,
        device: Optional[str] = None,
        online_dataset_save_path: Optional[str | Path] = None,
        merged_dataset_save_path: Optional[str | Path] = None,
        flat_save_path: Optional[str | Path] = None,
    ):
        self.surrogates = surrogates if isinstance(surrogates, list) else [surrogates]
        self._offline_dataset = offline_dataset
        self._online_dataset = BeamDataset()

        self.model_dir = Path(model_dir) if model_dir else None
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.min_samples = int(min_samples)
        self.online_mix_ratio = min(max(float(online_mix_ratio), 0.0), 1.0)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Backward-compatible flat_save_path means "save the newly collected
        # online samples", matching the old updater's internal dataset behavior.
        self.online_dataset_save_path = Path(
            online_dataset_save_path or flat_save_path
        ) if (online_dataset_save_path or flat_save_path) else None
        self.merged_dataset_save_path = (
            Path(merged_dataset_save_path) if merged_dataset_save_path else None
        )

        for surrogate in self.surrogates:
            surrogate.to(self.device)

        self._optimizers = [
            torch.optim.Adam(surrogate.parameters(), lr=lr)
            for surrogate in self.surrogates
        ]
        self._update_count = 0

    # ── Datasets / counters ───────────────────────────────────────────────────

    @property
    def online_dataset(self) -> BeamDataset:
        """New TraceWin samples collected during this run."""
        return self._online_dataset

    @property
    def offline_dataset(self) -> Optional[BeamDataset]:
        """Seed/pretraining dataset, if configured."""
        return self._offline_dataset

    @property
    def n_online_samples(self) -> int:
        """Number of new TraceWin samples collected in this run."""
        return len(self._online_dataset)

    @property
    def n_samples(self) -> int:
        """Backward-compatible alias for n_online_samples."""
        return self.n_online_samples

    # ── TraceWin ingestion ────────────────────────────────────────────────────

    def add_tracewin_result(self, result: BeamSimulationResult) -> bool:
        """Add one successful TraceWin result to the online dataset."""
        if result.source != "tracewin" or not result.success or result.beam_states is None:
            return False

        x, y, score = tracewin_result_to_flat_sample(result)
        self._online_dataset.append_flat_sample(x, y, score)
        return True

    def add_many_tracewin_results(self, results: List[BeamSimulationResult]) -> int:
        """Add many TraceWin results and return how many were accepted."""
        return sum(1 for result in results if self.add_tracewin_result(result))

    # Legacy names kept for older scripts/tests.
    def add(self, result: BeamSimulationResult) -> bool:
        return self.add_tracewin_result(result)

    def add_many(self, results: List[BeamSimulationResult]) -> int:
        return self.add_many_tracewin_results(results)

    # ── Fine-tuning ───────────────────────────────────────────────────────────

    def update(self) -> Optional[dict]:
        """Fine-tune each surrogate on online or mixed offline+online batches."""
        if self.n_online_samples < self.min_samples:
            print(
                f"[SurrogateDatasetUpdater] skip: {self.n_online_samples} "
                f"< {self.min_samples} campioni online minimi"
            )
            return None

        criterion = nn.MSELoss()
        stage_w = 1.0 / N_OUTPUT_STAGES
        final_losses = {}

        for i, (surrogate, optimizer) in enumerate(zip(self.surrogates, self._optimizers)):
            surrogate.train()
            for param in surrogate.parameters():
                param.requires_grad_(True)

            last_loss = 0.0
            for _ in range(self.epochs):
                stage_t, beam0_t, targets_t = self._collate_mixed(self.batch_size)
                preds = surrogate(stage_t, beam0_t)
                loss = sum(stage_w * criterion(pred, target)
                           for pred, target in zip(preds, targets_t))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(surrogate.parameters(), max_norm=1.0)
                optimizer.step()
                last_loss = float(loss.detach())

            surrogate.eval()
            final_losses[f"surrogate_{i}"] = last_loss

        self._update_count += 1
        print(
            f"[SurrogateDatasetUpdater] update #{self._update_count} "
            f"online={self.n_online_samples} "
            f"losses={{{', '.join(f'{k}: {v:.5f}' for k, v in final_losses.items())}}}"
        )

        if self.online_dataset_save_path is not None:
            self.save_online_dataset(self.online_dataset_save_path)
        if self.merged_dataset_save_path is not None:
            self.save_merged_dataset(self.merged_dataset_save_path)

        return final_losses

    def update_if_ready(self) -> Optional[dict]:
        """Call update() only when enough online samples have been collected."""
        if self.n_online_samples >= self.min_samples:
            return self.update()
        return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_online_dataset(self, path: Optional[str | Path] = None) -> int:
        """Save only the newly collected online TraceWin samples."""
        save_path = Path(path) if path is not None else self.online_dataset_save_path
        if save_path is None:
            raise ValueError("Specifica un path per save_online_dataset()")
        self._online_dataset.save_flat(save_path)
        return len(self._online_dataset)

    def save_merged_dataset(self, path: Optional[str | Path] = None) -> int:
        """Save offline + online samples as one flat dataset."""
        save_path = Path(path) if path is not None else self.merged_dataset_save_path
        if save_path is None:
            raise ValueError("Specifica un path per save_merged_dataset()")

        merged = self._online_dataset
        if self._offline_dataset is not None:
            merged = self._offline_dataset.merge(self._online_dataset)
        merged.save_flat(save_path)
        return len(merged)

    def save_surrogates(self, model_dir: Optional[str | Path] = None) -> None:
        """Save fine-tuned surrogate weights as surrogate_0.pt ... surrogate_N.pt."""
        save_dir = Path(model_dir) if model_dir is not None else self.model_dir
        if save_dir is None:
            raise ValueError("Specifica model_dir nel costruttore o in save_surrogates()")
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, surrogate in enumerate(self.surrogates):
            surrogate.save(
                str(save_dir / f"surrogate_{i}.pt"),
                extra={
                    "normalization_metadata": surrogate._norm_stats,
                    "update_count": self._update_count,
                    "online_samples": self.n_online_samples,
                },
            )
        print(
            f"[SurrogateDatasetUpdater] salvati {len(self.surrogates)} "
            f"surrogati fine-tunati in {save_dir}"
        )

    # Legacy names kept for older scripts/tests.
    def export_flat(self, path: str | Path) -> int:
        return self.save_online_dataset(path)

    def save(self, model_dir: Optional[str | Path] = None) -> None:
        self.save_surrogates(model_dir)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collate_mixed(self, n_total: int):
        """Build one training batch from online data, optionally mixed with offline."""
        n_total = int(n_total)
        n_online_available = len(self._online_dataset)

        if self._offline_dataset is None or len(self._offline_dataset) == 0:
            n_online = min(n_total, n_online_available)
            if n_online == 0:
                raise ValueError("No online samples available for fine-tuning")
            idx_on = np.random.choice(n_online_available, size=n_online, replace=True)
            stage, beam = self._online_dataset.get_training_batch(idx_on)
        else:
            n_online = min(int(n_total * self.online_mix_ratio), n_online_available)
            if n_online == 0:
                n_online = min(n_total, n_online_available)
            n_offline = n_total - n_online

            idx_off = np.random.choice(
                len(self._offline_dataset),
                size=n_offline,
                replace=True,
            )
            stage, beam = self._offline_dataset.get_training_batch(idx_off)

            if n_online > 0:
                idx_on = np.random.choice(n_online_available, size=n_online, replace=True)
                stage_on, beam_on = self._online_dataset.get_training_batch(idx_on)
                stage = [torch.cat([old, new], dim=0)
                         for old, new in zip(stage, stage_on)]
                beam = [torch.cat([old, new], dim=0)
                        for old, new in zip(beam, beam_on)]

        stage_t = [tensor.to(self.device) for tensor in stage]
        beam_all = [tensor.to(self.device) for tensor in beam]
        return stage_t, beam_all[0], beam_all[1:]


# Legacy alias kept so older imports and scripts continue to work.
SurrogateUpdater = SurrogateDatasetUpdater
