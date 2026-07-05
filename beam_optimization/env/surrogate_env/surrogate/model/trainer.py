"""Offline training for ModularMLP surrogate checkpoints.

This module creates surrogate models from scratch from BeamDataset files. It is
separate from SurrogateDatasetUpdater, which only fine-tunes existing models
with online TraceWin samples.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from beam_optimization.algorithms.utils.logger import Logger
from beam_optimization.config.paths import DEFAULT_BASE_SURROGATE_DIR, DEFAULT_SURROGATE_LOG_DIR
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


class SurrogateTrainer:
    """Train one or more ModularMLP surrogates from BeamDataset files."""

    def __init__(
        self,
        train_dataset_path: str | Path,
        val_dataset_path: Optional[str | Path] = None,
        output_dir: str | Path = DEFAULT_BASE_SURROGATE_DIR,
        *,
        n_models: int = 1,
        max_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        seed: int = 123,
        device: Optional[str | torch.device] = None,
        overwrite: bool = False,
        model_kwargs: Optional[dict] = None,
        log_dir: Optional[str | Path] = None,
        enable_tensorboard: bool = True,
    ):
        self.train_dataset_path = Path(train_dataset_path)
        self.val_dataset_path = Path(val_dataset_path) if val_dataset_path else None
        self.output_dir = Path(output_dir)
        self.n_models = int(n_models)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.overwrite = bool(overwrite)
        self.model_kwargs = dict(model_kwargs or {})
        self.enable_tensorboard = bool(enable_tensorboard)
        self.log_dir = (
            Path(log_dir)
            if log_dir is not None
            else DEFAULT_SURROGATE_LOG_DIR / self.train_dataset_path.parent.name
        )

        if self.n_models <= 0:
            raise ValueError("n_models must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def train(self) -> dict:
        """Train the requested surrogate checkpoints and return a summary."""
        train_dataset = BeamDataset.load(self.train_dataset_path)
        val_dataset = (
            BeamDataset.load(self.val_dataset_path)
            if self.val_dataset_path is not None
            else None
        )
        if len(train_dataset) == 0:
            raise ValueError("Cannot train a surrogate on an empty train dataset")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        norm_stats = compute_normalization_metadata(train_dataset)
        saved = []

        for local_index in range(self.n_models):
            model_seed = self.seed + local_index
            _seed_everything(model_seed)
            model = ModularMLP(norm_stats=norm_stats, **self.model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            save_path = self._checkpoint_path(local_index)
            logger = (
                Logger(self.log_dir / save_path.stem, algorithm="surrogate")
                if self.enable_tensorboard
                else None
            )

            try:
                history, best_state, best_val_loss = self._train_one(
                    model,
                    optimizer,
                    train_dataset,
                    val_dataset,
                    logger=logger,
                )
            finally:
                if logger is not None:
                    logger.close()
            model.load_state_dict(best_state)

            model.save(
                str(save_path),
                extra={
                    "model_config": _model_config(self.model_kwargs),
                    "normalization_metadata": norm_stats,
                    "training_metadata": {
                        "trainer": type(self).__name__,
                        "seed": model_seed,
                        "max_epochs": self.max_epochs,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                        "n_train_samples": len(train_dataset),
                        "n_val_samples": len(val_dataset) if val_dataset is not None else 0,
                        "history": history,
                    },
                    "best_val_loss": best_val_loss,
                    "train_dataset_path": str(self.train_dataset_path),
                    "val_dataset_path": (
                        str(self.val_dataset_path)
                        if self.val_dataset_path is not None
                        else None
                    ),
                },
            )
            saved.append(
                {
                    "path": str(save_path),
                    "best_val_loss": best_val_loss,
                    "final_train_loss": history[-1]["train_loss"],
                }
            )

        return {
            "output_dir": str(self.output_dir),
            "n_models": self.n_models,
            "checkpoints": saved,
        }

    def _train_one(
        self,
        model: ModularMLP,
        optimizer: torch.optim.Optimizer,
        train_dataset: BeamDataset,
        val_dataset: Optional[BeamDataset],
        logger: Optional[Logger] = None,
    ) -> tuple[list[dict], dict, float]:
        criterion = nn.MSELoss()
        history: list[dict] = []
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            losses = []
            grad_norms = []
            stage_loss_sums: list[float] | None = None
            n_stage_batches = 0
            indices = np.random.permutation(len(train_dataset))

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                stage_params, beam_states = train_dataset.get_training_batch(batch_idx)
                stage_params = [tensor.to(self.device) for tensor in stage_params]
                beam_states = [tensor.to(self.device) for tensor in beam_states]
                targets = beam_states[1:]

                preds = model(stage_params, beam_states[0])
                pred_targets = _prediction_pairs(preds, targets)
                loss_weight = 1.0 / len(pred_targets)
                stage_losses = [
                    criterion(pred, target)
                    for pred, target in pred_targets
                ]
                loss = sum(loss_weight * stage_loss for stage_loss in stage_losses)

                if stage_loss_sums is None:
                    stage_loss_sums = [0.0 for _ in stage_losses]
                for i, stage_loss in enumerate(stage_losses):
                    stage_loss_sums[i] += float(stage_loss.detach().cpu())
                n_stage_batches += 1

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                grad_norms.append(float(grad_norm.detach().cpu()))

            train_loss = float(np.mean(losses)) if losses else float("nan")
            grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else float("nan")
            stage_loss_means = (
                [value / n_stage_batches for value in stage_loss_sums]
                if stage_loss_sums is not None and n_stage_batches > 0
                else []
            )
            val_loss = (
                self._evaluate_loss(model, val_dataset, criterion)
                if val_dataset is not None and len(val_dataset) > 0
                else train_loss
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "grad_norm": grad_norm_mean,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

            if logger is not None:
                metrics = {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "grad_norm": grad_norm_mean,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": float(epoch),
                }
                for i, stage_loss in enumerate(stage_loss_means, start=1):
                    metrics[f"stage_{i}_train_loss"] = stage_loss
                logger.log(metrics, step=epoch)

        return history, best_state, best_val_loss

    def _evaluate_loss(
        self,
        model: ModularMLP,
        dataset: BeamDataset,
        criterion: nn.Module,
    ) -> float:
        model.eval()
        losses = []

        with torch.no_grad():
            for start in range(0, len(dataset), self.batch_size):
                stop = min(start + self.batch_size, len(dataset))
                indices = np.arange(start, stop)
                stage_params, beam_states = dataset.get_training_batch(indices)
                stage_params = [tensor.to(self.device) for tensor in stage_params]
                beam_states = [tensor.to(self.device) for tensor in beam_states]
                targets = beam_states[1:]
                preds = model(stage_params, beam_states[0])
                pred_targets = _prediction_pairs(preds, targets)
                loss_weight = 1.0 / len(pred_targets)
                loss = sum(loss_weight * criterion(pred, target)
                           for pred, target in pred_targets)
                losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else float("inf")

    def _checkpoint_path(self, local_index: int) -> Path:
        if self.overwrite:
            return self.output_dir / f"surrogate_{local_index}.pt"

        index = 0
        while True:
            candidate = self.output_dir / f"surrogate_{index}.pt"
            if not candidate.exists():
                return candidate
            index += 1


def train_surrogate(
    train_dataset_path: str | Path,
    val_dataset_path: Optional[str | Path] = None,
    output_dir: str | Path = DEFAULT_BASE_SURROGATE_DIR,
    *,
    n_models: int = 1,
    max_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 123,
    device: Optional[str | torch.device] = None,
    overwrite: bool = False,
    model_kwargs: Optional[dict] = None,
    log_dir: Optional[str | Path] = None,
    enable_tensorboard: bool = True,
) -> dict:
    """Convenience wrapper around SurrogateTrainer."""
    trainer = SurrogateTrainer(
        train_dataset_path,
        val_dataset_path,
        output_dir,
        n_models=n_models,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        overwrite=overwrite,
        model_kwargs=model_kwargs,
        log_dir=log_dir,
        enable_tensorboard=enable_tensorboard,
    )
    return trainer.train()


def compute_normalization_metadata(dataset: BeamDataset) -> dict:
    """Compute ModularMLP normalization statistics from a BeamDataset."""
    stage_params, beam_states = dataset.get_training_batch(np.arange(len(dataset)))
    return {
        "parameter_means": [tensor.mean(dim=0).detach().cpu() for tensor in stage_params],
        "parameter_variances": [
            tensor.var(dim=0, unbiased=False).detach().cpu()
            for tensor in stage_params
        ],
        "beam_state_means": [tensor.mean(dim=0).detach().cpu() for tensor in beam_states],
        "beam_state_variances": [
            tensor.var(dim=0, unbiased=False).detach().cpu()
            for tensor in beam_states
        ],
    }


def _prediction_pairs(preds, targets):
    if isinstance(preds, torch.Tensor):
        return [(preds, targets[-1])]
    return list(zip(preds, targets))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_config(model_kwargs: dict) -> dict:
    return {
        "hidden_sizes": list(model_kwargs.get("hidden_sizes", [256, 256, 256])),
        "dropout": float(model_kwargs.get("dropout", 0.15)),
        "latent_dim": int(model_kwargs.get("latent_dim", 64)),
        "out_hidden": list(model_kwargs.get("out_hidden", [256, 256])),
        "out_dropout": float(model_kwargs.get("out_dropout", 0.15)),
    }
