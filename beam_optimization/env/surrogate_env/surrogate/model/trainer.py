"""Offline training for ModularMLP surrogate checkpoints.

This module creates surrogate models from scratch from BeamDataset files. It is
separate from SurrogateDatasetUpdater, which only fine-tunes existing models
with online TraceWin samples.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from beam_optimization.algorithms.utils.logger import Logger
from beam_optimization.config.adige import BEAM_STATE_FEATURES, SCORE_WEIGHTS
from beam_optimization.config.paths import DEFAULT_BASE_SURROGATE_DIR, DEFAULT_SURROGATE_LOG_DIR
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.failure_classifier import (
    FailureClassifier,
    compute_pos_weight,
    derive_failure_labels,
)
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


# Maps each BEAM_STATE_FEATURES entry to the SCORE_WEIGHTS group it belongs to,
# so the training loss prioritizes features the same way score() does instead
# of weighting every feature equally regardless of its raw numeric scale.
_FEATURE_WEIGHT_GROUPS = {
    "npart_ratio": "npart_ratio",
    "ex": "emittance", "ey": "emittance",
    "x0": "offset", "y0": "offset",
    "x'0": "angle", "y'0": "angle",
    "SizeX": "size", "SizeY": "size",
}


def build_feature_loss_weights() -> torch.Tensor:
    """Per-feature loss weight vector, in BEAM_STATE_FEATURES order, from SCORE_WEIGHTS."""
    return torch.tensor(
        [SCORE_WEIGHTS[_FEATURE_WEIGHT_GROUPS[name]] for name in BEAM_STATE_FEATURES],
        dtype=torch.float32,
    )


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights).mean()


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
        weight_decay: float = 1e-4,
        patience: Optional[int] = 40,
        seed: int = 123,
        device: Optional[str | torch.device] = None,
        overwrite: bool = False,
        model_kwargs: Optional[dict] = None,
        log_dir: Optional[str | Path] = None,
        enable_tensorboard: bool = True,
        train_classifier: bool = True,
        classifier_patience: Optional[int] = 20,
    ):
        self.train_dataset_path = Path(train_dataset_path)
        self.val_dataset_path = Path(val_dataset_path) if val_dataset_path else None
        self.output_dir = Path(output_dir)
        self.n_models = int(n_models)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience) if patience is not None else None
        self.seed = int(seed)
        self.train_classifier = bool(train_classifier)
        self.classifier_patience = int(classifier_patience) if classifier_patience is not None else None
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
        self.feature_loss_weights = build_feature_loss_weights().to(self.device)

        if self.n_models <= 0:
            raise ValueError("n_models must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be positive when given")
        if self.classifier_patience is not None and self.classifier_patience <= 0:
            raise ValueError("classifier_patience must be positive when given")

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
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=15,
            )
            save_path = self._checkpoint_path(local_index)
            print(
                f"\nTraining {save_path.stem} on {self.device} "
                f"({len(train_dataset):,} train, "
                f"{len(val_dataset) if val_dataset is not None else 0:,} validation samples)",
                flush=True,
            )
            logger = (
                Logger(self.log_dir / save_path.stem, algorithm="surrogate")
                if self.enable_tensorboard
                else None
            )

            try:
                history, best_state, best_val_loss = self._train_one(
                    model,
                    optimizer,
                    scheduler,
                    train_dataset,
                    val_dataset,
                    logger=logger,
                    progress_label=save_path.stem,
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
                        "patience": self.patience,
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

        summary = {
            "output_dir": str(self.output_dir),
            "n_models": self.n_models,
            "checkpoints": saved,
        }

        if self.train_classifier:
            classifier_path = self._classifier_checkpoint_path()
            print(f"\nTraining {classifier_path.stem} on {self.device}", flush=True)
            model, history, best_val_loss = self._train_classifier(
                train_dataset, val_dataset, norm_stats,
            )
            model.save(
                str(classifier_path),
                extra={
                    "normalization_metadata": norm_stats,
                    "training_metadata": {
                        "trainer": type(self).__name__,
                        "seed": self.seed + self.n_models,
                        "max_epochs": self.max_epochs,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                        "classifier_patience": self.classifier_patience,
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
            summary["classifier"] = {
                "path": str(classifier_path),
                "best_val_loss": best_val_loss,
                "final_val_metrics": {
                    k: v for k, v in history[-1].items()
                    if k in ("precision", "recall", "f1")
                },
            }

        return summary

    def _train_one(
        self,
        model: ModularMLP,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        train_dataset: BeamDataset,
        val_dataset: Optional[BeamDataset],
        logger: Optional[Logger] = None,
        progress_label: str = "surrogate",
    ) -> tuple[list[dict], dict, float]:
        history: list[dict] = []
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        training_started = time.monotonic()

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
                    _weighted_mse(pred, target, self.feature_loss_weights)
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
                self._evaluate_loss(model, val_dataset)
                if val_dataset is not None and len(val_dataset) > 0
                else train_loss
            )
            scheduler.step(val_loss)
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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

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

            elapsed = time.monotonic() - training_started
            print(
                f"  [{progress_label}] epoch {epoch:03d}/{self.max_epochs} "
                f"train_loss={train_loss:.6g} val_loss={val_loss:.6g} "
                f"best_val={best_val_loss:.6g} lr={optimizer.param_groups[0]['lr']:.3g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            if self.patience is not None and epochs_without_improvement >= self.patience:
                print(
                    f"  [{progress_label}] early stopping: no val_loss improvement "
                    f"for {self.patience} epochs",
                    flush=True,
                )
                break

        return history, best_state, best_val_loss

    def _evaluate_loss(
        self,
        model: ModularMLP,
        dataset: BeamDataset,
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
                loss = sum(
                    loss_weight * _weighted_mse(pred, target, self.feature_loss_weights)
                    for pred, target in pred_targets
                )
                losses.append(float(loss.detach().cpu()))

        return float(np.mean(losses)) if losses else float("inf")

    def _checkpoint_path(self, local_index: int) -> Path:
        dataset_name = Path(self.train_dataset_path).resolve().parent.name
        if self.overwrite:
            return self.output_dir / f"surrogate_{dataset_name}_{local_index}.pt"

        index = 0
        while True:
            candidate = self.output_dir / f"surrogate_{dataset_name}_{index}.pt"
            if not candidate.exists():
                return candidate
            index += 1

    def _classifier_checkpoint_path(self) -> Path:
        """A single, shared checkpoint (not enumerated per ensemble member,
        unlike _checkpoint_path): all-particles-lost is a property of the
        physics/dataset, not per-model uncertainty, so one classifier is
        trained and reused everywhere. Always overwritten on retrain,
        independent of self.overwrite."""
        dataset_name = Path(self.train_dataset_path).resolve().parent.name
        return self.output_dir / f"failure_classifier_{dataset_name}.pt"

    def _train_classifier(
        self,
        train_dataset: BeamDataset,
        val_dataset: Optional[BeamDataset],
        norm_stats: dict,
    ) -> tuple[FailureClassifier, list[dict], float]:
        """Train the single, shared FailureClassifier once (not once per
        ensemble member). Reuses the same AdamW/ReduceLROnPlateau/early-
        stopping shape as _train_one(), but with a BCEWithLogitsLoss(pos_weight=...)
        classification loss and precision/recall/F1 tracked instead of a
        stage-weighted regression loss."""
        _seed_everything(self.seed + self.n_models)
        model = FailureClassifier(norm_stats=norm_stats).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15,
        )

        train_labels = derive_failure_labels(train_dataset.Y)
        pos_weight = compute_pos_weight(train_labels).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        history: list[dict] = []
        best_state = copy.deepcopy(model.state_dict())
        best_val_loss = float("inf")
        best_val_f1 = float("-inf")
        epochs_without_improvement = 0
        training_started = time.monotonic()

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            losses = []
            indices = np.random.permutation(len(train_dataset))

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                stage_params, beam_states = train_dataset.get_training_batch(batch_idx)
                stage_params = [tensor.to(self.device) for tensor in stage_params]
                beam0 = beam_states[0].to(self.device)
                labels = derive_failure_labels(train_dataset.Y[batch_idx]).to(self.device)

                logits = model(stage_params, beam0)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            train_loss = float(np.mean(losses)) if losses else float("nan")

            if val_dataset is not None and len(val_dataset) > 0:
                val_loss, val_metrics = self._evaluate_classifier(model, val_dataset, criterion)
            else:
                val_loss, val_metrics = train_loss, {
                    "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
                }

            scheduler.step(val_loss)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **val_metrics,
            })

            has_f1 = not np.isnan(val_metrics["f1"])
            improved = val_metrics["f1"] > best_val_f1 if has_f1 else val_loss < best_val_loss
            if improved:
                if has_f1:
                    best_val_f1 = val_metrics["f1"]
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            elapsed = time.monotonic() - training_started
            print(
                f"  [failure_classifier] epoch {epoch:03d}/{self.max_epochs} "
                f"train_loss={train_loss:.6g} val_loss={val_loss:.6g} "
                f"precision={val_metrics['precision']:.3g} recall={val_metrics['recall']:.3g} "
                f"f1={val_metrics['f1']:.3g} lr={optimizer.param_groups[0]['lr']:.3g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            if (
                self.classifier_patience is not None
                and epochs_without_improvement >= self.classifier_patience
            ):
                print(
                    f"  [failure_classifier] early stopping: no val improvement "
                    f"for {self.classifier_patience} epochs",
                    flush=True,
                )
                break

        model.load_state_dict(best_state)
        return model, history, best_val_loss

    def _evaluate_classifier(
        self,
        model: FailureClassifier,
        dataset: BeamDataset,
        criterion: torch.nn.Module,
        threshold: float = 0.5,
    ) -> tuple[float, dict]:
        model.eval()
        losses = []
        tp = fp = fn = tn = 0.0

        with torch.no_grad():
            for start in range(0, len(dataset), self.batch_size):
                stop = min(start + self.batch_size, len(dataset))
                indices = np.arange(start, stop)
                stage_params, beam_states = dataset.get_training_batch(indices)
                stage_params = [tensor.to(self.device) for tensor in stage_params]
                beam0 = beam_states[0].to(self.device)
                labels = derive_failure_labels(dataset.Y[indices]).to(self.device)

                logits = model(stage_params, beam0)
                losses.append(float(criterion(logits, labels).detach().cpu()))

                preds = (torch.sigmoid(logits) > threshold).float()
                tp += float(((preds == 1) & (labels == 1)).sum())
                fp += float(((preds == 1) & (labels == 0)).sum())
                fn += float(((preds == 0) & (labels == 1)).sum())
                tn += float(((preds == 0) & (labels == 0)).sum())

        val_loss = float(np.mean(losses)) if losses else float("inf")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
            else float("nan")
        )
        return val_loss, {"precision": precision, "recall": recall, "f1": f1}


def train_surrogate(
    train_dataset_path: str | Path,
    val_dataset_path: Optional[str | Path] = None,
    output_dir: str | Path = DEFAULT_BASE_SURROGATE_DIR,
    *,
    n_models: int = 1,
    max_epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: Optional[int] = 40,
    seed: int = 123,
    device: Optional[str | torch.device] = None,
    overwrite: bool = False,
    model_kwargs: Optional[dict] = None,
    log_dir: Optional[str | Path] = None,
    enable_tensorboard: bool = True,
    train_classifier: bool = True,
    classifier_patience: Optional[int] = 20,
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
        patience=patience,
        seed=seed,
        device=device,
        overwrite=overwrite,
        model_kwargs=model_kwargs,
        log_dir=log_dir,
        enable_tensorboard=enable_tensorboard,
        train_classifier=train_classifier,
        classifier_patience=classifier_patience,
    )
    return trainer.train()


def compute_normalization_metadata(dataset: BeamDataset) -> dict:
    """Compute ModularMLP normalization statistics from a BeamDataset.

    npart_ratio at output stages (index 1..N_OUTPUT_STAGES of beam_states) is
    logit-transformed before its mean/variance are computed -- see
    ModularMLP._apply_physical_bounds(), which correspondingly applies
    sigmoid() (not a clamp) to reconstruct that one column from the
    logit-space denormalized value. beam0 (stage 0, the network's input) is
    left untransformed: it is always exactly 1.0 in this dataset and is only
    ever consumed by _norm_beam(), not the output-side denorm+bounds path.
    """
    stage_params, beam_states = dataset.get_training_batch(np.arange(len(dataset)))
    npart_idx = BEAM_STATE_FEATURES.index("npart_ratio")

    transformed_beam_states = []
    for stage_idx, tensor in enumerate(beam_states):
        if stage_idx == 0:
            transformed_beam_states.append(tensor)
            continue
        columns = list(torch.unbind(tensor, dim=1))
        columns[npart_idx] = torch.logit(columns[npart_idx], eps=1e-4)
        transformed_beam_states.append(torch.stack(columns, dim=1))

    return {
        "parameter_means": [tensor.mean(dim=0).detach().cpu() for tensor in stage_params],
        "parameter_variances": [
            tensor.var(dim=0, unbiased=False).detach().cpu()
            for tensor in stage_params
        ],
        "beam_state_means": [tensor.mean(dim=0).detach().cpu() for tensor in transformed_beam_states],
        "beam_state_variances": [
            tensor.var(dim=0, unbiased=False).detach().cpu()
            for tensor in transformed_beam_states
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
