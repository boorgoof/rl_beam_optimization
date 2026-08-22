"""Offline surrogate evaluation on an independent BeamDataset split.

Reports native and target-standardized regression metrics by stage/feature,
valid-vs-failure score decomposition, bootstrap confidence intervals and the
true-vs-predicted final score correlation.
"""
from __future__ import annotations

import json
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from beam_optimization.config.adige import (
    ALL_PARTICLES_LOST_NPART_RATIO,
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    RL_MIN_NPART_RATIO,
    STAGE_MARKERS,
    score_function_metadata,
    score_tensor,
)
from beam_optimization.config.paths import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_SURROGATE_EVAL_OUTPUT,
    configure_matplotlib_cache,
    default_dataset_path,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _default_test_dataset_path() -> Path:
    """Return the newest numbered dataset that actually has a test split."""
    if DEFAULT_DATASET_ROOT.exists():
        candidates = [
            directory / "dataset_test.pt"
            for directory in DEFAULT_DATASET_ROOT.iterdir()
            if directory.is_dir() and directory.name.isdigit()
        ]
        existing = [path for path in candidates if path.exists()]
        if existing:
            return max(existing, key=lambda path: int(path.parent.name))
    return default_dataset_path(prefix="test")


def evaluate_surrogate(
    model: ModularMLP,
    dataset: BeamDataset,
    batch_size: int = 1024,
    device: Optional[str | torch.device] = None,
    plots_dir: Optional[str | Path] = None,
    plot_prefix: str = "surrogate",
    filter_score_plots_to_rl_valid: bool = False,
) -> dict:
    """Evaluate one surrogate on a dataset.

    Metrics cover beam-state errors by stage/feature and the score computed
    from the final predicted/target beam. Evaluation is batch-wise; only the
    two final-score vectors are retained for correlation and plots.

    `filter_score_plots_to_rl_valid` restricts the score_scatter/score_residuals
    plots (only those two -- the RMSE/NRMSE heatmaps stay over all samples) to
    samples with true `npart_ratio >= RL_MIN_NPART_RATIO`, and annotates them
    with `score_metrics_rl_valid` instead of `score_metrics`. It never changes
    any returned metric, only which samples the two score plots are drawn
    from -- the default (False) draws them over every sample, unchanged.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate a surrogate on an empty dataset")

    device_t = _resolve_device(device)
    model = model.to(device_t)
    model.eval()
    n_features = len(BEAM_STATE_FEATURES)
    sse_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.float64)
    sae_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.float64)
    target_sum_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.float64)
    target_sumsq_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.float64)
    count_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.int64)
    true_score_batches: list[np.ndarray] = []
    predicted_score_batches: list[np.ndarray] = []
    failure_label_batches: list[np.ndarray] = []
    true_npart_ratio_batches: list[np.ndarray] = []
    predicted_npart_ratio_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(dataset), int(batch_size)):
            stop = min(start + int(batch_size), len(dataset))
            indices = np.arange(start, stop)

            stage_params, beam_states = dataset.get_training_batch(indices)
            stage_params = [tensor.to(device_t) for tensor in stage_params]
            beam_states = [tensor.to(device_t) for tensor in beam_states]
            targets = beam_states[1:]

            preds = model(stage_params, beam_states[0])
            if isinstance(preds, torch.Tensor):
                pred_targets = [(N_OUTPUT_STAGES - 1, preds, targets[-1])]
            else:
                pred_targets = [
                    (idx, pred, target)
                    for idx, (pred, target) in enumerate(zip(preds, targets))
                ]

            for stage_idx, pred, target in pred_targets:
                diff = pred - target
                sse_stage_feature[stage_idx] += (
                    torch.sum(diff * diff, dim=0).detach().cpu().numpy()
                )
                sae_stage_feature[stage_idx] += (
                    torch.sum(torch.abs(diff), dim=0).detach().cpu().numpy()
                )
                target_sum_stage_feature[stage_idx] += (
                    torch.sum(target, dim=0).detach().cpu().numpy()
                )
                target_sumsq_stage_feature[stage_idx] += (
                    torch.sum(target * target, dim=0).detach().cpu().numpy()
                )
                count_stage_feature[stage_idx] += int(diff.shape[0])
                if stage_idx == N_OUTPUT_STAGES - 1:
                    true_score_batches.append(
                        score_tensor(target).detach().cpu().numpy().astype(np.float64)
                    )
                    predicted_score_batches.append(
                        score_tensor(pred).detach().cpu().numpy().astype(np.float64)
                    )
                    npart_index = BEAM_STATE_FEATURES.index("npart_ratio")
                    labels = (
                        target[:, npart_index] <= ALL_PARTICLES_LOST_NPART_RATIO
                    ).detach().cpu().numpy().astype(np.float64)
                    failure_label_batches.append(labels)
                    true_npart_ratio_batches.append(
                        target[:, npart_index].detach().cpu().numpy().astype(np.float64)
                    )
                    predicted_npart_ratio_batches.append(
                        pred[:, npart_index].detach().cpu().numpy().astype(np.float64)
                    )

    mse_stage_feature = _safe_divide(sse_stage_feature, count_stage_feature)
    mae_stage_feature = _safe_divide(sae_stage_feature, count_stage_feature)
    rmse_stage_feature = np.sqrt(mse_stage_feature)
    target_mean_stage_feature = _safe_divide(
        target_sum_stage_feature, count_stage_feature
    )
    target_variance_stage_feature = np.maximum(
        0.0,
        _safe_divide(target_sumsq_stage_feature, count_stage_feature)
        - np.square(target_mean_stage_feature),
    )
    target_std_stage_feature = np.sqrt(target_variance_stage_feature)
    standardized_mse_stage_feature = np.divide(
        mse_stage_feature,
        target_variance_stage_feature,
        out=np.full_like(mse_stage_feature, np.nan),
        where=target_variance_stage_feature > 1e-12,
    )
    nrmse_stage_feature = np.sqrt(standardized_mse_stage_feature)
    nmae_stage_feature = np.divide(
        mae_stage_feature,
        target_std_stage_feature,
        out=np.full_like(mae_stage_feature, np.nan),
        where=target_std_stage_feature > 1e-6,
    )

    sse_per_stage = np.sum(sse_stage_feature, axis=1)
    sae_per_stage = np.sum(sae_stage_feature, axis=1)
    count_per_stage = np.sum(count_stage_feature, axis=1)

    mse_per_stage = _safe_divide(sse_per_stage, count_per_stage)
    mae_per_stage = _safe_divide(sae_per_stage, count_per_stage)
    rmse_per_stage = np.sqrt(mse_per_stage)
    nrmse_per_stage = _nan_sqrt_mean(standardized_mse_stage_feature, axis=1)
    nrmse_all = _finite_or_none(_nan_sqrt_mean(standardized_mse_stage_feature))

    total_sse = float(sse_stage_feature.sum())
    total_count = int(count_stage_feature.sum())
    mse_all = total_sse / total_count

    sse_per_feature = np.sum(sse_stage_feature, axis=0)
    sae_per_feature = np.sum(sae_stage_feature, axis=0)
    count_per_feature = np.sum(count_stage_feature, axis=0)
    mse_per_feature = _safe_divide(sse_per_feature, count_per_feature)
    mae_per_feature = _safe_divide(sae_per_feature, count_per_feature)
    rmse_per_feature = np.sqrt(mse_per_feature)

    final_mse_per_feature = mse_stage_feature[-1]
    final_rmse_per_feature = rmse_stage_feature[-1]
    final_mae_per_feature = mae_stage_feature[-1]

    feature_metrics = {
        feature: {
            "mse_all_stages": _finite_or_none(mse_per_feature[index]),
            "rmse_all_stages": _finite_or_none(rmse_per_feature[index]),
            "mae_all_stages": _finite_or_none(mae_per_feature[index]),
            "mse_final_stage": _finite_or_none(final_mse_per_feature[index]),
            "rmse_final_stage": _finite_or_none(final_rmse_per_feature[index]),
            "mae_final_stage": _finite_or_none(final_mae_per_feature[index]),
            "nrmse_final_stage": _finite_or_none(nrmse_stage_feature[-1, index]),
            "nmae_final_stage": _finite_or_none(nmae_stage_feature[-1, index]),
            "target_std_final_stage": _finite_or_none(
                target_std_stage_feature[-1, index]
            ),
        }
        for index, feature in enumerate(BEAM_STATE_FEATURES)
    }

    true_scores = (
        np.concatenate(true_score_batches) if true_score_batches else np.empty(0)
    )
    predicted_scores = (
        np.concatenate(predicted_score_batches)
        if predicted_score_batches else np.empty(0)
    )
    failure_labels = (
        np.concatenate(failure_label_batches).astype(bool)
        if failure_label_batches else np.empty(0, dtype=bool)
    )
    true_npart_ratio = np.concatenate(true_npart_ratio_batches)
    predicted_npart_ratio = np.concatenate(predicted_npart_ratio_batches)
    score_metrics = _score_metrics(true_scores, predicted_scores)
    valid_mask = ~failure_labels
    score_metrics_valid = _score_metrics(
        true_scores[valid_mask], predicted_scores[valid_mask]
    )
    score_metrics_failures = _score_metrics(
        true_scores[failure_labels], predicted_scores[failure_labels]
    )
    true_rl_terminal = true_npart_ratio < RL_MIN_NPART_RATIO
    predicted_rl_terminal_regressor = predicted_npart_ratio < RL_MIN_NPART_RATIO
    score_metrics_rl_valid = _score_metrics(
        true_scores[~true_rl_terminal], predicted_scores[~true_rl_terminal]
    )
    score_metrics_rl_terminal = _score_metrics(
        true_scores[true_rl_terminal], predicted_scores[true_rl_terminal]
    )
    rl_terminal_metrics = {
        "threshold": float(RL_MIN_NPART_RATIO),
        "boundary_is_valid": True,
        "regressor": _binary_metrics(
            true_rl_terminal, predicted_rl_terminal_regressor
        ),
    }
    npart_ratio_bands = _npart_ratio_band_metrics(
        true_npart_ratio,
        predicted_npart_ratio,
    )

    final_mse = float(mse_per_stage[-1])
    results = {
        "n_samples": len(dataset),
        "mse_all": float(mse_all),
        "rmse_all": float(math.sqrt(mse_all)),
        "nrmse_all": nrmse_all,
        "mse_final_stage": final_mse,
        "rmse_final_stage": float(math.sqrt(final_mse)),
        "mse_per_stage": [_finite_or_none(v) for v in mse_per_stage],
        "rmse_per_stage": [_finite_or_none(v) for v in rmse_per_stage],
        "mae_per_stage": [_finite_or_none(v) for v in mae_per_stage],
        "nrmse_per_stage": [_finite_or_none(v) for v in nrmse_per_stage],
        "feature_names": list(BEAM_STATE_FEATURES),
        "stage_markers": list(STAGE_MARKERS[1:]),
        "feature_metrics": feature_metrics,
        "rmse_by_stage_and_feature": _matrix_to_json(rmse_stage_feature),
        "mse_by_stage_and_feature": _matrix_to_json(mse_stage_feature),
        "mae_by_stage_and_feature": _matrix_to_json(mae_stage_feature),
        "nrmse_by_stage_and_feature": _matrix_to_json(nrmse_stage_feature),
        "nmae_by_stage_and_feature": _matrix_to_json(nmae_stage_feature),
        "target_std_by_stage_and_feature": _matrix_to_json(target_std_stage_feature),
        "score_metrics": score_metrics,
        "score_metrics_valid": score_metrics_valid,
        "score_metrics_failures": score_metrics_failures,
        "score_metrics_rl_valid": score_metrics_rl_valid,
        "score_metrics_rl_terminal": score_metrics_rl_terminal,
        "sample_groups": {
            "n_valid": int(np.sum(valid_mask)),
            "n_failures": int(np.sum(failure_labels)),
        },
        "rl_sample_groups": {
            "n_valid": int(np.sum(~true_rl_terminal)),
            "n_terminal": int(np.sum(true_rl_terminal)),
        },
        "rl_terminal_metrics": rl_terminal_metrics,
        "npart_ratio_bands": npart_ratio_bands,
    }

    if plots_dir is not None:
        if filter_score_plots_to_rl_valid:
            score_plot_mask = ~true_rl_terminal
            score_plot_metrics = score_metrics_rl_valid
            score_title_suffix = f" (npart_ratio >= {RL_MIN_NPART_RATIO:g})"
        else:
            score_plot_mask = np.ones(true_scores.shape, dtype=bool)
            score_plot_metrics = score_metrics
            score_title_suffix = ""
        plots = _save_evaluation_plots(
            true_scores=true_scores[score_plot_mask],
            predicted_scores=predicted_scores[score_plot_mask],
            rmse_stage_feature=rmse_stage_feature,
            nrmse_stage_feature=nrmse_stage_feature,
            failure_labels=failure_labels[score_plot_mask],
            output_dir=Path(plots_dir),
            prefix=plot_prefix,
            score_metrics=score_plot_metrics,
            score_title_suffix=score_title_suffix,
        )
        results["plots"] = plots
    else:
        results["plots"] = {}
    return results


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(np.shape(numerator), np.nan, dtype=np.float64),
        where=np.asarray(denominator) > 0,
    )


def _finite_or_none(value) -> Optional[float]:
    value = float(value)
    return value if np.isfinite(value) else None


def _matrix_to_json(matrix: np.ndarray) -> list[list[Optional[float]]]:
    return [[_finite_or_none(value) for value in row] for row in matrix]


def _nan_sqrt_mean(values: np.ndarray, axis=None):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if axis is None:
        return float(np.sqrt(np.mean(values[finite]))) if np.any(finite) else float("nan")
    counts = np.sum(finite, axis=axis)
    sums = np.sum(np.where(finite, values, 0.0), axis=axis)
    means = np.divide(
        sums,
        counts,
        out=np.full(np.shape(sums), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return np.sqrt(means)


def _score_metrics(
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
    *,
    bootstrap_samples: int = 400,
) -> dict:
    if true_scores.size == 0 or predicted_scores.size == 0:
        return {
            key: None
            for key in (
                "mae", "rmse", "bias", "pearson_correlation", "r2",
                "true_mean", "true_std", "predicted_mean", "predicted_std",
            )
        }

    residuals = predicted_scores - true_scores
    centered_true = true_scores - np.mean(true_scores)
    centered_predicted = predicted_scores - np.mean(predicted_scores)
    true_ss = float(np.sum(centered_true * centered_true))
    predicted_ss = float(np.sum(centered_predicted * centered_predicted))
    residual_ss = float(np.sum(residuals * residuals))

    pearson = None
    if not np.isclose(true_ss, 0.0) and not np.isclose(predicted_ss, 0.0):
        pearson = float(
            np.sum(centered_true * centered_predicted)
            / math.sqrt(true_ss * predicted_ss)
        )
    r2 = (
        None
        if np.isclose(true_ss, 0.0) or np.isclose(predicted_ss, 0.0)
        else float(1.0 - residual_ss / true_ss)
    )

    metrics = {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(math.sqrt(np.mean(residuals * residuals))),
        "bias": float(np.mean(residuals)),
        "pearson_correlation": pearson,
        "r2": r2,
        "true_mean": float(np.mean(true_scores)),
        "true_std": float(np.std(true_scores)),
        "predicted_mean": float(np.mean(predicted_scores)),
        "predicted_std": float(np.std(predicted_scores)),
    }
    metrics["confidence_intervals_95"] = _bootstrap_score_intervals(
        true_scores,
        predicted_scores,
        n_resamples=bootstrap_samples,
    )
    return metrics



def _bootstrap_score_intervals(
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
    *,
    n_resamples: int,
) -> dict:
    """Deterministic non-parametric 95% intervals for the principal metrics."""
    n = int(true_scores.size)
    if n < 2 or n_resamples <= 0:
        return {}
    rng = np.random.default_rng(12_345)
    samples = {"mae": [], "rmse": [], "bias": [], "r2": []}
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        true = true_scores[idx]
        predicted = predicted_scores[idx]
        residual = predicted - true
        samples["mae"].append(float(np.mean(np.abs(residual))))
        samples["rmse"].append(float(np.sqrt(np.mean(np.square(residual)))))
        samples["bias"].append(float(np.mean(residual)))
        true_ss = float(np.sum(np.square(true - np.mean(true))))
        samples["r2"].append(
            float(1.0 - np.sum(np.square(residual)) / true_ss)
            if not np.isclose(true_ss, 0.0)
            else np.nan
        )
    intervals = {}
    for name, values in samples.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            lo, hi = np.quantile(finite, [0.025, 0.975])
            intervals[name] = {"low": float(lo), "high": float(hi)}
    return intervals


def _binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    tn = int(np.sum(~predictions & ~labels))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "n_samples": int(labels.size),
        "n_positive": int(np.sum(labels)),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _npart_ratio_band_metrics(
    true_ratio: np.ndarray,
    predicted_ratio: np.ndarray,
) -> list[dict]:
    """Regression and terminal-decision quality in operational transmission bands."""
    true_ratio = np.asarray(true_ratio, dtype=np.float64)
    predicted_ratio = np.asarray(predicted_ratio, dtype=np.float64)
    bands = [
        (
            "all_particles_lost",
            true_ratio <= ALL_PARTICLES_LOST_NPART_RATIO,
            f"npart_ratio <= {ALL_PARTICLES_LOST_NPART_RATIO:g}",
        ),
        (
            "rl_terminal_nonzero",
            (true_ratio > ALL_PARTICLES_LOST_NPART_RATIO)
            & (true_ratio < RL_MIN_NPART_RATIO),
            f"{ALL_PARTICLES_LOST_NPART_RATIO:g} < npart_ratio < {RL_MIN_NPART_RATIO:g}",
        ),
        (
            "rl_valid_near_boundary",
            (true_ratio >= RL_MIN_NPART_RATIO) & (true_ratio < 0.25),
            f"{RL_MIN_NPART_RATIO:g} <= npart_ratio < 0.25",
        ),
        (
            "rl_valid_above_025",
            true_ratio >= 0.25,
            "npart_ratio >= 0.25",
        ),
    ]
    rows = []
    predicted_terminal = predicted_ratio < RL_MIN_NPART_RATIO
    for name, mask, interval in bands:
        count = int(np.sum(mask))
        if count:
            residual = predicted_ratio[mask] - true_ratio[mask]
            true_mean = float(np.mean(true_ratio[mask]))
            predicted_mean = float(np.mean(predicted_ratio[mask]))
            mae = float(np.mean(np.abs(residual)))
            rmse = float(np.sqrt(np.mean(np.square(residual))))
            bias = float(np.mean(residual))
            regressor_terminal_rate = float(np.mean(predicted_terminal[mask]))
        else:
            true_mean = predicted_mean = mae = rmse = bias = None
            regressor_terminal_rate = None
        rows.append({
            "name": name,
            "interval": interval,
            "n_samples": count,
            "true_mean": true_mean,
            "predicted_mean": predicted_mean,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "regressor_terminal_rate": regressor_terminal_rate,
        })
    return rows



def _save_evaluation_plots(
    *,
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
    rmse_stage_feature: np.ndarray,
    nrmse_stage_feature: np.ndarray,
    failure_labels: np.ndarray,
    output_dir: Path,
    prefix: str,
    score_metrics: dict,
    score_title_suffix: str = "",
) -> dict[str, str]:
    configure_matplotlib_cache()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if true_scores.size and predicted_scores.size:
        lo = float(min(np.min(true_scores), np.min(predicted_scores)))
        hi = float(max(np.max(true_scores), np.max(predicted_scores)))
        if np.isclose(lo, hi):
            lo -= 1.0
            hi += 1.0

        fig, axis = plt.subplots(figsize=(6.5, 6.0))
        valid = ~failure_labels
        axis.scatter(
            true_scores[valid], predicted_scores[valid],
            s=13, alpha=0.5, edgecolors="none", label="valid",
        )
        if np.any(failure_labels):
            axis.scatter(
                true_scores[failure_labels], predicted_scores[failure_labels],
                s=18, alpha=0.55, edgecolors="none", color="crimson",
                label="all particles lost",
            )
        axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, label="ideal")
        axis.set_xlabel("True TraceWin score")
        axis.set_ylabel("Predicted surrogate score")
        axis.set_title(f"{prefix}: true vs predicted final score{score_title_suffix}")
        axis.grid(alpha=0.25)
        # Legend sits above the title, in the figure margin: data points are
        # always confined inside the axes rectangle, so nothing plotted can
        # ever land there regardless of the score distribution -- unlike a
        # placement over/beside the axes, which bbox_inches="tight" below
        # would crop right back onto the plot (that was the previous bug).
        axis.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.12),
            frameon=False, borderaxespad=0.0, ncol=3,
        )
        annotation = (
            f"RMSE={score_metrics['rmse']:.4g}\n"
            f"MAE={score_metrics['mae']:.4g}\n"
            f"Pearson={_format_optional(score_metrics['pearson_correlation'])}\n"
            f"R²={_format_optional(score_metrics['r2'])}"
        )
        # Same reasoning as the legend above: placed below the x-axis label,
        # in the figure margin, so it can never sit on top of a data point
        # regardless of where the score distribution falls.
        axis.text(
            0.0, -0.14, annotation, transform=axis.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.6"},
        )
        fig.tight_layout()
        scatter_path = output_dir / f"{prefix}_score_scatter.png"
        fig.savefig(scatter_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        paths["score_scatter"] = str(scatter_path)

        residuals = predicted_scores - true_scores
        fig, axis = plt.subplots(figsize=(7.5, 5.0))
        axis.scatter(true_scores, residuals, s=13, alpha=0.5, edgecolors="none")
        axis.axhline(0.0, linestyle="--", color="black", linewidth=1)
        axis.set_xlabel("True TraceWin score")
        axis.set_ylabel("Residual (predicted − true)")
        axis.set_title(f"{prefix}: final-score residuals{score_title_suffix}")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        residual_path = output_dir / f"{prefix}_score_residuals.png"
        fig.savefig(residual_path, dpi=170)
        plt.close(fig)
        paths["score_residuals"] = str(residual_path)

    heatmap = np.asarray(rmse_stage_feature, dtype=np.float64).T
    positive = heatmap[np.isfinite(heatmap) & (heatmap > 0)]
    norm = None
    if positive.size and float(np.max(positive)) > float(np.min(positive)):
        norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive)))

    fig, axis = plt.subplots(figsize=(12.0, 5.8))
    image = axis.imshow(heatmap, aspect="auto", cmap="viridis", norm=norm)
    axis.set_xticks(np.arange(N_OUTPUT_STAGES))
    axis.set_xticklabels([str(marker) for marker in STAGE_MARKERS[1:]], rotation=45)
    axis.set_yticks(np.arange(len(BEAM_STATE_FEATURES)))
    axis.set_yticklabels(BEAM_STATE_FEATURES)
    axis.set_xlabel("TraceWin stage marker")
    axis.set_ylabel("Beam feature")
    axis.set_title(f"{prefix}: RMSE by feature and stage")
    fig.colorbar(image, ax=axis, label="RMSE (feature native units, logarithmic color scale)")
    fig.tight_layout()
    heatmap_path = output_dir / f"{prefix}_rmse_heatmap.png"
    fig.savefig(heatmap_path, dpi=170)
    plt.close(fig)
    paths["rmse_heatmap"] = str(heatmap_path)

    normalized_heatmap = np.asarray(nrmse_stage_feature, dtype=np.float64).T
    masked = np.ma.masked_invalid(normalized_heatmap)
    fig, axis = plt.subplots(figsize=(12.0, 5.8))
    image = axis.imshow(masked, aspect="auto", cmap="magma", vmin=0.0)
    axis.set_xticks(np.arange(N_OUTPUT_STAGES))
    axis.set_xticklabels([str(marker) for marker in STAGE_MARKERS[1:]], rotation=45)
    axis.set_yticks(np.arange(len(BEAM_STATE_FEATURES)))
    axis.set_yticklabels(BEAM_STATE_FEATURES)
    axis.set_xlabel("TraceWin stage marker")
    axis.set_ylabel("Beam feature")
    axis.set_title(f"{prefix}: normalized RMSE by feature and stage")
    fig.colorbar(image, ax=axis, label="RMSE / target standard deviation")
    fig.tight_layout()
    normalized_path = output_dir / f"{prefix}_nrmse_heatmap.png"
    fig.savefig(normalized_path, dpi=170)
    plt.close(fig)
    paths["nrmse_heatmap"] = str(normalized_path)
    return paths



def _format_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4g}"


class _MeanEnsemble(torch.nn.Module):
    """Minimal evaluation-only wrapper returning the ensemble mean."""

    def __init__(self, models: list[ModularMLP]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, stage_params, beam0):
        predictions = [model(stage_params, beam0) for model in self.models]
        if isinstance(predictions[0], torch.Tensor):
            return torch.stack(predictions, dim=0).mean(dim=0)
        return [
            torch.stack([prediction[stage] for prediction in predictions], dim=0).mean(dim=0)
            for stage in range(len(predictions[0]))
        ]


def _checkpoint_provenance(path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "train_dataset_path": payload.get("train_dataset_path"),
        "val_dataset_path": payload.get("val_dataset_path"),
        "best_val_loss": payload.get("best_val_loss"),
        "model_config": payload.get("model_config", {}),
    }


def evaluate_surrogate_folder(
    model_dir: str | Path,
    dataset_path: str | Path,
    batch_size: int = 1024,
    device: Optional[str | torch.device] = None,
    save_path: Optional[str | Path] = None,
    plots_dir: Optional[str | Path] = None,
) -> dict:
    """Evaluate every surrogate_*.pt model in a directory."""
    model_dir = Path(model_dir)
    dataset_path = Path(dataset_path)
    model_paths = sorted(model_dir.glob("surrogate_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No surrogate_*.pt files found in {model_dir}")

    device_t = _resolve_device(device)
    dataset = BeamDataset.load(dataset_path)
    resolved_plots_dir = Path(plots_dir) if plots_dir is not None else None
    if resolved_plots_dir is None and save_path is not None:
        output_path = Path(save_path)
        resolved_plots_dir = output_path.parent / f"{output_path.stem}_plots"

    results = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
        "dataset_path": str(dataset_path),
        "dataset_provenance": {
            "path": str(dataset_path.resolve()),
            "size_bytes": int(dataset_path.stat().st_size),
            "modified_utc": datetime.fromtimestamp(
                dataset_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "n_samples": len(dataset),
        },
        "batch_size": int(batch_size),
        "device": str(device_t),
        "models": {},
        "score_function": score_function_metadata(),
    }

    loaded_models = []
    for model_path in model_paths:
        model = ModularMLP.load(str(model_path), device=str(device_t))
        loaded_models.append(model)
        metrics = evaluate_surrogate(
            model,
            dataset,
            batch_size=batch_size,
            device=device_t,
            plots_dir=resolved_plots_dir,
            plot_prefix=model_path.stem,
        )
        metrics["checkpoint_provenance"] = _checkpoint_provenance(model_path)
        results["models"][model_path.name] = metrics

    if len(loaded_models) > 1:
        ensemble = _MeanEnsemble(loaded_models)
        results["ensemble_mean"] = evaluate_surrogate(
            ensemble,
            dataset,
            batch_size=batch_size,
            device=device_t,
            plots_dir=resolved_plots_dir,
            plot_prefix="ensemble_mean",
        )
        results["ensemble_mean"]["ensemble_size"] = len(loaded_models)

    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all surrogate_*.pt models in a folder on a BeamDataset."
    )
    parser.add_argument("--model-dir", required=True, help="Folder containing surrogate_*.pt files.")
    parser.add_argument(
        "--dataset",
        default=str(_default_test_dataset_path()),
        help="Validation/test .pt dataset path.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", default=str(DEFAULT_SURROGATE_EVAL_OUTPUT),
        help="JSON output path (default: %(default)s).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Optional directory for score and RMSE plots. When omitted with "
            "--output, uses <output_stem>_plots next to the JSON."
        ),
    )
    args = parser.parse_args()

    results = evaluate_surrogate_folder(
        args.model_dir,
        args.dataset,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.output,
        plots_dir=args.plots_dir,
    )

    for model_name, metrics in results["models"].items():
        _print_model_report(model_name, metrics)
    if results.get("ensemble_mean"):
        _print_model_report("ensemble_mean", results["ensemble_mean"])


def _print_model_report(model_name: str, metrics: dict) -> None:
    print(f"\n{'=' * 96}\n{model_name}\n{'=' * 96}")
    print(
        f"samples={metrics['n_samples']:,}  mse_all={metrics['mse_all']:.6g}  "
        f"rmse_all={metrics['rmse_all']:.6g}  "
        f"nrmse_all={_format_optional(metrics.get('nrmse_all'))}  "
        f"mse_final={metrics['mse_final_stage']:.6g}  "
        f"rmse_final={metrics['rmse_final_stage']:.6g}"
    )

    score = metrics["score_metrics"]
    print("\nFinal-score metrics")
    print(
        f"  MAE={_format_optional(score['mae'])}  "
        f"RMSE={_format_optional(score['rmse'])}  "
        f"bias={_format_optional(score['bias'])}  "
        f"Pearson={_format_optional(score['pearson_correlation'])}  "
        f"R²={_format_optional(score['r2'])}"
    )
    print(
        f"  true mean/std={_format_optional(score['true_mean'])}/"
        f"{_format_optional(score['true_std'])}  predicted mean/std="
        f"{_format_optional(score['predicted_mean'])}/"
        f"{_format_optional(score['predicted_std'])}"
    )
    groups = metrics["sample_groups"]
    valid_score = metrics["score_metrics_valid"]
    failed_score = metrics["score_metrics_failures"]
    print(
        f"  physical non-failure, npart_ratio > "
        f"{ALL_PARTICLES_LOST_NPART_RATIO:g} (n={groups['n_valid']}): "
        f"MAE={_format_optional(valid_score['mae'])}  "
        f"RMSE={_format_optional(valid_score['rmse'])}  "
        f"R²={_format_optional(valid_score['r2'])}"
    )
    print(
        f"  failures-only (n={groups['n_failures']}): "
        f"MAE={_format_optional(failed_score['mae'])}  "
        f"RMSE={_format_optional(failed_score['rmse'])}"
    )

    rl_groups = metrics["rl_sample_groups"]
    rl_valid_score = metrics["score_metrics_rl_valid"]
    rl_terminal_score = metrics["score_metrics_rl_terminal"]
    print(
        f"  RL-valid, npart_ratio >= {RL_MIN_NPART_RATIO:g} "
        f"(n={rl_groups['n_valid']}): "
        f"MAE={_format_optional(rl_valid_score['mae'])}  "
        f"RMSE={_format_optional(rl_valid_score['rmse'])}  "
        f"R²={_format_optional(rl_valid_score['r2'])}"
    )
    print(
        f"  RL-terminal, npart_ratio < {RL_MIN_NPART_RATIO:g} "
        f"(n={rl_groups['n_terminal']}): "
        f"MAE={_format_optional(rl_terminal_score['mae'])}  "
        f"RMSE={_format_optional(rl_terminal_score['rmse'])}"
    )

    terminal = metrics["rl_terminal_metrics"]
    print(f"\nRL terminal decision (npart_ratio < {terminal['threshold']:g})")
    for label, key in (("regressor", "regressor"),):
        values = terminal.get(key)
        if values is None:
            continue
        confusion = values["confusion_matrix"]
        print(
            f"  {label}: precision={_format_optional(values['precision'])}  "
            f"recall={_format_optional(values['recall'])}  "
            f"specificity={_format_optional(values['specificity'])}  "
            f"F1={_format_optional(values['f1'])}  "
            f"TP/FP/FN/TN={confusion['tp']}/{confusion['fp']}/"
            f"{confusion['fn']}/{confusion['tn']}"
        )

    print("\nFinal npart_ratio by operational band")
    print(
        f"{'Band':<27} {'N':>7} {'true mean':>11} {'pred mean':>11} "
        f"{'MAE':>10} {'terminal %':>11}"
    )
    print("-" * 84)
    for band in metrics["npart_ratio_bands"]:
        terminal_rate = band["regressor_terminal_rate"]
        terminal_percent = (
            f"{100.0 * terminal_rate:.2f}" if terminal_rate is not None else "n/a"
        )
        print(
            f"{band['name']:<27} {band['n_samples']:>7} "
            f"{_format_optional(band['true_mean']):>11} "
            f"{_format_optional(band['predicted_mean']):>11} "
            f"{_format_optional(band['mae']):>10} {terminal_percent:>11}"
        )

    print("\nPer-feature errors (native feature units)")
    print(
        f"{'Feature':<14} {'RMSE all':>12} {'MAE all':>12} "
        f"{'RMSE final':>12} {'MAE final':>12}"
    )
    print("-" * 66)
    for feature in metrics["feature_names"]:
        values = metrics["feature_metrics"][feature]
        print(
            f"{feature:<14} "
            f"{_format_optional(values['rmse_all_stages']):>12} "
            f"{_format_optional(values['mae_all_stages']):>12} "
            f"{_format_optional(values['rmse_final_stage']):>12} "
            f"{_format_optional(values['mae_final_stage']):>12}"
        )

    print("\nPer-stage RMSE")
    for marker, rmse in zip(metrics["stage_markers"], metrics["rmse_per_stage"]):
        print(f"  marker {marker:>4}: {_format_optional(rmse)}")

    if metrics.get("plots"):
        print("\nPlots")
        for name, path in metrics["plots"].items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
