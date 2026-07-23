"""Offline surrogate evaluation on an independent BeamDataset split."""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_FEATURES,
    N_OUTPUT_STAGES,
    STAGE_MARKERS,
    score_tensor,
)
from beam_optimization.config.paths import (
    DEFAULT_DATASET_ROOT,
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
) -> dict:
    """Evaluate one surrogate on a dataset.

    Metrics cover beam-state errors by stage/feature and the score computed
    from the final predicted/target beam. Evaluation is batch-wise; only the
    two final-score vectors are retained for correlation and plots.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate a surrogate on an empty dataset")

    device_t = _resolve_device(device)
    model = model.to(device_t)
    model.eval()

    n_features = len(BEAM_STATE_FEATURES)
    sse_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.float64)
    sae_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.float64)
    count_stage_feature = np.zeros((N_OUTPUT_STAGES, n_features), dtype=np.int64)
    true_score_batches: list[np.ndarray] = []
    predicted_score_batches: list[np.ndarray] = []

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
                count_stage_feature[stage_idx] += int(diff.shape[0])
                if stage_idx == N_OUTPUT_STAGES - 1:
                    true_score_batches.append(
                        score_tensor(target).detach().cpu().numpy().astype(np.float64)
                    )
                    predicted_score_batches.append(
                        score_tensor(pred).detach().cpu().numpy().astype(np.float64)
                    )

    mse_stage_feature = _safe_divide(sse_stage_feature, count_stage_feature)
    mae_stage_feature = _safe_divide(sae_stage_feature, count_stage_feature)
    rmse_stage_feature = np.sqrt(mse_stage_feature)

    sse_per_stage = np.sum(sse_stage_feature, axis=1)
    sae_per_stage = np.sum(sae_stage_feature, axis=1)
    count_per_stage = np.sum(count_stage_feature, axis=1)

    mse_per_stage = _safe_divide(sse_per_stage, count_per_stage)
    mae_per_stage = _safe_divide(sae_per_stage, count_per_stage)
    rmse_per_stage = np.sqrt(mse_per_stage)

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
    score_metrics = _score_metrics(true_scores, predicted_scores)

    final_mse = float(mse_per_stage[-1])
    results = {
        "n_samples": len(dataset),
        "mse_all": float(mse_all),
        "rmse_all": float(math.sqrt(mse_all)),
        "mse_final_stage": final_mse,
        "rmse_final_stage": float(math.sqrt(final_mse)),
        "mse_per_stage": [_finite_or_none(v) for v in mse_per_stage],
        "rmse_per_stage": [_finite_or_none(v) for v in rmse_per_stage],
        "mae_per_stage": [_finite_or_none(v) for v in mae_per_stage],
        "feature_names": list(BEAM_STATE_FEATURES),
        "stage_markers": list(STAGE_MARKERS[1:]),
        "feature_metrics": feature_metrics,
        "rmse_by_stage_and_feature": _matrix_to_json(rmse_stage_feature),
        "mse_by_stage_and_feature": _matrix_to_json(mse_stage_feature),
        "mae_by_stage_and_feature": _matrix_to_json(mae_stage_feature),
        "score_metrics": score_metrics,
    }

    if plots_dir is not None:
        results["plots"] = _save_evaluation_plots(
            true_scores=true_scores,
            predicted_scores=predicted_scores,
            rmse_stage_feature=rmse_stage_feature,
            output_dir=Path(plots_dir),
            prefix=plot_prefix,
            score_metrics=score_metrics,
        )
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


def _score_metrics(true_scores: np.ndarray, predicted_scores: np.ndarray) -> dict:
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

    return {
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


def _save_evaluation_plots(
    *,
    true_scores: np.ndarray,
    predicted_scores: np.ndarray,
    rmse_stage_feature: np.ndarray,
    output_dir: Path,
    prefix: str,
    score_metrics: dict,
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
        axis.scatter(true_scores, predicted_scores, s=13, alpha=0.5, edgecolors="none")
        axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, label="ideal")
        axis.set_xlabel("True TraceWin score")
        axis.set_ylabel("Predicted surrogate score")
        axis.set_title(f"{prefix}: true vs predicted final score")
        axis.grid(alpha=0.25)
        axis.legend()
        annotation = (
            f"RMSE={score_metrics['rmse']:.4g}\n"
            f"MAE={score_metrics['mae']:.4g}\n"
            f"Pearson={_format_optional(score_metrics['pearson_correlation'])}\n"
            f"R²={_format_optional(score_metrics['r2'])}"
        )
        axis.text(
            0.03, 0.97, annotation, transform=axis.transAxes, va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        fig.tight_layout()
        scatter_path = output_dir / f"{prefix}_score_scatter.png"
        fig.savefig(scatter_path, dpi=170)
        plt.close(fig)
        paths["score_scatter"] = str(scatter_path)

        residuals = predicted_scores - true_scores
        fig, axis = plt.subplots(figsize=(7.5, 5.0))
        axis.scatter(true_scores, residuals, s=13, alpha=0.5, edgecolors="none")
        axis.axhline(0.0, linestyle="--", color="black", linewidth=1)
        axis.set_xlabel("True TraceWin score")
        axis.set_ylabel("Residual (predicted − true)")
        axis.set_title(f"{prefix}: final-score residuals")
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
    return paths


def _format_optional(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4g}"


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
        "model_dir": str(model_dir),
        "dataset_path": str(dataset_path),
        "batch_size": int(batch_size),
        "device": str(device_t),
        "models": {},
    }

    for model_path in model_paths:
        model = ModularMLP.load(str(model_path), device=str(device_t))
        results["models"][model_path.name] = evaluate_surrogate(
            model,
            dataset,
            batch_size=batch_size,
            device=device_t,
            plots_dir=resolved_plots_dir,
            plot_prefix=model_path.stem,
        )

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
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
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


def _print_model_report(model_name: str, metrics: dict) -> None:
    print(f"\n{'=' * 96}\n{model_name}\n{'=' * 96}")
    print(
        f"samples={metrics['n_samples']:,}  mse_all={metrics['mse_all']:.6g}  "
        f"rmse_all={metrics['rmse_all']:.6g}  "
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
