"""Evaluate surrogate models on a BeamDataset with MSE/RMSE metrics."""
from __future__ import annotations

import json
import math
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from beam_optimization.config.adige import N_OUTPUT_STAGES
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP


def _resolve_device(device: Optional[str | torch.device]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_surrogate(
    model: ModularMLP,
    dataset: BeamDataset,
    batch_size: int = 1024,
    device: Optional[str | torch.device] = None,
) -> dict:
    """Evaluate one surrogate on a dataset.

    Metrics are computed on beam-state predictions, not on the scalar score,
    over all 11 output stages.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate a surrogate on an empty dataset")

    device_t = _resolve_device(device)
    model = model.to(device_t)
    model.eval()

    sse_per_stage = np.zeros(N_OUTPUT_STAGES, dtype=np.float64)
    count_per_stage = np.zeros(N_OUTPUT_STAGES, dtype=np.int64)

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
                sse_per_stage[stage_idx] += float(torch.sum(diff * diff).cpu())
                count_per_stage[stage_idx] += int(diff.numel())

    mse_per_stage = np.divide(
        sse_per_stage,
        count_per_stage,
        out=np.full_like(sse_per_stage, np.nan, dtype=np.float64),
        where=count_per_stage > 0,
    )
    rmse_per_stage = np.sqrt(mse_per_stage)

    total_sse = float(sse_per_stage[count_per_stage > 0].sum())
    total_count = int(count_per_stage[count_per_stage > 0].sum())
    mse_all = total_sse / total_count

    final_mse = float(mse_per_stage[-1])
    return {
        "n_samples": len(dataset),
        "mse_all": float(mse_all),
        "rmse_all": float(math.sqrt(mse_all)),
        "mse_final_stage": final_mse,
        "rmse_final_stage": float(math.sqrt(final_mse)),
        "mse_per_stage": [float(v) if np.isfinite(v) else None for v in mse_per_stage],
        "rmse_per_stage": [float(v) if np.isfinite(v) else None for v in rmse_per_stage],
    }


def evaluate_surrogate_folder(
    model_dir: str | Path,
    dataset_path: str | Path,
    batch_size: int = 1024,
    device: Optional[str | torch.device] = None,
    save_path: Optional[str | Path] = None,
) -> dict:
    """Evaluate every surrogate_*.pt model in a directory."""
    model_dir = Path(model_dir)
    dataset_path = Path(dataset_path)
    model_paths = sorted(model_dir.glob("surrogate_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No surrogate_*.pt files found in {model_dir}")

    device_t = _resolve_device(device)
    dataset = BeamDataset.load(dataset_path)

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
    parser.add_argument("--dataset", required=True, help="Validation/test .pt dataset path.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = evaluate_surrogate_folder(
        args.model_dir,
        args.dataset,
        batch_size=args.batch_size,
        device=args.device,
        save_path=args.output,
    )

    for model_name, metrics in results["models"].items():
        print(
            f"{model_name}: "
            f"mse_all={metrics['mse_all']:.6g} "
            f"rmse_all={metrics['rmse_all']:.6g} "
            f"mse_final={metrics['mse_final_stage']:.6g} "
            f"rmse_final={metrics['rmse_final_stage']:.6g}"
        )


if __name__ == "__main__":
    main()
