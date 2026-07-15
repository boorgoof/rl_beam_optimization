"""
Bayesian Optimization command — finds a new candidate default parameter set.

Runs BayesianOptimizer (algorithms/baselines/bayesian_opt.py) against a
trained surrogate for --n-runs independent seeds, prints the best parameter
set found (for you to copy by hand into adige.py's default= fields), and
saves diagnostic plots.

Usage:
    python -m beam_optimization bayesian_opt \\
        --surrogate env/surrogate_env/surrogate/trained_models/base/surrogate_0.pt \\
        --dataset   env/dataset/base/dataset_base.pt \\
        --n-calls   200 \\
        --n-runs    3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from beam_optimization.config.adige import PARAMETERS, sensitivity_vec
from beam_optimization.config.paths import (
    DEFAULT_SINGLE_SURROGATE_MODEL,
    configure_matplotlib_cache,
    default_dataset_path,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.surrogate_env.surrogate.model.modular_mlp import ModularMLP
from beam_optimization.scripts.benchmark import (
    print_table,
    run_bo,
    save_convergence_plot,
    save_summary_plot,
)


def _print_best_params(best_params: dict) -> None:
    print("\nBest parameter set found (copy the values you want into adige.py defaults):")
    header = f"{'Parameter':<14} {'best':>14} {'default':>14} {'delta/sensitivity':>18}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    sens = sensitivity_vec()
    for i, p in enumerate(PARAMETERS):
        value = best_params[p.key]
        delta_sens = (value - p.default) / sens[i] if sens[i] != 0 else float("nan")
        print(f"{p.name:<14} {value:>14.6g} {p.default:>14.6g} {delta_sens:>18.3g}")
    print(sep)


def save_delta_plot(best_params: dict, output_json: str | Path) -> Path:
    """Bar chart of (best - default) / sensitivity per parameter.

    Normalizing by sensitivity makes the 16 physically heterogeneous
    parameters comparable on one axis (score-point-equivalent units).
    """
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sens = sensitivity_vec()
    names = [p.name for p in PARAMETERS]
    deltas = [
        (best_params[p.key] - p.default) / sens[i] if sens[i] != 0 else 0.0
        for i, p in enumerate(PARAMETERS)
    ]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.55 * len(names)), 5.0))
    colors = ["#4c78a8" if d >= 0 else "#e34948" for d in deltas]
    ax.bar(names, deltas, color=colors, alpha=0.86)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_title("Bayesian Optimization — shift from default (sensitivity units)")
    ax.set_ylabel("(best - default) / sensitivity")
    ax.set_xlabel("Parameter")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    path = Path(output_json).parent / "bayesian_opt_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate", default=str(DEFAULT_SINGLE_SURROGATE_MODEL))
    parser.add_argument("--dataset", default=str(default_dataset_path()))
    parser.add_argument("--n-calls", type=int, default=200,
                        help="Objective evaluations per run (capped at 200 inside BayesianOptimizer).")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Independent seeds; the best-scoring run's params are reported.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="beam_optimization/results/bayesian_opt.json")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Surrogate: {args.surrogate}")
    surrogate = ModularMLP.load(args.surrogate)
    surrogate.eval()

    print(f"Dataset:   {args.dataset}")
    dataset = BeamDataset.load(args.dataset)

    results: dict = {"bayesian_opt": []}
    for run in range(args.n_runs):
        seed = args.seed + run
        print(f"\nRun {run + 1}/{args.n_runs}  (seed={seed})")
        r = run_bo(surrogate, dataset, args.n_calls, seed)
        results["bayesian_opt"].append(r)
        print(f"  best={r['best_score']:.3f}")

    print_table(results)

    best_run = max(results["bayesian_opt"], key=lambda r: r["best_score"])
    _print_best_params(best_run["best_params"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": [
                    {"best_score": r["best_score"], "best_params": r["best_params"]}
                    for r in results["bayesian_opt"]
                ],
            },
            f,
            indent=2,
        )
    print(f"\nJSON summary saved -> {args.output}")

    convergence_path = save_convergence_plot(results, args.output, svg_horizon=1)
    if convergence_path is not None:
        print(f"Convergence plot saved -> {convergence_path}")
    summary_path = save_summary_plot(results, args.output)
    print(f"Summary plot saved     -> {summary_path}")
    delta_path = save_delta_plot(best_run["best_params"], args.output)
    print(f"Delta plot saved       -> {delta_path}")


if __name__ == "__main__":
    main()
