"""Calibrate one shared dataset/Bayesian exploration scale with TraceWin.

Starting from 0.4, candidate scales are tested in descending order.  Every
candidate is evaluated with both parameter distributions used by the project:

* dataset: independent Gaussian offsets with std = scale * sensitivity;
* Bayesian cold start: Sobol points inside default +/- scale * sensitivity.

The first (therefore largest) candidate for which both distributions reach the
requested valid-run rate is selected.  A run is valid when TraceWin itself
reports success (no crash/error); the target rate is the fraction of runs
(out of the sampled total) that must succeed, e.g. 0.90 means at least 9 out
of 10 samples must succeed. ``adige.py`` is never edited automatically.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import qmc

from beam_optimization.algorithms.baselines.bayesian_opt import hardware_aware_bounds
from beam_optimization.config.adige import (
    PARAM_KEYS,
    PARAMETERS,
    clip_params_to_hw,
    default_params,
    sensitivity_vec,
)


DEFAULT_START_SCALE = 0.4
DEFAULT_MIN_SCALE = 0.05
DEFAULT_SCALE_STEP = 0.05
DEFAULT_TARGET_SUCCESS_RATE = 0.90
DEFAULT_SAMPLES_PER_DISTRIBUTION = 32


def candidate_scales(
    start_scale: float = DEFAULT_START_SCALE,
    min_scale: float = DEFAULT_MIN_SCALE,
    scale_step: float = DEFAULT_SCALE_STEP,
) -> tuple[float, ...]:
    """Return a stable descending grid including the minimum scale."""
    if start_scale <= 0.0:
        raise ValueError("start_scale must be positive")
    if min_scale <= 0.0 or min_scale > start_scale:
        raise ValueError("min_scale must be positive and <= start_scale")
    if scale_step <= 0.0:
        raise ValueError("scale_step must be positive")

    values = []
    current = float(start_scale)
    tolerance = 1e-12 * max(1.0, start_scale)
    while current >= min_scale - tolerance:
        values.append(round(max(current, min_scale), 12))
        current -= scale_step
    if not math.isclose(values[-1], min_scale, rel_tol=0.0, abs_tol=tolerance):
        values.append(round(float(min_scale), 12))
    return tuple(dict.fromkeys(values))


def normalized_designs(
    n_samples: int,
    *,
    seed: Optional[int] = None,
    n_dimensions: int = len(PARAMETERS),
) -> dict[str, np.ndarray]:
    """Create reusable Gaussian and scrambled-Sobol normalized designs.

    ``seed`` is unset by default: each call draws a fresh, unseeded design
    from system entropy. Passing a seed only makes the two designs built by
    this one call reproducible relative to each other.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if n_dimensions < 1:
        raise ValueError("n_dimensions must be at least 1")

    gaussian = np.random.default_rng(seed).standard_normal(
        size=(n_samples, n_dimensions)
    )
    sobol_engine = qmc.Sobol(d=n_dimensions, scramble=True, seed=seed)
    if n_samples > 0 and n_samples & (n_samples - 1) == 0:
        sobol = sobol_engine.random_base2(int(math.log2(n_samples)))
    else:
        # Arbitrary counts are supported, although powers of two preserve the
        # balance properties used by the real cold-start command.
        sobol = sobol_engine.random(n_samples)
    return {
        "dataset_gaussian": np.asarray(gaussian, dtype=np.float64),
        "bayesian_sobol": np.asarray(sobol, dtype=np.float64),
    }


def parameter_sets_for_scale(
    scale: float,
    designs: dict[str, np.ndarray],
) -> dict[str, list[Dict[str, float]]]:
    """Map normalized designs to the exact dataset and Bayesian parameter spaces."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")

    defaults = default_params()
    default_vector = np.asarray([defaults[key] for key in PARAM_KEYS], dtype=np.float64)
    sensitivities = sensitivity_vec()
    gaussian = np.asarray(designs["dataset_gaussian"], dtype=np.float64)
    sobol = np.asarray(designs["bayesian_sobol"], dtype=np.float64)
    expected_shape = (gaussian.shape[0], len(PARAMETERS))
    if gaussian.ndim != 2 or gaussian.shape[1] != len(PARAMETERS):
        raise ValueError(f"invalid Gaussian design shape: {gaussian.shape}")
    if sobol.shape != expected_shape:
        raise ValueError(
            "Gaussian and Sobol designs must have the same shape; got "
            f"{gaussian.shape} and {sobol.shape}"
        )

    gaussian_vectors = default_vector + scale * sensitivities * gaussian
    dataset_params = [
        clip_params_to_hw({key: float(value) for key, value in zip(PARAM_KEYS, row)})
        for row in gaussian_vectors
    ]

    bounds = hardware_aware_bounds(PARAMETERS, scale)
    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    sobol_vectors = lower + sobol * (upper - lower)
    bayesian_params = [
        {key: float(value) for key, value in zip(PARAM_KEYS, row)}
        for row in sobol_vectors
    ]
    return {
        "dataset_gaussian": dataset_params,
        "bayesian_sobol": bayesian_params,
    }


def classify_result(result) -> tuple[bool, str]:
    """Classify TraceWin completion; valid means TraceWin itself succeeded."""
    if not bool(getattr(result, "success", False)):
        return False, "tracewin_failed"
    return True, "valid"


def _evaluate_distribution(
    simulator,
    params_list: Sequence[Dict[str, float]],
    *,
    distribution: str,
    scale: float,
    target_success_rate: float,
    verbose: bool,
) -> dict:
    n_success = 0
    n_total = len(params_list)
    required_successes = math.ceil(target_success_rate * n_total - 1e-12)
    maximum_failures = n_total - required_successes
    n_evaluated = 0
    stopped_early = False
    for index, params in enumerate(params_list, start=1):
        tracewin_params = getattr(simulator, "tracewin_params", None)
        if isinstance(tracewin_params, dict):
            tracewin_params.pop("random_seed", None)
        result = simulator.simulate(params)
        valid, _ = classify_result(result)
        n_success += int(valid)
        n_evaluated = index
        if verbose:
            print(
                f"  scale={scale:.3g} {distribution:<18} "
                f"sample {index}/{len(params_list)} success={valid} "
                f"score={float(result.score_val):.6g}",
                flush=True,
            )

        n_failures = n_evaluated - n_success
        if n_failures > maximum_failures:
            stopped_early = True
            if verbose:
                maximum_successes = n_success + (n_total - n_evaluated)
                print(
                    f"  stopping {distribution}: {n_failures} failures; "
                    f"at most {maximum_successes}/{n_total} successful "
                    f"({maximum_successes / n_total:.1%}), below target "
                    f"{target_success_rate:.1%}",
                    flush=True,
                )
            break

    n_failures = n_evaluated - n_success
    record = {
        "n_success": n_success,
        "n_failures": n_failures,
        "n_evaluated": n_evaluated,
        "n_total": n_total,
        "success_rate": n_success / n_evaluated,
        "target_reachable": not stopped_early,
        "stopped_early": stopped_early,
    }
    if verbose:
        print(
            f"  {distribution}: {n_success}/{n_total} successful "
            f"({record['success_rate']:.1%})",
            flush=True,
        )
    return record


def calibrate_exploration_scale(
    simulator,
    *,
    scales: Sequence[float],
    n_samples: int = DEFAULT_SAMPLES_PER_DISTRIBUTION,
    target_success_rate: float = DEFAULT_TARGET_SUCCESS_RATE,
    sample_seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Return the largest descending candidate safe for both distributions."""
    if not scales or any(scale <= 0.0 for scale in scales):
        raise ValueError("scales must contain positive values")
    if any(left <= right for left, right in zip(scales, scales[1:])):
        raise ValueError("scales must be strictly descending")
    if not 0.0 < target_success_rate <= 1.0:
        raise ValueError("target_success_rate must be in (0, 1]")

    designs = normalized_designs(n_samples, seed=sample_seed)
    scale_records = []
    selected_scale = None

    for scale_index, scale in enumerate(scales):
        if verbose:
            print(f"\nTesting shared exploration scale {scale:.6g}", flush=True)
        parameter_sets = parameter_sets_for_scale(float(scale), designs)
        distributions = {}
        distribution_names = ("dataset_gaussian", "bayesian_sobol")
        for distribution_index, distribution in enumerate(distribution_names, start=1):
            if verbose:
                print(
                    f"\n  Distribution {distribution_index}/2: {distribution}",
                    flush=True,
                )
            distributions[distribution] = _evaluate_distribution(
                simulator,
                parameter_sets[distribution],
                distribution=distribution,
                scale=float(scale),
                target_success_rate=target_success_rate,
                verbose=verbose,
            )
            if not distributions[distribution]["target_reachable"]:
                break
            if distribution_index == 1 and verbose:
                print(
                    f"  Same scale={scale:.6g}: checking bayesian_sobol "
                    "(validation 2/2, not a restart).",
                    flush=True,
                )

        worst_rate = min(
            record["success_rate"] for record in distributions.values()
        )
        accepted = (
            len(distributions) == 2
            and all(record["target_reachable"] for record in distributions.values())
            and worst_rate >= target_success_rate
        )
        scale_records.append({
            "scale": float(scale),
            "accepted": accepted,
            "worst_success_rate": worst_rate,
            "distributions": distributions,
        })
        if verbose:
            print(
                f"  RESULT scale={scale:.6g}: worst success rate "
                f"{worst_rate:.1%} -> {'ACCEPTED' if accepted else 'too large'}",
                flush=True,
            )
        if accepted:
            selected_scale = float(scale)
            break
        if verbose and scale_index + 1 < len(scales):
            print(
                f"  Scale={scale:.6g} rejected; trying lower "
                f"scale={scales[scale_index + 1]:.6g}.",
                flush=True,
            )

    return {
        "selected_scale": selected_scale,
        "target_success_rate": target_success_rate,
        "n_samples_per_distribution": n_samples,
        "sample_seed": sample_seed,
        "scales": scale_records,
        "status": "ok" if selected_scale is not None else "no_safe_scale_found",
    }


def save_report(report: dict, output: str | Path, *, run_config: dict) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "descending_shared_dataset_bayesian_success_calibration",
        "run_config": run_config,
        **report,
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    from beam_optimization.config.paths import PROJECT_ROOT, resolve_tracewin_project
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator

    parser = argparse.ArgumentParser(description=__doc__)
    tracewin_source = parser.add_mutually_exclusive_group()
    tracewin_source.add_argument("--workspace", default=None, metavar="PATH")
    tracewin_source.add_argument("--tracewin", default=None, metavar="INI")
    parser.add_argument("--calc-dir", default=None, metavar="PATH")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results/exploration_scale.json"),
        metavar="JSON",
    )
    parser.add_argument("--start-scale", type=float, default=DEFAULT_START_SCALE)
    parser.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE)
    parser.add_argument("--scale-step", type=float, default=DEFAULT_SCALE_STEP)
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=DEFAULT_TARGET_SUCCESS_RATE,
    )
    parser.add_argument(
        "--samples-per-distribution",
        type=int,
        default=DEFAULT_SAMPLES_PER_DISTRIBUTION,
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Design sampling seed; omitted: unseeded (drawn from system entropy).",
    )
    parser.add_argument("--tracewin-particles", type=int, default=10000)
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    try:
        workspace, project_file = resolve_tracewin_project(
            workspace=args.workspace,
            tracewin=args.tracewin,
        )
        scales = candidate_scales(
            args.start_scale,
            args.min_scale,
            args.scale_step,
        )
    except ValueError as exc:
        parser.error(str(exc))

    calc_dir = (
        Path(args.calc_dir).expanduser().resolve()
        if args.calc_dir
        else workspace / "exploration_scale_calc"
    )
    print(f"TraceWin workspace       : {workspace}")
    print(f"TraceWin project         : {project_file}")
    print(f"Calc dir                 : {calc_dir}")
    print(f"Candidate scales         : {', '.join(f'{s:g}' for s in scales)}")
    print(f"Samples/distribution     : {args.samples_per_distribution}")
    print(f"Target success rate      : {args.target_success_rate:.1%}")
    print("TraceWin seed            : unset")
    sample_seed_text = args.sample_seed if args.sample_seed is not None else "unseeded (random)"
    print(f"Design sample seed       : {sample_seed_text}")

    simulator = TraceWinSimulator(
        project_file=str(project_file),
        calc_dir=str(calc_dir),
        timeout=args.timeout,
        retries=args.retries,
        tracewin_params={"nbr_part1": int(args.tracewin_particles)},
        num_threads=args.tracewin_threads,
        initial_npart=args.tracewin_particles,
    )
    try:
        report = calibrate_exploration_scale(
            simulator,
            scales=scales,
            n_samples=args.samples_per_distribution,
            target_success_rate=args.target_success_rate,
            sample_seed=args.sample_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output = save_report(
        report,
        args.output,
        run_config={
            "workspace": str(workspace),
            "project": str(project_file),
            "calc_dir": str(calc_dir),
            "start_scale": args.start_scale,
            "min_scale": args.min_scale,
            "scale_step": args.scale_step,
            "target_success_rate": args.target_success_rate,
            "samples_per_distribution": args.samples_per_distribution,
            "sample_seed": args.sample_seed,
            "tracewin_seed": None,
            "tracewin_particles": args.tracewin_particles,
            "tracewin_threads": args.tracewin_threads,
            "timeout": args.timeout,
            "retries": args.retries,
        },
    )

    selected = report["selected_scale"]
    print(f"\nJSON saved to: {output}")
    if selected is None:
        print("No candidate reached the requested success rate.")
        return
    print("\nCopy-paste block for adige.py:")
    print(f"EXPLORATION_SCALE: float = {selected!r}")
    print("DATASET_SCALE: float = EXPLORATION_SCALE")
    print("BAYESIAN_SCALE: float = EXPLORATION_SCALE")


if __name__ == "__main__":
    main()
