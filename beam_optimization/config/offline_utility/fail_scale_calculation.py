"""Calibrate the Gaussian reset scale at which TraceWin physically fails 90% of the time.

Only the definitive, non-retryable physics failures declared by
TraceWinSimulator count toward the target. Technical or unknown failures abort
the calibration instead of biasing the measured scale.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from beam_optimization.config.adige import (
    PARAM_KEYS,
    TEST_RESET_SCALE,
    clip_params_to_hw,
    default_params,
    sensitivity_vec,
)
from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import (
    NON_RETRYABLE_PHYSICS_FAILURES,
)


DEFAULT_TARGET_FAILURE_RATE = 0.90
DEFAULT_SAMPLES = 32
DEFAULT_EXPANSION_FACTOR = 1.5
DEFAULT_MAX_SCALE = 10.0
DEFAULT_BISECTION_ITERATIONS = 5


def normalized_gaussian_design(n_samples: int, seed: Optional[int] = None) -> np.ndarray:
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    return np.random.default_rng(seed).standard_normal((n_samples, len(PARAM_KEYS)))


def parameter_sets_for_scale(scale: float, design: np.ndarray) -> list[dict[str, float]]:
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    design = np.asarray(design, dtype=np.float64)
    expected = (design.shape[0], len(PARAM_KEYS)) if design.ndim == 2 else None
    if design.ndim != 2 or design.shape != expected:
        raise ValueError(
            f"design must have shape (N, {len(PARAM_KEYS)}), got {design.shape}"
        )
    defaults = default_params()
    default_vec = np.asarray([defaults[key] for key in PARAM_KEYS], dtype=np.float64)
    vectors = default_vec + float(scale) * sensitivity_vec() * design
    return [
        clip_params_to_hw({key: float(value) for key, value in zip(PARAM_KEYS, row)})
        for row in vectors
    ]


def physical_failure_message(result) -> Optional[str]:
    """Return the original target physics error, or None when it is not one."""
    error = str(getattr(result, "error", "") or "")
    normalized = error.casefold()
    if any(message in normalized for message in NON_RETRYABLE_PHYSICS_FAILURES):
        return error
    return None


def classify_result(result) -> tuple[str, Optional[str]]:
    if bool(getattr(result, "success", False)):
        return "success", None
    message = physical_failure_message(result)
    if message is not None:
        return "physics_failure", message
    error = str(getattr(result, "error", "") or "unknown TraceWin failure")
    raise RuntimeError(
        "TraceWin returned a technical or unknown failure; calibration aborted "
        f"so it cannot be counted as physical beam loss: {error}"
    )


def evaluate_scale(
    simulator,
    params_list: Sequence[dict[str, float]],
    *,
    scale: float,
    target_failure_rate: float = DEFAULT_TARGET_FAILURE_RATE,
    verbose: bool = True,
) -> dict:
    if not 0.0 < target_failure_rate <= 1.0:
        raise ValueError("target_failure_rate must be in (0, 1]")
    n_total = len(params_list)
    if n_total < 1:
        raise ValueError("params_list must not be empty")
    required_failures = math.ceil(target_failure_rate * n_total - 1e-12)
    maximum_non_failures = n_total - required_failures
    n_physics_failures = 0
    samples = []
    stopped_early = False

    for index, params in enumerate(params_list, start=1):
        tracewin_params = getattr(simulator, "tracewin_params", None)
        if isinstance(tracewin_params, dict):
            tracewin_params.pop("random_seed", None)
        result = simulator.simulate(params)
        classification, message = classify_result(result)
        is_failure = classification == "physics_failure"
        n_physics_failures += int(is_failure)
        samples.append({
            "sample": index,
            "classification": classification,
            "success": bool(getattr(result, "success", False)),
            "score": float(getattr(result, "score_val", float("nan"))),
            "error": message,
            "params": {key: float(value) for key, value in params.items()},
        })
        if verbose:
            print(
                f"  scale={scale:.6g} sample {index}/{n_total} "
                f"physics_failure={is_failure} "
                f"score={float(getattr(result, 'score_val', float('nan'))):.6g}",
                flush=True,
            )

        n_non_failures = index - n_physics_failures
        if n_physics_failures >= required_failures:
            stopped_early = index < n_total
            break
        if n_non_failures > maximum_non_failures:
            stopped_early = index < n_total
            break

    accepted = n_physics_failures >= required_failures
    n_evaluated = len(samples)
    return {
        "scale": float(scale),
        "accepted": accepted,
        "required_failures": required_failures,
        "n_physics_failures": n_physics_failures,
        "n_non_failures": n_evaluated - n_physics_failures,
        "n_evaluated": n_evaluated,
        "n_total": n_total,
        "physical_failure_rate_lower_bound": n_physics_failures / n_total,
        "stopped_early": stopped_early,
        "samples": samples,
    }


def calibrate_fail_scale(
    simulator,
    *,
    start_scale: float = TEST_RESET_SCALE,
    max_scale: float = DEFAULT_MAX_SCALE,
    expansion_factor: float = DEFAULT_EXPANSION_FACTOR,
    bisection_iterations: int = DEFAULT_BISECTION_ITERATIONS,
    n_samples: int = DEFAULT_SAMPLES,
    target_failure_rate: float = DEFAULT_TARGET_FAILURE_RATE,
    sample_seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    if start_scale <= 0.0:
        raise ValueError("start_scale must be positive")
    if max_scale < start_scale:
        raise ValueError("max_scale must be >= start_scale")
    if expansion_factor <= 1.0:
        raise ValueError("expansion_factor must be > 1")
    if bisection_iterations < 0:
        raise ValueError("bisection_iterations must be >= 0")

    design = normalized_gaussian_design(n_samples, seed=sample_seed)
    evaluations: list[dict] = []

    def probe(scale: float) -> dict:
        if verbose:
            print(f"\nTesting fail scale {scale:.9g}", flush=True)
        record = evaluate_scale(
            simulator,
            parameter_sets_for_scale(scale, design),
            scale=scale,
            target_failure_rate=target_failure_rate,
            verbose=verbose,
        )
        evaluations.append(record)
        if verbose:
            print(
                f"  RESULT scale={scale:.9g}: "
                f"{record['n_physics_failures']}/{record['n_total']} guaranteed "
                f"physical failures -> {'TARGET REACHED' if record['accepted'] else 'below target'}",
                flush=True,
            )
        return record

    first = probe(float(start_scale))
    if first["accepted"]:
        selected = float(start_scale)
        lower = None
        upper = selected
    else:
        lower = float(start_scale)
        upper = None
        current = lower
        while current < max_scale:
            candidate = min(float(max_scale), current * float(expansion_factor))
            record = probe(candidate)
            if record["accepted"]:
                upper = candidate
                break
            lower = candidate
            current = candidate
            if math.isclose(current, max_scale, rel_tol=0.0, abs_tol=1e-15):
                break

        if upper is None:
            selected = None
        else:
            for _ in range(bisection_iterations):
                midpoint = (lower + upper) / 2.0
                record = probe(midpoint)
                if record["accepted"]:
                    upper = midpoint
                else:
                    lower = midpoint
            selected = float(upper)

    return {
        "selected_scale": selected,
        "status": "ok" if selected is not None else "no_failure_scale_found",
        "target_failure_rate": float(target_failure_rate),
        "n_samples": int(n_samples),
        "sample_seed": sample_seed,
        "bracket": {"lower_below_target": lower, "upper_reaches_target": upper},
        "evaluations": evaluations,
    }


def update_adige_fail_scale(scale: float, config_path: str | Path) -> Path:
    """Atomically replace the sole ALL_PARTICLE_LOST_SCALE declaration."""
    path = Path(config_path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^ALL_PARTICLE_LOST_SCALE(?:\s*:\s*[^=]+)?\s*=.*$", re.MULTILINE)
    replacement = f"ALL_PARTICLE_LOST_SCALE: float = {float(scale):.15e}"
    updated, count = pattern.subn(replacement, text)
    if count != 1:
        raise ValueError(
            f"Expected exactly one ALL_PARTICLE_LOST_SCALE declaration in {path}, found {count}"
        )
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, temp_name)
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return path


def save_report(report: dict, output: str | Path, *, run_config: dict) -> Path:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "gaussian_physics_failure_scale_calibration",
        "physics_failures": list(NON_RETRYABLE_PHYSICS_FAILURES),
        "run_config": run_config,
        **report,
    }
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    from beam_optimization.config.paths import (
        DEFAULT_FAIL_SCALE_OUTPUT, PROJECT_ROOT, resolve_tracewin_project,
    )
    from beam_optimization.env.tracewin_env.tracewin.tracewin_simulator import TraceWinSimulator

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--workspace", default=None, metavar="PATH")
    source.add_argument("--tracewin", default=None, metavar="INI")
    parser.add_argument("--calc-dir", default=None, metavar="PATH")
    parser.add_argument(
        "--output", default=str(DEFAULT_FAIL_SCALE_OUTPUT), metavar="JSON"
    )
    parser.add_argument("--start-scale", type=float, default=TEST_RESET_SCALE)
    parser.add_argument("--max-scale", type=float, default=DEFAULT_MAX_SCALE)
    parser.add_argument("--expansion-factor", type=float, default=DEFAULT_EXPANSION_FACTOR)
    parser.add_argument("--bisection-iterations", type=int, default=DEFAULT_BISECTION_ITERATIONS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--target-failure-rate", type=float, default=DEFAULT_TARGET_FAILURE_RATE)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--tracewin-particles", type=int, default=10000)
    parser.add_argument("--tracewin-threads", type=int, default=None, metavar="N")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Atomically write the measured value to config/adige.py.",
    )
    args = parser.parse_args()

    try:
        workspace, project_file = resolve_tracewin_project(
            workspace=args.workspace, tracewin=args.tracewin
        )
    except ValueError as exc:
        parser.error(str(exc))
    calc_dir = (
        Path(args.calc_dir).expanduser().resolve()
        if args.calc_dir
        else workspace / "fail_scale_calc"
    )
    print(f"TraceWin workspace       : {workspace}")
    print(f"TraceWin project         : {project_file}")
    print(f"Calc dir                 : {calc_dir}")
    print(f"Start / max scale        : {args.start_scale:g} / {args.max_scale:g}")
    print(f"Target physical failures : {args.target_failure_rate:.1%}")
    print(f"Samples per scale        : {args.samples}")

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
        report = calibrate_fail_scale(
            simulator,
            start_scale=args.start_scale,
            max_scale=args.max_scale,
            expansion_factor=args.expansion_factor,
            bisection_iterations=args.bisection_iterations,
            n_samples=args.samples,
            target_failure_rate=args.target_failure_rate,
            sample_seed=args.sample_seed,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    output = save_report(report, args.output, run_config=vars(args))
    print(f"\nJSON saved to: {output}")
    selected = report["selected_scale"]
    if selected is None:
        print("No scale reached the requested physical-failure rate; config unchanged.")
        return
    declaration = f"ALL_PARTICLE_LOST_SCALE: float = {selected:.15e}"
    print("\nCopy-paste block for adige.py:")
    print(declaration)
    if args.update_config:
        config_path = PROJECT_ROOT / "config/adige.py"
        update_adige_fail_scale(selected, config_path)
        print(f"Updated config atomically: {config_path}")


if __name__ == "__main__":
    main()
