"""Offline calculation of the three global RL/dataset scales for ADIGE.

Three scalars are shared by every parameter (in sensitivity units), derived in
this order because each depends on the previous one:

    dataset_scale -> reset_scale -> action_scale

dataset_scale is chosen freely first (how wide the surrogate's trust region
should be). reset_scale and action_scale are then derived so that, in the
worst case, a reset plus a full episode trajectory never leaves the dataset's
gaussian bell:

    k_sigma * reset_scale + action_scale * max_steps <= k_sigma_dataset * dataset_scale

Because sensitivity cancels out of this inequality, the constraint — and the
three scales themselves — are identical for every parameter; only the derived
per-parameter physical quantities (dataset_std_p, reset_std_p, step_max_p,
trajectory_max_p) depend on each parameter's sensitivity.

Hardware bounds (hw_min/hw_max) never enter these formulas — they are only a
clip applied when a concrete value is generated (dataset sampling, reset,
action step), not a constraint on the scales themselves.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
from beam_optimization.config.adige import DATASET_SCALE, MAX_STEPS, PARAMETERS, sensitivity_vec

# Single source of truth for every default in this module: compute_scales(),
# verify_constraints(), report(), and the CLI argparse defaults all reference
# these constants, so changing a default here changes it everywhere (there
# used to be separate, silently-diverging defaults in each function).
# dataset_scale defaults to adige.py's own DATASET_SCALE (not an independent
# hardcoded value), so a default run reproduces/verifies the currently
# configured scale instead of silently drifting from it.
DEFAULT_DATASET_SCALE: float = DATASET_SCALE
DEFAULT_K_SIGMA_DATASET: float = 2.5
DEFAULT_F_RESET: float = 0.25
DEFAULT_K_SIGMA: float = 2.5


def compute_scales(
    *,
    dataset_scale: float = DEFAULT_DATASET_SCALE,
    k_sigma_dataset: float = DEFAULT_K_SIGMA_DATASET,
    f_reset: float = DEFAULT_F_RESET,
    k_sigma: float = DEFAULT_K_SIGMA,
    target_scale: Optional[float] = None,
    max_steps: int = MAX_STEPS,
) -> dict:
    """Compute the three global scalars, in order: dataset_scale (given), reset_scale, action_scale.

    Args:
        dataset_scale: Chosen first, freely — width of the dataset's gaussian bell.
        k_sigma_dataset: How many dataset stddevs define the trust-region edge.
        f_reset: Fraction of the trust region budget reserved for the reset (rest goes to the trajectory).
        k_sigma: Worst-case reset excursion, in reset stddevs.
        target_scale: Fraction of dataset_scale the agent should typically cover in ~10 steps.
            Defaults to 0.4 * dataset_scale.
        max_steps: Episode horizon in RL steps.

    Returns:
        dict with dataset_scale, reset_scale, action_scale, action_scale_max.
    """
    if dataset_scale <= 0:
        raise ValueError(f"dataset_scale must be > 0, got {dataset_scale}")
    if k_sigma_dataset <= 0:
        raise ValueError(f"k_sigma_dataset must be > 0, got {k_sigma_dataset}")
    if not 0.0 < f_reset < 1.0:
        raise ValueError(f"f_reset must be in (0, 1), got {f_reset}")
    if k_sigma <= 0:
        raise ValueError(f"k_sigma must be > 0, got {k_sigma}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    if target_scale is None:
        target_scale = 0.4 * dataset_scale

    reset_scale = f_reset * k_sigma_dataset * dataset_scale / k_sigma
    action_scale_max = (1.0 - f_reset) * k_sigma_dataset * dataset_scale / max_steps
    action_scale_candidate = target_scale / 10.0
    action_scale = min(action_scale_candidate, action_scale_max)

    return {
        "dataset_scale": dataset_scale,
        "reset_scale": reset_scale,
        "action_scale": action_scale,
        "action_scale_max": action_scale_max,
    }


def verify_constraints(
    *,
    dataset_scale: float,
    reset_scale: float,
    action_scale: float,
    k_sigma_dataset: float = DEFAULT_K_SIGMA_DATASET,
    k_sigma: float = DEFAULT_K_SIGMA,
    max_steps: int = MAX_STEPS,
) -> List[str]:
    """Check the single global trust-region constraint and flag tight hardware clips.

    The core constraint is parameter-independent (sensitivity cancels out):

        k_sigma * reset_scale + action_scale * max_steps <= k_sigma_dataset * dataset_scale

    In addition, for parameters with known hardware bounds, warn if the
    dataset's gaussian bell (k_sigma_dataset * dataset_std_p) would be
    significantly clipped by hw_min/hw_max — this is only a diagnostic, since
    hardware is never a constraint on the scales themselves.
    """
    warnings: List[str] = []

    lhs = k_sigma * reset_scale + action_scale * max_steps
    rhs = k_sigma_dataset * dataset_scale
    if lhs > rhs:
        warnings.append(
            f"VIOLATION — k_sigma*reset_scale + action_scale*max_steps = {lhs:.6g} "
            f"> k_sigma_dataset*dataset_scale = {rhs:.6g}"
        )

    sens = sensitivity_vec()
    dataset_half_width = k_sigma_dataset * dataset_scale * sens
    for i, p in enumerate(PARAMETERS):
        half_width = dataset_half_width[i]
        if p.hw_min is not None and p.default - p.hw_min < half_width:
            warnings.append(
                f"  {p.name}: hw_min clips the dataset bell — "
                f"default-hw_min={p.default - p.hw_min:.4g} < k_sigma_dataset*dataset_std={half_width:.4g}"
            )
        if p.hw_max is not None and p.hw_max - p.default < half_width:
            warnings.append(
                f"  {p.name}: hw_max clips the dataset bell — "
                f"hw_max-default={p.hw_max - p.default:.4g} < k_sigma_dataset*dataset_std={half_width:.4g}"
            )

    return warnings


def report(
    *,
    dataset_scale: float = DEFAULT_DATASET_SCALE,
    k_sigma_dataset: float = DEFAULT_K_SIGMA_DATASET,
    f_reset: float = DEFAULT_F_RESET,
    k_sigma: float = DEFAULT_K_SIGMA,
    target_scale: Optional[float] = None,
    max_steps: int = MAX_STEPS,
) -> None:
    """Print the calibration report and a copy-paste block for adige.py."""
    scales = compute_scales(
        dataset_scale=dataset_scale,
        k_sigma_dataset=k_sigma_dataset,
        f_reset=f_reset,
        k_sigma=k_sigma,
        target_scale=target_scale,
        max_steps=max_steps,
    )
    reset_scale = scales["reset_scale"]
    action_scale = scales["action_scale"]

    print("\nScale Calculation")
    print(
        f"dataset_scale={dataset_scale}  k_sigma_dataset={k_sigma_dataset}  "
        f"f_reset={f_reset}  k_sigma={k_sigma}  max_steps={max_steps}"
    )
    print(
        f"-> reset_scale={reset_scale:.6g}  action_scale={action_scale:.6g} "
        f"(action_scale_max={scales['action_scale_max']:.6g})"
    )
    print()

    sens = sensitivity_vec()
    dataset_std = dataset_scale * sens
    reset_std = reset_scale * sens
    step_max = action_scale * sens
    trajectory_max = step_max * max_steps

    header = (
        f"{'Parameter':<14} {'dataset_std':>14} {'reset_std':>14} "
        f"{'step_max':>14} {'trajectory_max':>18}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, p in enumerate(PARAMETERS):
        print(
            f"{p.name:<14} {dataset_std[i]:>14.6g} {reset_std[i]:>14.6g} "
            f"{step_max[i]:>14.6g} {trajectory_max[i]:>18.6g}"
        )
    print(sep)

    warnings = verify_constraints(
        dataset_scale=dataset_scale,
        reset_scale=reset_scale,
        action_scale=action_scale,
        k_sigma_dataset=k_sigma_dataset,
        k_sigma=k_sigma,
        max_steps=max_steps,
    )
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(w)

    bar = "─" * 80
    print(f"\n{bar}")
    print("Copy-paste block for adige.py:")
    print(bar)
    print(f"DATASET_SCALE: float = {dataset_scale!r}")
    print(f"RESET_SCALE: float = {reset_scale:.15e}")
    print(f"ACTION_SCALE: float = {action_scale:.15e}")
    print(bar)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the action space design calibration report for ADIGE."
    )
    parser.add_argument(
        "--dataset-scale", type=float, default=DEFAULT_DATASET_SCALE,
        help="Dataset gaussian bell width, chosen first "
             "(default: %(default)s, i.e. adige.py's current DATASET_SCALE)"
    )
    parser.add_argument(
        "--k-sigma-dataset", type=float, default=DEFAULT_K_SIGMA_DATASET,
        help="Number of dataset stddevs defining the trust-region edge (default: %(default)s)"
    )
    parser.add_argument(
        "--f-reset", type=float, default=DEFAULT_F_RESET,
        help="Fraction of the trust-region budget reserved for reset (default: %(default)s)"
    )
    parser.add_argument(
        "--k-sigma", type=float, default=DEFAULT_K_SIGMA,
        help="Worst-case reset excursion, in reset stddevs (default: %(default)s)"
    )
    parser.add_argument(
        "--target-scale", type=float, default=None,
        help="Fraction of dataset_scale the agent should cover in ~10 steps (default: 0.4*dataset_scale)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_STEPS,
        help="Episode length in RL steps (default: %(default)s)"
    )
    args = parser.parse_args()

    report(
        dataset_scale=args.dataset_scale,
        k_sigma_dataset=args.k_sigma_dataset,
        f_reset=args.f_reset,
        k_sigma=args.k_sigma,
        target_scale=args.target_scale,
        max_steps=args.max_steps,
    )
