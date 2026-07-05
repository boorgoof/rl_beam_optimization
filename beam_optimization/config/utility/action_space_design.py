
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from beam_optimization.config.adige import PARAMETERS, sensitivity_vec


def _available_margin(param) -> float | None:
    """Return distance from default to nearest hardware edge, or None if unknown."""
    if param.hw_min is None or param.hw_max is None:
        return None
    return min(param.hw_max - param.default, param.default - param.hw_min)


def _positive_or_zero(value: float) -> float:
    return max(0.0, float(value))


def compute_action_scale_rl(
    *,
    target_trajectory_fraction: float = 0.4,
    target_steps: int = 10,
    max_steps: int = 20,
    margin: float = 1.3,
    reset_fraction: float = 0.3,
    fallback_scale: float = 1.0,
) -> np.ndarray:
    """Compute recommended action_scale_rl per parameter.

    The hardware-safe margin is split into reset and step budgets:

        reset_budget = reset_fraction * available_margin / margin
        step_budget = (1 - reset_fraction) * available_margin / margin

    The final action scale is:

        min(target_trajectory / (target_steps * sensitivity),
            step_budget / (max_steps * sensitivity))

    ``available_margin`` is the distance from the default to the nearest hardware
    edge, not half of the full hardware range. ``margin`` is a safety divisor:
    larger values are more conservative.

    Parameters without hardware bounds fall back to ``fallback_scale``.
    """
    if target_steps <= 0:
        raise ValueError(f"target_steps must be > 0, got {target_steps}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")
    if margin <= 0:
        raise ValueError(f"margin must be > 0, got {margin}")
    if not 0.0 <= reset_fraction <= 1.0:
        raise ValueError(f"reset_fraction must be in [0, 1], got {reset_fraction}")

    sens = sensitivity_vec()

    result = np.empty(len(PARAMETERS), dtype=np.float64)
    for i, p in enumerate(PARAMETERS):
        s = sens[i]
        available = _available_margin(p)
        if available is None:
            result[i] = fallback_scale
            continue
        if available <= 0 or s <= 0:
            result[i] = 0.0
            continue

        target_trajectory = target_trajectory_fraction * available
        candidate = target_trajectory / (target_steps * s)

        step_budget = (1.0 - reset_fraction) * available / margin
        cap = _positive_or_zero(step_budget / (s * max_steps))
        result[i] = _positive_or_zero(min(candidate, cap))

    return result


def compute_reset_scale(
    action_scale_rl: np.ndarray | None = None,
    *,
    margin: float = 1.3,
    reset_sigma: float = 3.0,
    reset_fraction: float = 0.3,
    fallback_reset_scale: float = 0.5,
) -> np.ndarray:
    """Compute recommended reset_scale per parameter.

    Reset is allocated a fixed fraction of the hardware-safe margin:

        reset_budget = reset_fraction * available_margin / margin
        reset_scale = reset_budget / (reset_sigma * sensitivity)

    Parameters without hardware bounds use ``fallback_reset_scale``.
    """
    if margin <= 0:
        raise ValueError(f"margin must be > 0, got {margin}")
    if reset_sigma < 0:
        raise ValueError(f"reset_sigma must be >= 0, got {reset_sigma}")
    if not 0.0 <= reset_fraction <= 1.0:
        raise ValueError(f"reset_fraction must be in [0, 1], got {reset_fraction}")

    sens = sensitivity_vec()

    result = np.empty(len(PARAMETERS), dtype=np.float64)
    for i, p in enumerate(PARAMETERS):
        s = sens[i]
        available = _available_margin(p)
        if available is None:
            result[i] = fallback_reset_scale
            continue
        if available <= 0 or s <= 0 or reset_sigma == 0:
            result[i] = 0.0
            continue

        reset_budget = reset_fraction * available / margin
        result[i] = _positive_or_zero(reset_budget / (reset_sigma * s))

    return result


def dataset_core_bounds(
    action_scale_rl: np.ndarray,
    reset_scale: np.ndarray,
    *,
    max_steps: int = 20,
    reset_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the dense core sampling zone for the surrogate dataset.

    The core zone covers a conservative reset excursion plus the maximum RL
    trajectory inside one episode:

        core_half_width =
            reset_sigma * reset_scale_p * sensitivity_p
            + action_scale_rl_p * sensitivity_p * max_steps
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")
    if reset_sigma < 0:
        raise ValueError(f"reset_sigma must be >= 0, got {reset_sigma}")

    sens = sensitivity_vec()
    action_scale_arr = np.asarray(action_scale_rl, dtype=np.float64)
    reset_scale_arr = np.asarray(reset_scale, dtype=np.float64)
    if action_scale_arr.shape != sens.shape:
        raise ValueError(f"action_scale_rl must have shape {sens.shape}, got {action_scale_arr.shape}")
    if reset_scale_arr.shape != sens.shape:
        raise ValueError(f"reset_scale must have shape {sens.shape}, got {reset_scale_arr.shape}")

    defaults = np.array([p.default for p in PARAMETERS], dtype=np.float64)
    half_width = reset_sigma * reset_scale_arr * sens + action_scale_arr * sens * max_steps
    return defaults - half_width, defaults + half_width


def verify_constraints(
    action_scale_rl: np.ndarray,
    reset_scale: np.ndarray,
    max_steps: int = 20,
    margin: float = 1.3,
    reset_sigma: float = 3.0,
) -> List[str]:
    """Check hardware safety constraints for each parameter.

    Constraint:

        reset_sigma * reset_scale * sensitivity
        + action_scale_rl * sensitivity * max_steps
        <= available_margin / margin

    where ``available_margin`` is distance from default to the nearest hardware
    edge. Parameters without bounds are reported as non-verifiable.
    """
    if margin <= 0:
        raise ValueError(f"margin must be > 0, got {margin}")

    sens = sensitivity_vec()
    action_scale_arr = np.asarray(action_scale_rl, dtype=np.float64)
    reset_scale_arr = np.asarray(reset_scale, dtype=np.float64)
    warnings: List[str] = []

    for i, p in enumerate(PARAMETERS):
        s = sens[i]
        reset_worst = reset_sigma * reset_scale_arr[i] * s
        trajectory = action_scale_arr[i] * s * max_steps
        total_excursion = reset_worst + trajectory

        available = _available_margin(p)
        if available is None:
            warnings.append(
                f"  {p.name}: hw bounds unknown — cannot verify safety constraint "
                f"(total excursion = {total_excursion:.4g} param units)"
            )
            continue
        if available <= 0:
            warnings.append(
                f"  {p.name}: INVALID — default={p.default:.4g} is outside or on hw bounds "
                f"[{p.hw_min:.4g}, {p.hw_max:.4g}]"
            )
            continue

        limit = available / margin
        if reset_worst > limit:
            warnings.append(
                f"  {p.name}: RESET TOO LARGE — reset_worst {reset_worst:.4g} > "
                f"limit {limit:.4g}; action_scale_rl cap is 0"
            )
        elif total_excursion > limit:
            warnings.append(
                f"  {p.name}: VIOLATION — total excursion {total_excursion:.4g} > "
                f"limit {limit:.4g} (available_margin={available:.4g}, safety_margin={margin})"
            )

    return warnings


def _fmt(value: float | None, width: int, precision: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return f"{'--':>{width}}"
    return f"{value:>{width}.{precision}g}"


def report(
    action_scale_rl: np.ndarray | None = None,
    reset_scale: np.ndarray | None = None,
    *,
    max_steps: int = 20,
    margin: float = 1.3,
    target_trajectory_fraction: float = 0.4,
    target_steps: int = 10,
    reset_sigma: float = 3.0,
    reset_fraction: float = 0.3,
) -> None:
    """Print a compact calibration report and a ParameterSpec copy-paste block."""
    if reset_scale is None:
        reset_scale = compute_reset_scale(
            margin=margin,
            reset_sigma=reset_sigma,
            reset_fraction=reset_fraction,
        )
    else:
        reset_scale = np.asarray(reset_scale, dtype=np.float64)

    if action_scale_rl is None:
        action_scale_rl = compute_action_scale_rl(
            target_trajectory_fraction=target_trajectory_fraction,
            target_steps=target_steps,
            max_steps=max_steps,
            margin=margin,
            reset_fraction=reset_fraction,
        )
    else:
        action_scale_rl = np.asarray(action_scale_rl, dtype=np.float64)

    sens = sensitivity_vec()
    step_max = action_scale_rl * sens
    trajectory_max = step_max * max_steps

    print("\nAction Space Design")
    print(f"max_steps={max_steps}  reset_fraction={reset_fraction}  reset_sigma={reset_sigma}")
    print()

    header = (
        f"{'Parameter':<14} {'action_scale_rl':>18} {'reset_scale':>14} "
        f"{'step_max':>14} {'trajectory_max':>18}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for i, p in enumerate(PARAMETERS):
        print(
            f"{p.name:<14} {action_scale_rl[i]:>18.6g} {reset_scale[i]:>14.6g} "
            f"{step_max[i]:>14.6g} {trajectory_max[i]:>18.6g}"
        )

    print(sep)

    bar = "─" * 80
    print(f"\n{bar}")
    print("Copy-paste block for adige.py:")
    print(bar)
    for i, p in enumerate(PARAMETERS):
        hw_min_s = "None" if p.hw_min is None else repr(p.hw_min)
        hw_max_s = "None" if p.hw_max is None else repr(p.hw_max)
        print(
            f'    ParameterSpec("{p.name}", "{p.key}", marker={p.marker}, '
            f'default={p.default!r}, sensitivity={p.sensitivity:.15e}, '
            f'hw_min={hw_min_s}, hw_max={hw_max_s}, '
            f'action_scale_rl={action_scale_rl[i]:.15e}, '
            f'reset_scale={reset_scale[i]:.15e}),'
        )
    print(bar)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the action space design calibration report for ADIGE."
    )
    parser.add_argument(
        "--max-steps", type=int, default=20,
        help="Episode length in RL steps (default: %(default)s)"
    )
    parser.add_argument(
        "--target-steps", type=int, default=10,
        help="Number of steps expected to cover the target trajectory (default: %(default)s)"
    )
    parser.add_argument(
        "--margin", type=float, default=1.3,
        help="Safety divisor for hw constraint; larger is more conservative (default: %(default)s)"
    )
    parser.add_argument(
        "--target-traj", type=float, default=0.4,
        metavar="FRACTION",
        help="Fraction of available margin used as typical useful trajectory (default: %(default)s)"
    )
    parser.add_argument(
        "--reset-sigma", type=float, default=3.0,
        help="Gaussian reset sigma multiplier used as worst-case design bound (default: %(default)s)"
    )
    parser.add_argument(
        "--reset-fraction", type=float, default=0.3,
        help="Fraction of hardware-safe margin reserved for reset randomization (default: %(default)s)"
    )
    args = parser.parse_args()

    report(
        max_steps=args.max_steps,
        margin=args.margin,
        target_trajectory_fraction=args.target_traj,
        target_steps=args.target_steps,
        reset_sigma=args.reset_sigma,
        reset_fraction=args.reset_fraction,
    )
