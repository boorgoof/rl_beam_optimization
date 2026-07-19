"""
Bayesian Optimization baseline.
Uses scikit-optimize (skopt) with a Gaussian Process surrogate.

BO maintains no memory between calls (stateless), so it must re-optimize from
scratch each time. The trained RL agent, by contrast, can infer actions instantly.

Usage:
    bounds = hardware_aware_bounds(PARAMETERS, bounds_scale=125.0)
    optimizer = BayesianOptimizer(n_calls=100, param_keys=param_keys, bounds=bounds)
    result = optimizer.optimize(objective_fn)
    print(result.best_params, result.best_score)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np


@dataclass
class BOResult:
    best_params: Dict[str, float]
    best_score: float
    score_history: List[float] = field(default_factory=list)
    n_calls: int = 0


class BayesianOptimizer:
    """Gaussian Process Bayesian Optimization over a configured parameter vector.

    Args:
        n_calls:    Total number of objective evaluations.
        n_initial:  Random initial points (before GP fits).
        acq_func:   Acquisition function ('EI', 'PI', 'LCB').
        seed:       Random seed.
        param_keys: Ordered parameter keys.
        bounds:     One (lower, upper) search interval per parameter, e.g.
                    from hardware_aware_bounds().
    """

    def __init__(
        self,
        n_calls: int = 100,
        n_initial: int = 20,
        acq_func: str = "EI",
        seed: int = 42,
        *,
        param_keys: Sequence[str],
        bounds: Sequence[tuple[float, float]],
    ):
        self.n_calls    = n_calls
        self.n_initial  = n_initial
        self.acq_func   = acq_func
        self.seed       = seed
        self.param_keys = tuple(param_keys)
        self.bounds     = tuple((float(lo), float(hi)) for lo, hi in bounds)

        if not self.param_keys:
            raise ValueError("param_keys must not be empty")
        if len(self.param_keys) != len(self.bounds):
            raise ValueError("param_keys and bounds must have the same length")

    def optimize(
        self,
        objective: Callable[[Dict[str, float]], float],
    ) -> BOResult:
        """Run BO.

        Args:
            objective: Function taking {key: value} dict and returning a score (maximize).

        Returns:
            BOResult with best parameters and history.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            raise ImportError("scikit-optimize is required: pip install scikit-optimize")

        space = [
            Real(lower, upper, name=key)
            for key, (lower, upper) in zip(self.param_keys, self.bounds)
        ]

        score_history: List[float] = []

        def _objective(x: list) -> float:
            params = {k: float(v) for k, v in zip(self.param_keys, x)}
            sc = objective(params)
            score_history.append(sc)
            return -sc   # skopt minimizes

        result = gp_minimize(
            func=_objective,
            dimensions=space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial,
            acq_func=self.acq_func,
            random_state=self.seed,
        )

        best_params = {k: float(v) for k, v in zip(self.param_keys, result.x)}
        return BOResult(
            best_params=best_params,
            best_score=-float(result.fun),
            score_history=score_history,
            n_calls=len(score_history),
        )


@dataclass(frozen=True)
class WarmStartSelection:
    """Dataset rows selected to initialize a Gaussian Process."""

    indices: tuple[int, ...]
    labels: tuple[str, ...]
    param_vectors: np.ndarray
    scores: np.ndarray


def hardware_aware_bounds(parameters, bounds_scale: float) -> list[tuple[float, float]]:
    """Build default ± sensitivity bounds intersected with hardware limits."""
    if bounds_scale <= 0:
        raise ValueError(f"bounds_scale must be positive, got {bounds_scale}")

    bounds = []
    for parameter in parameters:
        sensitivity = abs(float(parameter.sensitivity))
        if not np.isfinite(sensitivity) or sensitivity <= 0:
            raise ValueError(
                f"{parameter.name} has invalid sensitivity {parameter.sensitivity!r}"
            )
        lower = float(parameter.default) - bounds_scale * sensitivity
        upper = float(parameter.default) + bounds_scale * sensitivity
        if parameter.hw_min is not None:
            lower = max(lower, float(parameter.hw_min))
        if parameter.hw_max is not None:
            upper = min(upper, float(parameter.hw_max))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"{parameter.name} has an empty Bayesian search interval: "
                f"[{lower}, {upper}]"
            )
        bounds.append((lower, upper))
    return bounds


def select_warm_start(
    param_vectors,
    scores,
    *,
    parameters,
    bounds: Sequence[tuple[float, float]],
    n_best: int = 10,
    n_diverse: int = 30,
    seed: int = 42,
) -> WarmStartSelection:
    """Select high-scoring and diverse valid dataset rows deterministically."""
    if n_best < 0 or n_diverse < 0:
        raise ValueError("n_best and n_diverse must be non-negative")
    if n_best + n_diverse <= 0:
        raise ValueError("At least one warm-start point must be requested")

    vectors = np.asarray(param_vectors, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    n_dims = len(parameters)
    if vectors.ndim != 2 or vectors.shape[1] != n_dims:
        raise ValueError(
            f"param_vectors must have shape (N, {n_dims}), got {vectors.shape}"
        )
    if vectors.shape[0] != values.shape[0]:
        raise ValueError("param_vectors and scores must have the same row count")
    if len(bounds) != n_dims:
        raise ValueError("bounds and parameters must have the same length")

    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    tolerance = 1e-12 + 1e-6 * np.maximum(upper - lower, 1e-12)
    valid_mask = (
        np.isfinite(vectors).all(axis=1)
        & np.isfinite(values)
        & (vectors >= lower - tolerance).all(axis=1)
        & (vectors <= upper + tolerance).all(axis=1)
    )
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        raise ValueError("The dataset has no finite rows inside the Bayesian bounds")
    bounded_vectors = np.clip(vectors, lower, upper)

    # Stable score ordering: score descending, original dataset index ascending.
    # For duplicate parameter vectors keep the highest-scoring observation.
    order = np.lexsort((valid_indices, -values[valid_indices]))
    seen_vectors: set[bytes] = set()
    unique_ranked = []
    for index in valid_indices[order]:
        fingerprint = np.ascontiguousarray(bounded_vectors[index]).tobytes()
        if fingerprint in seen_vectors:
            continue
        seen_vectors.add(fingerprint)
        unique_ranked.append(int(index))
    ranked = np.asarray(unique_ranked, dtype=int)
    best_indices = list(ranked[: min(n_best, len(ranked))])

    remaining = [int(index) for index in ranked if int(index) not in best_indices]
    sensitivities = np.asarray(
        [abs(float(parameter.sensitivity)) for parameter in parameters],
        dtype=np.float64,
    )
    normalized = bounded_vectors / sensitivities
    rng = np.random.default_rng(seed)
    diverse_indices: list[int] = []

    if not best_indices and remaining:
        first_position = int(rng.integers(0, len(remaining)))
        diverse_indices.append(remaining.pop(first_position))

    while remaining and len(diverse_indices) < n_diverse:
        selected = best_indices + diverse_indices
        remaining_array = np.asarray(remaining, dtype=int)
        if selected:
            deltas = (
                normalized[remaining_array, None, :]
                - normalized[np.asarray(selected, dtype=int), :]
            )
            min_distances = np.linalg.norm(deltas, axis=2).min(axis=1)
            max_distance = float(min_distances.max())
            tied_positions = np.flatnonzero(
                np.isclose(min_distances, max_distance, rtol=1e-12, atol=1e-12)
            )
            chosen_position = int(rng.choice(tied_positions))
        else:
            chosen_position = int(rng.integers(0, len(remaining)))
        diverse_indices.append(remaining.pop(chosen_position))

    selected_indices = best_indices + diverse_indices
    labels = ["best"] * len(best_indices) + ["diverse"] * len(diverse_indices)
    return WarmStartSelection(
        indices=tuple(selected_indices),
        labels=tuple(labels),
        param_vectors=bounded_vectors[selected_indices].copy(),
        scores=values[selected_indices].copy(),
    )
