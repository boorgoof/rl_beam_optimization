"""
Particle Swarm Optimization (PSO) baseline.

Pure Python implementation — no external dependency beyond numpy.
Uses the standard inertia-weight PSO (Kennedy & Eberhart, 1995).

Usage:
    optimizer = PSOOptimizer(
        n_particles=30,
        n_iterations=100,
        param_keys=param_keys,
        default_values=default_values,
        sensitivity_values=sensitivity_values,
    )
    result = optimizer.optimize(objective_fn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np


@dataclass
class PSOResult:
    best_params: Dict[str, float]
    best_score: float
    score_history: List[float] = field(default_factory=list)
    n_evaluations: int = 0


class PSOOptimizer:
    """Particle Swarm Optimization over a configured parameter vector.

    Args:
        n_particles:   Swarm size.
        n_iterations:  Number of PSO iterations.
        bounds_scale:  Search range as multiple of sensitivity around default.
        w:             Inertia weight (momentum).
        c1:            Cognitive coefficient (personal best attraction).
        c2:            Social coefficient (global best attraction).
        seed:          Random seed.
        param_keys:    Ordered parameter keys.
        default_values: Ordered default parameter values.
        sensitivity_values: Ordered sensitivity values used to build bounds.
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 100,
        bounds_scale: float = 5.0,
        w: float = 0.729,
        c1: float = 1.494,
        c2: float = 1.494,
        seed: int = 42,
        *,
        param_keys: Sequence[str],
        default_values: Sequence[float],
        sensitivity_values: Sequence[float],
    ):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.bounds_scale = bounds_scale
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.param_keys = tuple(param_keys)
        self.default_values = np.asarray(default_values, dtype=np.float64)
        self.sensitivity_values = np.asarray(sensitivity_values, dtype=np.float64)

        if not self.param_keys:
            raise ValueError("param_keys must not be empty")
        if len(self.param_keys) != len(self.default_values):
            raise ValueError("param_keys and default_values must have the same length")
        if len(self.param_keys) != len(self.sensitivity_values):
            raise ValueError("param_keys and sensitivity_values must have the same length")
        self.n_params = len(self.param_keys)

    def optimize(
        self,
        objective: Callable[[Dict[str, float]], float],
    ) -> PSOResult:
        """Run PSO.

        Args:
            objective: Function taking {key: value} dict and returning a score (maximize).

        Returns:
            PSOResult with best parameters and iteration history.
        """
        rng = np.random.default_rng(self.seed)

        lo = self.default_values - self.bounds_scale * self.sensitivity_values
        hi = self.default_values + self.bounds_scale * self.sensitivity_values

        # Initialize swarm
        positions  = rng.uniform(lo, hi, size=(self.n_particles, self.n_params))
        velocities = rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1,
                                  size=(self.n_particles, self.n_params))

        p_best_pos   = positions.copy()
        p_best_score = np.full(self.n_particles, -np.inf)

        g_best_pos   = positions[0].copy()
        g_best_score = -np.inf

        score_history: List[float] = []
        n_eval = 0

        # Evaluate initial positions
        for i in range(self.n_particles):
            sc = objective(self._vec_to_params(positions[i]))
            n_eval += 1
            score_history.append(sc)
            p_best_score[i] = sc
            if sc > g_best_score:
                g_best_score = sc
                g_best_pos   = positions[i].copy()

        for iteration in range(self.n_iterations):
            r1 = rng.uniform(0, 1, size=(self.n_particles, self.n_params))
            r2 = rng.uniform(0, 1, size=(self.n_particles, self.n_params))

            velocities = (self.w  * velocities
                         + self.c1 * r1 * (p_best_pos - positions)
                         + self.c2 * r2 * (g_best_pos - positions))
            positions  = np.clip(positions + velocities, lo, hi)

            for i in range(self.n_particles):
                sc = objective(self._vec_to_params(positions[i]))
                n_eval += 1
                score_history.append(sc)
                if sc > p_best_score[i]:
                    p_best_score[i] = sc
                    p_best_pos[i]   = positions[i].copy()
                if sc > g_best_score:
                    g_best_score = sc
                    g_best_pos   = positions[i].copy()

            if (iteration + 1) % 10 == 0:
                print(f"  PSO iter {iteration+1:4d}  best_score={g_best_score:.4f}")

        return PSOResult(
            best_params=self._vec_to_params(g_best_pos),
            best_score=g_best_score,
            score_history=score_history,
            n_evaluations=n_eval,
        )

    def _vec_to_params(self, vec: np.ndarray) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.param_keys, vec)}
