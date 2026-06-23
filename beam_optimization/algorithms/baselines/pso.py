"""
Particle Swarm Optimization (PSO) baseline.

Pure Python implementation — no external dependency beyond numpy.
Uses the standard inertia-weight PSO (Kennedy & Eberhart, 1995).

Usage:
    optimizer = PSOOptimizer(n_particles=30, n_iterations=100)
    result = optimizer.optimize(objective_fn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from beam_optimization.config.adige import (
    PARAMETERS, PARAM_KEYS, N_PARAMS,
    default_params, params_to_vec, vec_to_params,
)


@dataclass
class PSOResult:
    best_params: Dict[str, float]
    best_score: float
    score_history: List[float] = field(default_factory=list)
    n_evaluations: int = 0


class PSOOptimizer:
    """Particle Swarm Optimization over the 16 beam parameters.

    Args:
        n_particles:   Swarm size.
        n_iterations:  Number of PSO iterations.
        bounds_scale:  Search range as multiple of sensitivity around default.
        w:             Inertia weight (momentum).
        c1:            Cognitive coefficient (personal best attraction).
        c2:            Social coefficient (global best attraction).
        seed:          Random seed.
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
    ):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.bounds_scale = bounds_scale
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

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

        defaults = np.array([p.default for p in PARAMETERS], dtype=np.float64)
        sens     = np.array([p.sensitivity for p in PARAMETERS], dtype=np.float64)
        lo       = defaults - self.bounds_scale * sens
        hi       = defaults + self.bounds_scale * sens

        # Initialize swarm
        positions  = rng.uniform(lo, hi, size=(self.n_particles, N_PARAMS))
        velocities = rng.uniform(-(hi - lo) * 0.1, (hi - lo) * 0.1,
                                  size=(self.n_particles, N_PARAMS))

        p_best_pos   = positions.copy()
        p_best_score = np.full(self.n_particles, -np.inf)

        g_best_pos   = positions[0].copy()
        g_best_score = -np.inf

        score_history: List[float] = []
        n_eval = 0

        # Evaluate initial positions
        for i in range(self.n_particles):
            sc = objective(vec_to_params(positions[i]))
            n_eval += 1
            score_history.append(sc)
            p_best_score[i] = sc
            if sc > g_best_score:
                g_best_score = sc
                g_best_pos   = positions[i].copy()

        for iteration in range(self.n_iterations):
            r1 = rng.uniform(0, 1, size=(self.n_particles, N_PARAMS))
            r2 = rng.uniform(0, 1, size=(self.n_particles, N_PARAMS))

            velocities = (self.w  * velocities
                         + self.c1 * r1 * (p_best_pos - positions)
                         + self.c2 * r2 * (g_best_pos - positions))
            positions  = np.clip(positions + velocities, lo, hi)

            for i in range(self.n_particles):
                sc = objective(vec_to_params(positions[i]))
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
            best_params=vec_to_params(g_best_pos),
            best_score=g_best_score,
            score_history=score_history,
            n_evaluations=n_eval,
        )
