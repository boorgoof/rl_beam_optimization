"""
Bayesian Optimization baseline.
Uses scikit-optimize (skopt) with a Gaussian Process surrogate.

BO maintains no memory between calls (stateless), so it must re-optimize from
scratch each time. The trained RL agent, by contrast, can infer actions instantly.

Usage:
    optimizer = BayesianOptimizer(n_calls=100)
    result = optimizer.optimize(objective_fn)
    print(result.best_params, result.best_score)
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
class BOResult:
    best_params: Dict[str, float]
    best_score: float
    score_history: List[float] = field(default_factory=list)
    n_calls: int = 0


class BayesianOptimizer:
    """Gaussian Process Bayesian Optimization over the 16 beam parameters.
    
    Args:
        n_calls:      Total number of objective evaluations.
        n_initial:    Random initial points (before GP fits).
        bounds_scale: Action range as multiple of sensitivity around default.
        acq_func:     Acquisition function ('EI', 'PI', 'LCB').
        seed:         Random seed.
    """

    def __init__(
        self,
        n_calls: int = 100,
        n_initial: int = 20,
        bounds_scale: float = 5.0,
        acq_func: str = "EI",
        seed: int = 42,
    ):
        self.n_calls     = n_calls
        self.n_initial   = n_initial
        self.bounds_scale = bounds_scale
        self.acq_func    = acq_func
        self.seed        = seed

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

        # Build search space: ±(bounds_scale × sensitivity) around default
        space = []
        defaults = default_params()
        for p in PARAMETERS:
            lo = p.default - self.bounds_scale * p.sensitivity
            hi = p.default + self.bounds_scale * p.sensitivity
            space.append(Real(lo, hi, name=p.key))

        score_history: List[float] = []

        def _objective(x: list) -> float:
            params = {k: v for k, v in zip(PARAM_KEYS, x)}
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

        best_params = {k: float(v) for k, v in zip(PARAM_KEYS, result.x)}
        return BOResult(
            best_params=best_params,
            best_score=-float(result.fun),
            score_history=score_history,
            n_calls=len(score_history),
        )
