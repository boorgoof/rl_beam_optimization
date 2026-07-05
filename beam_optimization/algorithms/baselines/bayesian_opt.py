"""
Bayesian Optimization baseline.
Uses scikit-optimize (skopt) with a Gaussian Process surrogate.

BO maintains no memory between calls (stateless), so it must re-optimize from
scratch each time. The trained RL agent, by contrast, can infer actions instantly.

Usage:
    optimizer = BayesianOptimizer(
        n_calls=100,
        param_keys=param_keys,
        default_values=default_values,
        sensitivity_values=sensitivity_values,
    )
    result = optimizer.optimize(objective_fn)
    print(result.best_params, result.best_score)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence


@dataclass
class BOResult:
    best_params: Dict[str, float]
    best_score: float
    score_history: List[float] = field(default_factory=list)
    n_calls: int = 0


class BayesianOptimizer:
    """Gaussian Process Bayesian Optimization over a configured parameter vector.
    
    Args:
        n_calls:      Total number of objective evaluations.
        n_initial:    Random initial points (before GP fits).
        bounds_scale: Search range as multiple of the (1-point) sensitivity around default.
        acq_func:     Acquisition function ('EI', 'PI', 'LCB').
        seed:         Random seed.
        param_keys:    Ordered parameter keys.
        default_values: Ordered default parameter values.
        sensitivity_values: Ordered sensitivity values used to build bounds.
    """

    def __init__(
        self,
        n_calls: int = 100,
        n_initial: int = 20,
        bounds_scale: float = 125.0,
        acq_func: str = "EI",
        seed: int = 42,
        *,
        param_keys: Sequence[str],
        default_values: Sequence[float],
        sensitivity_values: Sequence[float],
    ):
        self.n_calls     = n_calls
        self.n_initial   = n_initial
        self.bounds_scale = bounds_scale
        self.acq_func    = acq_func
        self.seed        = seed
        self.param_keys = tuple(param_keys)
        self.default_values = tuple(float(v) for v in default_values)
        self.sensitivity_values = tuple(float(v) for v in sensitivity_values)

        if not self.param_keys:
            raise ValueError("param_keys must not be empty")
        if len(self.param_keys) != len(self.default_values):
            raise ValueError("param_keys and default_values must have the same length")
        if len(self.param_keys) != len(self.sensitivity_values):
            raise ValueError("param_keys and sensitivity_values must have the same length")

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
        for key, default, sensitivity in zip(
            self.param_keys,
            self.default_values,
            self.sensitivity_values,
        ):
            lo = default - self.bounds_scale * sensitivity
            hi = default + self.bounds_scale * sensitivity
            space.append(Real(lo, hi, name=key))

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
