from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from beam_optimization.algorithms.baselines.bayesian_opt import (
    hardware_aware_bounds,
    select_warm_start,
)
from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    ERROR_SCORE,
    N_OUTPUT_STAGES,
    PARAMETERS,
    PARAM_KEYS,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.scripts.bayesian_opt import (
    _build_optimizer,
    run_tracewin_bayesian,
)


class _FakeOptimizer:
    def __init__(self, vector):
        self.vector = list(vector)
        self.told = []

    def ask(self):
        return list(self.vector)

    def tell(self, vector, objective):
        self.told.append((list(vector), float(objective)))


class _FakeSimulator:
    def __init__(self, outcomes, seen_seeds):
        self.tracewin_params = {}
        self._outcomes = outcomes
        self._seen_seeds = seen_seeds

    def simulate(self, params):
        seed = int(self.tracewin_params["random_seed"])
        self._seen_seeds.append(seed)
        success, score = self._outcomes[seed]
        beam_states = None
        if success:
            beam_states = np.zeros(
                (N_OUTPUT_STAGES + 1, BEAM_STATE_DIM),
                dtype=np.float32,
            )
            beam_states[:, 0] = 1.0
        return SimpleNamespace(
            params=params.copy(),
            beam_states=beam_states,
            score_val=float(score if success else ERROR_SCORE),
            success=success,
            source="tracewin",
            error=None if success else "synthetic TraceWin failure",
            timestamp=datetime.now(),
        )


def _one_row_dataset() -> BeamDataset:
    dataset = BeamDataset()
    x = np.zeros(9 + len(PARAMETERS), dtype=np.float32)
    x[9:] = np.asarray([parameter.default for parameter in PARAMETERS])
    y = np.zeros(N_OUTPUT_STAGES * BEAM_STATE_DIM, dtype=np.float32)
    dataset.append_flat_sample(x, y, 0.0)
    return dataset


class WarmStartTests(unittest.TestCase):
    def test_bounds_respect_hardware_limits(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        by_name = {
            parameter.name: bound for parameter, bound in zip(PARAMETERS, bounds)
        }
        eq02 = next(
            parameter for parameter in PARAMETERS if parameter.name == "AD.1EQ.02"
        )
        self.assertEqual(by_name["AD.1EQ.02"][1], min(
            eq02.default + 10.0 * abs(eq02.sensitivity),
            eq02.hw_max,
        ))
        for parameter, (lower, upper) in zip(PARAMETERS, bounds):
            if parameter.hw_min is not None:
                self.assertGreaterEqual(lower, parameter.hw_min)
            if parameter.hw_max is not None:
                self.assertLessEqual(upper, parameter.hw_max)

    def test_warm_start_is_deterministic_best_plus_diverse(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        rng = np.random.default_rng(123)
        lower = np.asarray([bound[0] for bound in bounds])
        upper = np.asarray([bound[1] for bound in bounds])
        vectors = rng.uniform(lower, upper, size=(55, len(PARAMETERS)))
        scores = np.arange(55, dtype=float)

        # Duplicate row 0 with a better score: the better observation must win.
        vectors = np.vstack([vectors, vectors[0]])
        scores = np.concatenate([scores, [1000.0]])
        first = select_warm_start(
            vectors,
            scores,
            parameters=PARAMETERS,
            bounds=bounds,
            n_best=10,
            n_diverse=30,
            seed=7,
        )
        second = select_warm_start(
            vectors,
            scores,
            parameters=PARAMETERS,
            bounds=bounds,
            n_best=10,
            n_diverse=30,
            seed=7,
        )
        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first.indices), 40)
        self.assertEqual(first.indices[0], 55)
        self.assertEqual(first.labels.count("best"), 10)
        self.assertEqual(first.labels.count("diverse"), 30)

    def test_missing_tracewin_seed_base_does_not_pass_random_seed(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        midpoint = [(lower + upper) / 2.0 for lower, upper in bounds]
        report = {
            "version": 1,
            "mode": "tracewin",
            "status": "running",
            "created_at": "test",
            "updated_at": "test",
            "config": {},
            "warm_start": [],
            "runs": [],
            "best_result": None,
        }
        seen_params = []

        class SimulatorWithoutSeed:
            def __init__(self):
                self.tracewin_params = {"nbr_part1": 10_000}

            def simulate(self, params):
                seen_params.append(dict(self.tracewin_params))
                beam_states = np.zeros(
                    (N_OUTPUT_STAGES + 1, BEAM_STATE_DIM),
                    dtype=np.float32,
                )
                beam_states[:, 0] = 1.0
                return SimpleNamespace(
                    params=params.copy(),
                    beam_states=beam_states,
                    score_val=5.0,
                    success=True,
                    source="tracewin",
                    error=None,
                    timestamp=datetime.now(),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch(
                "beam_optimization.scripts.bayesian_opt._build_optimizer",
                return_value=_FakeOptimizer(midpoint),
            ):
                result = run_tracewin_bayesian(
                    simulator_factory=lambda _run_index: SimulatorWithoutSeed(),
                    source_dataset=None,
                    bounds=bounds,
                    report=report,
                    output=root / "bayesian.json",
                    new_samples_output=root / "new.pt",
                    merged_dataset_output=None,
                    n_calls=1,
                    n_runs=1,
                    seed=42,
                    tracewin_seed_base=None,
                )

        self.assertNotIn("random_seed", seen_params[0])
        self.assertIsNone(result["runs"][0]["evaluations"][0]["tracewin_seed"])

    def test_optimizer_uses_gaussian_noise(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        optimizer = _build_optimizer(
            bounds,
            optimizer_seed=42,
            warm_start=[],
            evaluations=[],
        )
        self.assertEqual(optimizer.base_estimator_.noise, "gaussian")


class TraceWinLoopTests(unittest.TestCase):
    def test_variable_seeds_failure_persistence_and_resume(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        midpoint = [(lower + upper) / 2.0 for lower, upper in bounds]
        warm_params = {
            key: float(parameter.default)
            for key, parameter in zip(PARAM_KEYS, PARAMETERS)
        }
        report = {
            "version": 1,
            "mode": "tracewin",
            "status": "running",
            "created_at": "test",
            "updated_at": "test",
            "config": {},
            "warm_start": [
                {
                    "dataset_index": 0,
                    "selection": "best",
                    "params": warm_params,
                    "score": 1.0,
                }
            ],
            "runs": [],
            "best_result": None,
        }
        outcomes = {
            100: (True, 10.0),
            101: (False, ERROR_SCORE),
            102: (True, 20.0),
        }
        seen_seeds = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bayesian.json"
            new_output = root / "new.pt"
            merged_output = root / "merged.pt"
            source = _one_row_dataset()

            def simulator_factory(_run_index):
                return _FakeSimulator(outcomes, seen_seeds)

            with mock.patch(
                "beam_optimization.scripts.bayesian_opt._build_optimizer",
                return_value=_FakeOptimizer(midpoint),
            ):
                first = run_tracewin_bayesian(
                    simulator_factory=simulator_factory,
                    source_dataset=source,
                    bounds=bounds,
                    report=report,
                    output=output,
                    new_samples_output=new_output,
                    merged_dataset_output=merged_output,
                    n_calls=2,
                    n_runs=1,
                    seed=42,
                    tracewin_seed_base=100,
                )

            self.assertEqual(seen_seeds, [100, 101])
            self.assertEqual(
                [entry["tracewin_seed"] for entry in first["runs"][0]["evaluations"]],
                [100, 101],
            )
            self.assertEqual(len(BeamDataset.load(new_output)), 1)
            self.assertEqual(len(BeamDataset.load(merged_output)), 2)

            resumed = json.loads(output.read_text(encoding="utf-8"))
            with mock.patch(
                "beam_optimization.scripts.bayesian_opt._build_optimizer",
                return_value=_FakeOptimizer(midpoint),
            ):
                final = run_tracewin_bayesian(
                    simulator_factory=simulator_factory,
                    source_dataset=source,
                    bounds=bounds,
                    report=resumed,
                    output=output,
                    new_samples_output=new_output,
                    merged_dataset_output=merged_output,
                    n_calls=3,
                    n_runs=1,
                    seed=42,
                    tracewin_seed_base=100,
                )

            self.assertEqual(seen_seeds, [100, 101, 102])
            self.assertEqual(len(final["runs"][0]["evaluations"]), 3)
            self.assertEqual(final["best_result"]["tracewin_seed"], 102)
            self.assertEqual(final["best_result"]["score"], 20.0)
            self.assertEqual(len(BeamDataset.load(new_output)), 2)
            self.assertEqual(len(BeamDataset.load(merged_output)), 3)


if __name__ == "__main__":
    unittest.main()
