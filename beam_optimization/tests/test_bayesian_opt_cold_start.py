from __future__ import annotations

import inspect
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
)
from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    ERROR_SCORE,
    N_OUTPUT_STAGES,
    PARAMETERS,
    PARAM_KEYS,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.scripts import bayesian_opt_cold_start
from beam_optimization.scripts.bayesian_opt import (
    _build_optimizer,
    _params_from_vector,
    _random_unused_tracewin_seed,
    run_tracewin_bayesian,
)


class _FakeOptimizer:
    def __init__(self, vector):
        self.vector = list(vector)

    def ask(self):
        return list(self.vector)

    def tell(self, _vector, _objective):
        return None


class _FakeSimulator:
    def __init__(self, outcomes, seen_seeds):
        self.tracewin_params = {}
        self.outcomes = outcomes
        self.seen_seeds = seen_seeds

    def simulate(self, params):
        seed = int(self.tracewin_params["random_seed"])
        self.seen_seeds.append(seed)
        success, score = self.outcomes[seed]
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
            error=None if success else "synthetic cold-start failure",
            timestamp=datetime.now(),
        )


class SobolOptimizerTests(unittest.TestCase):
    def test_power_of_two_validation(self):
        self.assertTrue(bayesian_opt_cold_start._is_power_of_two(64))
        self.assertFalse(bayesian_opt_cold_start._is_power_of_two(50))
        self.assertFalse(bayesian_opt_cold_start._is_power_of_two(0))

    def test_random_optimizer_seed_is_saved_and_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cold.json"
            with mock.patch(
                "beam_optimization.scripts.bayesian_opt_cold_start.secrets.randbelow",
                return_value=123456,
            ):
                self.assertEqual(
                    bayesian_opt_cold_start._resolve_optimizer_seed(None, output),
                    123456,
                )
            output.write_text(
                json.dumps({
                    "mode": "tracewin_cold_start",
                    "config": {"seed": 987654},
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                bayesian_opt_cold_start._resolve_optimizer_seed(None, output),
                987654,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            payload["config"]["tracewin_seed_base"] = 4321
            output.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                bayesian_opt_cold_start._resolve_tracewin_seed_base(None, output),
                4321,
            )

    def test_random_tracewin_seed_does_not_reuse_completed_seed(self):
        report = {
            "runs": [{"evaluations": [{"tracewin_seed": 77}]}],
        }
        with mock.patch(
            "beam_optimization.scripts.bayesian_opt.secrets.randbelow",
            side_effect=[77, 88],
        ):
            self.assertEqual(_random_unused_tracewin_seed(report), 88)

    def test_sobol_points_precede_first_gp_model_and_resume_exactly(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        optimizer = _build_optimizer(
            bounds,
            optimizer_seed=42,
            warm_start=[],
            evaluations=[],
            initial_points=4,
            initial_point_generator="sobol",
        )
        evaluated = []
        for index in range(3):
            point = optimizer.ask()
            evaluated.append({"params": _params_from_vector(point), "score": index})
            optimizer.tell(point, -float(index))
            self.assertEqual(len(optimizer.models), 0)

        expected_fourth = optimizer.ask()
        resumed = _build_optimizer(
            bounds,
            optimizer_seed=42,
            warm_start=[],
            evaluations=evaluated,
            initial_points=4,
            initial_point_generator="sobol",
        )
        resumed_fourth = resumed.ask()
        np.testing.assert_allclose(resumed_fourth, expected_fourth)
        resumed.tell(resumed_fourth, -3.0)
        self.assertEqual(len(resumed.models), 1)

        evaluated.append({
            "params": _params_from_vector(resumed_fourth),
            "score": 3.0,
        })
        expected_first_gp = resumed.ask()
        resumed_gp = _build_optimizer(
            bounds,
            optimizer_seed=42,
            warm_start=[],
            evaluations=evaluated,
            initial_points=4,
            initial_point_generator="sobol",
        )
        np.testing.assert_allclose(resumed_gp.ask(), expected_first_gp)

    def test_cold_module_does_not_load_dataset_or_surrogate(self):
        source = inspect.getsource(bayesian_opt_cold_start)
        self.assertNotIn("BeamDataset.load", source)
        self.assertNotIn("ModularMLP", source)
        self.assertNotIn("--dataset", source)


class ColdStartLoopTests(unittest.TestCase):
    def test_failure_seed_phases_samples_and_resume(self):
        bounds = hardware_aware_bounds(PARAMETERS, 10.0)
        midpoint = [(lower + upper) / 2.0 for lower, upper in bounds]
        report = {
            "version": 1,
            "mode": "tracewin_cold_start",
            "status": "running",
            "created_at": "test",
            "updated_at": "test",
            "config": {},
            "warm_start": [],
            "runs": [],
            "best_result": None,
        }
        outcomes = {
            200: (False, ERROR_SCORE),
            201: (True, 10.0),
            202: (True, 20.0),
        }
        seen_seeds = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "cold.json"
            samples_output = root / "cold_samples.pt"

            def simulator_factory(_run_index):
                return _FakeSimulator(outcomes, seen_seeds)

            with mock.patch(
                "beam_optimization.scripts.bayesian_opt._build_optimizer",
                return_value=_FakeOptimizer(midpoint),
            ):
                first = run_tracewin_bayesian(
                    simulator_factory=simulator_factory,
                    source_dataset=None,
                    bounds=bounds,
                    report=report,
                    output=output,
                    new_samples_output=samples_output,
                    merged_dataset_output=None,
                    n_calls=2,
                    n_runs=1,
                    seed=42,
                    tracewin_seed_base=200,
                    initial_points=1,
                    initial_point_generator="sobol",
                )

            self.assertEqual(seen_seeds, [200, 201])
            evaluations = first["runs"][0]["evaluations"]
            self.assertEqual([entry["phase"] for entry in evaluations], [
                "sobol",
                "bayesian",
            ])
            self.assertFalse(evaluations[0]["success"])
            self.assertEqual(len(BeamDataset.load(samples_output)), 1)

            resumed = json.loads(output.read_text(encoding="utf-8"))
            with mock.patch(
                "beam_optimization.scripts.bayesian_opt._build_optimizer",
                return_value=_FakeOptimizer(midpoint),
            ):
                final = run_tracewin_bayesian(
                    simulator_factory=simulator_factory,
                    source_dataset=None,
                    bounds=bounds,
                    report=resumed,
                    output=output,
                    new_samples_output=samples_output,
                    merged_dataset_output=None,
                    n_calls=3,
                    n_runs=1,
                    seed=42,
                    tracewin_seed_base=200,
                    initial_points=1,
                    initial_point_generator="sobol",
                )

            self.assertEqual(seen_seeds, [200, 201, 202])
            self.assertEqual(final["best_result"]["tracewin_seed"], 202)
            self.assertEqual(len(BeamDataset.load(samples_output)), 2)
            self.assertEqual(list(root.glob("*merged*")), [])


if __name__ == "__main__":
    unittest.main()
