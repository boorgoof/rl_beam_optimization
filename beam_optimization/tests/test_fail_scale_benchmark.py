from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from beam_optimization.config.adige import (
    BEAM_STATE_DIM,
    N_OUTPUT_STAGES,
    N_PARAMS,
    PARAM_KEYS,
    PARAMETERS,
    TEST_RESET_SCALE,
    default_params,
    observation_dim,
    sensitivity_vec,
)
from beam_optimization.env.dataset import BeamDataset
from beam_optimization.env.simulation import BeamSimulationResult
from beam_optimization.scripts.fail_scale_benchmark import (
    optimum_distance,
    run_fail_scale_benchmark,
    sample_stress_params,
    save_csvs,
    save_episode_plot,
    shell_radius,
)


def _result(params, success, score, error=None):
    return BeamSimulationResult(
        params=dict(params), beam_states=None, score_val=score,
        success=success, error=error, source="test",
    )


def _dataset() -> BeamDataset:
    n = 8
    defaults = np.asarray(list(default_params().values()), dtype=np.float32)
    sensitivities = sensitivity_vec().astype(np.float32)
    param_rows = np.stack([
        defaults + sensitivities * (index - n / 2) * 0.01
        for index in range(n)
    ])
    X = np.concatenate([np.zeros((n, BEAM_STATE_DIM), dtype=np.float32), param_rows], axis=1)
    Y = np.zeros((n, N_OUTPUT_STAGES * BEAM_STATE_DIM), dtype=np.float32)
    dataset = BeamDataset()
    dataset.append_flat_samples(X, Y, np.zeros(n, dtype=np.float32))
    return dataset


class _Agent:
    def __init__(self):
        self.calls = 0

    def select_action(self, obs, training=False):
        self.calls += 1
        return np.zeros(N_PARAMS, dtype=np.float32)


class _Env:
    max_steps = 1

    def __init__(self):
        self.reset_calls = 0
        self._current_params = default_params()
        self.simulator = object()

    @property
    def current_params(self):
        return dict(self._current_params)

    def reset(self, seed=None, options=None):
        self.reset_calls += 1
        self._current_params = dict(options["initial_params"])
        if self.reset_calls == 1:
            result = _result(
                self._current_params, False, -999,
                "Error: Part of the beam distribution never reaches the end of the field map!",
            )
        else:
            result = _result(self._current_params, True, 10.0)
        return np.zeros(observation_dim(), dtype=np.float32), {
            "sim_result": result,
            "score": result.score_val,
        }

    def step(self, action):
        defaults = default_params()
        self._current_params = {
            key: defaults[key] + 0.5 * (self._current_params[key] - defaults[key])
            for key in PARAM_KEYS
        }
        result = _result(self._current_params, True, 20.0)
        return np.zeros(observation_dim(), dtype=np.float32), 10.0, False, True, {
            "sim_result": result, "score": 20.0, "step": 1,
        }


class FailScaleBenchmarkTests(unittest.TestCase):
    def test_shell_sampler_stays_between_three_sigma_edges_after_clipping(self):
        rng = np.random.default_rng(5)
        outer = 1.2
        for _ in range(100):
            params = sample_stress_params(
                rng, inner_scale=TEST_RESET_SCALE, outer_scale=outer
            )
            radius = shell_radius(params)
            self.assertGreater(radius, 3 * TEST_RESET_SCALE)
            self.assertLessEqual(radius, 3 * outer)
            for spec in PARAMETERS:
                if spec.hw_min is not None:
                    self.assertGreaterEqual(params[spec.key], spec.hw_min)
                if spec.hw_max is not None:
                    self.assertLessEqual(params[spec.key], spec.hw_max)

    def test_optimum_distance_is_rms_sensitivity_displacement(self):
        params = default_params()
        for spec in PARAMETERS:
            params[spec.key] += 2.0 * spec.sensitivity
        self.assertAlmostEqual(optimum_distance(params), 2.0)

    def test_failed_reset_is_recorded_and_resampled_before_policy_runs(self):
        env = _Env()
        agent = _Agent()
        reference = _result(default_params(), True, 25.0)
        report = run_fail_scale_benchmark(
            env, agent, _dataset(), outer_scale=1.2,
            episodes=1, max_reset_attempts=4, seed=7,
            reference_result=reference, verbose=False,
        )
        self.assertEqual(env.reset_calls, 2)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(report["reset_physics_failures"], 1)
        self.assertEqual(report["completed_valid_episodes"], 1)
        summary = report["episodes"][0]["summary"]
        self.assertGreater(summary["score_improvement"], 0)
        self.assertGreater(summary["optimum_distance_reduction"], 0)

    def test_csv_and_three_panel_plot_are_created(self):
        report = run_fail_scale_benchmark(
            _Env(), _Agent(), _dataset(), outer_scale=1.2,
            episodes=1, max_reset_attempts=4, seed=7,
            reference_result=_result(default_params(), True, 25.0),
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csvs = save_csvs(report, root / "report.json")
            plot = save_episode_plot(report["episodes"][0], 25.0, root / "plots")
            for path in [*map(Path, csvs.values()), plot]:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
