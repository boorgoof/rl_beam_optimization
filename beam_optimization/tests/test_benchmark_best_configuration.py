"""Persistence of the global best configuration in policy benchmarks."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from beam_optimization.config.adige import PARAM_KEYS
from beam_optimization.scripts.benchmark import (
    run_policy_episode,
    summarize_policy_episodes,
    write_policy_csvs,
)


def _params(offset: float) -> dict[str, float]:
    return {key: offset + index for index, key in enumerate(PARAM_KEYS)}


def _episode(
    episode: int,
    *,
    final_score: float,
    best_score: float,
    best_step: int,
    params: dict[str, float],
) -> dict:
    return {
        "algorithm": "sb3_sac",
        "episode": episode,
        "total_reward": final_score / 100.0,
        "final_score": final_score,
        "final_ex": 0.05,
        "final_ey": 0.05,
        "final_emittance": 0.05,
        "final_npart_ratio": 0.5,
        "n_steps": 3,
        "_best_observed_score": best_score,
        "_best_observed_step": best_step,
        "_best_observed_params": params,
    }


class BenchmarkBestConfigurationTests(unittest.TestCase):
    def test_policy_episode_captures_intermediate_environment_maximum(self):
        env = SimpleNamespace(
            state=SimpleNamespace(
                best_score=57.0,
                best_step=2,
                best_params=_params(10.0),
            ),
        )
        common_result = {
            "total_reward": 0.4,
            "final_score": 40.0,
            "final_features": {"ex": 0.04, "ey": 0.06, "npart_ratio": 0.8},
            "n_steps": 3,
            "steps": [],
        }
        with patch(
            "beam_optimization.scripts.benchmark.run_episode",
            return_value=common_result,
        ):
            row = run_policy_episode(env, object(), "sb3_sac", 42, 0)

        self.assertEqual(row["final_score"], 40.0)
        self.assertEqual(row["_best_observed_score"], 57.0)
        self.assertEqual(row["_best_observed_step"], 2)
        self.assertEqual(row["_best_observed_params"], _params(10.0))

    def test_summary_selects_global_maximum_not_final_score(self):
        first = _episode(
            0, final_score=30.0, best_score=55.0, best_step=2, params=_params(0.0)
        )
        second = _episode(
            1, final_score=50.0, best_score=60.0, best_step=1, params=_params(100.0)
        )

        result = summarize_policy_episodes([first, second])["sb3_sac"]

        self.assertEqual(result["best_observed_score"], 60.0)
        self.assertEqual(result["best_observed_episode"], 1)
        self.assertEqual(result["best_observed_step"], 1)
        self.assertEqual(result["best_observed_params"], _params(100.0))

    def test_tied_maximum_keeps_first_occurrence(self):
        first = _episode(
            0, final_score=30.0, best_score=60.0, best_step=2, params=_params(0.0)
        )
        second = _episode(
            1, final_score=50.0, best_score=60.0, best_step=1, params=_params(100.0)
        )

        result = summarize_policy_episodes([first, second])["sb3_sac"]

        self.assertEqual(result["best_observed_episode"], 0)
        self.assertEqual(result["best_observed_params"], _params(0.0))

    def test_summary_csv_has_all_parameter_columns_and_tracewin_tag(self):
        internal = _episode(
            0, final_score=30.0, best_score=55.0, best_step=2, params=_params(5.0)
        )
        summary = summarize_policy_episodes([internal])
        public = [{
            key: value for key, value in internal.items()
            if not key.startswith("_best_observed_")
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "benchmark.json"
            episodes_path, summary_path = write_policy_csvs(
                public, summary, output, tag="_tracewin"
            )

            self.assertEqual(episodes_path.name, "benchmark_policy_episodes_tracewin.csv")
            self.assertEqual(summary_path.name, "benchmark_policy_summary_tracewin.csv")
            with summary_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["best_observed_score"]), 55.0)
            self.assertEqual(int(rows[0]["best_observed_episode"]), 0)
            self.assertEqual(int(rows[0]["best_observed_step"]), 2)
            for key in PARAM_KEYS:
                self.assertIn(f"best_param_{key}", rows[0])
                self.assertEqual(
                    float(rows[0][f"best_param_{key}"]),
                    _params(5.0)[key],
                )


if __name__ == "__main__":
    unittest.main()
