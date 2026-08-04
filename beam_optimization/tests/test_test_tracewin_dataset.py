"""TraceWin qualitative tests use the explicitly selected KNN dataset."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

from beam_optimization.scripts import test as test_script


class TraceWinTestDatasetTests(unittest.TestCase):
    def test_cli_dataset_is_forwarded_to_tracewin_render_environment(self):
        args = SimpleNamespace(
            env="tracewin",
            reset_scale="test",
            tracewin_project="workspace/project.ini",
            calc_dir="results/test_calc",
            max_ep_steps=20,
            tracewin_timeout=180.0,
            dataset="dataset/018/dataset_all.pt",
            no_kill_stale=True,
        )
        selected_dataset = object()
        environment = object()

        with (
            mock.patch.object(
                test_script.BeamDataset,
                "load",
                return_value=selected_dataset,
            ) as dataset_load,
            mock.patch(
                "beam_optimization.env.tracewin_env.TraceWinEnv",
                return_value=environment,
            ) as env_class,
        ):
            result = test_script.make_env(args)

        self.assertIs(result, environment)
        dataset_load.assert_called_once_with(args.dataset)
        self.assertIs(
            env_class.call_args.kwargs["distance_dataset"],
            selected_dataset,
        )
        self.assertFalse(env_class.call_args.kwargs["kill_stale"])


if __name__ == "__main__":
    unittest.main()
