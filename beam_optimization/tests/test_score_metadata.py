from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from beam_optimization.config import adige
from beam_optimization.env.dataset.tracewin_dataset_builder import (
    TraceWinDatasetBuilder,
)
from beam_optimization.scripts.bayesian_opt import (
    _load_or_create_report,
    _new_report,
)


class ScoreMetadataTests(unittest.TestCase):
    def test_normalized_ast_ignores_formatting_comments_and_docstrings(self):
        compact = """
        def example(x):
            return x + 1
        """
        formatted = """
        def example(
            x,
        ):
            \"""Changed documentation only.\"""
            # Changed comment only.
            return (
                x
                + 1
            )
        """
        changed = """
        def example(x):
            return x - 1
        """

        self.assertEqual(
            adige._normalized_function_ast(compact),
            adige._normalized_function_ast(formatted),
        )
        self.assertNotEqual(
            adige._normalized_function_ast(compact),
            adige._normalized_function_ast(changed),
        )

    def test_metadata_contains_code_hash_and_covers_declared_configuration(self):
        metadata = adige.score_function_metadata()
        self.assertEqual(len(metadata["implementation_sha256"]), 64)
        canonical = dict(metadata)
        saved_digest = canonical.pop("sha256")
        expected = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(saved_digest, expected)

        with mock.patch.dict(
            adige.SCORE_WEIGHTS,
            {"emittance": adige.SCORE_WEIGHTS["emittance"] + 1.0},
        ):
            changed = adige.score_function_metadata()
        self.assertEqual(
            metadata["implementation_sha256"],
            changed["implementation_sha256"],
        )
        self.assertNotEqual(metadata["sha256"], changed["sha256"])


class ScoreMetadataDoesNotBlockResumeTests(unittest.TestCase):
    """score_function is informational only: a mismatched or missing tag
    must never prevent a resume (see score_provenance.py removal)."""

    def test_bayesian_resume_ignores_mismatched_or_missing_score_function(self):
        for mode in ("tracewin", "tracewin_cold_start"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "report.json"
                report = _new_report({"calls": 1}, [], mode=mode)

                legacy = copy.deepcopy(report)
                legacy.pop("score_function")
                output.write_text(json.dumps(legacy), encoding="utf-8")
                resumed = _load_or_create_report(
                    output, config={"calls": 1}, warm_start=[], mode=mode,
                )
                self.assertEqual(resumed["config"], {"calls": 1})

                mismatched = copy.deepcopy(report)
                mismatched["score_function"]["sha256"] = "0" * 64
                output.write_text(json.dumps(mismatched), encoding="utf-8")
                resumed = _load_or_create_report(
                    output, config={"calls": 1}, warm_start=[], mode=mode,
                )
                self.assertEqual(resumed["score_function"]["sha256"], "0" * 64)

    def test_dataset_builder_resume_ignores_mismatched_or_missing_score_function(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "dataset"
            builder = TraceWinDatasetBuilder(
                object(), output_dir=output_dir, target_samples=1, seed=123,
            )
            state = builder._load_or_create_state()

            for mutation in (
                lambda payload: payload.pop("score_function"),
                lambda payload: payload["score_function"].update({"sha256": "f" * 64}),
            ):
                payload = copy.deepcopy(state)
                mutation(payload)
                builder.state_path.write_text(json.dumps(payload), encoding="utf-8")
                resumed = TraceWinDatasetBuilder(
                    object(), output_dir=output_dir, target_samples=1, seed=123,
                )
                self.assertEqual(resumed._load_or_create_state()["config"], state["config"])


if __name__ == "__main__":
    unittest.main()
