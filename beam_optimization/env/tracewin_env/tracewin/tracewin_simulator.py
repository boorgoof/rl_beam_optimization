"""
It is a class that maps ADIGE parameters to a BeamSimulationResult using TraceWin via SSH to run the program with permission of the user comunian.
Note:
pyTraceWin_wrapper runs TraceWin via SSH as `comunian@localhost` (license requirement).
Each call to simulate() resets the calc_dir and launches a fresh process.
See TRACEWIN_SETUP.md for system-level prerequisites.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .pyTraceWin_wrapper import TraceWin
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator

from beam_optimization.config.adige import (
    BEAM_STATE_FEATURES, ERROR_SCORE, STAGE_MARKERS, INITIAL_NPART,
    default_params, score,
)


# TraceWin messages that describe a definitive physical outcome for the
# current machine configuration. Re-running the same configuration cannot
# repair these outcomes, so they return a failed BeamSimulationResult directly
# instead of entering the technical-error retry loop.
NON_RETRYABLE_PHYSICS_FAILURES = (
    "all particles are lost",
    "synchronous particle never reaches the end of the field map",
    "part of the beam distribution never reaches the end of the field map",
)


def _non_retryable_physics_failure(stdout: str, stderr: str) -> Optional[str]:
    """Return the original matching error line, or None if retry is appropriate."""
    for line in f"{stdout}\n{stderr}".splitlines():
        normalized = line.casefold()
        if any(message in normalized for message in NON_RETRYABLE_PHYSICS_FAILURES):
            return line.strip()
    return None


class TraceWinSimulator(BeamSimulator):
    """Run TraceWin simulations and return structured BeamSimulationResult objects.

    TraceWin runs directly on the original project workspace (no local copy):
    the workspace is made world-writable so `comunian` can use it, and any
    file TraceWin generates there (.cal, *_new.ini) is removed before and
    after every simulation so the shared workspace never accumulates
    artifacts.

    Args:
        project_file:  Path to the .ini TraceWin project file.
        calc_dir:      Working directory for TraceWin output files.
                       Created automatically; given world-write permissions so
                       TraceWin (running as `comunian` via SSH) can write there.
        timeout:       Seconds before aborting a single TraceWin call.
        retries:       Retry attempts after a technical failure (total = retries + 1).
                       Definitive physics failures listed in
                       NON_RETRYABLE_PHYSICS_FAILURES are never retried.
        retry_sleep:   Seconds to wait between retries.
        kill_stale:    If True, kill leftover `comunian` TraceWin processes before
                       each simulation (prevents resource conflicts).
        tracewin_params: Extra TraceWin CLI options forwarded as key=value
                       (e.g. {"random_seed": 42} for reproducible Monte Carlo).
        num_threads:   TraceWin nbr_thread (None = all CPUs). Use 1 for
                       bit-reproducible runs.
        initial_npart: Number of particles used to normalize npart_ratio.
                       Defaults to the project-wide INITIAL_NPART.
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str,
        timeout: float = 100.0,
        retries: int = 2,
        retry_sleep: float = 5.0,
        kill_stale: bool = True,
        tracewin_params: Optional[Dict[str, object]] = None,
        num_threads: Optional[int] = None,
        initial_npart: int = INITIAL_NPART,
    ):
        project_path = Path(project_file).resolve()
        self.project_file = str(project_path)
        self.calc_dir = str(Path(calc_dir).resolve())
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.retry_sleep = float(retry_sleep)
        self.kill_stale = bool(kill_stale)
        self.tracewin_params: Dict[str, object] = dict(tracewin_params or {})
        self.num_threads = num_threads
        self.initial_npart = int(initial_npart)
        self._source_project_dir = str(project_path.parent)
        self._project_filename = project_path.name
        self._sim_count = 0

        if not Path(self.project_file).exists():
            raise FileNotFoundError(
                f"TraceWin project file not found: {self.project_file}"
            )

        Path(self.calc_dir).mkdir(parents=True, exist_ok=True)
        os.chmod(self.calc_dir, 0o777)

    def simulate(self, params: Optional[Dict[str, float]] = None) -> BeamSimulationResult:
        """Run one TraceWin simulation."""
        full_params = default_params()
        if params:
            full_params.update(params)

        if self.kill_stale:
            _kill_stale_tracewin_processes()

        self._sim_count += 1
        last_exc: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                return self._run_once(full_params)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    print(
                        f"  [TraceWin] attempt {attempt + 1} failed: {exc}"
                        f"  — retrying in {self.retry_sleep}s"
                    )
                    time.sleep(self.retry_sleep)

        return self._failed_result(full_params, str(last_exc))

    @property
    def n_simulations(self) -> int:
        """Return the number of simulations run so far."""
        return self._sim_count

    def _reset_calc_dir(self):
        """Delete and recreate calc_dir with open permissions for comunian."""
        if os.path.exists(self.calc_dir):
            shutil.rmtree(self.calc_dir, ignore_errors=True)
        os.makedirs(self.calc_dir, exist_ok=True)
        os.chmod(self.calc_dir, 0o777)

    def _prepare_workspace(self) -> str:
        """Make the project workspace ready for TraceWin and return its .ini path."""
        source_dir = Path(self._source_project_dir)
        self._make_world_accessible(source_dir)
        self._clean_runtime_project_artifacts(source_dir)
        return self.project_file

    def _clean_runtime_project_artifacts(self, project_dir: Path) -> None:
        """Remove generated files (.cal, *_new.ini) from the project workspace."""
        for path in project_dir.iterdir():
            if path.name == self._project_filename:
                continue
            if path.name.endswith(".cal") or path.name.endswith("_new.ini"):
                path.unlink(missing_ok=True)

    def _make_world_accessible(self, root: Path) -> None:
        """Let the comunian SSH user read/write the project workspace."""
        os.chmod(root, 0o777)
        for path in root.rglob("*"):
            try:
                os.chmod(path, 0o777 if path.is_dir() else 0o666)
            except OSError:
                pass

    def _run_once(self, params: Dict[str, float]) -> BeamSimulationResult:
        """Run one TraceWin simulation and return the results."""
        self._reset_calc_dir()
        runtime_project_file = self._prepare_workspace()

        try:
            tw = TraceWin(runtime_project_file, self.calc_dir)
            success = tw.run(
                timeout=self.timeout,
                elem_params=params,
                other_params=self.tracewin_params,
                num_threads=self.num_threads,
            )

            if not success:
                physics_failure = _non_retryable_physics_failure(
                    tw.last_stdout,
                    tw.last_stderr,
                )
                if physics_failure is not None:
                    return self._failed_result(params, physics_failure)
                raise RuntimeError(
                    f"TraceWin failed.\n"
                    f"  stdout: {tw.last_stdout[:300]}\n"
                    f"  stderr: {tw.last_stderr[:300]}"
                )

            df = tw.results()
            beam_states, final_beam = self._extract_beam_states(df)
            score_val = score(final_beam)

            return BeamSimulationResult(
                params=params.copy(),
                beam_states=beam_states,
                final_beam=final_beam,
                score_val=score_val,
                success=True,
                source="tracewin",
                metadata={
                    "project_file": self.project_file,
                    "calc_dir": self.calc_dir,
                    "sim_count": self._sim_count,
                    "initial_npart": self.initial_npart,
                },
            )
        finally:
            self._clean_runtime_project_artifacts(Path(self._source_project_dir))

    def _extract_beam_states(self, df):
        """Extract beam states at each STAGE_MARKER from partran1.out."""
        n = len(STAGE_MARKERS)
        beam_states = np.zeros((n, len(BEAM_STATE_FEATURES)), dtype=np.float32)

        for stage_index, marker in enumerate(STAGE_MARKERS):
            hits = df[df["##"] == _normalize_tracewin_marker(marker)]
            if hits.empty:
                hits = df[(df["##"] - marker).abs() < 0.5]
            if hits.empty:
                continue
            row = hits.iloc[-1]
            for feature_index, feature in enumerate(BEAM_STATE_FEATURES):
                beam_states[stage_index, feature_index] = (
                    _read_beam_feature_from_tracewin_row(
                        row,
                        feature,
                        self.initial_npart,
                    )
                )

        last = df.iloc[-1]
        final_beam = {
            feature: _read_beam_feature_from_tracewin_row(
                last,
                feature,
                self.initial_npart,
            )
            for feature in BEAM_STATE_FEATURES
        }

        return beam_states, final_beam

    def _failed_result(self, params: Dict[str, float], error: str) -> BeamSimulationResult:
        """Return a BeamSimulationResult for a failed TraceWin call."""
        return BeamSimulationResult(
            params=params.copy(),
            beam_states=None,
            final_beam=None,
            score_val=ERROR_SCORE,
            success=False,
            error=error,
            source="tracewin",
            metadata={
                "project_file": self.project_file,
                "calc_dir": self.calc_dir,
                "sim_count": self._sim_count,
                "initial_npart": self.initial_npart,
            },
        )


def _normalize_tracewin_marker(value) -> int | float:
    """Normalize a TraceWin marker value before matching output rows."""
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def _read_beam_feature_from_tracewin_row(
    row,
    feature: str,
    initial_npart: int = INITIAL_NPART,
) -> float:
    """Read one beam-state feature from a TraceWin output DataFrame row."""
    if feature == "npart_ratio":
        npart = float(row.get("npart", 0.0))
        return npart / initial_npart if initial_npart > 0 else 0.0
    return float(row.get(feature, 0.0))


def _kill_stale_tracewin_processes():
    """Kill any leftover TraceWin processes running as `comunian`."""
    TraceWin._kill_remote_tracewin_processes()
