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
        retries:       Retry attempts after the first failure (total = retries + 1).
        retry_sleep:   Seconds to wait between retries.
        kill_stale:    If True, kill leftover `comunian` TraceWin processes before
                       each simulation (prevents resource conflicts).
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str,
        timeout: float = 180.0,
        retries: int = 2,
        retry_sleep: float = 5.0,
        kill_stale: bool = True,
    ):
        # Initialize the TraceWinSimulator with the given parameters.
        project_path = Path(project_file).resolve()
        self.project_file = str(project_path)
        self.calc_dir     = str(Path(calc_dir).resolve())
        self.timeout      = float(timeout)
        self.retries      = int(retries)
        self.retry_sleep  = float(retry_sleep)
        self.kill_stale   = bool(kill_stale)
        self._source_project_dir = str(project_path.parent)
        self._project_filename = project_path.name
        self._sim_count   = 0

        # if the project file does not exist, raise a FileNotFoundError.
        if not Path(self.project_file).exists():
            raise FileNotFoundError(
                f"TraceWin project file not found: {self.project_file}"
            )
        
        # Create the calculation directory with open permissions for comunian.
        Path(self.calc_dir).mkdir(parents=True, exist_ok=True)
        os.chmod(self.calc_dir, 0o777)


    def simulate(self, params: Optional[Dict[str, float]] = None) -> BeamSimulationResult:
        """Run one TraceWin simulation.

        Args:
            params: {STAGE_PARAM_KEYS: value} dict. {"ele[2][5]": 0.365663,"ele[4][5]": 0.168963, ...}
            if None, fall back to the default values defined in adige.py.

        Returns:
            BeamSimulationResult with beam states at all STAGE_MARKERS, score, and TraceWin metadata.
        """
        
        # use default parameters if none are provided
        full_params = default_params()
        # update the default parameters with the provided parameters
        if params:
            full_params.update(params)

        # Kill any leftover TraceWin processes if kill_stale is True
        if self.kill_stale:
            _kill_stale_tracewin_processes()

        # Increment the simulation count and initialize last_exc to None
        self._sim_count += 1
        last_exc: Optional[Exception] = None

        # try to run the simulation using _run_once(), retrying up to self.retries times if it fails 
        # (if retris = 2 it will try 3 times in total. first attempt + 2 retries   )
        # if ok return the result
        for attempt in range(self.retries + 1):
            try:
                return self._run_once(full_params)
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    print(f"  [TraceWin] attempt {attempt+1} failed: {exc}  — retrying in {self.retry_sleep}s")
                    time.sleep(self.retry_sleep)

        # If we reach here, return a failed result 
        return self._failed_result(full_params, str(last_exc))

    @property
    def n_simulations(self) -> int:
        """Return the number of simulations run so far."""
        return self._sim_count

    # Internal methods to run TraceWin correctly in the workspace with user comunian.
    # ---------
    def _reset_calc_dir(self):
        """Delete and recreate calc_dir with open permissions for comunian."""
        if os.path.exists(self.calc_dir):
            shutil.rmtree(self.calc_dir, ignore_errors=True)
        os.makedirs(self.calc_dir, exist_ok=True)
        os.chmod(self.calc_dir, 0o777)

    def _prepare_workspace(self) -> str:
        """Make the project workspace ready for TraceWin and return its .ini path.

        TraceWin runs directly on the original project workspace (shared,
        network-mounted). The workspace is made world-writable so `comunian`
        can use it, and any artifact left over from a previous run is removed
        first so TraceWin always starts from a clean state.
        """
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

        # Reset the calculation directory and prepare the project workspace for TraceWin.
        self._reset_calc_dir()
        runtime_project_file = self._prepare_workspace()

        try:
            # Run TraceWin with the given parameters and timeout.
            tw = TraceWin(runtime_project_file, self.calc_dir)
            success = tw.run(timeout=self.timeout, elem_params=params)

            # If TraceWin failed, raise an exception.
            if not success:
                raise RuntimeError(
                    f"TraceWin failed.\n"
                    f"  stdout: {tw.last_stdout[:300]}\n"
                    f"  stderr: {tw.last_stderr[:300]}"
                )

            # Extract the results from TraceWin and compute the score final.
            df = tw.results()
            beam_states, final_beam = self._extract_beam_states(df)
            score_val = score(final_beam)

            # Return the simulation result.
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
                },
            )
        finally:
            # TraceWin writes generated files (.cal, *_new.ini) straight into the shared workspace. Remove them now so the workspace is always left clean, whether the run succeeded or raised.
            self._clean_runtime_project_artifacts(Path(self._source_project_dir))

    # Internal methods for building the BeamSimulationResult from TraceWin output.
    # ------------
    def _extract_beam_states(self, df):
        """Extract beam states at each STAGE_MARKER from partran1.out.

        Returns:
            beam_states: (N_stages, 9) float32 ndarray
            final_beam:  dict of feature name → float 
        """
        # Build (N_stages, 9) array by matching element indices in the ## column
        n = len(STAGE_MARKERS)
        beam_states = np.zeros((n, len(BEAM_STATE_FEATURES)), dtype=np.float32)

        for si, marker in enumerate(STAGE_MARKERS):
            hits = df[df["##"] == _normalize_tracewin_marker(marker)]
            if hits.empty:
                # Fuzzy fallback: accept values within ±0.5 of the marker
                hits = df[(df["##"] - marker).abs() < 0.5]
            if hits.empty:
                continue
            row = hits.iloc[-1]
            for vi, var in enumerate(BEAM_STATE_FEATURES):
                beam_states[si, vi] = _read_beam_feature_from_tracewin_row(row, var)

        # Final beam state from last row
        last = df.iloc[-1]
        final_beam = {
            var: _read_beam_feature_from_tracewin_row(last, var)
            for var in BEAM_STATE_FEATURES
        }

        return beam_states, final_beam

    def _failed_result(self, params: Dict[str, float], error: str) -> BeamSimulationResult:
        """ define the BeamSimulationResult to return in case of failure. """
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
            },
        )


# Helpers 
#--------------
def _normalize_tracewin_marker(value) -> int | float:
    """Normalize a TraceWin marker value before matching the partran1.out rows."""
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return value


def _read_beam_feature_from_tracewin_row(row, feature: str) -> float:
    """Read one beam-state feature from a TraceWin output DataFrame row."""
    if feature == "npart_ratio":
        npart = float(row.get("npart", 0.0))
        return npart / INITIAL_NPART if INITIAL_NPART > 0 else 0.0
    return float(row.get(feature, 0.0))


def _kill_stale_tracewin_processes():
    """Kill any leftover TraceWin processes running as `comunian`.

    Prevents resource conflicts when a previous simulation timed out and left
    a TraceWin process still running inside the SSH session.
    """
    TraceWin._kill_remote_tracewin_processes()
