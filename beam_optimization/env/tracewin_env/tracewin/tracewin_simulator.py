"""
It is a class that maps ADIGE parameters to a BeamSimulationResult using TraceWin via SSH to run the program with permission of the user comunian. 
Note:
pyTraceWin_wrapper runs TraceWin via SSH as `comunian@localhost` (license requirement).
Each call to simulate() resets the calc_dir and launches a fresh process.
See TRACEWIN_SETUP.md for system-level prerequisites.
"""
from __future__ import annotations

import os
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .pyTraceWin_wrapper import TraceWin
from beam_optimization.env.simulation import BeamSimulationResult, BeamSimulator

from beam_optimization.config.adige import (
    BEAM_STATE_FEATURES, STAGE_MARKERS, INITIAL_NPART,
    default_params, score,
)

ERROR_SCORE = -999.0


# Legacy name kept for compatibility with existing imports.
SimResult = BeamSimulationResult



class TraceWinSimulator(BeamSimulator):
    """Run TraceWin simulations and return structured BeamSimulationResult objects.

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
        use_local_project_cache:
                       If True, TraceWin receives a local copy of the project
                       workspace, while calc_dir stays where the caller asked.
                       This avoids TraceWin writing generated project files into
                       network-mounted workspaces.
        local_project_cache_root:
                       Root folder for the local project cache.
    """

    def __init__(
        self,
        project_file: str,
        calc_dir: str,
        timeout: float = 180.0,
        retries: int = 2,
        retry_sleep: float = 5.0,
        kill_stale: bool = True,
        use_local_project_cache: bool = True,
        local_project_cache_root: Optional[str] = None,
    ):
        # Initialize the TraceWinSimulator with the given parameters. 
        project_path = Path(project_file).resolve()
        self.project_file = str(project_path)
        self.calc_dir     = str(Path(calc_dir).resolve())
        self.timeout      = float(timeout)
        self.retries      = int(retries)
        self.retry_sleep  = float(retry_sleep)
        self.kill_stale   = bool(kill_stale)
        self.use_local_project_cache = bool(use_local_project_cache)
        self.local_project_cache_root = str(
            Path(local_project_cache_root or "/tmp/tracewin_project_cache").resolve()
        )
        self._source_project_dir = str(project_path.parent)
        self._project_filename = project_path.name
        self._runtime_project_file = self.project_file
        self._runtime_project_dir = self._source_project_dir
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

    #Internal methods for use tracewin (Nota questa parte è dovuta solo perche ho avuto probleme nel runnare tracewin con utente comunian)
    # ------------
    #  simulate() calls _run_once()
    #  _run_once() calls _reset_calc_dir() and _prepare_runtime_project() which retunrn the file .ini in the cache to run using TraceWin wrapper.
    #  prepare_runtime_project() calls 
    #     - _local_cache_dir() decide the local cache directory to use for containing the project workspace copy
    #     - _project_signature() compute a hash describing the current project workspace state (to decide whether the cached project workspace is still valid, otherwise it will rebuild the cache)
    #     - _copy_project_workspace() copy the project workspace into the local cache.
    #     - _should_copy_project_path() decide what should be copied to the cache.
    #     - _clean_runtime_project_artifacts() remove generated files from the cached
    #     - _make_world_accessible() let the comunian SSH user read/write the local project cache.

    def _reset_calc_dir(self):
        """Delete and recreate calc_dir with open permissions for comunian."""
        if os.path.exists(self.calc_dir):
            shutil.rmtree(self.calc_dir, ignore_errors=True)
        os.makedirs(self.calc_dir, exist_ok=True)
        os.chmod(self.calc_dir, 0o777)

    def _prepare_runtime_project(self) -> str:
        """Return the project file that TraceWin should open for this run.

        TraceWin can create or modify files inside the project workspace when
        it runs. To keep the original workspace untouched, this method can copy
        the whole project workspace into a local cache and return the cached
        project ``.ini`` file that is used to run TraceWin.  
        
        Note that however the simulation outputs are still written to ``self.calc_dir`` correctly because it has comunian permissions.
        """
        
        # If the cache is disabled, TraceWin opens the original project file.
        if not self.use_local_project_cache:
            self._runtime_project_file = self.project_file
            self._runtime_project_dir = self._source_project_dir
            return self._runtime_project_file

        # Otherwise, identify the cache folder and compute the current project signature.
        source_dir = Path(self._source_project_dir)
        cache_dir = self._local_cache_dir()
        signature = self._project_signature(source_dir)
        signature_file = cache_dir / ".tracewin_cache_signature"

        # Reuse the cache only if it exists and its saved signature still matches.
        cache_is_current = (
            cache_dir.exists()
            and (cache_dir / self._project_filename).exists()
            and signature_file.exists()
            and signature_file.read_text(encoding="utf-8").strip() == signature
        )

        # If the original workspace changed, rebuild the cached copy from scratch.
        if not cache_is_current:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._copy_project_workspace(source_dir, cache_dir)
            signature_file.write_text(signature, encoding="utf-8")

        # TraceWin runs as comunian, so the cached workspace must be writable by that user.
        self._make_world_accessible(cache_dir)
        self._clean_runtime_project_artifacts(cache_dir)

        # Return only the cached .ini path: TraceWin receives this file as entrypoint.
        self._runtime_project_dir = str(cache_dir)
        self._runtime_project_file = str((cache_dir / self._project_filename).resolve())
        return self._runtime_project_file

    def _local_cache_dir(self) -> Path:
        """Return the local cache directory for this TraceWin project.

        The directory name is built from:
        - the project file stem, for readability;
        - a short hash of the full project path, to avoid collisions between
          projects with the same filename in different folders.

        Example:
            ``/path/to/condensed.ini`` ->
            ``/tmp/tracewin_project_cache/condensed_<hash>``
        """
        digest = hashlib.sha1(self.project_file.encode("utf-8")).hexdigest()[:12]
        stem = Path(self.project_file).stem
        return Path(self.local_project_cache_root) / f"{stem}_{digest}"

    def _project_signature(self, source_dir: Path) -> str:
        """Return a hash describing the current project workspace state.

        The signature is used to decide whether the cached project workspace is
        still valid. 

        If any relevant project file is added, removed, renamed, resized, or
        modified, this signature changes and the cache is rebuilt.
        """
        h = hashlib.sha1()
        for path in sorted(source_dir.rglob("*")):
            # The signature must describe the same files that would be copied.
            if path.is_dir() or not self._should_copy_project_path(path, source_dir):
                continue
            stat = path.stat()
            rel = path.relative_to(source_dir).as_posix()

            # Relative path + size + mtime are enough to detect relevant cache changes.
            h.update(rel.encode("utf-8"))
            h.update(str(stat.st_size).encode("ascii"))
            h.update(str(stat.st_mtime_ns).encode("ascii"))
        return h.hexdigest()

    def _copy_project_workspace(self, source_dir: Path, cache_dir: Path) -> None:
        """Copy the project workspace into the local cache.

        Only files accepted by ``_should_copy_project_path`` are copied. This
        keeps the cached workspace focused on project inputs and avoids copying
        old TraceWin outputs or generated files.
        """
        for path in sorted(source_dir.rglob("*")):
            if not self._should_copy_project_path(path, source_dir):
                continue
            rel = path.relative_to(source_dir)
            dst = cache_dir / rel
            if path.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)

    def _should_copy_project_path(self, path: Path, source_dir: Path) -> bool:
        """Return True if a workspace path should be copied to the cache.

        The cache should contain the project inputs, not previous simulation
        outputs or generated artifacts.

        Skipped paths:
        - output/cache folders such as ``calc`` and ``calc_env_dashboard``;
        - Python cache folders;
        - TraceWin generated ``.cal`` files;
        - generated ``*_new.ini`` files, except when that file is the selected
          project file itself.
        """
        rel_parts = path.relative_to(source_dir).parts
        if any(part in {"calc", "calc_env_dashboard", "__pycache__"} for part in rel_parts):
            return False

        if path.is_dir():
            return True

        name = path.name
        if name == ".tracewin_cache_signature":
            return False
        if name.endswith(".cal"):
            return False
        if name.endswith("_new.ini") and name != self._project_filename:
            return False
        return True

    def _clean_runtime_project_artifacts(self, cache_dir: Path) -> None:
        """Remove generated files from the cached project before each run."""
        for path in cache_dir.iterdir():
            if path.name == self._project_filename:
                continue
            if path.name.endswith(".cal") or path.name.endswith("_new.ini"):
                path.unlink(missing_ok=True)

    def _make_world_accessible(self, root: Path) -> None:
        """Let the comunian SSH user read/write the local project cache."""
        os.chmod(root, 0o777)
        for path in root.rglob("*"):
            try:
                os.chmod(path, 0o777 if path.is_dir() else 0o666)
            except OSError:
                pass

    def _run_once(self, params: Dict[str, float]) -> BeamSimulationResult:
        """Run one TraceWin simulation and return the results."""
        
        # Reset the calculation directory and prepare the runtime project file for TraceWin (copy the project file and obtain the runtime project file).
        self._reset_calc_dir()
        runtime_project_file = self._prepare_runtime_project()

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
                "runtime_project_file": self._runtime_project_file,
                "runtime_project_dir": self._runtime_project_dir,
                "use_local_project_cache": self.use_local_project_cache,
                "calc_dir": self.calc_dir,
                "sim_count": self._sim_count,
            },
        )
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
                "runtime_project_file": self._runtime_project_file,
                "runtime_project_dir": self._runtime_project_dir,
                "use_local_project_cache": self.use_local_project_cache,
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
    try:
        result = subprocess.run(
            ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
             "comunian@localhost",
             "pkill -u comunian -x TraceWin || true; "
             "pkill -u comunian -f '[x]vfb-run.*TraceWin' || true"],
            timeout=10,
            capture_output=True,
        )
    except Exception:
        pass  # Non-critical — if SSH fails the main simulation will also fail
